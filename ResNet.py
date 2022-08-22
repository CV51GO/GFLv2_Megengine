import megengine
import megengine.module as M
import numpy as np


def build_conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    return M.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

def build_norm_layer(num_features):
    layer = M.BatchNorm2d(num_features, eps=1e-5)
    return layer

class ResLayer(M.Sequential):

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    M.AvgPool2d(
                        kernel_size=stride,
                        stride=stride))
            downsample.extend([
                build_conv_layer(
                    in_channels=inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(planes * block.expansion)
            ])
            downsample = M.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)

class Bottleneck(M.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):

        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.bn1 = build_norm_layer(planes)
        self.bn2 = build_norm_layer(planes)
        self.bn3 = build_norm_layer(planes * self.expansion)

        self.conv1 = build_conv_layer(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        
        self.conv2 = build_conv_layer(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            bias=False)
       
        self.conv3 = build_conv_layer(
            in_channels=planes,
            out_channels=planes * self.expansion,
            kernel_size=1,
            bias=False)

        self.relu = M.ReLU()
        self.downsample = downsample

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out

        out = _inner_forward(x)
        out = self.relu(out)
        return out

class ResNet(M.Module):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True):

        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.frozen_stages = frozen_stages
        self.avg_down = avg_down
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)


        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                avg_down=self.avg_down,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            setattr(self, layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)


    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = build_conv_layer(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1= build_norm_layer(stem_channels)
        self.relu = M.ReLU()
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

if __name__ == "__main__":
    net = ResNet(depth=50,   
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,)
    input = megengine.Tensor(np.random.randn(1, 3, 224, 224))
    output = net(input)
    print(net)
    print(type(output))
    print([item.shape for item in output])


