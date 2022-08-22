import collections
from itertools import repeat
from functools import partial
import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
import megengine.functional.nn as nn

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse
_pair = _ntuple(2)

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = F.clip(x1, lower=0, upper=max_shape[1])
        y1 = F.clip(y1, lower=0, upper=max_shape[0])
        x2 = F.clip(x2, lower=0, upper=max_shape[1])
        y2 = F.clip(y2, lower=0, upper=max_shape[0])
    return F.stack([x1, y1, x2, y2], -1)

def nms(boxes, scores, iou_threshold):
    inds = nn.nms(boxes, scores, iou_threshold)
    dets = F.concat((boxes[inds], scores[inds].reshape(-1, 1)), axis=1)
    return dets, inds

def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.astype(np.float32) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    if len(boxes_for_nms) < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        total_mask = F.zeros(scores.shape, dtype=np.bool)
        for id in megengine.tensor(np.unique(idxs.numpy())):
            mask = megengine.tensor(((idxs == id).numpy().nonzero())[0]).reshape(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
        keep = megengine.tensor(total_mask.numpy().nonzero()[0]).reshape(-1)
        keep = keep[F.argsort(scores[keep], descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]
    return F.concat([boxes, scores[:, None]], -1), keep

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    num_classes = multi_scores.shape[1] - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = F.broadcast_to(multi_bboxes[:, None], (multi_scores.shape[0], num_classes, 4))
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[F.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)].reshape(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = megengine.tensor(valid_mask.numpy().nonzero()[1])

    if bboxes.size == 0:
        bboxes = F.zeros((0, 5))
        labels = F.zeros((0, ), dtype=np.int32)
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


class Scale(M.Module):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = megengine.Parameter(megengine.tensor(scale, dtype=np.float32))

    def forward(self, x):
        return x * self.scale

class Integral(M.Module):
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.project = F.linspace(0, self.reg_max, self.reg_max + 1)

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), axis=1)
        x = F.linear(x, self.project.reshape(1,-1)).reshape(-1,4)
        return x

def build_conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    return M.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

class ConvModule(M.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        conv_padding = padding

        self.conv = build_conv_layer(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=conv_padding, 
            bias=bias)

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.groups = self.conv.groups
        
        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels

            gn = M.GroupNorm(
                num_groups=norm_cfg['num_groups'],
                num_channels=norm_channels
            ) 
            self.norm_name = 'gn'
            setattr(self, self.norm_name, gn)

        if self.with_activation:
            self.activate = M.ReLU()
    
    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x



class AnchorGenerator(object):

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
   
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        scales = octave_scales * octave_base_scale
        self.scales = megengine.Tensor(scales)
      
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = megengine.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return [base_anchors.shape[0] for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        return len(self.strides)

    def gen_base_anchors(self):

        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center
        h_ratios = F.sqrt(ratios)

        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).reshape(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).reshape(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).reshape(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).reshape(-1)

        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = F.stack(base_anchors, axis=-1)
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = F.repeat(x.reshape(1 ,-1), len(y), axis=0).reshape(-1)
        yy = F.repeat(y, len(x))
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_sizes, device='cuda'):
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        feat_h, feat_w = featmap_size
        feat_h = int(feat_h)
        feat_w = int(feat_w)

        shift_x = F.arange(0, feat_w, device=device) * stride[0]
        shift_y = F.arange(0, feat_h, device=device) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = F.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors

class GFocalHead(M.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_max=16,
                 reg_topk=4,
                 reg_channels=64,
                 add_mean=True,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                test_cfg = dict(
                        nms_pre=1000,
                        min_bbox_size=0,
                        score_thr=0.05,
                        nms=dict(type='nms', iou_threshold=0.6),
                        max_per_img=100)):
        super(GFocalHead, self).__init__()
        self.stacked_convs = stacked_convs
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        if add_mean:
            self.total_dim += 1
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.cls_out_channels = num_classes + 1
        self.test_cfg = test_cfg
        anchor_generator.pop('type')
        self.anchor_generator = AnchorGenerator(**anchor_generator)
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()
        self.integral = Integral(self.reg_max)

    def _init_layers(self):
        self.relu = M.ReLU()
        self.cls_convs = list()
        self.reg_convs = list()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = M.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = M.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = list(
            [Scale(1.0) for _ in self.anchor_generator.strides])
        conf_vector = [M.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [M.Conv2d(self.reg_channels, 1, 1), M.Sigmoid()]
        self.reg_conf = M.Sequential(*conf_vector)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat))
        N, C, H, W = bbox_pred.shape
        prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max+1, H, W), axis=2)
        prob_topk_raw = F.topk(F.transpose(prob, (0,1,3,4,2)).reshape(-1, 17), self.reg_topk, descending=True)[0].reshape(prob.shape[0:2]+prob.shape[3:]+(self.reg_topk,))
        prob_topk = F.transpose(prob_topk_raw,(0,1,4,2,3))
        if self.add_mean:
            stat = F.concat([prob_topk, prob_topk.mean(axis=2, keepdims=True)], axis=2)
        else:
            stat = prob_topk
        quality_score = self.reg_conf(stat.reshape(N, -1, H, W))
        cls_score = F.sigmoid(self.gfl_cls(cls_feat)) * quality_score
        return cls_score, bbox_pred

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride, anchors in zip(
                cls_scores, bbox_preds, self.anchor_generator.strides,
                mlvl_anchors):
            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]
            assert stride[0] == stride[1]
            scores = F.transpose(cls_score, (1,2,0)).reshape(
                -1, self.cls_out_channels)
            bbox_pred = F.transpose(bbox_pred, (1,2,0))
            bbox_pred = self.integral(bbox_pred) * stride[0]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores = F.max(scores, axis=1)
                _, topk_inds = F.topk(max_scores, nms_pre, descending=True)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = F.concat(mlvl_bboxes)

        if rescale:
            mlvl_bboxes /= megengine.tensor(scale_factor)

        mlvl_scores = F.concat(mlvl_scores)
        padding = F.zeros([mlvl_scores.shape[0]]+[1])
        mlvl_scores = F.concat([mlvl_scores, padding], axis=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg['score_thr'], cfg['nms'],
                                                    cfg['max_per_img'])
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores
    def anchor_center(self, anchors):
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return F.stack([anchors_cx, anchors_cy], axis=-1)



if __name__ == '__main__':
    gfocalhead = GFocalHead(        
                        num_classes=80,
                        in_channels=256,
                        stacked_convs=4,
                        feat_channels=256,
                        anchor_generator=dict(
                            type='AnchorGenerator',
                            ratios=[1.0],
                            octave_base_scale=8,
                            scales_per_octave=1,
                            strides=[8, 16, 32, 64, 128]),
                        reg_max=16,
                        reg_topk=4,
                        reg_channels=64,
                        add_mean=True,
                        )

    input = (megengine.Tensor(np.random.randn(1, 256, 100, 136)),
            megengine.Tensor(np.random.randn(1, 256, 50, 68)),
            megengine.Tensor(np.random.randn(1, 256, 25, 34)),
            megengine.Tensor(np.random.randn(1, 256, 13, 17)),
            megengine.Tensor(np.random.randn(1, 256, 7, 9)))

    output = gfocalhead(input)
    print(type(output))
    print([item.shape for item in output[0]])
    print([item.shape for item in output[0]])
