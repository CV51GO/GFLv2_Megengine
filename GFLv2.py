import numpy as np
import megengine
import megengine.module as M
import megengine.hub as hub

from ResNet import ResNet
from FPN import FPN
from GFocalHead import GFocalHead

def bbox2result(bboxes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, megengine.Tensor):
            bboxes = bboxes.numpy()
            labels = labels.numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

class GFLv2(M.Module):
    def __init__(self):
        super(GFLv2, self).__init__()

        self.backbone = ResNet(depth=50,
                               num_stages=4,
                               out_indices=(0, 1, 2, 3),
                               frozen_stages=1,
                               norm_cfg=dict(type='BN', requires_grad=True),
                               norm_eval=True)
        self.neck = FPN(in_channels=[256, 512, 1024, 2048],
                        out_channels=256,
                        start_level=1,
                        add_extra_convs='on_output',
                        num_outs=5)
        self.bbox_head = GFocalHead(num_classes=80,
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
                                test_cfg=dict(nms_pre=1000,
                                              min_bbox_size=0,
                                              score_thr=0.05,
                                              nms=dict(type='nms', iou_threshold=0.6),
                                              max_per_img=100))


    def forward(self, img, img_metas):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, 
                                               img_metas, 
                                               rescale=True)
        bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]

        return bbox_results

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x

@hub.pretrained(
"https://studio.brainpp.com/api/v1/activities/3/missions/52/files/d2ce9780-da58-46c1-a509-9ecd0cdbff0d"
)
def get_Megengine_GFLv2_model():
    model_megengine = GFLv2()
    # model_megengine.load_state_dict(megengine.load('./megengine_GFLv2.pkl'))
    return model_megengine
