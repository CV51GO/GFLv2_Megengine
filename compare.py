import os
import time
import datetime
from pathlib import Path
import sys
sys.path.append('./official_GFLV2/GFocalV2')
import numpy as np
import cv2

def _scale_size(size, scale):
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)

def rescale_size(old_size, scale, return_scale=False):
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size

# 图片路径
img_root = './data'
imgname_list = os.listdir(img_root)


# 官方模型初始化
from mmdet.models import build_detector
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
import torch

config_file = './official_GFLV2/GFocalV2/configs/gfocal/gfocal_r50_fpn_1x.py'
checkpoint_file = './official_GFLV2/GFocalV2/gfocal_r50_fpn_1x.pth'
cfg = Config.fromfile(config_file)
model_torch = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model_torch, checkpoint_file, map_location='cpu')
model_torch = MMDataParallel(model_torch, device_ids=[0])
model_torch.eval()


# megengine模型初始化
import megengine
import megengine.functional as F
from GFLv2 import get_Megengine_GFLv2_model
model_megengine = get_Megengine_GFLv2_model(pretrained=True)
model_megengine.eval()

for imgname in imgname_list:
    print(f'inference {imgname}')
    # 1.读取图片
    img_path = str(Path(img_root)/imgname)
    img = cv2.imread(img_path)

    # 2.预处理
    # 2.1 resize
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale=(1333, 800), return_scale=True)
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    resized_img = resized_img.copy().astype(np.float32)
    new_h, new_w = resized_img.shape[:2]
    w_scale = new_w / w
    h_scale = new_h / h
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                            dtype=np.float32)
    # 2.2 归一化
    mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
    std = np.array([58.395, 57.12 , 57.375], dtype=np.float32)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB, resized_img)  # inplace
    cv2.subtract(resized_img, mean, resized_img)  # inplace
    cv2.multiply(resized_img, stdinv, resized_img)  # inplace   
    # 2.3 padding
    divisor = 32
    pad_h = int(np.ceil(resized_img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(resized_img.shape[1] / divisor)) * divisor
    padding = (0, 0, pad_w - resized_img.shape[1], pad_h - resized_img.shape[0])
    padding_img = cv2.copyMakeBorder(
        resized_img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        cv2.BORDER_CONSTANT,
        value=0)
    # 3.官方推理：
    data = dict()
    net_input_torch = torch.tensor(padding_img.transpose(2, 0, 1)).unsqueeze(0)
    data['img'] = [net_input_torch]
    data['img_metas'] = [[{'img_shape': resized_img.shape, 
                        'scale_factor': scale_factor, 
                        }]]
    torch_start_time = time.time()
    with torch.no_grad():
        result = model_torch(return_loss=False, rescale=True, **data)
    torch_end_time = time.time()
    print("torch inference time: {:.3f}s".format(torch_end_time-torch_start_time))
    # print(result)

    # 4.Megengine推理
    net_input_megengine =(F.expand_dims(megengine.tensor(padding_img.transpose(2, 0, 1)), 0))
    megengine_start_time = time.time()
    out = model_megengine(net_input_megengine, data['img_metas'][0])
    megengine_end_time = time.time()
    print("megengine inference time: {:.3f}s".format(megengine_end_time-megengine_start_time))


    # 5.比较官方推理和Megengine推理的结果
    for torch_out, megengine_out in zip(result[0], out[0]):
        np.testing.assert_allclose(torch_out, megengine_out, rtol=1e-3)
    # print(out)
    print('pass!')
