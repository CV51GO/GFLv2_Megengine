## 介绍

本项目是论文《Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection》的Megengine实现。该论文的官方实现地址：https://github.com/implus/GFocalV2

## 环境安装

依赖于CUDA10

```
conda create -n GFLv2 python=3.7
pip install -r requirements.txt
```

下载官方的权重gfocal_r50_fpn_1x.pth：https://drive.google.com/file/d/1wSE9-c7tcQwIDPC6Vm_yfOokdPfmYmy7/view?usp=sharing
，将下载后的文件置于./official_GFLV2/GFocalV2/路径下。 

## 使用方法

安装完环境后，直接运行`python compare.py`。

`compare.py`文件对官方实现和Megengine实现的推理结果进行了对比。

运行`compare.py`时，会读取`./data`中存放的图片进行推理。`compare.py`中实现了Megengine框架和官方使用的Pytorch框架的推理，并判断两者推理结果的一致性。

## 模型加载示例

在使用模型时，使用如下代码即可加载模型和权重：
```python
import megengine.hub as hub
megengine_model = hub.load('CV51GO/GFLv2_Megengine','get_Megengine_GFLv2_model',pretrained=True)
```

