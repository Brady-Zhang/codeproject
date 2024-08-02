## Unet and deeplabv3+ Convolutional Networks for WildScenes dataset 2D image Segmentation in Pytorch
---
### Table of Contents
1. [Introduction](#Introduction)
2. [Related code](#Relatedcode)
3. [Performance](#Performance)
4. [Environment](#Environment)
5. [Download](#Download)
6. [How2train](#How2train)
7. [miou](#miou)
8. [Reference](#Reference)

## Introduction
**This repo include five models used for image Segmentation, not include pspnet.**  
| models |
| :----- |
U-NET (WITH GHOSTCONV AND EMA) |  
U-NET (VGG) |
U-NET (RESNET50) |
DEEPLABV3+(XCEPTION) |
DEEPLABV3+(MOBILENET) |

## Related code
| model | path |
| :----- | :----- |
Unet | https://github.com/Brady-Zhang/codeproject/unet  
PSPnet | https://github.com/Brady-Zhang/codeproject
deeplabv3+ | https://github.com/Brady-Zhang/codeproject

### 性能情况
**unet并不适合VOC此类数据集，其更适合特征少，需要浅层特征的医药数据集之类的。**
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| VOC12+SBD | [unet_vgg_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_vgg_voc.pth) | VOC-Val12 | 512x512| 58.78 | 
| VOC12+SBD | [unet_resnet_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_resnet_voc.pth) | VOC-Val12 | 512x512| 67.53 | 
