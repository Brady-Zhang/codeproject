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

## Relatedcode
| model | path |
| :----- | :----- |
U-NET (WITH GHOSTCONV AND EMA) | https://github.com/Brady-Zhang/codeproject 
U-NET (VGG) | https://github.com/Brady-Zhang/codeproject
U-NET (RESNET50) | https://github.com/Brady-Zhang/codeproject
DEEPLABV3+(XCEPTION) | https://github.com/Brady-Zhang/codeproject
DEEPLABV3+(MOBILENET) | https://github.com/Brady-Zhang/codeproject

### Performance
| training dataset | per-trained weight | testing dataset | input image size | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| 27% of wildscence 2d set | [no weight] | 3% of wildscence 2d set | 512x512| 15.41 | 
| 27% of wildscence 2d set | [unet_resnet_voc.pth](https://drive.google.com/file/d/1a_yvh3iUMYchsZqD9VX6fxdOk7CJ4xZM/view?usp=sharing) | 3% of wildscence 2d set | 512x512| 40.43 | 
| 27% of wildscence 2d set | [unet_vgg_voc.pth](https://drive.google.com/file/d/1a_yvh3iUMYchsZqD9VX6fxdOk7CJ4xZM/view?usp=sharing) | 3% of wildscence 2d set | 512x512| 42.22 | 
| 27% of wildscence 2d set | [deeplab_xception.pth](https://drive.google.com/file/d/1a_yvh3iUMYchsZqD9VX6fxdOk7CJ4xZM/view?usp=sharing) | 3% of wildscence 2d set | 512x512| 31.40 | 
| 27% of wildscence 2d set | [deeplab_mobilenetv2.pth](https://drive.google.com/file/d/1a_yvh3iUMYchsZqD9VX6fxdOk7CJ4xZM/view?usp=sharing) | 3% of wildscence 2d set | 512x512| 35.01 | 
### Environment
pytorch==2.1.1   
### Download
All per-trained weights: https://drive.google.com/file/d/1a_yvh3iUMYchsZqD9VX6fxdOk7CJ4xZM/view?usp=sharing
Selected dataset from Wildscence: https://drive.google.com/drive/folders/1KFc15zSS_cc2t_an2YYDZHhuhA7BS7MS?usp=sharing
