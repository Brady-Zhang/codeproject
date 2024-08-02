## Unet and deeplabv3+ Convolutional Networks for WildScenes dataset 2D image Segmentation in Pytorch
---
### Table of Contents
1. [Introduction](#Introduction)
2. [Related code](#Relatedcode)
3. [Performance](#Performance)
4. [Environment](#Environment)
5. [Download](#Download)
6. [How2train](#How2train)
7. [predictions](#predictions)
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
### How2train
#### data processing
1. download our selected five data folders from https://drive.google.com/drive/folders/1KFc15zSS_cc2t_an2YYDZHhuhA7BS7MS?usp=sharing in the path /codeproject/unet-pytorch-main/VOCdevkit/VOC2007

2. run dataspilt.ipynb(/codeproject/unet-pytorch-main/VOCdevkit/VOC2007/dataspilt.ipynb) to spilt the image into JPEGImages and indexlabel in SegmentationClass files.

3. you can choose run voc_annotation.py(/codeproject/unet-pytorch-main/voc_annotation.py) to spilt the data into training and testing set and get train.txt and val.txt two files(already contains the train.txt and val.txt). 

4. run sp.py(/codeproject/unet-pytorch-main/VOCdevkit/VOC2007/sp.py) to spilt the JPEGImages and SegmentationClass files into training and testing set according to the train.txt and val.txt.

#### training and testing 
##### U-NET (WITH GHOSTCONV AND EMA)
1. go to /codeproject/unet
2. simplely run main.py in /codeproject/unet/main.py and you will get the predict result as well.

##### U-NET (VGG) and U-NET (RESNET50)
1. go to the /codeproject/unet-pytorch-main
2. download per-trained weights: https://drive.google.com/file/d/1a_yvh3iUMYchsZqD9VX6fxdOk7CJ4xZM/view?usp=sharing
3. make unet_resnet_voc.pth and unet_vgg_voc.pth under model_data
4. go to the /codeproject/unet-pytorch-main/train.py, change the backbone to vgg or resnet 50
5. in /codeproject/unet-pytorch-main/train.py  change the model_path to model_data/unet_resnet_voc.pth or model_data/unet_vgg_voc.pth
6. run the train.py(/codeproject/unet-pytorch-main/train.py)
7. after training, go to /codeproject/unet-pytorch-main/unet.py, change the model_path to 'logs/best_epoch_weights.pth' which is the best weight we get from training
8. go to /codeproject/unet-pytorch-main/get_miou.py and run it, it can have the predictions and miou outputs under /codeproject/unet-pytorch-main/miou_out

##### DEEPLABV3+(XCEPTION) and DEEPLABV3+(MOBILENET)
1. go to the /codeproject/deeplabv3-plus-pytorch-main
2. download per-trained weights: https://drive.google.com/file/d/1a_yvh3iUMYchsZqD9VX6fxdOk7CJ4xZM/view?usp=sharing
3. make deeplab_xception_voc.pth and deeplab_mobilenetv2.pth under model_data
4. go to the /codeproject/deeplabv3-plus-pytorch-main/train.py, change the backbone to xception or mobilenet
5. in /codeproject/deeplabv3-plus-pytorch-main/train.py  change the model_path to model_data/deeplab_xception_voc.pth or model_data/deeplab_mobilenetv2.pth
6. run the train.py(/codeproject/deeplabv3-plus-pytorch-main/train.py)
7. after training, go to /codeproject/deeplabv3-plus-pytorch-main/deeplab.py, change the model_path to 'logs/best_epoch_weights.pth' which is the best weight we get from training
8. go to /codeproject/deeplabv3-plus-pytorch-main/get_miou.py and run it, it can have the predictions and miou outputs under /codeproject/deeplabv3-plus-pytorch-main/miou_out


## predictions
--------------------------actual indexlabel --------------------------------------------------- prediction----------------------------
U-NET (VGG)
![trainvis](4981722578630_.pic.jpg)

## Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus
https://github.com/bubbliiiing/unet-pytorch



















