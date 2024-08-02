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
### How2train
#### data processing
1. download our selected five data folders from https://drive.google.com/drive/folders/1KFc15zSS_cc2t_an2YYDZHhuhA7BS7MS?usp=sharing in the path /codeproject/unet-pytorch-main/VOCdevkit/VOC2007

2. run dataspilt.ipynb(/codeproject/unet-pytorch-main/VOCdevkit/VOC2007/dataspilt.ipynb) to spilt the image into JPEGImages and indexlabel in SegmentationClass files.

3. you can choose run voc_annotation.py(/codeproject/unet-pytorch-main/voc_annotation.py) to spilt the data into training and testing set and get train.txt and val.txt two files(already contains the train.txt and val.txt). 

4. run sp.py(/codeproject/unet-pytorch-main/VOCdevkit/VOC2007/sp.py) to spilt the JPEGImages and SegmentationClass files into training and testing set according to the train.txt and val.txt.


#### data processing
##### U-NET (WITH GHOSTCONV AND EMA)
1. simplely run main.py in /codeproject/unet/main.py



### 预测步骤
#### 一、使用预训练权重
##### a、VOC预训练权重
1. 下载完库后解压，如果想要利用voc训练好的权重进行预测，在百度网盘或者release下载权值，放入model_data，运行即可预测。  
```python
img/street.jpg
```    
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。    
##### b、医药预训练权重
1. 下载完库后解压，如果想要利用医药数据集训练好的权重进行预测，在百度网盘或者release下载权值，放入model_data，修改unet.py中的model_path和num_classes；
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_medical.pth',
    #--------------------------------#
    #   所需要区分的类的个数+1
    #--------------------------------#
    "num_classes"   : 2,
    #--------------------------------#
    #   所使用的的主干网络：vgg、resnet50   
    #--------------------------------#
    "backbone"      : "vgg",
    #--------------------------------#
    #   输入图片的大小
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"         : True,
    #--------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------#
    "cuda"          : True,
}
```
2. 运行即可预测。  
```python
img/cell.png
```
#### 二、使用自己训练的权重
1. 按照训练步骤训练。    
2. 在unet.py文件里面，在如下部分修改model_path、backbone和num_classes使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_voc.pth',
    #--------------------------------#
    #   所需要区分的类的个数+1
    #--------------------------------#
    "num_classes"   : 21,
    #--------------------------------#
    #   所使用的的主干网络：vgg、resnet50   
    #--------------------------------#
    "backbone"      : "vgg",
    #--------------------------------#
    #   输入图片的大小
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   blend参数用于控制是否
    #   让识别结果和原图混合
    #--------------------------------#
    "blend"         : True,
    #--------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------#
    "cuda"          : True,
}
```
3. 运行predict.py，输入    
```python
img/street.jpg
```   
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。    

### 评估步骤
1、设置get_miou.py里面的num_classes为预测的类的数量加1。  
2、设置get_miou.py里面的name_classes为需要去区分的类别。  
3、运行get_miou.py即可获得miou大小。  

## Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus




















