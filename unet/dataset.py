import numpy as np
import torch
import random
from torch.utils.data import Dataset
import os
import cv2

class CustomDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.images = os.path.join(self.path, 'image')
        self.labels = os.path.join(self.path, 'indexLabel')
        
        self.images_path = [f for f in os.listdir(self.images) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        self.labels_path = [f for f in os.listdir(self.labels) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        
        # 定义标签映射
        self.label_mapping = {
            0: 0,   # unlabelled
            1: 6,   # asphalt -> other-terrain
            2: 2,   # dirt
            3: 3,   # mud
            4: 4,   # water
            5: 5,   # gravel
            6: 6,   # other-terrain
            7: 7,   # tree-trunk
            8: 8,   # tree-foliage
            9: 9,   # bush
            10: 10, # fence
            11: 11, # structure
            12: 16, # pole -> other-object
            13: -1, # vehicle (to be removed)
            14: 14, # rock
            15: 15, # log
            16: 16, # other-object
            17: 17, # sky
            18: 18, # grass
        }

    def __getitem__(self, index):
        image_path = os.path.join(self.images, self.images_path[index])
        label_path = os.path.join(self.labels, self.labels_path[index])
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} does not exist.")
        
        frame = cv2.imread(image_path, 1)
        if frame is None:
            raise ValueError(f"Failed to read image from {image_path}.")
        
        frame = frame.astype("float32")  # 作为彩色图片读取
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0  # 更改数据通道为RGB并归一化

        label = cv2.imread(label_path, 0)  # 作为灰度图读取
        if label is None:
            raise ValueError(f"Failed to read label from {label_path}.")
        
        image, label = self.RandomCrop(frame, label)  # 图片太大，随机裁剪

        image = torch.tensor(image).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.int64)

        # 标签映射
        label = self.map_labels(label)

        return image, label

    def map_labels(self, label):
        mapped_label = torch.zeros_like(label)
        for src, dst in self.label_mapping.items():
            if dst == -1:
                continue  # 跳过需要去除的标签
            mapped_label[label == src] = dst

        # 重新排序标签，去除重复的类
        unique_labels = torch.unique(mapped_label)
        new_label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        for old_label, new_label in new_label_mapping.items():
            mapped_label[mapped_label == old_label] = new_label

        return mapped_label

    def RandomCrop(self, img, seg):
        # 随机裁剪
        height, width, c = img.shape
        h = 256
        w = 256
        x = random.uniform(width / 4, 3 * width / 4)
        y = random.uniform(height / 4, 3 * height / 4)

        # 左上角
        crop_xmin = np.clip(x - (w / 2), a_min=0, a_max=width).astype(np.int32)
        crop_ymin = np.clip(y - (h / 2), a_min=0, a_max=height).astype(np.int32)
        # 右下角
        crop_xmax = np.clip(x + (w / 2), a_min=0, a_max=width).astype(np.int32)
        crop_ymax = np.clip(y + (h / 2), a_min=0, a_max=height).astype(np.int32)

        cropped_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
        cropped_seg = seg[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        return cropped_img, cropped_seg

    def __len__(self):
        return len(self.images_path)



# import numpy as np
# import torch
# import random
# from torch.utils.data import Dataset
# import os
# import cv2

# class dataset(Dataset):
#     def __init__(self, path):

#         self.path=path

#         self.images=self.path+'/image'
#         self.labels=self.path+'/indexLabel'


#         self.images_path = os.listdir(self.images)
#         self.labels_path = os.listdir(self.labels)

#     def __getitem__(self, index):

#         frame=cv2.imread(self.images+'/'+self.images_path[index],1).astype("float32")#作为彩色图片读取
#         frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255.0#由于opencv读取数据通道顺序默认为BGR，所以更改数据通道为BGR，然后再降数据归一化

#         label=cv2.imread(self.labels+'/'+self.labels_path[index],0)#读取label值,作为灰度图读取

#         image,label=self.RandomCrop(frame,label)#图片太大，给随机切分一下


#         image = torch.tensor(image).permute(2,0,1)
#         label = torch.tensor(label,dtype=torch.int64)-1#由于是从1-18,需要给转换为0-17

#         # 标签映射，将16和17映射到有效标签15
#         label = torch.where(label == 16, 15, label)
#         label = torch.where(label == 17, 15, label)
#         return image,label

#     def RandomCrop(self,img, seg):
#         # 随机裁剪
#         height, width, c = img.shape
#         # x,y代表裁剪后的图像的中心坐标，h,w表示裁剪后的图像的高，宽，由于Unet的特性，最好给弄成2的整数
#         h = 256
#         w = 256
#         x = random.uniform(width / 4, 3 * width / 4)
#         y = random.uniform(height / 4, 3 * height / 4)

#         # 左上角
#         crop_xmin = np.clip(x - (w / 2), a_min=0, a_max=width).astype(np.int32)
#         crop_ymin = np.clip(y - (h / 2), a_min=0, a_max=height).astype(np.int32)
#         # 右下角
#         crop_xmax = np.clip(x + (w / 2), a_min=0, a_max=width).astype(np.int32)
#         crop_ymax = np.clip(y + (h / 2), a_min=0, a_max=height).astype(np.int32)

#         cropped_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
#         cropped_seg = seg[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

#         return cropped_img, cropped_seg

#     def __len__(self):
#         return len(os.listdir(self.images))

