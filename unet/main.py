
from unet import U_Net
from dataset import CustomDataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# # 检查CUDA是否可用
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据加载器
train_dataloader = DataLoader(CustomDataset('train'), batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(CustomDataset('val'), batch_size=16, shuffle=True, num_workers=8,pin_memory=True)

# 初始化模型并将其移动到GPU
net = U_Net(img_ch=3, output_ch=16).to(device)  # 修改输出通道数为16
# 确认模型在GPU上
print(f"Model is on device: {next(net.parameters()).device}")
# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 定义训练的轮数
epochs = 100

# 定义实际的类别名称
class_names = {
    0: "background",
    1: "dirt",
    2: "mud",
    3: "water",
    4: "gravel",
    5: "other-terrain",
    6: "tree-trunk",
    7: "tree-foliage",
    8: "bush",
    9: "fence",
    10: "structure",
    11: "rock",
    12: "log",
    13: "other-object",
    14: "sky",
    15: "grass"
}

# IOU calculation function for each class
def IOU_per_class(pred, target, nclass=16):
    ious = {}
    pred = pred.view(-1)
    target = target.view(-1)
    for i in range(nclass):
        pred_ins = pred == i
        target_ins = target == i
        inser = (pred_ins & target_ins).sum().float()
        union = pred_ins.sum().float() + target_ins.sum().float() - inser
        if union == 0:
            ious[class_names[i]] = float('nan')
        else:
            ious[class_names[i]] = (inser / (union + 1e-10)).item()
    return ious

# Validation function
def validate(val_dl, net, nclass=16):
    net.eval()
    ious = {name: [] for name in class_names.values()}
    val_losses = []
    with torch.no_grad():
        for input, target in tqdm(val_dl):
            input = input.to(device)
            target = target.to(device)

            output = net(input)
            loss = criterion(output, target)
            val_losses.append(loss.item())

            pred = torch.argmax(output, dim=1)
            class_ious = IOU_per_class(pred, target, nclass)
            for name, iou in class_ious.items():
                ious[name].append(iou)

    avg_ious = {name: np.nanmean(iou_list) for name, iou_list in ious.items()}
    miou = np.nanmean(list(avg_ious.values()))
    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_ious, miou, avg_val_loss

# Training loop
train_losses = []
test_losses = []
mean_ious = []

for epoch in range(epochs):
    net.train()
    mean_loss = []
    for data, label in tqdm(train_dataloader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label)
        mean_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    train_loss = sum(mean_loss) / len(mean_loss)
    train_losses.append(train_loss)
    
    print('EPOCH:', epoch)
    print("Train LOSS:", train_loss)
    
    # Validation phase
    avg_ious, miou, val_loss = validate(test_dataloader, net)
    mean_ious.append(miou)
    test_losses.append(val_loss)

    print('Validation Loss:', val_loss)
    print('Mean IOU:', miou)

    # Print IOU for each class
    print('Class-wise IOU:')
    for class_name, class_iou in avg_ious.items():
        print(f'{class_name}: {class_iou}')
    
# 训练完成后保存分类的IOU图像
avg_ious, miou, _ = validate(test_dataloader, net)


# 画每类的mean IOU
plt.figure()
plt.bar(range(len(avg_ious)), list(avg_ious.values()))
plt.xticks(range(len(avg_ious)), list(avg_ious.keys()), rotation=90)
plt.xlabel('Class')
plt.ylabel('Mean IOU')
plt.title('Class-wise IOU')
plt.savefig('class_wise_iou.png')
plt.close()

# 画train和test的loss折线图
plt.figure()
plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.savefig('train_test_loss.png')
plt.close()

# 画总iou的变化折线图
plt.figure()
plt.plot(range(epochs), mean_ious, label='Mean IOU')
plt.xlabel('Epoch')
plt.ylabel('Mean IOU')
plt.title('Mean IOU over Epochs')
plt.legend()
plt.savefig('mean_iou.png')
plt.close()

print("训练完成，图像已保存。")


# from unet import U_Net
# from UnetAtt import U_Net_v1
# from dataset import dataset
# import torch.nn as nn
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt

# # 检查CUDA是否可用
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

# # 定义损失函数
# criterion = nn.CrossEntropyLoss()

# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 创建数据加载器
# train_dataloader = DataLoader(dataset('/root/unet/train'), batch_size=2, shuffle=True)
# test_dataloader = DataLoader(dataset('/root/unet/val'), batch_size=2, shuffle=True)

# # 初始化模型并将其移动到GPU
# net = U_Net_v1(img_ch=3, output_ch=16).to(device)  # 修改输出通道数为16

# # 定义优化器
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# # 定义训练的轮数
# epochs = 100

# # 定义实际的类别名称
# class_names = {
#     0: "background",
#     1: "dirt",
#     2: "mud",
#     3: "water",
#     4: "gravel",
#     5: "other-terrain",
#     6: "tree-trunk",
#     7: "tree-foliage",
#     8: "bush",
#     9: "fence",
#     10: "structure",
#     11: "rock",
#     12: "log",
#     13: "other-object",  # other-object
#     14: "sky",
#     15: "grass"
# }

# # 修改IOU计算函数以适应16个类别
# def IOU_per_class(pred, target, nclass=16):
#     ious = {}
#     pred = pred.view(-1)
#     target = target.view(-1)
#     for i in range(nclass):
#         pred_ins = pred == i
#         target_ins = target == i
#         inser = (pred_ins & target_ins).sum().float()
#         union = pred_ins.sum().float() + target_ins.sum().float() - inser
#         if union == 0:
#             ious[class_names[i]] = float('nan')
#         else:
#             ious[class_names[i]] = (inser / (union + 1e-10)).item()
#     return ious

# # 验证函数
# def validate(val_dl, net, nclass=16):
#     net.eval()
#     ious = {name: [] for name in class_names.values()}
#     with torch.no_grad():
#         for input, target in tqdm(val_dl):
#             input = input.to(device)
#             target = target.to(device)

#             output = net(input)
#             pred = torch.argmax(output, dim=1)
#             class_ious = IOU_per_class(pred, target, nclass)
#             for name, iou in class_ious.items():
#                 ious[name].append(iou)

#     avg_ious = {name: np.nanmean(iou_list) for name, iou_list in ious.items()}
#     miou = np.nanmean(list(avg_ious.values()))
#     return avg_ious, miou

# # 训练循环
# train_losses = []
# test_losses = []
# mean_ious = []

# for epoch in range(epochs):
#     net.train()
#     mean_loss = []
#     for data, label in tqdm(train_dataloader):
#         data = data.to(device)
#         label = label.to(device)
#         optimizer.zero_grad()
#         output = net(data)
#         loss = criterion(output, label)
#         mean_loss.append(loss.item())
#         loss.backward()
#         optimizer.step()
    
#     train_loss = sum(mean_loss) / len(mean_loss)
#     train_losses.append(train_loss)
    
#     print('EPOCH:', epoch)
#     print("LOSS:", train_loss)
    
#     # 验证阶段
#     avg_ious, miou = validate(test_dataloader, net)
#     mean_ious.append(miou)
#     test_losses.append(train_loss)  # 修正记录为loss
    
#     print('Mean IOU:', miou)
    
# # 训练完成后保存分类的IOU图像
# avg_ious, miou = validate(test_dataloader, net)

# # 画每类的mean IOU
# plt.figure()
# plt.bar(range(len(avg_ious)), list(avg_ious.values()))
# plt.xticks(range(len(avg_ious)), list(avg_ious.keys()), rotation=90)
# plt.xlabel('Class')
# plt.ylabel('Mean IOU')
# plt.title('Class-wise IOU')
# plt.savefig('class_wise_iou.png')
# plt.close()

# # 画train和test的loss折线图
# plt.figure()
# plt.plot(range(epochs), train_losses, label='Train Loss')
# plt.plot(range(epochs), test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Train and Test Loss')
# plt.legend()
# plt.savefig('train_test_loss.png')
# plt.close()

# # 画总iou的变化折线图
# plt.figure()
# plt.plot(range(epochs), mean_ious, label='Mean IOU')
# plt.xlabel('Epoch')
# plt.ylabel('Mean IOU')
# plt.title('Mean IOU over Epochs')
# plt.legend()
# plt.savefig('mean_iou.png')
# plt.close()

# print("训练完成，图像已保存。")

