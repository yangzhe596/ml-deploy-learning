"""
Day 2: 数据加载
===============

学习目标：
1. 理解 Dataset 和 DataLoader 的作用
2. 学会自定义 Dataset 类
3. 掌握 transforms 数据增强
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# ============================================================
# Part 1: 自定义 Dataset 类
# ============================================================

class SimpleDataset(Dataset):
    """
    自定义 Dataset 必须实现三个方法：
    - __init__: 初始化数据
    - __len__: 返回数据集大小
    - __getitem__: 返回单个样本
    """
    def __init__(self, num_samples=100):
        # 生成随机数据：100 个样本，每个样本 3 个特征
        self.data = torch.randn(num_samples, 3)
        # 生成标签：0 或 1
        self.labels = torch.randint(0, 2, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集
dataset = SimpleDataset(num_samples=100)
print(f"数据集大小: {len(dataset)}")
print(f"第一个样本: {dataset[0]}")

# ============================================================
# Part 2: DataLoader
# ============================================================

"""
DataLoader 的作用：
- batch_size: 每次取多少个样本
- shuffle: 是否打乱顺序（训练时需要，测试时不需要）
- num_workers: 用几个进程加载数据（0 = 主进程）
"""

# batch_size=16: 每次取 16 个样本
# shuffle=True: 每个 epoch 打乱顺序
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 遍历一个 batch
for batch_data, batch_labels in dataloader:
    print(f"\nBatch data shape: {batch_data.shape}")  # (16, 3)
    print(f"Batch labels shape: {batch_labels.shape}")  # (16,)
    break  # 只看第一个 batch

# ============================================================
# Part 3: transforms 数据增强
# ============================================================

"""
transforms 的作用：
- 对图像进行预处理和数据增强
- 常用操作：Resize, Crop, Flip, Normalize, ToTensor
"""

# 定义 transforms 流水线
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为 tensor，归一化到 [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
])

# 示例：处理一个随机图像
fake_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
print(f"\n原始图像 shape: {fake_image.shape}")

# 应用 transforms
tensor_image = transform(fake_image)
print(f"Transform 后 shape: {tensor_image.shape}")  # (3, 32, 32)
print(f"值范围: [{tensor_image.min():.2f}, {tensor_image.max():.2f}]")

# ============================================================
# Part 4: 完整示例 - 带 transforms 的 Dataset
# ============================================================

class ImageDataset(Dataset):
    """模拟图像数据集"""
    def __init__(self, num_samples=50, transform=None):
        self.num_samples = num_samples
        self.transform = transform
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 模拟图像：32x32x3
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        label = np.random.randint(0, 10)  # 10 个类别
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 创建带 transforms 的数据集
image_dataset = ImageDataset(num_samples=50, transform=transform)
image_dataloader = DataLoader(image_dataset, batch_size=8, shuffle=True)

for images, labels in image_dataloader:
    print(f"\n图像 batch shape: {images.shape}")  # (8, 3, 32, 32)
    print(f"标签 batch shape: {labels.shape}")  # (8,)
    break

print("\nDay 2 完成！")
