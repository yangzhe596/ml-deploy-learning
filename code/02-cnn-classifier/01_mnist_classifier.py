"""
Day 4: CNN 基础 + MNIST 分类器
==============================

学习目标：
1. 理解卷积层（Conv2d）和池化层（MaxPool2d）
2. 搭建 CNN 分类器
3. 训练 MNIST 手写数字识别
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# Part 1: 理解卷积和池化
# ============================================================

"""
卷积层 nn.Conv2d：
- 用卷积核（filter）扫描图像，提取特征
- 参数：in_channels, out_channels, kernel_size

池化层 nn.MaxPool2d：
- 缩小特征图尺寸，减少计算量
- 参数：kernel_size（通常为 2）
"""

# 示例：看卷积和池化的效果
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
pool = nn.MaxPool2d(kernel_size=2)

# 模拟一张 28x28 的灰度图
fake_image = torch.randn(1, 1, 28, 28)  # (batch, channels, height, width)

conv_out = conv(fake_image)
pool_out = pool(conv_out)

print(f"原始尺寸: {fake_image.shape}")
print(f"卷积后: {conv_out.shape}")   # (1, 16, 28, 28) - 通道数变了
print(f"池化后: {pool_out.shape}")   # (1, 16, 14, 14) - 尺寸减半

# ============================================================
# Part 2: 定义 CNN 模型
# ============================================================

class MNIST_CNN(nn.Module):
    """
    CNN 结构：
    输入(1,28,28) → Conv(1→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → FC → 输出(10)
    """
    def __init__(self):
        super().__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1通道 → 32通道
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # 28x28 → 14x14

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32通道 → 64通道
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  # 14x14 → 7x7

        # 全连接层
        self.flatten = nn.Flatten()   # 64*7*7 → 3136
        self.fc = nn.Linear(64 * 7 * 7, 10)  # 3136 → 10（10个数字类别）

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = MNIST_CNN()
print("\n模型结构：")
print(model)

# ============================================================
# Part 3: 加载 MNIST 数据集
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

train_dataset = datasets.MNIST(
    root='../../datasets',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='../../datasets',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"\n训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# ============================================================
# Part 4: 训练
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n开始训练...")

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 200 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# ============================================================
# Part 5: 测试
# ============================================================

print("\n测试模型...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"测试集准确率: {accuracy:.2f}%")

# 保存模型
import os
save_dir = os.path.join(os.path.dirname(__file__), '../../models')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'mnist_cnn.pth')
torch.save(model.state_dict(), save_path)
print(f"模型已保存到 {save_path}")

print("\nDay 4 完成！")
