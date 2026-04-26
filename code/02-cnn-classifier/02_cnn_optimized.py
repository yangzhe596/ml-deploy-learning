"""
Day 5: CNN 优化（BatchNorm + Dropout）
======================================

学习目标：
1. 理解 BatchNorm 的作用
2. 理解 Dropout 的作用
3. 对比有/无 BatchNorm 的训练效果
4. 比较不同优化器（SGD vs Adam）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# Part 1: 定义两个模型对比
# ============================================================

class CNN_Basic(nn.Module):
    """基础 CNN（Day 4 的版本）"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CNN_Optimized(nn.Module):
    """
    优化版 CNN：
    - BatchNorm: 加速训练，稳定梯度
    - Dropout: 防止过拟合
    """
    def __init__(self):
        super().__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)     # BatchNorm: 对 32 个通道做归一化
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)     # BatchNorm
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # 全连接层
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)    # Dropout: 随机丢弃 50% 神经元
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(x)               # 训练时随机丢弃
        x = self.fc(x)
        return x

# ============================================================
# Part 2: 训练函数
# ============================================================

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=3):
    """训练并返回训练历史"""
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        # 训练
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 测试
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")

    return history

# ============================================================
# Part 3: 准备数据
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../../datasets', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../../datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

criterion = nn.CrossEntropyLoss()

# ============================================================
# Part 4: 对比实验
# ============================================================

print("=" * 60)
print("实验 1: 基础 CNN + Adam")
print("=" * 60)

model_basic = CNN_Basic()
optimizer_basic = optim.Adam(model_basic.parameters(), lr=0.001)
history_basic = train_model(model_basic, train_loader, test_loader, optimizer_basic, criterion)

print("\n" + "=" * 60)
print("实验 2: 优化 CNN（BatchNorm + Dropout）+ Adam")
print("=" * 60)

model_optimized = CNN_Optimized()
optimizer_optimized = optim.Adam(model_optimized.parameters(), lr=0.001)
history_optimized = train_model(model_optimized, train_loader, test_loader, optimizer_optimized, criterion)

print("\n" + "=" * 60)
print("实验 3: 优化 CNN + SGD")
print("=" * 60)

model_sgd = CNN_Optimized()
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
history_sgd = train_model(model_sgd, train_loader, test_loader, optimizer_sgd, criterion)

# ============================================================
# Part 5: 结果对比
# ============================================================

print("\n" + "=" * 60)
print("最终结果对比")
print("=" * 60)
print(f"基础 CNN + Adam:       {history_basic['test_acc'][-1]:.2f}%")
print(f"优化 CNN + Adam:       {history_optimized['test_acc'][-1]:.2f}%")
print(f"优化 CNN + SGD:        {history_sgd['test_acc'][-1]:.2f}%")

print("\nDay 5 完成！")
