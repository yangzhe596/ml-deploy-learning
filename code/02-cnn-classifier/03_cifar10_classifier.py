"""
Day 6: CIFAR-10 + 数据增强
===========================

学习目标：
1. 加载 CIFAR-10 数据集
2. 理解数据增强的作用
3. 训练更深的网络
4. 可视化混淆矩阵

注：CIFAR-10 下载较慢，本示例用模拟数据验证流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ============================================================
# Part 1: 数据准备（模拟 CIFAR-10）
# ============================================================

"""
CIFAR-10: 10 个类别，每类 6000 张 32x32 彩色图像
类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
比 MNIST 难得多，需要数据增强
"""

# 模拟数据
X_train = torch.randn(5000, 3, 32, 32)
y_train = torch.randint(0, 10, (5000,))
X_test = torch.randn(1000, 3, 32, 32)
y_test = torch.randint(0, 10, (1000,))

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# ============================================================
# Part 2: 定义更深的 CNN
# ============================================================

class DeepCNN(nn.Module):
    """
    更深的 CNN：
    - 3 个卷积块
    - BatchNorm + Dropout
    - 适合 CIFAR-10
    """
    def __init__(self):
        super().__init__()
        # 卷积块 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # 卷积块 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # 卷积块 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        # 全连接层
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = DeepCNN()
print("\n模型结构：")
print(model)

# ============================================================
# Part 3: 训练
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

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)

    # 测试
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
          f"Train Acc: {accuracy:.2f}%, Test Acc: {test_acc:.2f}%")

# ============================================================
# Part 4: 混淆矩阵
# ============================================================

print("\n计算混淆矩阵...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
num_classes = 10
confusion = np.zeros((num_classes, num_classes), dtype=int)
for pred, label in zip(all_preds, all_labels):
    confusion[label][pred] += 1

# 打印混淆矩阵
print("\n混淆矩阵：")
print("预测 →", end="")
for i in range(num_classes):
    print(f"{classes[i]:>6}", end="")
print()

for i in range(num_classes):
    print(f"{classes[i]:>6}", end="")
    for j in range(num_classes):
        print(f"{confusion[i][j]:>6}", end="")
    print()

# 保存模型
import os
save_dir = os.path.join(os.path.dirname(__file__), '../../models')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'cifar10_cnn.pth')
torch.save(model.state_dict(), save_path)
print(f"\n模型已保存到 {save_path}")

print("\nDay 6 完成！")
