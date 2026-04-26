"""
Day 7: 评估指标 + 阶段总结
===========================

学习目标：
1. 理解 accuracy、precision、recall、F1
2. 错误样本分析
3. Stage 1 总结
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# ============================================================
# Part 1: 加载训练好的模型
# ============================================================

class MNIST_CNN(nn.Module):
    """Day 4 的模型结构（无 BatchNorm）"""
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

model = MNIST_CNN()

# 加载模型（如果存在）
import os
model_path = os.path.join(os.path.dirname(__file__), '../../models/mnist_cnn.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("加载已训练的模型")
else:
    print("模型不存在，需要先运行 Day 4 训练")

# ============================================================
# Part 2: 计算评估指标
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='../../datasets', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Accuracy（准确率）
accuracy = np.mean(all_preds == all_labels)
print(f"\nAccuracy: {accuracy:.4f}")

# Precision、Recall、F1（每个类别）
num_classes = 10
print("\n每个类别的指标：")
print(f"{'类别':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")

for cls in range(num_classes):
    # True Positive: 预测为 cls 且实际为 cls
    tp = np.sum((all_preds == cls) & (all_labels == cls))
    # False Positive: 预测为 cls 但实际不是 cls
    fp = np.sum((all_preds == cls) & (all_labels != cls))
    # False Negative: 预测不是 cls 但实际是 cls
    fn = np.sum((all_preds != cls) & (all_labels == cls))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{cls:>6} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

# ============================================================
# Part 3: 错误样本分析
# ============================================================

print("\n错误样本分析：")

# 找出预测错误的样本
wrong_idx = np.where(all_preds != all_labels)[0]
print(f"错误样本数量: {len(wrong_idx)} / {len(all_labels)}")

# 显示前 5 个错误样本
print("\n前 5 个错误样本：")
for i in range(min(5, len(wrong_idx))):
    idx = wrong_idx[i]
    print(f"样本 {idx}: 真实={all_labels[idx]}, 预测={all_preds[idx]}")

# 统计最常见的错误类型
print("\n最常见的错误类型：")
error_pairs = list(zip(all_labels[wrong_idx], all_preds[wrong_idx]))
from collections import Counter
error_counts = Counter(error_pairs)
for (true, pred), count in error_counts.most_common(5):
    print(f"  真实={true}, 预测={pred}: {count} 次")

print("\nDay 7 完成！")
print("\n" + "="*50)
print("Stage 1 总结")
print("="*50)
print("已掌握：")
print("- PyTorch 张量操作和自动求导")
print("- Dataset、DataLoader、transforms")
print("- nn.Module 模型定义和训练循环")
print("- CNN 基础（Conv2d、MaxPool2d）")
print("- BatchNorm、Dropout 优化技巧")
print("- 评估指标（Accuracy、Precision、Recall、F1）")
print("\n下一步：Stage 2 - YOLOv8 目标检测")
