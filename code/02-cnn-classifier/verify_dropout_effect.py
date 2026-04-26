"""
验证 Dropout 对训练结果的影响
对比：有 Dropout vs 无 Dropout
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 无 Dropout
class CNN_NoDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 有 Dropout
class CNN_WithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_and_evaluate(model, train_loader, test_loader, name, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n{'='*50}")
    print(f"训练: {name}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        # 训练
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

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

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, "
              f"差距: {train_acc - test_acc:.2f}%")

    return train_acc, test_acc

# 准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../../datasets', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../../datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 训练两个模型
model_no_dropout = CNN_NoDropout()
train_no, test_no = train_and_evaluate(model_no_dropout, train_loader, test_loader, "无 Dropout")

model_with_dropout = CNN_WithDropout()
train_with, test_with = train_and_evaluate(model_with_dropout, train_loader, test_loader, "有 Dropout")

# 总结
print(f"\n{'='*50}")
print("总结")
print(f"{'='*50}")
print(f"无 Dropout - 训练: {train_no:.2f}%, 测试: {test_no:.2f}%, 差距: {train_no-test_no:.2f}%")
print(f"有 Dropout - 训练: {train_with:.2f}%, 测试: {test_with:.2f}%, 差距: {train_with-test_with:.2f}%")
print(f"\n如果差距变小，说明 Dropout 减少了过拟合")
