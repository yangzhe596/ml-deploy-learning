"""
Day 3: 模型定义 + 训练循环
==========================

学习目标：
1. 理解 nn.Module 的作用
2. 掌握完整的训练循环（forward → loss → backward → step）
3. 学会模型保存/加载（state_dict）
"""

import torch
import torch.nn as nn

# ============================================================
# Part 1: 定义模型
# ============================================================

class SimpleNet(nn.Module):
    """
    简单的全连接网络
    结构：输入(3) → 隐藏层(16) → 输出(2)
    """
    def __init__(self):
        super().__init__()  # 必须调用父类构造函数
        self.fc1 = nn.Linear(3, 16)   # 第一层：3 → 16
        self.relu = nn.ReLU()         # 激活函数
        self.fc2 = nn.Linear(16, 2)   # 第二层：16 → 2

    def forward(self, x):
        """前向传播：定义数据如何流过网络"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNet()
print("模型结构：")
print(model)

# ============================================================
# Part 2: 准备数据（用 Day 2 学的 Dataset + DataLoader）
# ============================================================

from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, num_samples=200):
        self.data = torch.randn(num_samples, 3)
        # 标签：如果 x[0] > 0 则标签为 1，否则为 0
        self.labels = (self.data[:, 0] > 0).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = SimpleDataset(num_samples=200)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ============================================================
# Part 3: 定义损失函数和优化器
# ============================================================

"""
损失函数：衡量预测值和真实值的差距
- nn.CrossEntropyLoss: 分类任务常用
- nn.MSELoss: 回归任务常用

优化器：根据梯度更新模型参数
- optim.SGD: 随机梯度下降
- optim.Adam: 自适应学习率，更常用
"""

criterion = nn.CrossEntropyLoss()           # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器

# ============================================================
# Part 4: 训练循环（核心！）
# ============================================================

print("\n开始训练...")

num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0

    for batch_data, batch_labels in dataloader:
        # 1. 前向传播：计算预测值
        outputs = model(batch_data)

        # 2. 计算损失
        loss = criterion(outputs, batch_labels)

        # 3. 反向传播：计算梯度
        optimizer.zero_grad()  # 清零梯度（Day 1 学过为什么）
        loss.backward()        # 计算梯度

        # 4. 更新参数
        optimizer.step()       # 根据梯度更新权重

        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ============================================================
# Part 5: 测试模型
# ============================================================

print("\n测试模型...")
model.eval()  # 切换到评估模式（关闭 Dropout、BatchNorm 等）

test_data = torch.tensor([[1.0, 0.5, -0.3],   # x[0] > 0 → 预测 1
                          [-0.5, 0.8, 0.2]])   # x[0] < 0 → 预测 0

with torch.no_grad():  # 测试时不需要计算梯度
    predictions = model(test_data)
    predicted_classes = torch.argmax(predictions, dim=1)

print(f"输入: {test_data}")
print(f"预测类别: {predicted_classes}")
print(f"期望: [1, 0]")

# ============================================================
# Part 6: 模型保存和加载
# ============================================================

"""
state_dict: 保存模型的参数（权重和偏置）
- 不保存模型结构，只保存参数
- 更灵活，推荐使用
"""

# 保存
torch.save(model.state_dict(), "models/simple_net.pth")
print("\n模型已保存到 models/simple_net.pth")

# 加载
loaded_model = SimpleNet()
loaded_model.load_state_dict(torch.load("models/simple_net.pth"))
loaded_model.eval()

# 验证加载的模型
with torch.no_grad():
    loaded_predictions = loaded_model(test_data)
    loaded_classes = torch.argmax(loaded_predictions, dim=1)
    print(f"加载后预测: {loaded_classes}")

print("\nDay 3 完成！")
