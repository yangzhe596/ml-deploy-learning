# Day 3: 模型定义 + 训练循环

## 学习内容

### 1. 定义模型
```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

**核心**：继承 `nn.Module`，必须实现 `__init__` 和 `forward`。

### 2. 训练循环（5 步）
```python
for batch_data, batch_labels in dataloader:
    outputs = model(batch_data)           # 1. 前向传播
    loss = criterion(outputs, batch_labels) # 2. 计算损失
    optimizer.zero_grad()                  # 3. 清零梯度
    loss.backward()                        # 4. 反向传播
    optimizer.step()                       # 5. 更新参数
```

### 3. 损失函数和优化器
```python
criterion = nn.CrossEntropyLoss()           # 分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

### 4. 模型保存/加载
```python
# 保存（推荐）
torch.save(model.state_dict(), "model.pth")

# 加载
model = SimpleNet()
model.load_state_dict(torch.load("model.pth"))
```

**为什么用 state_dict？**
- 只保存参数，不保存结构
- 更灵活，兼容性更好
- 文件更小

### 5. 测试模式
```python
model.eval()          # 切换到评估模式
with torch.no_grad(): # 不计算梯度
    predictions = model(test_data)
```

## 关键概念

| 概念 | 说明 |
|------|------|
| `nn.Module` | 所有模型的基类 |
| `forward()` | 定义前向传播 |
| `zero_grad()` | 清零梯度（在 backward 前） |
| `state_dict()` | 保存/加载模型参数 |
| `model.eval()` | 切换到评估模式 |

## 验证问题

1. **训练循环的 5 个步骤是什么？**
   - 前向传播 → 计算损失 → 清零梯度 → 反向传播 → 更新参数

2. **为什么先 zero_grad() 再 backward()？**
   - PyTorch 默认累加梯度
   - 不清零会导致梯度累加错误

3. **model.eval() 和 torch.no_grad() 分别做什么？**
   - `model.eval()`: 改变 Dropout/BN 行为
   - `torch.no_grad()`: 不计算梯度，节省内存
