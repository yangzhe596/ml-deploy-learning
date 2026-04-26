# Day 1: 张量 + 自动求导

## 学习内容

### 1. 张量创建
```python
torch.zeros(2, 3)      # 全零
torch.eye(3)           # 单位矩阵
torch.tensor([1,2,3])  # 从列表创建
torch.randn(2, 2)      # 随机正态分布
```

### 2. 张量运算
- 矩阵乘法：`mat_a @ mat_b`
- 转置：`mat_a.T`
- 变形：`mat_a.reshape(3, 2)`

### 3. GPU 操作
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_tensor = tensor.to(device)
```

### 4. 自动求导
```python
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 2 = 8
```

### 5. 梯度清零
```python
optimizer.zero_grad()  # 必须在 backward() 前清零
```

**原因**：PyTorch 默认累加梯度，不清零会导致梯度累加错误。

## 关键概念

| 概念 | 说明 |
|------|------|
| `requires_grad=True` | 告诉 PyTorch 追踪这个张量的操作 |
| `backward()` | 反向传播，计算梯度 |
| `zero_grad()` | 清零梯度，防止累加 |

## 验证问题

1. **为什么需要梯度清零？**
   - PyTorch 默认累加梯度
   - 不清零会导致多个 batch 的梯度累加，更新错误

2. **多变量求导的结果？**
   - z = x² + y²
   - dz/dx = 2x, dz/dy = 2y
