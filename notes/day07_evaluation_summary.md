# Day 7: 评估指标 + 阶段总结

## 学习内容

### 1. 评估指标

**Accuracy（准确率）**
```
Accuracy = 正确预测数 / 总样本数
```

**Precision（精确率）**
```
Precision = TP / (TP + FP)
预测为正的样本中，实际为正的比例
```

**Recall（召回率）**
```
Recall = TP / (TP + FN)
实际为正的样本中，被预测为正的比例
```

**F1 Score**
```
F1 = 2 * Precision * Recall / (Precision + Recall)
精确率和召回率的调和平均
```

### 2. 错误样本分析

- 找出预测错误的样本
- 分析最常见的错误类型
- 理解模型的弱点

### 3. Stage 1 总结

**已掌握：**
- PyTorch 张量操作和自动求导
- Dataset、DataLoader、transforms
- nn.Module 模型定义和训练循环
- CNN 基础（Conv2d、MaxPool2d）
- BatchNorm、Dropout 优化技巧
- 评估指标（Accuracy、Precision、Recall、F1）

**核心代码模式：**
```python
# 训练循环
for batch in dataloader:
    outputs = model(batch)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

## 关键概念

| 概念 | 说明 |
|------|------|
| Accuracy | 整体准确率 |
| Precision | 预测为正的准确率 |
| Recall | 实际为正的召回率 |
| F1 | 精确率和召回率的平衡 |

## 下一步

Stage 2：YOLOv8 目标检测
