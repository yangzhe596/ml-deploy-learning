# Day 4: CNN 基础 + MNIST 分类器

## 学习内容

### 1. 卷积层 Conv2d
```python
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
```

| 参数 | 说明 |
|------|------|
| `in_channels` | 输入通道数（灰度图=1，RGB=3） |
| `out_channels` | 输出通道数（卷积核数量） |
| `kernel_size` | 卷积核大小 |
| `padding` | 填充（保持尺寸不变） |

### 2. 池化层 MaxPool2d
```python
nn.MaxPool2d(kernel_size=2)  # 尺寸减半
```

**作用**：减少计算量，提取主要特征。

### 3. CNN 结构
```
输入(1,28,28)
  → Conv1(1→32) + ReLU + MaxPool → (32,14,14)
  → Conv2(32→64) + ReLU + MaxPool → (64,7,7)
  → Flatten → 3136
  → FC → 10
```

### 4. MNIST 训练结果
- **测试准确率**: 98.99%
- **训练轮数**: 3 epochs

## 关键概念

| 概念 | 说明 |
|------|------|
| 卷积 | 用卷积核扫描图像，提取特征 |
| 池化 | 缩小特征图，减少计算量 |
| padding | 填充，保持输出尺寸 |
| Flatten | 展平多维特征图为一维 |

## 验证问题

1. **padding=1 的作用？**
   - 在图像周围补一圈 0
   - 保持卷积后尺寸不变
   - 更好地提取边缘特征

2. **为什么用 CNN 而不是全连接层？**
   - CNN 参数更少（权重共享）
   - CNN 保留空间结构
   - 全连接层参数爆炸，丢失空间信息

3. **model.train() 和 model.eval() 的区别？**
   - train(): Dropout 随机丢弃，BN 用 batch 统计量
   - eval(): Dropout 关闭，BN 用 running 统计量
