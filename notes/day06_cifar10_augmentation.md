# Day 6: CIFAR-10 + 数据增强

## 学习内容

### 1. CIFAR-10 数据集
- 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
- 每类 6000 张 32x32 彩色图像
- 比 MNIST 难得多，需要数据增强

### 2. 数据增强
```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),        # 随机裁剪
    transforms.RandomHorizontalFlip(),            # 随机水平翻转
    transforms.ColorJitter(brightness=0.2),       # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

**为什么只在训练集做？**
- 训练集需要多样性，防止过拟合
- 测试集需要一致性，保证评估准确

### 3. 更深的 CNN
```
输入(3,32,32)
  → Conv1(3→32) + BN + ReLU + MaxPool → (32,16,16)
  → Conv2(32→64) + BN + ReLU + MaxPool → (64,8,8)
  → Conv3(64→128) + BN + ReLU + MaxPool → (128,4,4)
  → Flatten → 2048
  → Dropout(0.5) → FC(256) → FC(10)
```

### 4. 混淆矩阵
- 对角线：正确预测
- 非对角线：错误预测
- 可以看出哪些类别容易混淆

## 关键概念

| 概念 | 说明 |
|------|------|
| 数据增强 | 增加训练数据多样性 |
| 混淆矩阵 | 展示分类结果的矩阵 |
| 更深的网络 | 提取更复杂的特征 |

## 验证问题

1. **为什么 Day 6 用 3 层卷积而 Day 4 只用 2 层？**
   - CIFAR-10 更复杂，需要更深的网络提取特征

2. **数据增强为什么只在训练集做？**
   - 训练集需要多样性
   - 测试集需要一致性

3. **混淆矩阵的对角线代表什么？**
   - 对角线 = 正确预测
   - 非对角线 = 错误预测
