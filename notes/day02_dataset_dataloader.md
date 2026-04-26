# Day 2: 数据加载

## 学习内容

### 1. 自定义 Dataset
```python
class SimpleDataset(Dataset):
    def __init__(self):    # 初始化数据
    def __len__(self):     # 返回数据集大小
    def __getitem__(self, idx):  # 返回单个样本
```

**核心**：继承 `torch.utils.data.Dataset`，必须实现这 3 个方法。

### 2. DataLoader
```python
DataLoader(dataset, batch_size=16, shuffle=True)
```

| 参数 | 说明 |
|------|------|
| `batch_size` | 每次取多少样本 |
| `shuffle` | 是否打乱顺序（训练=True，测试=False） |
| `num_workers` | 多进程加载（0=主进程） |

### 3. transforms 数据增强
```python
transform = transforms.Compose([
    transforms.ToTensor(),           # 图像 → tensor，归一化到 [0,1]
    transforms.Normalize(mean, std)  # 标准化
])
```

**注意**：`ToTensor()` 会将 HWC → CHW（通道在前）

## 关键概念

| 概念 | 说明 |
|------|------|
| Dataset | 数据集抽象类，定义如何获取数据 |
| DataLoader | 自动按 batch 取数据、打乱、多进程 |
| transforms | 图像预处理和数据增强流水线 |
| HWC → CHW | PyTorch 要求通道在前（C, H, W） |

## 验证问题

1. **自定义 Dataset 必须实现哪 3 个方法？**
   - `__init__`: 初始化数据
   - `__len__`: 返回数据集大小
   - `__getitem__`: 返回单个样本

2. **shuffle=True 和 shuffle=False 分别用在什么场景？**
   - True: 训练集（打乱顺序，提高鲁棒性）
   - False: 测试集（顺序固定，方便复现和调试）
