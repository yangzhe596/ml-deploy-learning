"""
验证 BatchNorm 和 Dropout 的效果
"""

import torch
import torch.nn as nn

# ============================================================
# 验证 1: Dropout 的效果
# ============================================================

print("=" * 50)
print("验证 Dropout")
print("=" * 50)

dropout = nn.Dropout(0.5)
x = torch.ones(10)  # 10 个值为 1 的张量

# 训练模式
dropout.train()
train_out = dropout(x)
print(f"训练模式 - 输入: {x}")
print(f"训练模式 - 输出: {train_out}")
print(f"训练模式 - 被丢弃的数量: {(train_out == 0).sum().item()}")

# 评估模式
dropout.eval()
eval_out = dropout(x)
print(f"\n评估模式 - 输入: {x}")
print(f"评估模式 - 输出: {eval_out}")
print(f"评估模式 - 被丢弃的数量: {(eval_out == 0).sum().item()}")

# ============================================================
# 验证 2: BatchNorm 的效果
# ============================================================

print("\n" + "=" * 50)
print("验证 BatchNorm")
print("=" * 50)

bn = nn.BatchNorm2d(3)  # 3 个通道

# 模拟一个 batch 的数据
x = torch.randn(4, 3, 2, 2)  # batch=4, channels=3, 2x2

# 训练模式
bn.train()
train_out = bn(x)
print(f"训练模式 - 输入均值: {x.mean(dim=(0,2,3)).detach().numpy().round(2)}")
print(f"训练模式 - 输出均值: {train_out.mean(dim=(0,2,3)).detach().numpy().round(2)}")
print(f"训练模式 - running_mean: {bn.running_mean.detach().numpy().round(2)}")

# 评估模式
bn.eval()
eval_out = bn(x)
print(f"\n评估模式 - 输入均值: {x.mean(dim=(0,2,3)).detach().numpy().round(2)}")
print(f"评估模式 - 输出均值: {eval_out.mean(dim=(0,2,3)).detach().numpy().round(2)}")
print(f"评估模式 - running_mean: {bn.running_mean.detach().numpy().round(2)}")

print("\n结论：")
print("- Dropout: 训练时随机丢弃，评估时不丢弃")
print("- BatchNorm: 训练时用 batch 统计量，评估时用 running 统计量")
