import torch

# ============================================================
# Part 1: 张量创建
# ============================================================

zeros_tensor = torch.zeros(2, 3)
print(zeros_tensor)
print(f"shape: {zeros_tensor.shape}")

eye_matrix = torch.eye(3)
print(eye_matrix)
print(f"shape: {eye_matrix.shape}")

range_tensor = torch.tensor([1, 2, 3, 4, 5])
print(range_tensor)
print(f"shape: {range_tensor.shape}")

rand_tensor = torch.randn(2, 2)
print(rand_tensor)
print(f"shape: {rand_tensor.shape}")

# ============================================================
# Part 2: 张量运算
# ============================================================

mat_a = torch.randn(2, 3)
mat_b = torch.randn(3, 2)

# 矩阵乘法: (2x3) @ (3x2) = (2x2)
mat_mul = mat_a @ mat_b
print(f"\n矩阵乘法 shape: {mat_mul.shape}")

# 转置: (2x3) -> (3x2)
mat_transpose = mat_a.T
print(f"转置 shape: {mat_transpose.shape}")

# Reshape: (2x3) -> (3x2)
mat_reshape = mat_a.reshape(3, 2)
print(f"Reshape shape: {mat_reshape.shape}")

# ============================================================
# Part 3: GPU 操作
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n当前设备: {device}")

gpu_tensor = rand_tensor.to(device)
print(f"GPU 张量设备: {gpu_tensor.device}")

# ============================================================
# Part 4: 自动求导
# ============================================================

# requires_grad=True: 告诉 PyTorch 追踪这个张量的操作
x = torch.tensor(3.0, requires_grad=True)
print(f"\nx = {x.item()}")
print(f"requires_grad: {x.requires_grad}")

# 前向传播：定义 y = x^2 + 2x + 1
y = x**2 + 2*x + 1
print(f"y = x^2 + 2*x + 1 = {y.item()}")

# 反向传播：计算梯度
y.backward()

# 验证：dy/dx = 2x + 2 = 2*3 + 2 = 8
print(f"自动求导梯度: {x.grad.item()}")
print(f"手动计算梯度 (2*3+2): 8")

# ============================================================
# Part 5: 多变量求导
# ============================================================

x_adv = torch.tensor(2.0, requires_grad=True)
y_adv = torch.tensor(3.0, requires_grad=True)

# z = x^2 + y^2
z = x_adv**2 + y_adv**2
print(f"\nz = x^2 + y^2 = {z.item()}")

z.backward()

# dz/dx = 2x = 4, dz/dy = 2y = 6
print(f"dz/dx = 2*x = {x_adv.grad.item()}")
print(f"dz/dy = 2*y = {y_adv.grad.item()}")

# 梯度清零
x_adv.grad.zero_()
print(f"梯度清零后 dz/dx: {x_adv.grad.item()}")

print("\n思考题：为什么训练神经网络时需要梯度清零？")