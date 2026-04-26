# ML 模型训练与部署学习路线

## 学习者背景
- 自动驾驶仿真领域算法工程师
- 有 C++ 和 Python 基础
- 每天学习时间：1 小时
- 学习方式：阅读代码 + 实操为主

## 学习目标

### 最终目标
从零走通一个 **YOLOv8n 多类别目标检测模型**的完整部署流程：
1. 训练 YOLOv8n 目标检测模型
2. 模型转换（PyTorch → ONNX → TensorRT）
3. 模型量化（INT8 量化）
4. 自定义算子开发（CUDA kernel）
5. 端到端部署推理（3090 GPU）

### 阶段目标

| 阶段 | 天数 | 目标 | 产出 |
|------|------|------|------|
| 1 | 1-7 | PyTorch 基础 + CNN 分类 | MNIST/CIFAR-10 分类器 |
| 2 | 8-17 | YOLOv8 目标检测训练 | 自定义数据集检测器 |
| 3 | 18-22 | ONNX 导出深入 | ONNX 模型 + 推理流水线 |
| 4 | 23-30 | TensorRT 部署 | TensorRT engine（Python + C++） |
| 5 | 31-37 | 模型量化 | INT8 量化模型 |
| 6 | 38-49 | 自定义 CUDA 算子 | CUDA kernel + PyTorch 绑定 |
| 7 | 50-58 | 端到端部署项目 | 完整检测部署方案 |

## 目录结构

```
ml-deploy-learning/
├── README.md                 # 本文件
├── plans/                    # 详细学习计划
│   └── daily-plan.md         # 每日学习计划
├── code/                     # 代码实践
│   ├── 01-pytorch-basics/    # PyTorch 基础
│   ├── 02-cnn-classifier/    # CNN 分类器
│   ├── 03-onnx-export/       # ONNX 导出
│   ├── 04-tensorrt-basics/   # TensorRT 部署
│   ├── 05-object-detection/  # 目标检测
│   ├── 06-model-quantization/# 模型量化
│   ├── 07-custom-operators/  # 自定义算子
│   └── 08-final-project/     # 最终项目
├── datasets/                 # 数据集
├── models/                   # 保存的模型
└── notes/                    # 学习笔记
```

## 环境配置

### 已搭建环境（2026-04-26）

**环境名称**：`ml-deploy`（conda 环境）

| 项目 | 版本 |
|------|------|
| OS | Debian 13 (trixie) |
| GPU | NVIDIA GeForce RTX 3090 (24GB) |
| NVIDIA Driver | 595.58.03 |
| CUDA (driver) | 13.2 |
| Conda | miniforge3 26.1.1 |
| Python | 3.11.15 (conda-forge) |
| PyTorch | 2.11.0+cu130 |
| TorchVision | 0.26.0+cu130 |
| TorchAudio | 2.11.0+cu130 |
| ONNX | 1.21.0 |
| ONNX Runtime GPU | 1.25.0 (CUDA + TensorRT providers) |
| ONNX Sim | 0.4.36 |
| Ultralytics | 8.4.41 |
| OpenCV | 4.13.0 |
| NumPy | 2.4.4 |
| Matplotlib | 3.10.9 |
| TensorBoard | 2.20.0 |
| Jupyter | 5.9.1 |
| TensorRT | ⬜ 未安装（Stage 4 再装） |

### 使用环境

环境已搭建完成，直接激活即可使用：

```bash
# 激活 conda 环境
conda activate ml-deploy

# 验证环境
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

import onnxruntime as ort
print(f'ONNX Runtime providers: {ort.get_available_providers()}')

from ultralytics import YOLO
print('Ultralytics: OK')
"
```

### 环境搭建记录（已完成）

如需重新搭建环境，可参考以下步骤：

```bash
# 1. 创建 conda 环境
conda create -n ml-deploy python=3.11 -y
conda activate ml-deploy

# 2. 安装 PyTorch（从 PyTorch 官方索引，自动匹配 CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 ONNX 相关（先装 cmake，onnxsim 编译依赖它）
pip install cmake
pip install onnx onnxruntime-gpu onnxsim

# 4. 安装 YOLOv8
pip install ultralytics

# 5. 安装其他工具
pip install matplotlib pillow numpy opencv-python-headless tensorboard jupyter
```

> **注意**：PyTorch + CUDA 包较大（~2GB），下载慢时可耐心等待或使用代理。
> `onnxsim` 需要 cmake 来编译，务必先装 cmake。

### TensorRT 安装（Stage 4 时再装）

TensorRT 需要从 NVIDIA 官网下载，安装较复杂，建议在学习到 Stage 4 时再配置：

```bash
# 方式 1：pip 安装（如果可用）
pip install tensorrt

# 方式 2：从 NVIDIA 官网下载 .tar 或 .deb 安装
# 参考：https://docs.nvidia.com/deeplearning/tensorrt/install-guide/
```

## 学习流程

### 两种学习模式

根据学习进度和时间，选择合适的模式：

#### 模式 A：自己写代码

1. **AI 给任务描述**：只告诉你要做什么，不给代码
2. **你先尝试**：自己写代码，写完贴给 AI
3. **AI 给反馈**：指出问题，给提示，不直接给答案
4. **验证运行**：运行代码，确认结果正确

**适用场景**：时间充裕，想深入理解

#### 模式 B：学习已写代码（默认，推荐）

1. **AI 写完整代码**：包含详细注释
2. **你阅读理解**：逐行阅读，理解每行代码的作用
3. **AI 提问验证**：通过提问确认你理解了关键概念
4. **验证运行**：运行代码，确认结果正确

**适用场景**：时间紧张，快速掌握核心概念

### 每日学习流程

```
1. 回顾（5 分钟）
   └── 回顾前一天的内容

2. 学习（40 分钟）
   ├── 选择学习模式（A 或 B）
   ├── 按模式流程学习
   └── 记录问题和笔记

3. 验证（10 分钟）
   ├── 运行代码验证结果
   └── 回答 AI 的验证问题

4. 记录（5 分钟）
   └── 在 notes/ 目录记录关键知识点
```

### 验证标准

每完成一天的学习，确保你能回答以下问题：
- 这个知识点的核心概念是什么？
- 代码中每一行的作用是什么？
- 如果修改某个参数，会发生什么？
- 这个知识点和前面学过的有什么关联？

## 学习建议

1. **每天 1 小时**：保持连续性比单次长时间学习更重要
2. **代码优先**：先跑通代码，再理解原理
3. **记录笔记**：在 `notes/` 目录记录关键知识点
4. **遇到问题**：先尝试自己解决，记录问题和解决方案
5. **定期回顾**：每周回顾一次，确保理解到位
