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

## 环境要求

### 基础环境
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（如果有的话）
- ONNX Runtime
- TensorRT 8.6+（NVIDIA GPU）

### 推荐安装
```bash
# PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision torchaudio

# ONNX 相关
pip install onnx onnxruntime onnxsim

# TensorRT（需要从 NVIDIA 官网下载）
# pip install tensorrt  # 或使用 .whl 安装

# 其他工具
pip install matplotlib pillow numpy opencv-python
```

## 学习建议

1. **每天 1 小时**：保持连续性比单次长时间学习更重要
2. **代码优先**：先跑通代码，再理解原理
3. **记录笔记**：在 `notes/` 目录记录关键知识点
4. **遇到问题**：先尝试自己解决，记录问题和解决方案
5. **定期回顾**：每周回顾一次，确保理解到位
