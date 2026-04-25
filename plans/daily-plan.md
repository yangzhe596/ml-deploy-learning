# 每日学习计划（60 天）

> 每天 1 小时，代码实操为主
> 部署平台：3090 GPU
> 目标：YOLOv8n 多类别目标检测完整部署（训练 → ONNX → TensorRT → 量化 → 自定义算子）
> 学习者背景：自动驾驶仿真工程师，有 C++ / Python 基础

## 核心设计原则

1. **先有模型再学部署**：先训练 YOLOv8 拿到真实模型，再学 ONNX/TRT，学习动机清晰
2. **去除重复**：ONNX/TRT 只学一次，直接用 YOLO 模型练，不再用 MNIST 踩坑
3. **压缩水日**：安装包、看 Netron 等 30 分钟能搞定的事不单独占一天
4. **CUDA 时间充足**：12 天，从 kernel 到 NMS 有足够缓冲
5. **利用你的背景**：C++ 部分加速推进，仿真图像作为检测输入

---

## 第一阶段：PyTorch 基础（第 1 周，7 天）

> 有 Python 基础，压缩张量/autograd 到 1 天，重点在训练流程

### Day 1：张量 + 自动求导
- **实操**：
  - 张量创建、运算、变形、GPU 操作
  - autograd 计算图、反向传播、梯度清零
- **产出**：`01-pytorch-basics/01_tensor_autograd.py`

### Day 2：数据加载
- **实操**：
  - 自定义 Dataset 类
  - DataLoader（batch, shuffle, num_workers）
  - transforms 数据增强
- **产出**：`01-pytorch-basics/02_dataset_dataloader.py`

### Day 3：模型定义 + 训练循环
- **实操**：
  - nn.Module 定义全连接网络
  - 完整训练循环（forward → loss → backward → step）
  - 模型保存/加载（state_dict）
- **产出**：`01-pytorch-basics/03_model_training.py`

### Day 4：CNN 基础 + MNIST 分类器
- **实操**：
  - nn.Conv2d、nn.MaxPool2d、特征图可视化
  - 搭建 CNN 分类器，训练 MNIST
- **产出**：`02-cnn-classifier/01_mnist_classifier.py`

### Day 5：CNN 优化（BatchNorm + Dropout）
- **实操**：
  - 添加 BatchNorm、Dropout
  - 对比有/无 BN 的训练曲线
  - 尝试不同优化器（SGD vs Adam）
- **产出**：`02-cnn-classifier/02_cnn_optimized.py`

### Day 6：CIFAR-10 + 数据增强
- **实操**：
  - CIFAR-10 数据加载 + 增强（RandomCrop, HorizontalFlip, ColorJitter）
  - 训练更深的网络
  - 混淆矩阵可视化
- **产出**：`02-cnn-classifier/03_cifar10_classifier.py`

### Day 7：评估指标 + 阶段总结
- **实操**：
  - accuracy、precision、recall、F1 计算
  - 错误样本分析
  - 整理 Stage 1 代码，确认能独立复现
- **产出**：`02-cnn-classifier/04_evaluation.py`

**阶段验证**：能独立搭建 CNN、训练、评估，理解 PyTorch 训练全流程。

---

## 第二阶段：目标检测 YOLOv8（第 2-3 周，10 天）

> 先拿到一个真实可用的检测模型，后续部署才有意义

### Day 8：环境搭建 + YOLOv8 初体验
- **实操**：
  - pip install ultralytics
  - 加载 yolov8n 预训练模型，推理示例图片
  - 理解输出格式（boxes, scores, classes）
- **产出**：`05-object-detection/01_yolov8_basics.py`

### Day 9：YOLO 数据格式 + COCO8
- **实操**：
  - 理解 YOLO 格式（txt: class x_center y_center w h）
  - 加载 COCO8 数据集
  - 数据可视化（画 bounding box）
- **产出**：`05-object-detection/02_dataset.py`

### Day 10：YOLOv8 训练
- **实操**：
  - COCO8 训练（epochs=50, imgsz=640）
  - 训练参数调优（lr, batch_size, augment）
  - TensorBoard 监控训练过程
- **产出**：`05-object-detection/03_training.py`

### Day 11：模型评估
- **实操**：
  - model.val() 评估
  - mAP 指标解读（mAP50, mAP50-95）
  - PR 曲线、混淆矩阵
- **产出**：`05-object-detection/04_evaluation.py`

### Day 12：推理 + 结果可视化
- **实操**：
  - 单图/批量推理
  - 结果可视化（画框 + 标签 + 置信度）
  - 推理速度测试（FPS）
- **产出**：`05-object-detection/05_inference.py`

### Day 13：自定义数据集准备
- **实操**：
  - 选 3-5 个类别的小数据集（50-100 张）
  - 标注工具使用（LabelImg 或 CVAT）
  - 转换为 YOLO 格式
  - 写 data.yaml
- **产出**：`05-object-detection/06_custom_dataset.py`

### Day 14：自定义数据集训练
- **实操**：
  - 用自己的数据集训练 YOLOv8n
  - 调参（epochs, lr, augmentation）
  - 对比预训练 vs 从头训练
- **产出**：`05-object-detection/07_custom_training.py`

### Day 15：模型导出 ONNX
- **实操**：
  - model.export(format='onnx')
  - 动态/静态 shape 配置
  - Netron 可视化 ONNX 结构
  - ONNX Runtime 推理验证
- **产出**：`05-object-detection/08_export_onnx.py`

### Day 16：TensorRT 初步部署
- **实操**：
  - ONNX → TensorRT engine（trtexec 或 Python API）
  - FP16 推理
  - 对比 PyTorch vs TensorRT 速度
- **产出**：`05-object-detection/09_tensorrt_deploy.py`

### Day 17：阶段总结
- **实操**：
  - 整理完整流程：数据 → 训练 → 评估 → 导出 → 部署
  - 记录踩坑和解决方案
  - 准备下一阶段的 ONNX 深入学习
- **产出**：`notes/stage2_summary.md`

**阶段验证**：能用自己的数据训练 YOLOv8n，导出 ONNX，用 TensorRT 推理。

---

## 第三阶段：ONNX 深入（第 3 周，5 天）

> 不再用 MNIST，直接用 Day 15 导出的 YOLO ONNX 模型练习

### Day 18：ONNX 基础 + 导出细节
- **实操**：
  - ONNX 规范核心概念（算子、图、tensor）
  - torch.onnx.export 参数详解（opset_version, input_names, dynamic_axes）
  - 导出失败排查（不支持的算子处理）
- **产出**：`03-onnx-export/01_onnx_export.py`

### Day 19：ONNX Runtime 推理
- **实操**：
  - ONNX Runtime 基本推理流程
  - CPU vs GPU（CUDAExecutionProvider）
  - 批量推理 + 性能对比
- **产出**：`03-onnx-export/02_onnx_inference.py`

### Day 20：ONNX 优化
- **实操**：
  - onnxsim 简化模型（去除冗余节点）
  - 算子融合、常量折叠
  - 优化前后节点数/推理速度对比
- **产出**：`03-onnx-export/03_onnx_optimize.py`

### Day 21：ONNX 模型调试
- **实操**：
  - 数值精度对比（PyTorch vs ONNX 输出 diff）
  - 中间结果验证
  - 常见导出问题及解决方案
- **产出**：`03-onnx-export/04_onnx_debug.py`

### Day 22：ONNX 完整流水线
- **实操**：
  - 从 PyTorch 到 ONNX Runtime 的完整流程
  - 前处理 + 推理 + 后处理（NMS）整合
  - 用真实图片端到端验证
- **产出**：`03-onnx-export/05_complete_pipeline.py`

**阶段验证**：能独立完成 PyTorch → ONNX 导出 → 优化 → 推理验证。

---

## 第四阶段：TensorRT 部署（第 4-5 周，8 天）

> 有 C++ 基础，C++ API 部分可加速

### Day 23：TensorRT 基础 + 环境配置
- **实操**：
  - TensorRT 安装确认（检查版本兼容性）
  - 理解 TensorRT 工作流（parse → build → serialize → deserialize）
  - 第一个 TensorRT 程序（Python API）
- **产出**：`04-tensorrt-basics/01_trt_basics.py`

### Day 24：ONNX → TensorRT 转换
- **实操**：
  - ONNX 模型转 TensorRT engine
  - 静态 shape vs 动态 shape
  - Engine 序列化/反序列化
  - trtexec 命令行工具
- **产出**：`04-tensorrt-basics/02_onnx_to_trt.py`

### Day 25：TensorRT 推理
- **实操**：
  - 加载 engine、分配 GPU 内存
  - 执行推理、结果解析
  - CUDA Stream 异步推理
- **产出**：`04-tensorrt-basics/03_trt_inference.py`

### Day 26：TensorRT 性能优化
- **实操**：
  - FP16 量化推理
  - Workspace size 配置
  - 批量推理优化
  - PyTorch vs ONNX RT vs TensorRT 性能对比
- **产出**：`04-tensorrt-basics/04_trt_optimization.py`

### Day 27：TensorRT C++ API
- **实操**：
  - C++ TensorRT 环境搭建（CMake）
  - Engine 构建 + 推理（C++）
  - 与 Python API 性能对比
- **产出**：`04-tensorrt-basics/05_cpp_trt.cpp`

### Day 28：YOLOv8 TensorRT 完整部署
- **实操**：
  - YOLOv8 ONNX → TensorRT engine
  - 前处理（letterbox resize）+ 推理 + 后处理（NMS）
  - 用真实图片端到端验证
- **产出**：`04-tensorrt-basics/06_yolo_trt_deploy.py`

### Day 29：C++ 完整部署
- **实操**：
  - C++ 实现 YOLOv8 TensorRT 推理
  - OpenCV 集成（图像读取 + 预处理 + 结果绘制）
  - 性能测试
- **产出**：`04-tensorrt-basics/07_cpp_yolo_deploy.cpp`

### Day 30：阶段总结
- **实操**：
  - 整理 TensorRT 部署流程
  - 性能对比报告（PyTorch vs ONNX RT vs TRT Python vs TRT C++）
  - 记录踩坑和解决方案
- **产出**：`notes/stage4_summary.md`

**阶段验证**：能用 C++ 和 Python 两种方式将 YOLOv8 部署到 TensorRT，完成端到端推理。

---

## 第五阶段：模型量化（第 5-6 周，7 天）

### Day 31：量化基础概念
- **实操**：
  - 理解量化原理（FP32 → INT8）
  - 量化 vs 精度 trade-off
  - PTQ（训练后量化）vs QAT（量化感知训练）
- **产出**：`06-model-quantization/01_quantization_basics.py`

### Day 32：PyTorch 量化
- **实操**：
  - 动态量化（torch.quantization.quantize_dynamic）
  - 静态量化（prepare → calibrate → convert）
  - 量化模型精度对比
- **产出**：`06-model-quantization/02_pytorch_quantization.py`

### Day 33：ONNX 量化
- **实操**：
  - onnxruntime.quantization 量化
  - 校准数据集准备
  - 量化前后精度 + 速度对比
- **产出**：`06-model-quantization/03_onnx_quantization.py`

### Day 34：TensorRT INT8 量化
- **实操**：
  - TensorRT INT8 校准器实现
  - 自定义 calibrator 类
  - INT8 engine 构建 + 推理
- **产出**：`06-model-quantization/04_trt_int8.py`

### Day 35：YOLOv8 量化部署
- **实操**：
  - YOLOv8 模型 INT8 量化
  - 量化精度分析（mAP 对比）
  - 量化性能对比（FP32 vs FP16 vs INT8）
- **产出**：`06-model-quantization/05_yolo_quantization.py`

### Day 36：量化最佳实践
- **实操**：
  - 量化敏感层分析
  - 混合精度策略
  - 量化问题排查清单
- **产出**：`06-model-quantization/06_quantization_best_practices.py`

### Day 37：阶段总结
- **实操**：
  - 量化方案对比报告
  - 精度-速度-模型大小 trade-off 分析
  - 记录踩坑和解决方案
- **产出**：`notes/stage5_summary.md`

**阶段验证**：能对 YOLOv8 进行 INT8 量化，理解精度损失和性能提升的关系。

---

## 第六阶段：自定义 CUDA 算子（第 6-8 周，12 天）

> 有 C++ 基础，CUDA 学习曲线会更平缓。12 天给够缓冲。

### Day 38：CUDA 基础
- **实操**：
  - CUDA 环境确认（nvcc, nvidia-smi）
  - 第一个 kernel（向量加法）
  - 理解 thread, block, grid
- **产出**：`07-custom-operators/01_cuda_basics.cu`

### Day 39：CUDA 内存模型
- **实操**：
  - 全局内存 vs 共享内存
  - 内存合并（coalesced access）
  - cudaMemcpy 设备-主机传输
- **产出**：`07-custom-operators/02_cuda_memory.cu`

### Day 40：向量运算 kernel
- **实操**：
  - 向量加法、点积
  - 归约（reduction）操作
  - 性能分析（nvprof / nsight）
- **产出**：`07-custom-operators/03_vector_ops.cu`

### Day 41：矩阵乘法 kernel
- **实操**：
  - 基础版矩阵乘法
  - 共享内存优化版（tiling）
  - 性能对比
- **产出**：`07-custom-operators/04_matmul.cu`

### Day 42：PyTorch C++ Extension
- **实操**：
  - 编写 C++ 算子
  - torch.utils.cpp_extension 编译
  - Python 端调用自定义算子
- **产出**：`07-custom-operators/05_pytorch_extension.py`

### Day 43：CUDA 加速 PyTorch 算子
- **实操**：
  - 将 CUDA kernel 封装为 PyTorch 算子
  - 前向 + 反向传播实现
  - autograd 注册
- **产出**：`07-custom-operators/06_cuda_pytorch_op.cu`

### Day 44：IoU 计算 CUDA 实现
- **实操**：
  - CPU 版 IoU 实现
  - CUDA kernel 版 IoU
  - 性能对比
  - 正确性验证
- **产出**：`07-custom-operators/07_iou_cuda.cu`

### Day 45：NMS CUDA 实现
- **实操**：
  - NMS 算法理解
  - CUDA 并行 NMS 实现
  - 与 torchvision.ops.nms 对比
- **产出**：`07-custom-operators/08_nms_cuda.cu`

### Day 46：TensorRT Plugin 基础
- **实操**：
  - TensorRT Plugin 架构理解
  - 实现一个简单 Plugin（如自定义激活函数）
  - Plugin 注册和使用
- **产出**：`07-custom-operators/09_trt_plugin.cpp`

### Day 47：NMS TensorRT Plugin
- **实操**：
  - 将 NMS 实现为 TensorRT Plugin
  - 集成到 YOLOv8 推理流程
  - 端到端验证
- **产出**：`07-custom-operators/10_nms_trt_plugin.cpp`

### Day 48：算子性能优化
- **实操**：
  - 共享内存优化
  - Warp 级别操作
  - Occupancy 优化
  - 性能分析工具使用
- **产出**：`07-custom-operators/11_optimization.cu`

### Day 49：阶段总结
- **实操**：
  - 整理 CUDA 算子代码
  - 性能对比报告（CPU vs CUDA vs TensorRT）
  - 记录踩坑和解决方案
- **产出**：`notes/stage6_summary.md`

**阶段验证**：能编写 CUDA kernel，封装为 PyTorch 算子，实现 NMS 等检测相关算子。

---

## 第七阶段：端到端部署项目（第 8-9 周，9 天）

> 把所有学到的串起来，做一个完整的检测部署方案

### Day 50：部署架构设计
- **实操**：
  - 设计部署流程图
  - 确定技术选型（TensorRT + C++ / Python）
  - 准备测试数据（真实图片/视频）
- **产出**：`08-final-project/01_architecture.md`

### Day 51：图像预处理流水线
- **实操**：
  - 图像读取 + letterbox resize
  - 归一化 + HWC→CHW
  - 批量预处理优化
- **产出**：`08-final-project/02_preprocessing.py`

### Day 52：推理引擎封装
- **实操**：
  - 封装 TensorRT 推理类
  - 支持动态 batch
  - 错误处理和日志
- **产出**：`08-final-project/03_inference_engine.py`

### Day 53：后处理 + NMS
- **实操**：
  - 解码检测结果
  - NMS 后处理
  - 使用自定义 CUDA NMS
  - 结果可视化
- **产出**：`08-final-project/04_postprocessing.py`

### Day 54：视频流检测
- **实操**：
  - 视频读取（OpenCV VideoCapture）
  - 逐帧检测 + 结果绘制
  - FPS 统计和输出
- **产出**：`08-final-project/05_video_detection.py`

### Day 55：HTTP API 服务
- **实操**：
  - FastAPI / Flask 搭建检测服务
  - 图片上传 + 检测 + 返回结果
  - 并发请求处理
- **产出**：`08-final-project/06_api_server.py`

### Day 56：C++ 完整部署
- **实操**：
  - C++ 实现完整检测流程
  - OpenCV 集成
  - 多线程优化（预处理/推理/后处理流水线）
- **产出**：`08-final-project/07_cpp_deploy.cpp`

### Day 57：性能测试 + 优化
- **实操**：
  - 端到端延迟测试
  - 各阶段耗时分析
  - 瓶颈优化
  - 性能对比报告
- **产出**：`08-final-project/08_performance_report.py`

### Day 58：项目总结 + 文档
- **实操**：
  - 编写部署文档
  - 整理全部代码
  - 总结学习路线
  - 规划后续深入方向
- **产出**：`08-final-project/09_deployment_doc.md`

**阶段验证**：能完成从图片/视频输入到检测结果输出的完整部署方案。

---

## 缓冲时间（2 天）

Day 59-60 作为缓冲，用于：
- 补之前没完成的内容
- 深入某个感兴趣的方向
- 复习和知识串联

---

## 进度跟踪

| 阶段 | 天数 | 内容 | 状态 |
|------|------|------|------|
| 1 | 1-7 | PyTorch 基础 + CNN 分类 | ⬜ |
| 2 | 8-17 | YOLOv8 目标检测训练 | ⬜ |
| 3 | 18-22 | ONNX 导出深入 | ⬜ |
| 4 | 23-30 | TensorRT 部署 | ⬜ |
| 5 | 31-37 | 模型量化 | ⬜ |
| 6 | 38-49 | 自定义 CUDA 算子 | ⬜ |
| 7 | 50-58 | 端到端部署项目 | ⬜ |
| - | 59-60 | 缓冲 | ⬜ |

**总计 60 天（约 8.5 周），每天 1 小时**

---

## 环境准备清单（开始前完成）

- [ ] Python 3.8+，pip install torch torchvision torchaudio
- [ ] CUDA toolkit（匹配 PyTorch CUDA 版本）
- [ ] cuDNN
- [ ] pip install onnx onnxruntime onnxsim
- [ ] TensorRT（从 NVIDIA 官网下载，匹配 CUDA 版本）
- [ ] pip install ultralytics
- [ ] pip install opencv-python matplotlib numpy
- [ ] pip install labelimg（标注工具，Day 13 用）
- [ ] nvidia-smi 确认 GPU 可用

## 学习资源

- PyTorch 教程：https://pytorch.org/tutorials/
- Ultralytics 文档：https://docs.ultralytics.com/
- ONNX 文档：https://onnx.ai/onnx/
- TensorRT 文档：https://docs.nvidia.com/deeplearning/tensorrt/
- CUDA 编程指南：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 李沐《动手学深度学习》：https://zh.d2l.ai/
