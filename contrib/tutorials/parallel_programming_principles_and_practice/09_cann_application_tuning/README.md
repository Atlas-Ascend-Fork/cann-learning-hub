![CANN 应用调优：Dis-GMRES + MuduoXinyu](images/readme_cover.png)

# CANN 应用调优

通过 GMRES 和 MuduoXinyu 学习性能分析：先运行基线，找到主要耗时，再比较参数或实现的变化。两部分均可在单卡上完成。

## 开始前

了解 C++、ACL 调用和基本性能统计。GMRES 需要 CSR 与向量运算基础；MuduoXinyu 需要模型推理基础，并按 9.6 节下载源码、模型和 tokenizer。

## 学习目标

- 用 msprof 查看 GMRES 的主要耗时。
- 比较 restart=4 与 30，解释收敛与计算成本的变化。
- 运行 MuduoXinyu 两种 Attention 实现，检查输出 token 并统计吞吐量。
- 根据本次结果解释是否加速。

## 学习顺序

按下表打开 Notebook。每个实验已包含相关原理、编译运行和结果分析；做完实验后再完成章节练习。

| 顺序 | Notebook | 内容 |
|---|---|---|
| 1 | [GMRES 调优实验](01_distributed_gmres/09_gmres.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 2 | [MuduoXinyu Attention 实验](02_muduoxinyu_optimization/09_muduo.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 3 | [第 09 章练习](09_practice.ipynb) | 检查理解、完成编程与分析题 |

## 实验环境

课程使用 CANNLab Atlas A3（SoC `Ascend910_9362`）、CANN 9.0，镜像为 `cann_9.0.0-py3.11-A3-arm-20260829`，Notebook 内核为 `Python 3.11.4`。平台使用方法见[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。

GMRES 使用单卡，按已有方向逐个减去投影完成正交化（MGS）。MuduoXinyu 的 A 为 FP32 多算子，B 为 FP16 FlashAttentionV4；两者同时改变精度和算子组织，比较结论针对完整实现。先检查功能与数据，再判断性能。

## 阅读源码

9.2 节采集性能数据后，可从以下文件寻找耗时对应的计算：

- [npu_gmres.cpp](01_distributed_gmres/src/dis_gmres/src/npu_gmres.cpp)：查看一次 GMRES 迭代怎样调用 SpMV 和向量运算。
- [gmres_ops.cpp](01_distributed_gmres/src/dis_gmres/kernels/gmres_ops.cpp)：查看 SpMV、Dot、Norm、AXPY 和 Scale 的设备计算。
- [npu_compute.cpp](01_distributed_gmres/src/dis_gmres/src/npu_compute.cpp)：查看设备内存、运行时编译和 Kernel 启动。

分析时将源码中的调用与本次 msprof 输出对应起来，再比较改变 restart 前后的结果。
