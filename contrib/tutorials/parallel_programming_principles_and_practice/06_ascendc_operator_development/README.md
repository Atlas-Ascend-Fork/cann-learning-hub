![算子开发：SpMV + RoPE](images/readme_cover.png)

# Ascend C 算子开发

本章包含 SpMV 和 RoPE 两部分：先编译运行 FP32 SpMV，阅读分行 Kernel，再运行 RoPE 的向量计算；BF16 精度、分区与数据复用放在扩展部分。两部分完成后一起做章节练习。

## 开始前

完成第 02、04 章，了解 C++、CSR 和 ACL 调用。RoPE 还需要张量布局和三角函数基础。

## 学习目标

- 解释 CSR 与 RoPE 旋转对的数据布局。
- 编译运行 Ascend C 算子并检查误差。
- 比较精度、分区和数据复用的影响。
- 阅读 SIMD 与 SIMT 的并行计算方式。

## 学习顺序

按下表打开 Notebook。每个实验已包含相关原理、编译运行和结果分析；做完实验后再完成章节练习。

| 顺序 | Notebook | 内容 |
|---|---|---|
| 1 | [SpMV 实验](01_ascendc_spmv/06_spmv.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 2 | [RoPE 实验](02_rope_operator/06_rope.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 3 | [第 06 章练习](06_practice.ipynb) | 检查理解、完成编程与分析题 |

## 实验环境

课程使用 CANNLab Atlas A3（SoC `Ascend910_9362`）、CANN 9.0，镜像为 `cann_9.0.0-py3.11-A3-arm-20260829`，Notebook 内核为 `Python 3.11.4`。平台使用方法见[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。

A3 上运行 SpMV FP32 和 RoPE SIMD。SpMV 的 FP16、BF16 和 persistent 使用 CPU 原型；RoPE SIMT 是 Ascend 950 模板，实测需要对应硬件。

## 阅读源码

阅读 [spmv.hpp](01_ascendc_spmv/src/ascend_spmv/include/spmv.hpp)，看 `prepare()` 准备哪些数据、`run()` 执行哪些计算。

- [ascendc_spmv.cpp](01_ascendc_spmv/src/ascend_spmv/src/ascendc_spmv.cpp)：查看 FP32 实现怎样复制设备数据、启动 Kernel 和回读结果。
- [npu_spmv_context.cpp](01_ascendc_spmv/src/ascend_spmv/src/npu_spmv_context.cpp)：查看 CPU 原型怎样选择分区并复用矩阵数据。

6.2 节运行、6.5 节读取结果后，用本次 CSV 中的初始化时间、计算时间和误差分析这些做法的影响。
