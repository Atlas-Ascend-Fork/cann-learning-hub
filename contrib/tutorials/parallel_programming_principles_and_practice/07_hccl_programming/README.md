![HCCL 编程](images/readme_cover.png)

# HCCL 编程

在 SpMV 中使用 HCCL 广播输入、收集输出，理解设备计算与集合通信如何配合。先完成单卡数据流检查，再根据资源选做多卡比较。

## 开始前

完成第 03、04 章，理解 CSR 和 MPI 集合通信。准备至少一张可用的昇腾卡，以及 CANN、ACL RTC 和 HCCL。

## 学习目标

- 设置进程编号、设备编号和通信组。
- 解释 Broadcast、AllGather 与补零。
- 检查结果，区分通信、传输和本地计算时间。
- 有多卡时比较增加设备后的耗时变化。

## 学习顺序

按下表打开 Notebook。每个实验已包含相关原理、编译运行和结果分析；做完实验后再完成章节练习。

| 顺序 | Notebook | 内容 |
|---|---|---|
| 1 | [HCCL 通信与 SpMV实验](07_lab.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 2 | [第 07 章练习](07_practice.ipynb) | 检查理解、完成编程与分析题 |

## 实验环境

课程使用 CANNLab Atlas A3（SoC `Ascend910_9362`）、CANN 9.0，镜像为 `cann_9.0.0-py3.11-A3-arm-20260829`，Notebook 内核为 `Python 3.11.4`。平台使用方法见[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。

单卡可完成计算、通信初始化和结果检查。多卡实验至少需要两张卡；目前课程已有的 A3 验证记录覆盖单卡，多卡需在相应资源上运行。

## 阅读源码

7.4 节运行单卡程序后，阅读 [spmv_distributed_hccl.cpp](src/hccl_spmv/src/spmv_distributed_hccl.cpp)，找到 Broadcast、局部 SpMV 和 AllGather 的调用顺序。

再看 [hccl_context.cpp](src/hccl_spmv/src/hccl_context.cpp)，了解每个进程怎样选择设备、建立通信组。多卡运行时，将这些设置与各 rank 的日志对照。
