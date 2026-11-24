![MPI 编程](images/readme_cover.png)

# MPI 编程

用 MPI 将 SpMV 分给多个进程，学习分发矩阵、广播输入向量和汇集计算结果。与 OpenMP 线程不同，MPI 进程需要显式传递数据。

## 开始前

完成第 02 章，理解 CSR 和加速比。环境中需有 `mpicxx` 与 `mpirun`。

## 学习目标

- 理解进程编号、通信组和集合通信。
- 比较按行数与按非零元数量分区。
- 分别分析通信、局部计算和总时间。

## 学习顺序

按下表打开 Notebook。每个实验已包含相关原理、编译运行和结果分析；做完实验后再完成章节练习。

| 顺序 | Notebook | 内容 |
|---|---|---|
| 1 | [MPI SpMV实验](03_lab.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 2 | [第 03 章练习](03_practice.ipynb) | 检查理解、完成编程与分析题 |

## 实验环境

课程使用 CANNLab Atlas A3（SoC `Ascend910_9362`）、CANN 9.0，镜像为 `cann_9.0.0-py3.11-A3-arm-20260829`，Notebook 内核为 `Python 3.11.4`。平台使用方法见[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。

默认在单机 CPU 上运行，进程数不超过分配的核心数。多节点实验按实验手册选做。

## 阅读源码

3.4 节运行后，阅读 [spmv_mpi.cpp](src/mpi_spmv/src/spmv_mpi.cpp)，依次找到矩阵分发、向量广播、各进程的局部计算和结果汇集。

矩阵的生成与读取见共用源码 [csr_matrix.cpp](../common/sparse/src/csr_matrix.cpp)。将 3.3 节手算的分区与程序输出对照，检查每个进程分到了哪些行。
