![OpenMP 编程](images/readme_cover.png)

# OpenMP 编程

从串行稀疏矩阵向量乘（SpMV）出发，用 OpenMP 将矩阵的行分配给多个 CPU 线程，比较线程数和调度方式对耗时的影响。

## 开始前

完成第 01 章，了解数组和循环。准备鲲鹏毕昇 Host 编译器、CMake 和 OpenMP 运行库。

## 学习目标

- 用 CSR 格式表示稀疏矩阵并手算 SpMV。
- 设置线程数和调度方式，检查并行结果。
- 计算加速比与效率，分析负载不均和内存访问开销。

## 学习顺序

按下表打开 Notebook。每个实验已包含相关原理、编译运行和结果分析；做完实验后再完成章节练习。

| 顺序 | Notebook | 内容 |
|---|---|---|
| 1 | [OpenMP SpMV实验](02_lab.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 2 | [第 02 章练习](02_practice.ipynb) | 检查理解、完成编程与分析题 |

## 实验环境

课程使用 CANNLab Atlas A3（SoC `Ascend910_9362`）、CANN 9.0，镜像为 `cann_9.0.0-py3.11-A3-arm-20260829`，Notebook 内核为 `Python 3.11.4`。平台使用方法见[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。

本章在主机 CPU 上运行，使用第 01 章安装的毕昇 Host 3.2.0。线程数不超过分配的 CPU 核心数。2.4～2.6 节已提供编译器配置单元格，点击运行后再编译，无需从第 01 章复制代码。

若 WebIDE 一直显示 `Waiting for Jupyter Session to be idle`，且代码单元还没有执行编号，先点击“中断”，将 Notebook 另存为一个新名称，再打开副本运行。

## 阅读源码

2.4 节编译运行后，可对照以下源码理解结果：

- [cpu_omp_spmv.cpp](src/openmp_spmv/src/cpu_omp_spmv.cpp)：看矩阵的行如何分配给线程，以及每个线程怎样累加一行。
- [spmv_benchmark.cpp](src/openmp_spmv/benchmark/spmv_benchmark.cpp)：看串行时间、并行时间和误差怎样计算。

矩阵生成和串行计算使用共用目录 [common/sparse](../common/sparse)，后面的线程数比较使用相同输入。
