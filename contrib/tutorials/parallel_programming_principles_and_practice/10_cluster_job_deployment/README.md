# 线性求解器部署与分析

部署并运行 Ascend-Xyce 线性求解器测试程序，比较 CPU 单线程、OpenMP 和设备求解器。学习准备依赖、构建、配置输入输出和分析日志。

## 开始前

完成第 09 章，了解 C++ 与 CMake。CPU 求解器随本章提供，设备求解器引用第 09 章源码，请保留完整课程目录。

## 学习目标

- 追踪应用、适配器和求解器的调用。
- 完成编译运行，检查收敛和误差。
- 比较矩阵装配、求解和总时间。
- 按日志定位故障；有集群队列时选做作业提交。

## 学习顺序

按下表打开 Notebook。每个实验已包含相关原理、编译运行和结果分析；做完实验后再完成章节练习。

| 顺序 | Notebook | 内容 |
|---|---|---|
| 1 | [线性求解器部署实验](10_lab.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 2 | [第 10 章练习](10_practice.ipynb) | 检查理解、完成编程与分析题 |

## 实验环境

课程使用 CANNLab Atlas A3（SoC `Ascend910_9362`）、CANN 9.0，镜像为 `cann_9.0.0-py3.11-A3-arm-20260829`，Notebook 内核为 `Python 3.11.4`。平台使用方法见[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。

默认在 CANNLab 的 Notebook 中逐个点击运行代码单元格，不需要调度器。实验手册也保留了终端命令。测试程序比较线性求解器；完整 Xyce 电路仿真另需 Xyce、第三方依赖和电路网表，属于扩展内容。

## 阅读源码

先看 [xyce_benchmark.cpp](src/ascend_xyce/benchmark/xyce_benchmark.cpp)怎样配置输入和启动测试，再看 [xyce_adapter.cpp](src/ascend_xyce/src/xyce_adapter.cpp)怎样构造方程、选择求解器并检查结果。

CPU 求解器随本章提供，设备求解器使用第 09 章的 GMRES。10.2 节说明它们的调用关系，10.3 节完成构建，10.4 节运行并保存 CSV。
