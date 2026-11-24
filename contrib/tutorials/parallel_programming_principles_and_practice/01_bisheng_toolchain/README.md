![毕昇编译器安装](images/readme_cover.png)

# 毕昇工具链

检查 CANN、编译器与设备环境，编译运行一个 C++ 程序。本章为后续 OpenMP、MPI、ACL 和算子开发实验准备工具。

## 开始前

会使用 Linux 终端，了解基础 C/C++。先完成环境检查，再按 1.3 节安装鲲鹏毕昇 Host 编译器。

## 学习目标

- 查看工具版本、安装位置和设备信息。
- 区分主机 C++ 与 Ascend C 编译。
- 编译运行示例，并根据报错定位缺失依赖。

## 学习顺序

按下表打开 Notebook。每个实验已包含相关原理、编译运行和结果分析；做完实验后再完成章节练习。

| 顺序 | Notebook | 内容 |
|---|---|---|
| 1 | [毕昇工具链实验](01_lab.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 2 | [第 01 章练习](01_practice.ipynb) | 检查理解、完成编程与分析题 |

## 实验环境

课程使用 CANNLab Atlas A3（SoC `Ascend910_9362`）、CANN 9.0，镜像为 `cann_9.0.0-py3.11-A3-arm-20260829`，Notebook 内核为 `Python 3.11.4`。平台使用方法见[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。

本课程使用鲲鹏毕昇 Host 3.2.0（ARM64）。1.3 节提供[华为云镜像软件包](https://mirrors.huaweicloud.com/kunpeng/archive/compiler/bisheng_compiler/)的下载和解压单元格，安装后可直接继续编译实验。
