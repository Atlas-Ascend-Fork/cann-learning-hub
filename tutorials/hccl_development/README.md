# HCCL 集合通信系列教程

本教程面向昇腾集合通信库 HCCL（Huawei Collective Communication Library）初学者，从单卡训练的瓶颈讲起，帮助读者理解集合通信的由来与 HCCL 的定位；随后认全常用集合通信算子，理解 Ring、Mesh、Halving-Doubling 三种算法的差别与适用场景；并带领读者在**普通 CPU 环境**（无需昇腾 NPU）中运行 HCCL-VM 虚拟通信平台，亲手跑通通信测试、解读结果，完成一次「问题定位 → 修改验证」的完整闭环。后续章节将覆盖 HCCL 软件架构与编程模型、AICPU_TS / CCU / AIV 引擎，以及 AI 集群中的 HCCL 问题定位。

教程按章节划分，每个章节包含以下内容：

- Notebooks：包含课程知识点与课堂小练，适用于自主学习或讲师引导式教学，可在 cannlab 环境里运行，也可在自行搭建的 JupyterLab 环境中使用。
- images：包含 Notebook 引用的原理示意图、算法动图与实跑截图。
- answer：包含章节练习参考答案。

## 相关仓库

| 仓库 | 地址 |
| --- | --- |
| HCCL | https://gitcode.com/cann/hccl |
| HCOMM | https://gitcode.com/cann/hcomm |

课程中的集合通信算子、算法与 `hccl_test` 测试工具均来自上述两个仓库，欢迎各位开发者前往体验、提出 Issue 或参与共建。

## 课程结构

### HCCL 简介与集合通信基础（初级）

| Notebook | Link | 状态 |
| --- | --- | --- |
| 1.1 本章介绍 | [查看 Notebook](./01_hccl_intro_collective_comm/01.01_chapter_intro.ipynb) | ✅ 已发布 |
| 1.2 集群通信介绍 | [查看 Notebook](./01_hccl_intro_collective_comm/01.02_cluster_comm_intro.ipynb) | ✅ 已发布 |
| 1.3 HCCL | [查看 Notebook](./01_hccl_intro_collective_comm/01.03_hccl_intro.ipynb) | ✅ 已发布 |

### HCCL 基础算子与算法（初级）

| Notebook | Link | 状态 |
| --- | --- | --- |
| 2.1 本章介绍 | [查看 Notebook](./02_hccl_operators_algorithms/02.01_chapter_intro.ipynb) | ✅ 已发布 |
| 2.2 HCCL 算子介绍 | [查看 Notebook](./02_hccl_operators_algorithms/02.02_operators.ipynb) | ✅ 已发布 |
| 2.3 HCCL 算法简介 | [查看 Notebook](./02_hccl_operators_algorithms/02.03_algorithms.ipynb) | ✅ 已发布 |

### 北极星工具 HCCL-VM：安装与应用（初级）

| Notebook | Link | 状态 |
| --- | --- | --- |
| 3.1 本章介绍 | [查看 Notebook](./03_hccl_vm_guide/03.01_chapter_intro.ipynb) | ✅ 已发布 |
| 3.2 平台背景与原理 | [查看 Notebook](./03_hccl_vm_guide/03.02_hccl_vm_platform_overview.ipynb) | ✅ 已发布 |
| 3.3 获取并解压 HCCL-VM 绿色包 | [查看 Notebook](./03_hccl_vm_guide/03.03_cloud_pack_setup.ipynb) | ✅ 已发布 |
| 3.4 认识测试用例：hccl_test 的用法与结果解读 | [查看 Notebook](./03_hccl_vm_guide/03.04_test_cases.ipynb) | ✅ 已发布 |
| 3.5 跑通第一个通信测试并解读结果 | [查看 Notebook](./03_hccl_vm_guide/03.05_first_comm_test.ipynb) | ✅ 已发布 |
| 3.6 配置自己的通信测试：集群、通信域与 rank 映射 | [查看 Notebook](./03_hccl_vm_guide/03.06_configure_tests.ipynb) | ✅ 已发布 |
| 3.7 通信用例实战：四个用例走完一组测试 | [查看 Notebook](./03_hccl_vm_guide/03.07_communication_cases.ipynb) | ✅ 已发布 |
| 3.8 问题定位：按步骤排查与日志阅读 | [查看 Notebook](./03_hccl_vm_guide/03.08_debug_and_verify.ipynb) | ✅ 已发布 |
| 3.9 本地 WSL 环境与自建绿色包（选看） | [查看 Notebook](./03_hccl_vm_guide/03.09_appendix_local_env.ipynb) | ✅ 已发布 |

### 后续章节（待发布）

以下章节的 PDF 课件见配套的 slides 目录，Notebook 版正在按「小节标题 + 实跑验证」的同一标准逐章转写，小节划分与链接将在各章转写完成后补充。

| 章节 | Notebook | 状态 |
| --- | --- | --- |
| 4. HCCL 软件架构与编程模型 | — | 🔄 待发布 |
| 5. AICPU_TS 引擎 | — | 🔄 待发布 |
| 6. CCU 引擎 | — | 🔄 待发布 |
| 7. AIV 引擎 | — | 🔄 待发布 |
| 8. AI 集群 HCCL 问题定位 | — | 🔄 待发布 |

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | 第 1~2 章为理论讲解，无需任何设备；第 3 章基于 HCCL-VM 通信仿真平台，普通 CPU 即可运行，**无需昇腾 NPU** |
| 云端环境（推荐） | **cannlab 环境**：CPU、架构 aarch64；自带 CANN 9.0.0（**包不依赖它**，见下） |
| 本地环境（选看） | Windows 11 + WSL2 + Ubuntu 24.04（x86_64），需自装 CANN |
| 绿色包 | `hccl_vm_pack.tar.gz`，约 82 MB，见仓库 Release 附件 |
| Python | 3.x（Notebook 使用 python3 内核） |

## 第 3 章 HCCL-VM在线体验注意事项

第 3 章提供**两条路径**：
- **云端（主线）**：在 cannlab 环境里下载一个 82 MB 的绿色包即可运行，**不需要安装 CANN、不需要编译**，约 5 分钟出结果；
- **本地（选看）**：在自己电脑的 WSL 里装 CANN、编译 HCCL-VM，可以改源码、重新打包，见 [3.9](./03_hccl_vm_guide/03.09_appendix_local_env.ipynb)。

> **注意：**
> - 第 3 章云端路径已在 **cannlab 环境（aarch64 + CANN 9.0.0）** 上完整实测；本地路径在 **x86_64** 上实测。
>   其它架构/版本使用可能存在问题，欢迎开发者提出 issue 或 PR 进行共建。
> - 绿色包与**架构**绑定（仅 aarch64）；与运行环境的 CANN 版本无关——包内自带库并由 RPATH 强制优先。

从 [3.3 获取并解压 HCCL-VM 绿色包](./03_hccl_vm_guide/03.03_cloud_pack_setup.ipynb) 开始，完成绿色包的下载安装，
再按 [3.5 跑通第一个通信测试并解读结果](./03_hccl_vm_guide/03.05_first_comm_test.ipynb) 的顺序点运行，
最后看到 `CHECKER_RUN_SUMMARY  All Success` 即为跑通。
