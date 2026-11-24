[![并行编程原理与实践（基于国产平台）](images/readme_cover.png)](https://e.huawei.com/cn/talent/learning/#/zone?customizedZoneId=CGuWmOTPyk8D6pEvQS7esX9GJCg)

---

# 并行编程原理与实践（基于国产平台）

## 课程简介

本课程围绕国产昇腾平台上的并行编程与应用实践展开，依次介绍毕昇编译工具链、OpenMP、MPI、AscendCL、Ascend C、HCCL，以及 CANN 应用调试、调优和集群作业部署。课程包含 10 个实验，共 22 课时。

## 适合人群

面向具备 C/C++ 或 Python 基础，希望学习并行编程、昇腾算子调用与 CANN 应用开发的学习者。建议按照下表顺序学习；各实验的具体前置要求见对应章节 README。

## 学习目标

完成课程后，学习者能够：

- 使用毕昇编译器完成基础 Host 工程构建；
- 理解 OpenMP、MPI 与 HCCL 的并行和通信模型；
- 通过 C/C++、Python 接口调用昇腾算子，并完成 Ascend C 算子开发；
- 使用 CANN 工具定位程序问题，分析应用瓶颈；
- 理解 CANN/HPC 应用的构建部署、作业运行与结果管理流程。

## 课程支持的硬件产品

| 项目 | 要求 |
|---|---|
| 已验证硬件 | Atlas A3（SoC Ascend910_9362） |
| CANN 版本 | 9.0.0 |
| Python | 3.11 |
| 多卡要求 | HCCL 多卡实验需要 2 张及以上 NPU；单卡环境可完成 world1 学习路径 |

## 已验证的在线体验环境

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
|---|---|---|---|
| CANNLab 云开发环境 | `cann_9.0.0-py3.11-A3-arm-20260829` | Python 3.11.4 | 按章节 README 的环境说明运行 |

环境创建方法参见 [CANNLab 环境体验指南](../../../docs/CANNLab_env_experience_guide.md)。

## 课程章节目录

| 顺序 | 实验名称 | 课时 | 章节链接 |
|---|---|---:|---|
| 01 | 毕昇编译器安装 | 2 | [01_bisheng_toolchain](01_bisheng_toolchain) |
| 02 | OpenMP 编程 | 2 | [02_openmp_programming](02_openmp_programming) |
| 03 | MPI 编程 | 2 | [03_mpi_programming](03_mpi_programming) |
| 04 | ACL C/C++ 算子库调用 | 2 | [04_acl_cpp_operator_calls](04_acl_cpp_operator_calls) |
| 05 | ACL Python 算子库调用 | 2 | [05_acl_python_operator_calls](05_acl_python_operator_calls) |
| 06 | 算子开发（SpMV + RoPE） | 4 | [06_ascendc_operator_development](06_ascendc_operator_development) |
| 07 | HCCL 编程 | 2 | [07_hccl_programming](07_hccl_programming) |
| 08 | CANN 编程错误调试 | 2 | [08_cann_debugging_tools](08_cann_debugging_tools) |
| 09 | CANN 应用调优（Dis-GMRES + MuduoXinyu） | 2 | [09_cann_application_tuning](09_cann_application_tuning) |
| 10 | 线性求解器部署与分析（Xyce） | 2 | [10_cluster_job_deployment](10_cluster_job_deployment) |

第 06 章包含 SpMV 和 RoPE 两个部分，第 09 章包含 Dis-GMRES 和 MuduoXinyu 两个部分。具体学习顺序见各章 README。

OpenMP、MPI、Ascend C、GMRES 与 Xyce 共用 `common/sparse` 中的 CSR 数据与 CPU 参考实现。各章的编译和运行步骤见对应 Notebook。

## 第一次做实验

请克隆完整仓库，以便使用各章需要的共用源码。第 10 章还会使用第 09 章的 GMRES 求解器。

```bash
git clone https://gitcode.com/cann/cann-learning-hub.git
cd cann-learning-hub/contrib/tutorials/parallel_programming_principles_and_practice
```

下载后，先完成 [CANN 开发环境检查](01_bisheng_toolchain/01_lab.ipynb#s-01-02_cann_development_environment)。在 CANNLab 中，选择平台提供的 CANN Python 环境运行 Notebook。

每章 README 都列出了学习顺序。按顺序打开其中的 `.ipynb` 文件，从上到下阅读说明、执行代码。实验命令默认从当前 `.ipynb` 文件所在文件夹执行；使用终端时，先进入这个文件夹。

| 实验 | 环境与依赖 | 主要观察结果 |
|---|---|---|
| 01 毕昇 | 鲲鹏毕昇编译器、CANN 工具链 | 工具版本、编译命令、程序输出 |
| 02 OpenMP | C++ 编译器、CMake、OpenMP | 不同线程数和调度方式下的误差、耗时与加速比 |
| 03 MPI | MPI 开发包、`mpicxx`、`mpirun`，以及运行进程所需的 CPU 资源 | 不同进程数下的误差、计算时间、通信时间和总时间 |
| 04 ACL C/C++ | NPU、CANN、ops-sparse 稀疏算子库；安装步骤见本章实验手册 | GEMM 和 SpMV 的计算结果、相对误差与耗时 |
| 05 ACL Python | NPU、pyACL、NumPy、ATC；单算子模型按本章步骤生成 | GEMM 和 SpMV 与 NumPy 结果的误差、执行耗时及所用计算单元 |
| 06 Ascend C | NPU、CANN 的 RTC 运行时编译功能 | SpMV 的误差和耗时；RoPE 四组输入的校验结果与耗时 |
| 07 HCCL | NPU、CANN、HCCL；多卡扩展实验需至少两张 NPU | 单卡计算的误差和耗时；选做多卡实验时记录各进程日志、通信时间和总时间 |
| 08 调试工具 | msSanitizer、msDebug；使用 msDebug 前需平台开通调试权限 | 故障诊断、源码位置、退出码，以及修复前后的检查结果 |
| 09 应用调优 | NPU、CANN；Muduo 另需源码、stories110M 模型和配套分词器文件 | GMRES 的残差、解误差和耗时；Muduo 的生成结果和 A/B 吞吐量 |
| 10 部署管理 | NPU、CANN、CMake，以及包含第 09 章和共用源码的完整课程目录 | 三种求解器的收敛状态、解误差、耗时和结果 CSV |

先完成第 01 章的环境检查，再按各章步骤运行。遇到依赖缺失时，查看该章的准备说明。

## 作者团队

计卫星、高建花、王一拙、杨怡雪、傅俊林
