![ACL C/C++ 算子库调用](images/readme_cover.png)

# ACL C/C++ 算子库调用

用 C++ 调用 GEMM 和 SpMV，学习设备数据准备、算子执行、结果校验和资源释放。GEMM 使用 ACLNN，SpMV 使用 ops-sparse。

## 开始前

完成第 01 章，了解矩阵乘和 CSR。准备 CANN Toolkit；SpMV 另需 ops-sparse，可在 [4.4 节](04_spmv.ipynb#s-04-04_acl_sparse_spmv)点击检查和安装。

## 学习目标

- 管理设备、上下文、Stream 和内存。
- 分别运行 GEMM 与 SpMV，并与 CPU 结果比较。
- 说明两种 API 在资源创建、执行和释放上的差异。

## 学习顺序

按下表打开 Notebook。每个实验已包含相关原理、编译运行和结果分析；做完实验后再完成章节练习。

| 顺序 | Notebook | 内容 |
|---|---|---|
| 1 | [ACL C++ 算子调用：GEMM](04_gemm.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 2 | [ACL C++ 算子调用：SpMV](04_spmv.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 3 | [第 04 章练习](04_practice.ipynb) | 检查理解、完成编程与分析题 |

## 实验环境

课程使用 CANNLab Atlas A3（SoC `Ascend910_9362`）、CANN 9.0，镜像为 `cann_9.0.0-py3.11-A3-arm-20260829`，Notebook 内核为 `Python 3.11.4`。平台使用方法见[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。

使用 `ACL_C_STUB=OFF` 构建设备版本。编译前确认 CANN 和 ops-sparse 头文件、动态库已就绪。

本章使用 ops-sparse 的固定提交 [e0015bb](https://gitcode.com/cann/ops-sparse/commit/e0015bb76090487165f43dbaf2aae37648887a4b)。4.4 节会检查安装目录；缺少库时，点击安装单元格即可获取该版本源码、构建并安装。

## 阅读源码

4.3、4.4 节分别运行两个程序，再对照源码查看输入准备、算子调用、结果回读和资源释放：

- [GEMM 示例](src/acl_operator_calls/GEMM-acl/src/main.cpp)：调用 ACLNN 矩阵乘。
- [SpMV 示例](src/acl_operator_calls/SpMV-acl/src/main.cpp)：用 CSR 描述稀疏矩阵，再调用 ops-sparse。

4.5 节比较这两种调用方式需要创建和释放哪些对象。
