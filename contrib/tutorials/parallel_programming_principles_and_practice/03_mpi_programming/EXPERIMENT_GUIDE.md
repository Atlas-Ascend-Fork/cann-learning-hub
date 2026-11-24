# MPI SpMV 实验手册

## 1. 目标、代码与通信流程

本实验在 **CPU** 上执行局部 CSR SpMV，用 MPI 完成按 CSR 行划分计算任务、向量广播和结果汇集。初始化使用 `MPI_Init`、`MPI_Comm_rank`、`MPI_Comm_size`；每个进程（rank）获得一个行区间，`MPI_Bcast` 广播完整 `x`，`MPI_Gatherv` 按不等行数收集 `local_y`。

```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &process_count);
MPI_Bcast(x->data(), static_cast<int>(x->size()), MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv(local_y.data(), local_rows, MPI_FLOAT, /* ... */, MPI_COMM_WORLD);
```

## 2. 编译与单机实验

在第 03 章目录打开终端。进程数不超过分配的 CPU 核心数，使用不同 CSV 文件保存各次结果。

```bash
cd src/mpi_spmv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build -j
mkdir -p results
mpirun -np 1 ./build/spmv_mpi --matrix U1 --repeat 3 --csv results/U1_1p.csv
mpirun -np 2 ./build/spmv_mpi --matrix U1 --repeat 3 --csv results/U1_2p.csv
mpirun -np 4 ./build/spmv_mpi --matrix U1 --repeat 3 --csv results/U1_4p.csv
```

单机多进程用于验证 Rank 划分、Broadcast/Gatherv 和正确性，不能替代多节点扩展性实验。

## 3. 多节点实验（选做）

创建 `hostfile`：

```text
node01 slots=2
node02 slots=2
```

两台节点使用相同程序路径和 MPI 环境后执行：

```bash
mpirun -np 4 --hostfile hostfile ./build/spmv_mpi
```

## 4. 验证、记录与分析

每次运行都与当次 CPU 串行结果比较，检查绝对 L2 误差小于 `1e-6`，同时记录最大绝对误差。逐进程记录通信域规模（size）、row range、compute/communication/total time；负载不均衡定义为 `(max_rank_compute-avg_rank_compute)/avg_rank_compute`，这里先求每个 rank 的平均 compute，再跨 rank 取最大值和平均值；取数用 `.per_rank.csv`，而非汇总行的 compute。

| Ranks | Compute (ms) | Communication (ms) | Total (ms) | Imbalance | Absolute L2 error | Pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | | | | | | |
| 2 | | | | | | |
| 4 | | | | | | |

计时先取每轮各 rank 的最大值，再对重复取平均。total 包含每轮广播、计算与汇集；一次性 CSR 分发和轮首 Barrier 不在 total 内。各阶段最大值可能来自不同 rank，不要求它们之和等于 total。

练习：手算同一个小矩阵按行数与按 nnz 划分的负载差异；现有程序按 nnz 划分，没有分区切换参数。再用实测解释通信占比随 rank 数的变化。参考答案见 `answer/`。
