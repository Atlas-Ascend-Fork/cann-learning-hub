# OpenMP SpMV 实验手册

## 1. 实验目标与后端

本实验在 CPU 上用 OpenMP 并行计算 CSR SpMV，比较 1、2、4 线程的结果和时间；资源足够时再加 8 线程。通过运行参数改变配置即可。

## 2. 核心原理

核心循环位于 `src/openmp_spmv/src/cpu_omp_spmv.cpp`：

```cpp
#pragma omp parallel for schedule(runtime)
for (std::int32_t row = 0; row < matrix.rows; ++row) {
    float sum = 0.0F;
    for (std::int32_t index = matrix.row_ptr[row];
         index < matrix.row_ptr[row + 1]; ++index) {
        sum += matrix.values[index] * x[matrix.col_idx[index]];
    }
    (*y)[row] = sum;
}
```

各线程共享只读的 CSR 数组和 `x`；`row`、`index`、`sum` 是每次迭代私有变量。每个迭代只写不同的 `y[row]`，因此没有写冲突。`OMP_NUM_THREADS` 控制线程数，`OMP_SCHEDULE` 控制 `schedule(runtime)` 的分配策略。行长度均匀时 `static` 开销较低；长尾矩阵可比较 `dynamic` 的负载均衡收益。

课程提供 6 个矩阵：U1/U2 的行长度较均匀，L1/L2 的长行集中在连续区域，B1/B2 为块状稀疏结构。后缀 1/2 表示两档规模；同一轮比较只改变线程数或调度策略。

## 3. 编译与运行

先完成第 01 章的编译器安装。在第 02 章目录打开终端，设置编译器路径后再编译；线程数不超过分配的 CPU 核心数。

```bash
export PATH="$HOME/opt/BiShengCompiler-3.2.0-aarch64-linux/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/BiShengCompiler-3.2.0-aarch64-linux/lib:$HOME/opt/BiShengCompiler-3.2.0-aarch64-linux/lib/aarch64-unknown-linux-gnu:$LD_LIBRARY_PATH"
clang++ --version
cd src/openmp_spmv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j

OMP_NUM_THREADS=1 OMP_SCHEDULE=static ./build/bin/spmv_benchmark --matrix U1 --warmup 1 --repeat 10 --csv results/U1_1t.csv
OMP_NUM_THREADS=2 OMP_SCHEDULE=static ./build/bin/spmv_benchmark --matrix U1 --warmup 1 --repeat 10 --csv results/U1_2t.csv
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./build/bin/spmv_benchmark --matrix U1 --warmup 1 --repeat 10 --csv results/U1_4t.csv

# 调度策略对照：均匀矩阵 U1 与长尾矩阵 L1
OMP_NUM_THREADS=4 OMP_SCHEDULE=dynamic,16 ./build/bin/spmv_benchmark --matrix U1 --warmup 1 --repeat 10 --csv results/U1_dynamic.csv
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./build/bin/spmv_benchmark --matrix L1 --warmup 1 --repeat 10 --csv results/L1_static.csv
OMP_NUM_THREADS=4 OMP_SCHEDULE=dynamic,16 ./build/bin/spmv_benchmark --matrix L1 --warmup 1 --repeat 10 --csv results/L1_dynamic.csv
```

核对输出中的实际线程数。U1 的 static 结果使用前面生成的 `U1_4t.csv`；只有 2 个可用核心时，将调度比较中的线程数统一改为 2。资源足够时可另测 8 线程，并使用新的 CSV 文件名。

## 4. 正确性、性能与练习

以串行 `CpuSingleBackend` 输出为 reference；记录程序报告的相对误差，并以 `1e-6` 为通过阈值。其中 `T1` 是串行参考时间 `cpu_single_ms`，`Tp` 是使用 p 个线程时的 `cpu_openmp_ms`。程序每次运行重新测量串行时间，直接使用本次 CSV 的两个时间计算。OpenMP 一线程仍有调度开销，加速比不必为 1。对每个线程数计算 `Speedup=T1/Tp`、`Efficiency=Speedup/p`。有效带宽按程序采用的字节模型除以计算时间计算，并注明模型假设。

| Threads | Time (ms) | Relative error | Pass | Speedup | Efficiency | Bandwidth (GB/s) |
|---:|---:|---:|:---:|---:|---:|---:|
| 1 | | | | | | |
| 2 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |

练习：保持矩阵、预热和重复次数不变，比较 `static` 与 `dynamic,16`；解释为何 SpMV 的加速受内存带宽、调度开销和行长度分布限制。参考答案见 `answer/`。
