# SpMV 开发实验手册

## 实验内容

运行 FP32 Ascend C SpMV，并用 CPU 原型比较低精度存储、按非零元分区和矩阵数据复用。FP32 在 NPU 上执行；FP16、BF16 和 persistent 目前在 CPU 上执行。

## 构建与运行

在 `01_ascendc_spmv` 目录打开终端，确认 CANN 环境已加载，再执行：

```bash
cd src/ascend_spmv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPMV_REAL_ASCENDC=ON
cmake --build build -j
./build/bin/spmv_benchmark --matrix U1 --warmup 1 --repeat 3 --csv results/U1.csv
```

检查日志 `Actual Backend=Ascend C RTC` 和 CSV 的 `actual_backend=ascend_c`。FP32 相对误差应为有限数且小于 `1e-6`。再把矩阵改为 U2，结果文件改为 `results/U2.csv`，比较规模变化后的结果。

## 理解数据流

`AscendCSpmvBackend` 分配设备内存，复制 CSR 和 x，运行 SpMV，等待计算结束，再回读 y 与 CPU 参考结果比较。RTC 表示在运行时编译和加载 Kernel。

记录输入传输、`ascendc_launch_to_complete_ms` 和输出传输。中间一项包含提交和同步等待，纯 Kernel 时间需要性能分析工具测量。

## 分析精度与复用

FP16/BF16 原型用于观察存储与误差的取舍，记录本次误差，不设置统一通过阈值。压缩比为 FP32 CSR 字节数除以低精度 CSR 字节数，数值越大表示越省空间。例如 84 字节压到 64 字节，压缩比为 1.3125，节省约 23.8%。

| 实现 | 执行位置 | 相对误差 | 初始化时间 | 每次计算总时间 |
|---|---|---:|---:|---:|
| CPU 单线程 | CPU | | | |
| OpenMP | CPU | | | |
| FP32 Ascend C | NPU | | | |
| FP16/BF16 原型 | CPU | | | |
| 数据复用原型 | CPU | | | |

结合 6.4 节计算调用 1、10、1000 次的成本，再完成 6.7 节练习。
