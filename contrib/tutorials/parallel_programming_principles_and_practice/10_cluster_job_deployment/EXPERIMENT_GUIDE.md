# 线性求解器部署实验手册

## 实验目标

编译运行 `xyce_benchmark`，比较 CPU 单线程、OpenMP 和 Ascend Device 三种求解器。本程序构造线性方程并调用求解器；完整 Xyce 电路仿真还需读取网表并执行仿真。

CPU 实现随本章提供，设备实现使用第 09 章的 GMRES。请保留完整课程目录，在 CANN 环境完成下面的步骤。

## 编译与运行

在第 10 章目录打开终端：

```bash
cd src/ascend_xyce
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASCEND_XYCE_HOST_ONLY=OFF
cmake --build build -j
./build/bin/xyce_benchmark --matrix U1 --warmup 0 --repeat 1 --csv results/U1.csv
```

先用 U1 和一次计时检查流程。运行成功后，把 `--repeat 1` 改为 `--repeat 5` 重测；需要保留前一次结果时，同时更换 CSV 文件名。

## 检查结果

程序退出码为 0，CSV 对当前矩阵包含三行求解结果。逐行检查：

- `converged=1`，`final_residual` 为有限数且不超过 `1e-6`。
- `solution_error_vs_expected` 为有限数且不超过 `1e-3`。
- OpenMP 和 Device 的 `solution_error_vs_cpu` 不超过 `1e-3`；CPU 行的 0 表示与自身比较。

| 求解器 | 迭代数 | 残差 | 解误差 | 线性求解时间 | 总时间 |
|---|---:|---:|---:|---:|---:|
| CPU 单线程 | | | | | |
| OpenMP | | | | | |
| Ascend Device | | | | | |

总时间包含线性求解，线性求解又包含 SpMV。比较时说明范围，避免重复相加。

## 分析设备耗时（选做）

在同一终端执行：

```bash
msprof --application="./build/bin/xyce_benchmark --matrix U1 --warmup 0 --repeat 5 --csv results/U1_profile.csv" --output=profiling/U1 --runtime-api=on --task-time=on
msprof --export=on --output=profiling/U1
```

`msprof` 在输出目录下创建每次采集的子目录。对照设备任务时间和程序输出，观察求解时间主要花在哪里。

## 有集群队列时选做

把编译与运行命令写入作业脚本，按所在集群要求设置工作目录、CPU、内存、时限和日志路径。使用平台规定的提交命令（如 `sbatch` 或 `qsub`），保存作业编号并查看排队、运行和结束状态。CANNLab 终端实验不需要这一步。

如需选做完整 Xyce 构建，再准备 Xyce 及其第三方依赖，并使用工程中的 `fetch_xyce.sh`、`build_xyce.sh`。它与上面的求解器测试分别记录结果。
