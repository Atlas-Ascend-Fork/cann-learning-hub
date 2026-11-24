# GMRES 单卡调优实验手册

## 准备与目标

使用 CANNLab A3、CANN 9.0，先完成本节环境检查。GMRES 求解 `Ax=b`；本实验比较 restart=4 与 30，观察搜索空间大小如何影响迭代次数、正交化和总时间。

SpMV、Dot、Norm、AXPY 和 Scale 在设备端计算，Host 处理小型 Hessenberg 矩阵和 Givens 旋转。

## 编译并运行基线

在 `01_distributed_gmres` 目录打开终端，依次执行：

```bash
cd src/dis_gmres
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDIS_GMRES_FORCE_STUB=OFF
cmake --build build -j
./build/bin/dis_gmres --matrix U2 --rank 0 --world-size 1 --device 0 --orthogonalization mgs --warmup 0 --repeat 3 --restart 30 --max-iterations 300 --tolerance 1e-6
```

检查 `compute backend = Ascend C RTC Device GMRES`、收敛状态、真实相对残差和解误差。残差应不超过 `1e-6`，解误差应小于 `1e-3`，程序正常退出。

`warmup=0` 表示不额外丢弃求解样本。每次求解仍会创建内部 RTC/Device 状态；矩阵读取、分区和通信组初始化在重复求解前完成。

## 采集和分析

在同一终端使用相同参数采集：

```bash
msprof --application="./build/bin/dis_gmres --matrix U2 --rank 0 --world-size 1 --device 0 --orthogonalization mgs --warmup 0 --repeat 3 --restart 30 --max-iterations 300 --tolerance 1e-6" --output=./profiling/restart30 --runtime-api=on --task-time=on
msprof --export=on --output=./profiling/restart30
```

重跑可沿用输出目录，导出命令使用同一目录。回到 9.2 节 Notebook，将目录设置格改为 `%env PROFILE_DIR=profiling/restart30` 并运行，再运行“查看耗时最多的 API 和算子”。代码读取最新一组 `api_statistic_*.csv` 和 `op_statistic_*.csv`；记录主要耗时和来源文件。API 与算子表的时间可能重叠，不能相加为总时间。

## 改变重启长度

把普通运行和采集命令中的 `--restart 30` 改为 `--restart 4`，采集目录改为 `profiling/restart4`，其余参数保持一致。分析第二次采集前，将同一个目录设置单元格改为 `%env PROFILE_DIR=profiling/restart4` 并运行，再运行分析代码。两次都通过误差检查后填写：

| 设置 | 实际迭代数 | 真实残差 | 解误差 | 总时间 | Dot/AXPY 时间 |
|---|---:|---:|---:|---:|---:|
| restart=30 | | | | | |
| restart=4 | | | | | |

解释本次哪种设置更快，以及差异来自迭代次数还是每轮计算成本。当前 Device 实现使用 MGS，CGS 尚未实现。

## 多卡扩展（选做）

有多卡资源时，再按 9.5 节比较 rows/nnz 分区和不同 rank 数。单卡两种分区都取得完整矩阵，本次单卡实验只比较 restart。
