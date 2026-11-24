# HCCL + Ascend C SpMV 实验手册

## 先完成单卡路径

从本章目录执行。单卡不需要填写设备 IP 或 rank table。

```bash
set -euo pipefail
cd src/hccl_spmv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHCCL_SPMV_STUB=OFF
cmake --build build -j
./build/bin/hccl_spmv --rank 0 --world-size 1 --device 0 --matrix U1 --warmup 1 --repeat 3
```

确认 CMake 输出 `HCCL backend=1`。运行后检查 `Actual Compute Backend=Ascend C FP32 RTC`、`Communication Backend=ACL+HCCL`、有限的 CPU Reference Error 和 `Correctness=PASS`；误差应小于 `1e-6`。

## 理解数据流和计时

rank 0 把 x 传到 Device，HCCL Broadcast 分发 x；每个 rank 用本地 CSR 分片执行 Ascend C SpMV；AllGather 收集经过 padding 的 local_y，Host 回读后按有效行取结果。padding 让各 rank 发送同样长度，填充行不参与校验。

每轮先取各 rank 耗时的最大值，再对重复取平均。local SpMV launch-to-complete 已含同步等待，Synchronization 是其子项，不能再加一次。不同字段的最大值可能来自不同 rank，因此直接记录 total，不强求各项相加等于 total。

## 多卡扩展（可选）

有两张可用卡时，在同一目录执行：

```bash
bash scripts/run.sh --npus 2 --matrix U1 --warmup 1 --repeat 3
```

单机默认使用 root-info 初始化。脚本打印本次日志目录 `results/run.XXXXXX`，里面保留各 rank 的日志。若使用 rank table，按实际 HCCN IP 运行 `scripts/generate_rank_table.py`，并给 `run.sh` 传入匹配设备数量的 `--rank-table` 文件。

记录设备数、误差、local SpMV、HCCL、传输和 total。先做同输入的正确性检查，再解释增加设备后的快慢；单卡实验用于检查完整数据流，多卡实验才用于讨论通信扩展性。
