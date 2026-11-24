# CANNLab 历史实测（2026-10-29）

本页保留 2026-10-29 的旧版运行结果。当前实验使用已集成 A/B 实现的 `course-attention-ab` 版本，操作步骤以 [Notebook](../02_muduoxinyu_optimization/09_muduo.ipynb) 为准。

GMRES 与 MuduoXinyu 已在 CANNLab 的 `cann_pr737_a3` 实例跑通。两项功能检查均通过；Muduo 本轮 B 比 A 的平均吞吐低 1.2966%，没有测到加速。以下数字来自本次原始日志，不使用 HiDevLab 的历史数字替代。

## 环境与版本

| 项目 | 本次配置 |
| --- | --- |
| 平台 | CANNLab，Chrome → WebIDE |
| 实例 | `cann_pr737_a3`，A3，1×NPU 910C、40 vCPU、240 GiB |
| 镜像 | `cann_9.0.0-py3.11-A3-arm-20260829` |
| CANN / Python / 驱动 | `/home/developer/Ascend/cann-9.0.0` / 3.11.4 / 25.5.5 |
| 程序设备 | 逻辑设备 0；开机时物理卡 1、芯片物理编号 2/3，健康 OK |
| 本次目录 | `/mnt/workspace/cann-demo-20261029` |
| 课程提交 | `054c9bfb4397ad9ac505f5d1cfa1685147087e05` |
| Muduo 仓库 | https://gitcode.com/bnujwx/muduoXinyu |
| Muduo 提交 | `703b1ebdfe0129239c740b3ed737fe71cacc6672` |

已有课程工作区与旧实验目录保持原样。本次单独获取课程目录进行实验。

模型复用 `/mnt/workspace/pr737_a3_final/stories110M.bin`，分词器为同目录 `tokenizer.bin`，两者与 HiDevLab 验证的真实资产完全一致：

```text
stories110M.bin  438381596 bytes
tokenizer.bin    433869 bytes
```

## GMRES 结果

真实 ACL+HCCL 后端构建成功，ctest 1/1 通过，耗时 18.63 秒。U1 为 100000 阶、1099985 个非零元素；单卡、MGS、restart=30、max_iterations=300、tolerance=1e-6、warmup=0。

| 指标 | 结果 |
| --- | --- |
| 迭代次数 | 32 |
| 相对残差 | 8.803283e-7，通过 ≤1e-6 |
| 相对解误差 | 1.069179e-6，通过 <1e-3 |
| 单次 solve | 3235.835938 ms |
| repeat=3 平均 solve | 3208.927002 ms |
| 单次 SpMV / Dot | 183.178848 / 602.348328 ms |
| 单次 AXPY / Norm | 40.923206 / 45.464443 ms |
| 单次同步 | 867.680542 ms |
| HCCL / AllReduce / AllGather | 0 ms / 0 次 / 0 次 |

阶段计时与同步计时重叠，不相加。日志明确显示 `compute backend = Ascend C RTC Device GMRES`。三次重复的 CSV 保存平均时间和最后一次的迭代、误差；不提供逐轮完整分布。

msprof 采集和导出退出码为 0，保存了非空的 `op_statistic`、`op_summary`、`task_time` 和 `api_statistic`。自动导出与显式导出形成两组带时间戳的 CSV，分析时选一组，不能重复计数。单卡采集中的 Cluster Tuning 报告提示未完成；不能宣称完成多卡通信分析。Timeline 尚未逐项分析。

## MuduoXinyu 结果

`make -j8 npu` 构建成功。短功能 A、短功能 B、性能 B、性能 A 四个用例均正常退出。

| 指标 | A：FP32 多算子 | B：FP16 FlashAttention |
| --- | --- | --- |
| 有效采样 1 | 167.135 tokens/s | 169.034 tokens/s |
| 有效采样 2 | 168.079 tokens/s | 160.594 tokens/s |
| 有效采样 3 | 170.243 tokens/s | 169.275 tokens/s |
| 均值 | 168.4857 tokens/s | 166.3010 tokens/s |
| CV | 0.7722% | 2.4273% |

B 相对 A 为 **−1.2966%**。两阶段 A/B token 均一致，路径标记正确；短检查 FA=144、Cast=576、fallback=0、flash_sync=0。`functional_pass`、`performance_valid`、`overall_pass` 为 true，`performance_beneficial` 为 false。

现场结论：“功能通过，性能数据有效，但本轮 FlashAttention 方案没有带来吞吐提升。它同时改变精度、缓存和算子组织，且存在转换与启动成本；要确定哪项成本占主导，还需要进一步测量。”这组数据也不足以证明 B 在所有情况下都会更慢。

## 重新实验

请按当前 [操作指南](operation_guide.md) 和 Notebook 操作。

## 原始记录与本次修复

远端的 `cann-demo-20261029` 保存环境文件、构建日志、两组 GMRES 日志与 CSV、`muduo-ab-first`、`profile-u1` 和演示脚本。远端压缩包为 `/mnt/workspace/cann-demo-20261029-evidence.zip`。

本地备份位于仓库根目录 `work/cann-demo-20261029-evidence.zip`，解压到 `work/cannlab-evidence-20261029/`。共 131 个文件，压缩包 CRC 检查通过。

首次用交互终端的 `for ... printf ... > demo-env.sh` 保存变量时，WebIDE 的命令跟踪控制字符混入文件，导致开头一次 source 报错。验证进程仍继承了已导出的完整变量，后续真实构建、运行与结果检查正常完成，原始报错保留在 `validation.log` 中。随后改为 Python 直接写文件，并在新 Bash 中验证路径恢复成功；性能采集使用修复后的文件。新环境准备指南也已改用这一保存方式。

HiDevLab 和 CANNLab 的驱动、系统、编译环境不同，两次独立运行不能直接当作平台速度排名或优化前后加速比。本页保留当时的结果；课堂读数以本轮 Notebook 输出为准。

## 关机状态

实验与备份完成后已停止 `cann_pr737_a3`，平台显示 Stopped、运行实例数为 0。
