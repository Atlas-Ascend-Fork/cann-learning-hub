# CANNLab 演示操作指南

## 1. 打开实验

在 [CANNLab 环境列表](https://gitcode.com/org/cann/cannlab/environment)启动实验环境，等待 Running，点击 Connect 进入 WebIDE。选择 `Python 3.11.4` 内核。

使用完整课程仓库，在 `09_cann_application_tuning` 中打开下面的 Notebook。准备、编译和运行命令都在 Notebook 中，按顺序点击执行。

| 演示内容 | Notebook |
|---|---|
| GMRES | [01_distributed_gmres/09_gmres.ipynb](../01_distributed_gmres/09_gmres.ipynb) |
| MuduoXinyu | [02_muduoxinyu_optimization/09_muduo.ipynb](../02_muduoxinyu_optimization/09_muduo.ipynb) |
| restart 对照与 A/B 分析 | [09_practice.ipynb](../09_practice.ipynb) |

## 2. GMRES

课前按 Notebook 完成编译和一次性能采集。

现场先打开 `npu_gmres.cpp`，指出 SpMV、Dot、AXPY 和 Norm，再运行“建立单卡基线”。默认使用 U2、restart=30、3 次重复；时间预算较短时，可在课前统一改为 `--repeat 1` 并彩排。

运行后先看 `converged`、残差和解误差，再读总时间。接着展示“查看耗时最多的 API 和算子”，解释 Dot 与同步开销。

需要做前后对照时，在章节练习中运行 restart=4 与 30，比较实际迭代数、误差和耗时。程序输出是单次均值，msprof 表是本次采集的累计量；默认 `--repeat 3` 时覆盖三次求解。修改重复次数后，按实际次数换算，并注意单位。

## 3. MuduoXinyu

课前准备 stories110M 模型、配套 tokenizer 和源码，下载 `course-attention-ab` 教学版本并完成 NPU 编译。下载和编译不放到录制过程中。

现场打开 Notebook，先运行路径设置格，再依次运行第 3 步“运行两种实现”和第 4 步“读取结论”。

脚本先做两条路径的短功能检查，再各做 1 次预热和 3 次统计。运行期间解释：A 为 FP32 多算子，B 为 FP16 FlashAttentionV4；两者使用相同的模型、提示词和生成长度。

结果出来后，读出功能检查、A/B 平均吞吐量、波动和相对变化。没有加速时就按实测结果讲。需要查看调用次数或报错时，打开输出目录中的四份日志。

重跑会更新 `OUTPUT_DIR`。需要保留前一次结果作比较时，再换一个目录名。课前先跑一遍确认耗时；录制时间不够时，展示课前保存的真实结果并说明来源。

## 4. 录制结束

保存 Notebook 的输出。回到 CANNLab 环境列表，点击 Stop，并等待 Stopped。

## 常见问题

| 现象 | 处理 |
|---|---|
| 找不到 `src/` | 确认打开了表中对应的 Notebook，终端则先进入它所在目录 |
| CANN 命令或动态库找不到 | 重新选择平台内核；自行启动 Jupyter 时，先加载 CANN 环境再启动服务 |
| 模型下载失败 | 单独重试模型下载，或上传已有的 stories110M 与配套 tokenizer，并修改路径设置 |
| 程序运行出错 | 查看本次输出和日志，再处理报错指出的问题 |
