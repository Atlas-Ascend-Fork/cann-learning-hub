# MuduoXinyu A/B 实验

MuduoXinyu 原项目来自 [BNU-SYS](https://gitee.com/bnu-sys/muduoXinyu)。课程使用[公开教学副本](https://gitcode.com/bnujwx/muduoXinyu)的 `course-attention-ab` 版本，源码已包含两条 Attention 实现。原项目署名和许可证见源码仓库。

按[实验 Notebook](../09_muduo.ipynb#s-02-02_multi_operator_vs_flash_attention)下载源码、准备模型、编译并运行 A/B。下载源码和下载模型分为两格，模型下载失败后可单独重试。

模型来自 [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas)，Notebook 通过 Hugging Face 镜像下载；tokenizer 来自 [llama2.c](https://github.com/karpathy/llama2.c)。本实验使用 stories110M（12 层）及配套 tokenizer，模型约 419 MiB；合成的 smoke_tiny.bin 不适用于本实验。

| 文件 | 作用 |
|---|---|
| `run_ab_benchmark.sh` | 执行 A/B 并保存日志和结果 |
| `analyze_results.py` | 检查 token、执行路径和工作量，计算吞吐量均值与波动 |

构建日志在 `results/build.log`，A/B 结果在 `results/attention_ab`。重跑会更新结果目录；需要保留前一次结果时，更换 `OUTPUT_DIR`。

查看 `result.json` 中的功能检查、数据有效性和吞吐量变化。四份运行日志用于排查问题，`commands.log` 保存实际命令。

Path A 为 FP32 多算子，Path B 为 FP16 FlashAttentionV4。结果描述这两套实现的差异；单独研究融合效果时，需使用相同精度的对照。
