# MuduoXinyu：HiDevLab 历史实测

本页保留 2026-10-28 的旧版运行结果。当前实验使用已集成 A/B 实现的 `course-attention-ab` 版本，操作步骤以 [Notebook](../02_muduoxinyu_optimization/09_muduo.ipynb) 为准。

2026-10-28，通过 Chrome → HiDevLab → WebIDE，在 DevEnv_193623 的 A3 / CANN 9.0.0 环境完成课程完整 A/B 实验。编译成功，四个运行用例退出码均为 0，最终 `RUN_AB=PASS`。

实验结束后已确认平台显示 DevEnv_193623「已关机」，运行中实例为 0。

## 本次使用的代码和模型

- 仓库：https://gitcode.com/bnujwx/muduoXinyu
- 提交：`703b1ebdfe0129239c740b3ed737fe71cacc6672`，来自旧课程基线 `60c6371cd30894d9896dfa979b86c6f892b6cbda`。除 README 来源说明外，文件与旧基线一致。
- 课程仓库提交：`054c9bfb4397ad9ac505f5d1cfa1685147087e05`。
- 模型：[karpathy/tinyllamas 的 stories110M.bin](https://huggingface.co/karpathy/tinyllamas/blob/main/stories110M.bin)，438381596 字节。
- 分词器：[karpathy/llama2.c 的 tokenizer.bin](https://github.com/karpathy/llama2.c/blob/master/tokenizer.bin)，433869 字节。

两个文件先从原始来源下载到本地，再通过 WebIDE 文件夹右键 `Upload...` 上传。平台直连 Hugging Face 失败，镜像下载较慢，未使用其未完成文件。使用的是下面实验所需的 stories110M 和配套 tokenizer。


模型检查确认 12 层、dim=768、12 个注意力头、32000 词表、最大序列长度 1024；tokenizer 的词表条目数匹配。本次没有使用合成 smoke checkpoint。

## 实测结果

执行顺序为 smoke A → smoke B → perf B → perf A。功能阶段 12 步；性能阶段 120 步、四条 hello，第一个样本预热，后三个样本统计。

| 指标 | 结果 |
| --- | --- |
| A/B token 完全一致 | 功能与性能阶段均通过 |
| FlashAttention 调用次数 | 144 |
| Cast 调用次数 | 576 |
| fallback / FlashAttention 显式同步次数 | 0 / 0 |
| A 平均吞吐率 / CV | 196.8047 tokens/s / 0.3123% |
| B 平均吞吐率 / CV | 206.4053 tokens/s / 8.5080% |
| B 相对 A 的平均变化 | +4.8783% |
| functional_pass / performance_valid | true / true |

这次可以确认功能跑通。B 的波动大于平均提升，不能据此声称稳定加速；课程分析器的 `performance_valid` 不包含波动阈值或显著性检验。讲解时同时展示均值和 CV，保留这一轮结果。

## 结果记录

首次结果保存在 `/workspace/cann-demo/muduo-ab-first/`。本地备份为 `work/muduo-hidevlab-evidence.zip`，包含当时的命令、运行日志和构建日志。

重新实验请按当前 [Notebook](../02_muduoxinyu_optimization/09_muduo.ipynb) 操作。
