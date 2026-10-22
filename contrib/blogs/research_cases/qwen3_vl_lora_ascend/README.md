# 单卡昇腾约 53 分钟完成 Qwen3-VL-2B LoRA 微调：从训练启动到定位效果验证

> **作者**：樵明朗（北京航空航天大学 电子信息工程学院）
>
> **协作**：陈兴昊（北京航空航天大学 电子信息工程学院）
>
> **单位**：北京航空航天大学 MC2 实验室
>
> **联系方式**：minglangqiao@buaa.edu.cn
>
> **发布时间**：2026.10
>
> **关键词**：CANN / 模型训练 / LoRA微调

## 一、方案说明

想快速验证 Qwen 在自己的任务上能不能训起来，我的经验是：先选小模型、明确一个训练目标，把训练、保存权重、推理验证这条链路跑通，再扩大模型和数据规模。

这里分享一次已完成的 Qwen3-VL-2B-Instruct 微调记录。任务是视觉定位：输入图片和问题，让模型输出 `<find>...</find>` 和归一化的 `<bbox>[x1, y1, x2, y2]</bbox>`。训练使用单张昇腾 NPU，只在语言模型的 L08 层挂 LoRA，冻结视觉编码器和对齐模块。

例如，问“图中的鸟是否具有某种眼部和胸部特征”，需要模型指出相关的头部、胸部区域，方便检查回答依据是否落在图中对应位置。这类需求也出现在图文问答和图片内容审核中。本例重点验证视觉定位环节：输出简短的证据描述和区域坐标，再分别检查格式与框的位置。

这次训练完成 1 个 epoch，共 188 个优化器 step，训练器记录的运行时间为 3,193.8474 秒，约 **53 分 14 秒**，统计至末尾的 loss 验证；随后完成了权重合并和 6,000 条生成式定位评测。最终正常退出，保存了 `checkpoint-188`，之后也完成了模型合并与推理评测。

## 二、环境准备

### 2.1 本次运行配置

| 项目 | 本次配置 |
|---|---|
| 模型 | Qwen3-VL-2B-Instruct |
| 设备 | 训练使用单张昇腾 NPU，训练队列分配物理卡 4；补充设备截图显示型号为 910B3 |
| 服务器 NPU 配置 | 补充设备截图显示 8 张 910B3，每卡 HBM 标称 65,536 MB |
| npu-smi / Version | 补充设备截图显示均为 25.5.1 |
| 主机架构 | aarch64 |
| Python | 3.10 环境 |
| 训练框架 | ms-swift 4.3.0.dev0，本地源码安装 |
| CANN | 日志中加载的路径为 cann-9.0.0-beta.2 |
| 精度 | bfloat16 |
| 微调方式 | LoRA，L08 单层，rank=8、alpha=32、dropout=0.05 |
| 可训练参数 | 约 31.13 万，占总参数的 0.0146% |
| Attention | flash_attn，启用 padding_free |
| 序列处理 | max_length=2048，启用 packing |
| 显存记录 | 训练日志中的 memory(GiB) 为 42.08 |

本次训练使用单张 NPU，设备信息来自补充截图，训练资源占用以日志记录为准。迁移环境时，先核对 PyTorch、torch_npu、CANN 与模型实现是否兼容。

### 2.2 先检查环境，再启动训练

在已经安装匹配版本依赖的环境中执行：

```bash
npu-smi info
python --version
python -m pip show torch torch-npu transformers ms-swift deepspeed
command -v swift
python -c 'import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__); print(torch.npu.is_available())'
```

### 2.3 准备模型和数据

模型可从 [Qwen3-VL-2B-Instruct 官方模型页](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) 获取，下载后检查配置、tokenizer、processor 和权重分片是否完整。

数据使用 JSONL，每行一条图文对话。下面是实际训练数据中一条样本的结构，图片路径改为可移植的示例路径；assistant 内容是训练标签：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a precise vision assistant. You MUST use <find>...</find> for reasoning and <bbox>...</bbox> for coordinates. All coordinates MUST be normalized to [0,1]."
    },
    {
      "role": "user",
      "content": "<image>Question: Does the bird in the picture have white eye and buff breast?\nIdentify the visual evidence and answer in this format:\n<find>...</find><bbox>[x1, y1, x2, y2]</bbox>"
    },
    {
      "role": "assistant",
      "content": "<find>Target: bird's eye and breast. Region: bird's head and chest. Why: white eye and buff breast visible.</find>\n<bbox>[0.3000, 0.1000, 0.5500, 0.6000]</bbox>"
    }
  ],
  "images": ["./images/bird.jpg"]
}
```

实际保存时，上面这个对象要压成一行。训练前先检查图片路径和坐标格式，尤其要保证训练与评测使用同一套提示词、同一种坐标约定。

这里有一个很容易误读日志的地方：**原始样本数、长度过滤后的样本数、packing 后的序列数是三件事。** 本次运行的实际记录如下：

| 数据 | 原始条数 | 长度过滤后 | packing 后序列数 |
|---|---:|---:|---:|
| 训练集 | 21,000 | 17,077 | 6,011 |
| loss 验证集 | 6,000 | 5,933 | 1,879 |

本例采用 `truncation_strategy=delete`，21,000 条原始训练样本经过长度过滤后保留 17,077 条，再打包为 6,011 个序列，对应日志中的 `size=6011`。后文另对 6,000 条 validation 样本进行生成式定位评测。

## 三、启动训练

### 3.1 训练参数

下面按历史日志还原主要训练参数，模型、数据和输出目录统一使用示例路径；运行时替换为本地准备好的模型和数据，并先完成环境适配。以下相对路径均以运行命令的工作目录为基准；设备编号通过 `ASCEND_RT_VISIBLE_DEVICES` 指定，示例默认选择可用设备 0，历史运行使用物理卡 4。

```bash
#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE=1
export MASTER_PORT=34008
export HCCL_SOCKET_IF_PORT=0

MODEL_DIR=./models/Qwen3-VL-2B-Instruct
TRAIN_DATA=./qwen/training.jsonl
LOSS_DATA=./qwen/loss.jsonl
OUT_DIR=./outputs/qwen3vl2b_l08_r8
mkdir -p "$OUT_DIR"

swift sft \
  --model "$MODEL_DIR" \
  --model_type qwen3_vl \
  --template qwen3_vl \
  --dataset "$TRAIN_DATA" \
  --val_dataset "$LOSS_DATA" \
  --split_dataset_ratio 0 \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --target_regex '.*language_model\.layers\.8\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$' \
  --lora_rank 8 --lora_alpha 32 --lora_dropout 0.05 \
  --freeze_vit true --freeze_aligner true \
  --num_train_epochs 1 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --max_length 2048 --truncation_strategy delete \
  --attn_impl flash_attn --padding_free true --packing true \
  --gradient_checkpointing true --vit_gradient_checkpointing false \
  --deepspeed zero2 \
  --eval_steps 500 --save_steps 500 --save_total_limit 2 \
  --logging_steps 5 \
  --dataset_num_proc 32 --packing_num_proc 16 --dataloader_num_workers 4 \
  --load_from_cache_file true --create_checkpoint_symlink true \
  --output_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/train.log"
```

本次有效 batch 是 `2 × 16 × 1 = 32` 个 packing 后的序列，每个序列可包含多条原始样本。6,011 个打包序列对应约 188 个优化器 step，与最终 checkpoint 的步数一致。

本次选择 L08，迁移任务时可根据验证效果调整训练层，或尝试全层 LoRA。

### 3.2 为什么这套配置适合先跑通

2B 模型和单层 LoRA 将可训练参数控制在较小范围，适合先验证数据与训练链路。冻结视觉部分可以减少相关训练开销；packing 则能减少短样本的 padding 浪费。实际显存需求还包括基础模型、视觉 token 和激活，应结合输入规模检查。

先运行几个 step，确认 loss、梯度和数据加载都正常，再跑完整 epoch。正式训练时保留 `pipefail`，这样通过 `tee` 保存日志时，训练命令失败也能正确返回失败状态。

### 3.3 换成自己的数据，优先调整什么

下面是本次配置中最值得关注的几项，以及迁移任务时的调整建议。建议每次只调整一类参数，保留相同的验证集，方便判断变化来自哪里。

| 参数 | 本次取值 | 作用与调整建议 |
|---|---|---|
| `max_length` / `truncation_strategy` | `2048` / `delete` | 先查看编码后的长度分布和过滤比例；过滤较多时检查图像 token 与文本长度，再结合长度分布调整上限 |
| 单卡 batch / 梯度累积 | `2` / `16` | 显存紧张时先减小单卡 batch；可增加梯度累积维持近似的有效 batch，但通常会影响耗时 |
| `learning_rate` / `warmup_ratio` | `1e-4` / `0.05` | 先观察 loss 和梯度是否异常；若持续波动，结合数据检查尝试降低学习率 |
| `target_regex` / `lora_rank` | L08 / `8` | 先确认目标模块匹配且可训练参数符合预期；训练稳定后，再用同口径验证集评估是否需要扩大适配范围或 rank |
| `packing` / `padding_free` | 均为 `true` | 用于减少短样本填充开销，依赖兼容的 attention 实现；修改时成组检查，具体报错见 5.2 |

我的调整顺序是先检查数据与长度过滤，再确定设备能承受的 batch，接着观察学习率与训练稳定性，最后评估是否需要扩大 LoRA 的训练范围。2B 与单层 LoRA 在这里是快速验证的起点，是否满足实际任务仍要由生成结果判断。

## 四、训练日志与效果验证

### 4.1 完成训练的证据

训练日志中的关键字段节选如下，路径作了缩写：

```text
step 1/188:   loss=2.96618772, token_acc=0.55160350
step 185/188: loss=0.53073454, token_acc=0.84714842

eval_loss=0.54720300
eval_token_acc=0.84776430
epoch=1.0, global_step/max_steps=188/188

[INFO:swift] Saving model checkpoint to .../checkpoint-188
train_runtime=3193.8474
train_loss=0.97730037
memory(GiB)=42.08
[queue] END ... returncode=0
```

这里 `0.5307` 是末次记录的训练 loss，`0.9773` 是整个训练过程的平均 loss。配置的保存间隔是 500 step，总步数为 188；本次在训练结束时完成验证并保存 checkpoint，最终 `trainer_state.json` 记录 `global_step=max_steps=188`、`epoch=1.0`。

### 4.2 看日志趋势，判断训练是否稳定

训练 loss 从首个记录点的 2.9662 降到末次记录的 0.5307，整体呈下降趋势；最终验证 loss 为 0.5472，`eval_token_acc=84.78%` 表示验证集上的 token 级准确率。接下来通过生成式评测检查区域定位效果。

### 4.3 保存权重之后，还要做推理评测

训练成功后，先用基础模型和 adapter 加载检查输出；需要独立权重时，再执行 LoRA 合并。示例：

```bash
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
swift export \
  --adapters ./outputs/qwen3vl2b_l08_r8/实际运行子目录/checkpoint-188 \
  --merge_lora true \
  --output_dir ./outputs/qwen3vl2b_l08_r8_merged
```

`实际运行子目录` 要替换成训练输出的真实路径，基础模型目录也必须仍可访问。本次生成式评测使用合并后的模型和 vLLM 后端，完成了 6,000 条样本推理。

### 4.4 先用一条样本检查推理链路

迁移到自己的环境时，可以先准备一个只有一行的 `./qwen/smoke.jsonl`：沿用 2.3 的数据结构和提示词，准备一条包含 system、user 和 images 字段的记录，图片换成实际存在的测试图片。建议选一张未参与训练、便于人工判断区域的图片。

以下为复现时使用的单条推理检查命令。完成上面的权重合并后，可使用 Transformers 后端检查输出；后文的历史批量评测使用 vLLM。

```bash
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
mkdir -p ./outputs/qwen3vl2b_smoke
swift infer \
  --model ./outputs/qwen3vl2b_l08_r8_merged \
  --model_type qwen3_vl \
  --template qwen3_vl \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --infer_backend transformers \
  --val_dataset ./qwen/smoke.jsonl \
  --max_batch_size 1 \
  --max_new_tokens 256 \
  --temperature 0 \
  --result_path ./outputs/qwen3vl2b_smoke/predictions.jsonl
```

检查结果文件中的 `response`：应同时包含 `<find>...</find>` 和 `<bbox>[x1, y1, x2, y2]</bbox>`，四个坐标应满足 `0 ≤ x1 < x2 ≤ 1`、`0 ≤ y1 < y2 ≤ 1`。将坐标乘以原图宽高后，人工确认矩形是否覆盖问题对应的区域。单条检查通过后，再批量统计输出格式和定位质量。

### 4.5 批量定位评测与适用范围

| 指标 | 训练后实测值 |
|---|---:|
| 预测总数 / 匹配参考样本数 | 6,000 / 6,000 |
| 输出可解析率 | 99.97% |
| 有效 bbox 比例 | 99.75% |
| Mean IoU，相对教师框 | 0.3772 |
| Acc@IoU ≥ 0.3 | 54.14% |
| Acc@IoU ≥ 0.5 | 36.91% |

本表展示训练后指标，参考框来自教师标注。可解析率、有效 bbox 比例以全部 6,000 条样本为分母；IoU 和两个阈值准确率沿用历史评测脚本口径，对可计算 IoU 的样本统计，反映预测框与教师参考框的重合程度。

本次实践完成了训练、权重保存、合并和推理验证，大部分输出能按约定格式解析。后续可重点检查低 IoU 样本，从标注质量、输入分辨率和训练配置入手改进定位效果。

换成高分辨率图片、更长文本或不同任务后，先检查数据保留率、资源占用和定位质量，再决定是否增加训练范围。

## 五、常见问题与报错

下面六例整理自独立历史任务，覆盖训练参数、LoRA 注入、图像预处理、运行环境和后续评测。运行库导入案例来自项目共用的昇腾推理环境，其余案例来自 Qwen 任务。日志节选集中呈现关键字段，完整出处和原始行号保存在同目录 `failure_excerpts.txt`。

### 5.1 分布式训练还没开始，默认端口已经被占用

一次 Qwen2.5-VL-7B 的 L00 训练任务在分布式 rendezvous 初始化时退出：

```text
torch.distributed.DistNetworkError: The server socket has failed to listen
on any local network address. port: 29500, useIpv6: false,
code: -98, name: EADDRINUSE, message: address already in use
[queue] END ... returncode=1
```

这条错误指向本机监听端口冲突。多组训练即使用了不同 NPU，也可能共用同一个默认 `MASTER_PORT`。

先查看端口占用：

```bash
ss -ltnp 'sport = :29500'
```

为各任务分配互不冲突的端口，并确认配置传到了实际的 torchrun 命令中。例如前文成功案例使用 `MASTER_PORT=34008`。结束占用端口的进程前，先确认其归属，避免影响其他正常任务。

### 5.2 关闭了 Flash Attention，却忘记同步关闭 packing

一组 Qwen3-VL-2B 训练任务启动时同时传入了下面三个参数：

```bash
--attn_impl eager --padding_free true --packing true
```

在参数初始化阶段，框架直接拒绝了该组合：

```text
ValueError: The "packing" feature requires a flash attention implementation.
Please use one of: "flash_attn", "flash_attention_2",
"flash_attention_3", "flash_attention_4".
```

这是参数初始化阶段的配置依赖冲突。排查算子兼容问题时，若把 attention 改成 eager，应同步关闭依赖 Flash Attention 的 packing 配置。

可以选择两条路线：在已验证兼容的环境中使用框架支持的 Flash Attention 实现；或者以 eager 做兼容性验证，同时关闭 `packing` 和 `padding_free`，重新评估显存与训练效率。本文成功运行使用的是前一种组合。具体实现需与框架版本和昇腾环境匹配。

### 5.3 LoRA 正则多转义了一层：数据处理完了，adapter 却挂不上

一次 Qwen2.5-VL-7B 的 L00 运行已经完成数据编码，却在 PEFT 注入 LoRA 时失败：

```text
ValueError: Target modules
.*model\\.language_model\\.layers\\.0\\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$
not found in the base model. Please check the target modules and try again.
```

这里可以直接对比同一日志文件中的两次启动参数。失败时每个点号前是两个反斜杠，后续启动改为单个反斜杠。对于直接传给正则引擎的字符串，匹配字面点号应使用 `\.`；多加一层转义会改变匹配含义。

下面是 Bash 单引号参数的写法，点号前保留一个反斜杠：

```bash
--target_regex '.*model\.language_model\.layers\.0\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
```

后续启动日志出现了 `PeftModelForCausalLM`，并记录 `0.7209M Trainable [0.0087%]`，确认 adapter 已成功注入。

模型版本不同，模块前缀也可能不同。实际操作时先打印目标层的模块名，用同一个正则做匹配计数；匹配数量为 0 就停止启动。若经过 Python、JSON 或任务队列模板传递参数，应检查实际进程接收到的正则字符串。

### 5.4 图像长宽比超出预处理限制

一组 Qwen3-VL-2B 定位与回答评测在图片编码阶段失败。调用链为 `replace_tag → fetch_image → smart_resize`：

```text
ValueError: absolute aspect ratio must be smaller than 200, got 271.0
Shutdown initiated (timeout=0)
Aborting 787 requests
```

另一份 Qwen2.5-VL-7B 的失败记录也出现了同类错误，长宽比为 `426.0`。报错发生在图像预处理阶段，图片检查需覆盖长宽比等形状约束。

对包含预测裁剪图的流程，应在生成 crop 后检查宽高、最小边长，以及 `max(width, height) / min(width, height)`。可保证裁剪宽高至少为 1，并对长宽比超过 50 的 crop 使用边缘填充。日志已定位到图片长宽比超限，具体异常图片需进一步检查。

普通等比例缩放保持原有长宽比，这类问题可采用填充策略处理。如果选择跳过或回退原图，应统计数量并固定评测口径。处理后先单独验证异常样本，再恢复批量推理。

### 5.5 checkpoint 已经保存，后续推理仍然可能 OOM

一次 Qwen2.5-VL-7B 的 L10 任务日志先记录了训练完成：

```text
[INFO:swift] Saving model checkpoint to .../checkpoint-998
train_runtime=5286.0255, epoch=1.0, global_step/max_steps=998/998
```

随后，同一任务日志进入 `swift/cli/infer.py` 的生成式评测，在视觉编码器的 `eager_attention_forward → softmax` 调用中报错：

```text
[rank1]: RuntimeError: NPU out of memory. Tried to allocate 22.15 GiB
(NPU 1; 60.96 GiB total capacity; 49.20 GiB already allocated;
49.20 GiB current active; 9.81 GiB free;
50.03 GiB reserved in total by PyTorch)
[queue] END ... returncode=1
```

训练阶段已完成并保存权重，报错发生在后续生成式评测的视觉 attention 路径。排查时应核对推理进程实际使用的 attention 实现；本次堆栈显示为 eager 路径。

本次申请的 22.15 GiB 超过了 9.81 GiB 剩余空间。可先降低推理 batch，并按任务需要限制图像 token 规模，用少量样本验证后再恢复批量评测，同时记录速度和效果变化。

已有 checkpoint 时，可先验证能否正常加载，再单独重跑失败的推理节点。把训练、合并和评测分别记录状态，能避免因为一个后续节点失败而重新训练。

### 5.6 Python 包已安装，仍然可能找不到昇腾运行库

项目共用的昇腾 vLLM 服务启动日志中，有一类在导入 `torch_npu` 时就失败的记录：

```text
ImportError: libhccl.so: cannot open shared object file: No such file or directory
RuntimeError: Failed to load the backend extension: torch_npu.
```

错误发生在导入 `torch_npu` 时，动态库加载器找不到 `libhccl.so`。可依次检查 CANN 是否完整安装、当前 shell 是否加载了环境脚本，以及容器里是否能访问对应库目录。

先在**真正执行训练或推理的那个 shell** 中加载与安装位置对应的 CANN 环境，再验证导入。例如：

```bash
source "${CANN_ENV:?请将 CANN_ENV 设置为匹配版本的 set_env.sh 完整路径}"
python -c 'import torch, torch_npu; print(torch.npu.is_available())'
```

`CANN_ENV` 应按实际安装位置设置，并与当前 PyTorch、torch_npu 版本匹配。

参考：昇腾论坛实践帖的方案、环境、执行、验证和排错组织方式；本文的训练数据与结论来自真实实验记录。
