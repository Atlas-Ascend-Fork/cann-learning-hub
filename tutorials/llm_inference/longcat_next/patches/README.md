# LongCat-Next 单卡推理 patch 使用说明

本目录提供 LongCat-Next 单卡推理实现的上游 patch：将其应用到
[cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 仓库后，
即可在多卡样例之外运行单卡（CPU offload）推理。

## 1. patch 文件信息

| 项 | 值 |
| --- | --- |
| 文件位置 | 本目录 `longcat-next-single-npu.patch`（git format-patch 格式，约 67KB） |
| 来源 | 从本地 cann-recipes-infer 克隆的 `longcat-next-single-npu` 分支导出，基于该仓 master `969160e` |
| 内容 | 16 文件（+875 / -108）：单卡 CPU offload 核心代码（modeling / model_infer / components）、4 个单卡 YAML、3 个请求清单、README 单卡章节 |

本 patch 为基于infer仓master分支 `969160e` 的时点快照，patch 为单个 git 提交，包含以下改动：

- `models/longcat_next/models/modeling_longcat_next.py` — MoE 专家权重与 N-gram 嵌入的 CPU offload 路径（`CPUExpertStore` / `CPUOffloadedEmbedding`）、单卡配置校验
- `models/longcat_next/models/model_infer.py` — 单卡入口与请求清单（request manifest）驱动逻辑、权重目录友好报错
- `models/longcat_next/models/components/*.py` ×6 — 组件适配
- `models/longcat_next/config/*single_npu*.yaml` ×4 — 冒烟 / 文本 / 多模态 / 图像生成配置
- `models/longcat_next/requests/{text,multimodal,image_gen}.json` — 随附请求清单（多卡 yaml 亦可复用 `text.json`）
- `models/longcat_next/README.md` — 新增"单卡推理（CPU offload）"章节

## 2. 环境要求

| 项目 | 要求 |
| --- | --- |
| 硬件 | Atlas A3 系列单卡（64GB HBM）；主机内存 ≥ 64GB 可用（权重 mmap 与 CPU 专家计算，推荐 ≥ 128GB） |
| CANN | ≥ 8.5.1 |
| Python | 3.12，`torch==2.8.0`、`torch_npu==2.8.0.post4` |
| 模型权重 | `meituan-longcat/LongCat-Next`（151GB BF16，含 `assets/`、`image_decoder/`、`cosy24k_vocoder/`） |
| Python 依赖 | 按 `models/longcat_next/requirements.txt` 安装；若使用 `datasets==3.6.0` 且钉 `numpy==1.26.4`，需同时钉 `pyarrow==25.0.1`（否则 import 报 numpy 版本冲突） |

## 3. 使用步骤

### 3.1 应用 patch

本 patch 基于上游 master 的 `969160e` 提交制作，推荐在该基线上应用（与课程 Notebook 的自动流程一致）：

```bash
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
git checkout 969160e                                  # 固定基线提交
git am <本教程仓路径>/tutorials/llm_inference/longcat_next/patches/longcat-next-single-npu.patch
```

应用成功后 `git log -1` 应显示：`[feat] 新增 LongCat-Next 单卡 CPU offload 推理样例`。

> 在 `969160e` 及其之后的 master 上 `git am` 均可干净应用；若在其他基线上报冲突，可先 `git am --abort`，再尝试
> `git apply --3way <本教程仓路径>/tutorials/llm_inference/longcat_next/patches/longcat-next-single-npu.patch` 手动解决。
>
> 课程 Notebook（第 2 章）会自动完成"全量克隆 → checkout `969160e` → `git am`"并幂等复用，无需手工执行本节步骤。

### 3.2 环境配置（一次性）

```bash
# CANN 环境（若 set_env.sh 中 cann_path 为占位符，需先改成本机路径）
sed -i "s|your_cann_pkgs_path|/usr/local/Ascend/ascend-toolkit/latest|" executor/scripts/set_env.sh

# Python 环境与可见 NPU
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0        # 选空闲卡
```

### 3.3 配置权重与素材

```bash
# 1) 修改 yaml 的 model_path（必改；单卡 yaml 中默认占位 /path/to/LongCat-Next）
sed -i "s|model_path: /path/to/LongCat-Next|model_path: /path/to/your/LongCat-Next|" \
    models/longcat_next/config/longcat_next_multimodal_single_npu_offload_smoke.yaml

# 2) 多模态素材软链（图理解/语音任务需要；权重目录自带官方 assets/）
ln -sfn /path/to/your/LongCat-Next/assets models/longcat_next/requests/assets
```

### 3.4 运行推理

`--yaml` 传 `config/` 下的文件名（不支持绝对路径）：

```bash
bash executor/scripts/infer.sh --model longcat_next --yaml longcat_next_multimodal_single_npu_offload_smoke.yaml
```

可选配置速查：

| YAML | 用途 | 参考耗时（A3 单卡） |
| --- | --- | --- |
| `..._offload_smoke.yaml` | 冒烟基线（`requests/text.json` 的 3-token 短版本） | 生成约 30s（另含约 2-3 分钟权重加载） |
| `..._text_single_npu_offload.yaml` | 文本生成 128-token | 约 5 分钟 |
| `..._multimodal_single_npu_offload.yaml` | 图理解 / 语音转写 / 语音合成 | 约 16 分钟 |
| `..._image_gen_single_npu_offload.yaml` | 图像生成（16×16 视觉码 → 448×448 PNG） | 约 8 分钟 |

结果保存在 `models/longcat_next/res/<日期>/longcat_next_<yaml名>/`：`log_0.log` 日志、`results/<请求id>/` 下的 `text.txt` / `image*.png` / `audio*.wav` / `result.json`。

### 3.5 撤销 patch

```bash
git reset --hard HEAD~1     # 删除该提交（工作区需无其他未提交改动）
```

## 4. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `RuntimeError: The path does not exist: .../your_cann_pkgs_path` | `executor/scripts/set_env.sh` 的 `cann_path` 是占位符，按 3.2 节修改 |
| `FileNotFoundError: 权重目录不存在: /path/to/LongCat-Next` | yaml 的 `model_config.model_path` 未改，按 3.3 节修改 |
| `world_size=1 requires both enable_moe_cpu_offload and enable_ngram_cpu_offload` | 框架强制约束：单卡必须双 offload（151GB 权重 > 64GB HBM），使用随附单卡 yaml 即已开启 |
| `YAML file not found: .../config/<绝对路径>` | `--yaml` 只接受 `config/` 下的文件名，不传绝对路径 |
| 多卡运行报 `aicpu exception` | 先确认 NPU 卡未被其他任务占用（共享机器常见）；本 patch 已通过与 master 原版对照证实不引入多卡回归 |

