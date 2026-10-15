# LongCat-Next 单卡推理实践

本教程以美团开源的原生多模态大模型 `LongCat-Next` 为例，展示如何在**单张 Atlas A3 NPU** 上使用 cann-recipes-infer 离线推理框架部署完整权重约 151GB（BF16）的多模态 MoE 模型：通过 **MoE / N-gram 权重 CPU offload** 解决 151GB 权重与 64GB HBM 的容量矛盾，完成 Baseline 冒烟、文本生成与图理解/语音/图像生成等多模态任务。课程主流程采用 recipes 的典型用法：第 2 章首次运行时自动**克隆上游 cann-recipes-infer 仓并打上单卡推理 patch**（上游仓内置的是多卡实现），此后查看并修改克隆仓 `models/longcat_next/config/` 下的 YAML 配置，通过 `executor/scripts/infer.sh` 拉起离线推理，再从 `res/` 与 `prof/` 目录读取日志、结果与性能产物。

教程包含以下内容：
- Notebooks：包含环境准备、推理仓克隆与 patch 应用、YAML 修改、`infer.sh` 启动、请求清单编写与多模态任务体验步骤，可在 CANNLab 中运行，也可在本地 Jupyter 环境中执行。
- SRC：包含 Notebook 辅助模块（推理仓 bootstrap、运行 YAML 准备、结果目录定位、请求指标收集、Profiler 产物解析、统一依赖检查）。

## 关于 cann-recipes-infer

[cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 是 CANN 推理优化样例仓，面向 LLM 与多模态模型推理业务中的典型模型和加速算法，提供基于 CANN 平台的优化样例，帮助开发者快速完成 NPU 推理部署与性能验证。仓库以 `executor/` 提供统一推理执行框架，以 `models/` 提供不同模型的脚本与 YAML 配置，以 `module/` 和 `ops/` 提供基础层、量化模块以及 AscendC、PyPTO、TileLang 等算子样例。

本课程在运行时克隆该仓库（全量约 300MB，固定基线提交 `969160e`），并通过 `git am` 应用课程的单卡推理 patch（见 [patches/README.md](./patches/README.md)）：patch 在上游 LongCat-Next **多卡 TP** 样例之上增加 **单卡 CPU offload** 推理路径与配套 YAML/请求清单。克隆与 patch 应用过程幂等（已就绪直接复用、异常自动恢复），由第 2 章 Notebook 自动完成，无需手工操作；后续章节与后续会话直接复用同一克隆。学习本课程后，可以在克隆仓中查看 LongCat 系列的多卡 TP 部署（`config/longcat_next_*_tp*.yaml`）与更多模型的优化实践。

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A3 训练/推理系列产品（单卡 64GB HBM） |
| CANN 版本 | ≥ 8.5.1（教程在 8.5.1 上验证；样例建议 9.1.0） |
| Python | 3.12（配套 conda 环境，见下文环境准备） |
| PyTorch | `torch==2.8.0`、`torch_npu==2.8.0.post4`、`torchaudio==2.8.0` |
| 主机内存 | ≥ 64GB 可用（权重 mmap 与 CPU 专家计算，推荐 ≥ 128GB） |
| 磁盘空间 | 权重下载目录需 ≥ 160GB 可用（权重约 151GB）+ 推理仓克隆约 300MB（运行目录内，一次性）+ 结果目录若干 GB |
| 网络 | 首次运行需访问 gitcode.com（克隆 cann-recipes-infer）与 ModelScope/HuggingFace（下载权重）；此后复用本地缓存，离线可运行 |

## 环境准备

LongCat-Next 权重体积大（151GB），首次运行 Notebook 时会**自动从远程下载**（支持断点续传），不适配共享在线体验镜像；请在具备本地 NPU 与足够磁盘空间的主机上运行。第 2 章首次运行时还会把上游 cann-recipes-infer 仓克隆到运行目录（全量约 300MB）并应用单卡推理 patch，此后复用本地克隆，离线可运行。完整步骤：

1. 安装 CANN 软件包并 source 环境（`/usr/local/Ascend/ascend-toolkit/set_env.sh` 或等价脚本）。
2. 创建独立 conda 环境并安装依赖（torch_npu 需从 gitcode release 手动安装 whl，PyPI 不发布）：

```bash
conda create -y -n cann-recipes-longcat python=3.12
conda activate cann-recipes-longcat
pip install <torch_npu-2.8.0.post4-cp312-cp312-manylinux_2_28_aarch64.whl 的本地路径>
pip install -r tutorials/llm_inference/longcat_next/requirements.txt
```

3. 为该环境注册 Jupyter kernel（否则 Notebook 页面选不到该 Python）：

   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name cann-recipes-longcat \
       --display-name "Python 3 (cann-recipes-longcat)"
   ```

4. 模型权重获取（二选一）：

   - **自动下载（默认）**：不做任何设置，第 2 章首次运行时自动从 ModelScope 下载 `meituan-longcat/LongCat-Next`（约 151GB，视带宽约 25 分钟~4 小时，支持断点续传，下载中断后重跑会续传）。下载相关环境变量：

     | 变量 | 默认值 | 说明 |
     | --- | --- | --- |
     | `LONGCAT_NEXT_DOWNLOAD_BACKEND` | `modelscope` | 下载通道：`modelscope` 或 `hf`（Hugging Face，默认走 hf-mirror.com 镜像，可用 `HF_ENDPOINT` 覆盖） |
     | `LONGCAT_NEXT_DOWNLOAD_DIR` | `~/.cache/modelscope` | 下载缓存目录；**所在磁盘需 ≥ 160GB 可用**，空间不足会在下载前明确报错 |

     首次解析成功（下载或本地指定）后，路径会缓存在运行目录 `Sources/model_inference_optimization/longcat_next/resolved_model_path.txt`，**后续章节与后续会话自动复用**，不会重复下载；删除该缓存文件（或整个 `Sources` 运行目录）可重置。

   - **使用本地已有权重**：设置环境变量跳过下载：

     ```bash
     export LONGCAT_NEXT_MODEL_PATH=/path/to/LongCat-Next
     ```

     权重目录需包含 `model.safetensors.index.json`、`assets/`（官方演示素材）、`image_decoder/` 与 `cosy24k_vocoder/`（生成头权重）。
5. 启动 Jupyter 并选择 `Python 3 (cann-recipes-longcat)` kernel，同时确认空闲 NPU：

   ```bash
   export ASCEND_RT_VISIBLE_DEVICES=0   # 按需选择空闲健康的卡
   jupyter notebook --no-browser
   ```

> **注意**：
> - 第 2 章会按本目录 [`requirements.txt`](./requirements.txt) 检查 1-6 章统一使用的 Python 依赖：仅安装缺失项或替换不匹配的版本；第 3-6 章只复用该环境，不再安装依赖。
> - Notebook 的运行目录为仓库根下 `Sources/model_inference_optimization/longcat_next`（自动创建），用于保存推理仓克隆（`cann-recipes-infer/`）、运行 YAML 与指标；该目录已被 gitignore，可随时删除重跑（删除后第 2 章会重新克隆并打 patch）。

## 模型权重

- 模型标识：`meituan-longcat/LongCat-Next`（MIT License，15 个 safetensors 分片，ModelScope 与 HuggingFace 均已上架）。
- 获取方式：默认在第 2 章首次运行时自动从 ModelScope 下载（151GB，断点续传）；已下载或已有本地权重的，设置 `LONGCAT_NEXT_MODEL_PATH` 直接使用。
- 路径复用：首次解析成功后路径会写入运行目录的 `resolved_model_path.txt` 缓存，第 3-6 章及后续会话直接复用（打印"复用已解析的模型路径"），不重复下载也不重复联网校验。
- 使用方式：Notebook 直接加载解析出的权重目录，不做任何转换，也不写权重目录。
- 演示素材：权重目录自带官方 `assets/`（book.png、math1.wav、system_audio.wav、vc_zh3.wav 等），Notebook 环境准备阶段会自动创建软链 `requests/assets` 指向它，多模态请求按相对路径引用。

## 端到端复现

Notebook 环境中按顺序打开并运行。各章推理运行含约 2-3 分钟权重加载，总时长约 20-30 分钟（**首次运行第 2 章另需克隆推理仓（约 300MB）与下载 151GB 权重**，权重下载视带宽约 25 分钟~4 小时）：

1. `01_chapter_intro.ipynb`
2. `02_baseline_inference.ipynb`（小说续写 3-token 冒烟，约 2 分钟）
3. `03_text_generation.ipynb`（128-token 文本生成，约 6 分钟）
4. `04_multimodal_tasks.ipynb`（图理解/语音转写/语音合成约 16 分钟 + 图像生成约 8 分钟）

每章的推理启动方式一致：第 2 章完成一次性准备（克隆 cann-recipes-infer 到运行目录 `Sources/model_inference_optimization/longcat_next/cann-recipes-infer/`，`git checkout` 固定基线 `969160e` 并 `git am` 单卡 patch），此后各章把课程 YAML 写入克隆仓的 `models/longcat_next/config/`（写入当前模型路径），再显式展示并执行（`--yaml` 按上游约定传 `config/` 下的文件名）：

```bash
cd <运行目录>/cann-recipes-infer
bash executor/scripts/infer.sh --model longcat_next --yaml <运行yaml文件名>
```

recipes 日志与结果保存在克隆仓的 `models/longcat_next/res/<日期>/longcat_next_<yaml名>/`：`log_0.log` 为日志，`results/<请求id>/` 下是每条请求的 `text.txt` / `image*.png` / `audio*.wav` 与结构化 `result.json`（生成 token 数、自回归耗时、采样参数等）。开启 Profiling 后产物位于同目录 `prof/prefill`、`prof/decode` 下。

## 代码与权重说明

- 本实践的 Notebook 与 Notebook 辅助代码位于本目录 `src/inference_scripts/`：`recipe_workflow.py`（推理仓克隆与 patch 应用 bootstrap、运行 YAML 准备、结果目录定位、请求指标收集、Profiler 产物解析）与 `environment_check.py`（统一依赖检查）。
- 推理框架来自运行时克隆的上游 cann-recipes-infer 仓（固定基线 `969160e` + 单卡 patch，位于运行目录 `Sources/model_inference_optimization/longcat_next/cann-recipes-infer/`，布局与上游仓一致：executor（core / model_loader / offline / utils / scripts）、module、dataset 与 models/）。克隆与 patch 状态记录在运行目录 `infer_repo_state.json`，删除整个运行目录即可完全重置。
- LongCat-Next 模型实现位于克隆仓的 `models/longcat_next/models/`：`modeling_longcat_next.py`（主干 + MLA + shortcut MoE + 单卡 CPU offload 路径，由 patch 提供）、`model_infer.py`（多模态请求清单驱动入口）、`components/`（视觉/音频编码器、生成头、image_refiner、cosy24k vocoder）。
- 单卡配置由 patch 提供，位于克隆仓的 `models/longcat_next/config/`：`*_single_npu_offload_smoke.yaml` 冒烟基线（小说续写请求的 3-token 短版本）、`*_text_single_npu_offload.yaml` 文本生成、`*_multimodal_single_npu_offload.yaml` 多模态、`*_image_gen_single_npu_offload.yaml` 图像生成；同目录的 `*_tp*.yaml` 为上游多卡配置。
- 模型权重使用官方原始权重，无需转换；推理过程只读权重目录。

## 回馈上游：单卡实现 patch

本教程的单卡 CPU offload 实现已整理为 patch（`patches/longcat-next-single-npu.patch`，基于上游 master `969160e`），课程 Notebook 会在克隆仓上自动 `git am` 该 patch——这也是向社区分发单卡能力的标准方式：开发者 clone 上游仓、checkout `969160e` 后 `git am` 该 patch，即可运行 LongCat-Next 单卡推理；应用步骤、环境配置要点与常见问题见 [patches/README.md](./patches/README.md)。上游合入后本 patch 可弃用。

## 许可与第三方说明

本目录代码遵循 Apache License 2.0；其中源自 LongCat-Next（MIT License）与 HuggingFace 的部分见 [`Notice`](./Notice) 与 [`LICENSE.txt`](./LICENSE.txt)。

## Notebooks

<table>
<tr><th>Notebook</th><th>说明</th></tr>
<tr><td>1. 章节介绍</td><td>模型架构、单卡 CPU offload 原理与课程安排</td></tr>
<tr><td>2. Baseline 冒烟跑通</td><td>环境与工作区准备（克隆 cann-recipes-infer + 单卡 patch）、YAML 字段讲解、infer.sh 启动与结果读取</td></tr>
<tr><td>3. 文本生成体验</td><td>请求清单格式、采样参数与 128-token 小说续写</td></tr>
<tr><td>4. 多模态任务</td><td>图理解、语音转写、语音合成与图像生成</td></tr>
</table>
