# 基于 BERT 的 IMDB 影评情感分类实战

## 1. 整体案例简介

本项目演示如何在华为昇腾 NPU 上，基于预训练的 **bert-base-uncased** 模型，在 **IMDB 影评数据集**上完成情感二分类任务（正面 / 负面），覆盖以下完整环节：

- CANN 环境加载与 NPU 可用性验证
- IMDB 数据集下载、格式解析与 Tokenize 预处理
- 预训练 BERT 模型与分类头的加载
- 训练超参数配置与模型训练
- 测试集准确率评估
- 训练前后推理效果对比，直观展现 Fine-tuning 的价值

## 2. 学习目标

完成本案例学习后，你应当能够：

1. 了解 IMDB 数据集的目录结构、数据格式与加载预处理流程，掌握将原始文本转化为 BERT 输入张量（input_ids / attention_mask / labels）的方法；
2. 掌握基于预训练 BERT 模型构建文本分类任务的方法，理解 `BertForSequenceClassification` 的结构与分类头的作用；
3. 掌握在昇腾 NPU 上配置 `TrainingArguments` 超参数、使用 `Trainer` 启动训练，并进行模型评估与推理的完整流程。

## 3. 环境准备

### 3.1 安装依赖

本案例依赖以下 Python 库，可通过 `pip` 统一安装：

```bash
pip install "torch>=2.5" torch_npu "transformers>=4.40" "accelerate>=0.30" "tqdm>=4.60" "numpy>=1.24"
```

| 库名 | 用途 |
| --- | --- |
| `torch` | PyTorch 深度学习框架 |
| `torch_npu` | 昇腾 NPU 适配层，使 PyTorch 能调用 NPU 算力 |
| `transformers` | Hugging Face 模型加载与训练（BERT、Trainer 等） |
| `accelerate` | 训练加速与设备调度 |
| `tqdm` | 进度条可视化 |
| `numpy` | 数值计算与指标统计 |

> 若使用 CANNLab 环境或魔搭（ModelScope）在线环境，上述库通常已预装，可跳过安装步骤。

## 4. 前置知识

学习本案例前，建议具备以下知识储备：

- **BERT 模型架构**：了解 BERT 的输入编码（Token Embedding / Segment Embedding / Position Embedding）、Masked Language Model 与 Next Sentence Prediction 预训练任务。推荐阅读 [BERT 论文](https://arxiv.org/pdf/1810.04805)。
- **Transformer Encoder 结构**：BERT 基于 Transformer 的 Encoder，需了解多头自注意力机制（Multi-Head Self-Attention）与前馈网络（Feed-Forward Network）的基本结构。
- **IMDB 数据集**：了解其数据格式与情感分类任务定义，参考 [IMDB 数据集](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)。
- **PyTorch 基础**：了解 `Dataset`、`DataLoader`、`Trainer` 等基本用法。

**IMDB 数据集背景介绍：**

IMDB 情感分析数据集是 NLP 领域广泛使用的电影评论情感分类任务数据集，最初由斯坦福大学发布，用于评估文本分类模型的情感识别能力。其数据格式以 `.txt` 文件组织，训练集和测试集各有 25000 条评论（50% 正面，50% 负面），此外还有 50000 条未标注评论可用于无监督学习。每条数据由 text（评论文本）和 label 组成，label 为 1 表示正面评论，为 0 表示负面评论。

## 5. 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上 |
| Python | 3.11 |
| PyTorch | `torch` + `torch_npu` |

## 6. 已验证的在线体验环境

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| CANNLab 云开发环境 | cann_9.0.0_py3.11-A2-arm | Python 3.11.4 | 参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md) 创建环境并运行 Notebook |

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)，选择对应 CANN 版本文档。

## 7. 课程章节目录

打开 `bert_sentiment_analysis.ipynb`，按照 notebook 中的章节顺序逐步学习：

| 小节 | 标题 | 内容说明 |
| --- | --- | --- |
| 1 | 内容概述 | 案例背景、BERT 模型简介与本案例涵盖的知识点 |
| 2 | 整体流程概览 | 端到端流程图与各环节简介 |
| 3 | 环境准备与依赖安装 | CANN 环境加载、依赖安装、NPU 可用性验证 |
| 4 | 模型介绍 | BERT 预训练原理（MLM / NSP）、模型加载与关键代码解读 |
| 5 | 数据集介绍与下载 | IMDB 数据集结构、下载解压、文本 Tokenize 预处理 |
| 6 | 训练模型与代码介绍 | 评估指标、训练超参数、Trainer 训练、测试集评估 |
| 7 | 训练前后推理效果对比 | 同一组示例文本在训练前 / 后模型上的推理结果对比 |
