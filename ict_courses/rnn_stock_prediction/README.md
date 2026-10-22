# 基于 LSTM 的股票价格走势预测实战

## 1. 整体案例简介

股票市场是一个复杂的、非线性的动态系统，其价格波动受到宏观经济、公司基本面、市场情绪等多种因素的影响。准确预测股票价格是金融领域最具挑战性的任务之一。传统的时间序列分析方法（如 ARIMA）难以捕捉复杂的非线性关系。

循环神经网络（Recurrent Neural Networks, RNN）及其变体，如长短期记忆网络（LSTM），能够有效捕捉时间序列数据中的长期依赖关系，因此被广泛应用于自然语言处理、语音识别以及金融时间序列预测等领域。LSTM 通过引入门控机制（输入门、遗忘门、输出门）解决了传统 RNN 中的梯度消失问题，能够选择性地保留或遗忘历史信息，特别适合处理具有长期依赖特征的时间序列数据。

本项目演示如何在华为昇腾 NPU 上，使用 **PyTorch** 深度学习框架（通过 `torch_npu` 适配层使能 NPU 加速）构建 **两层 LSTM** 模型，以"贵州茅台"（股票代码：600519）的历史日交易数据为样本，预测其未来收盘价走势。案例覆盖以下完整环节：

- CANN 环境加载与 NPU 可用性验证
- 使用 akshare 获取股票历史交易数据（网络不可用时自动从 AtomGit 下载离线数据）
- 数据归一化与时间窗口数据集构建
- LSTM 模型设计、训练与性能评估
- 预测结果可视化分析

## 2. 学习目标

完成本案例学习后，你应当能够：

1. 掌握使用 akshare 获取金融时间序列数据，并完成归一化、时间窗口构建等预处理流程，将原始股价数据转化为适用于 RNN 的监督学习问题；
2. 掌握使用 PyTorch 构建两层 LSTM 模型的方法，理解 LSTM 的门控机制（输入门、遗忘门、输出门）与输入输出张量格式（batch_size, seq_length, input_size）；
3. 掌握在昇腾 NPU 上配置损失函数、优化器、训练超参数并手写训练循环启动训练，以及通过 RMSE 指标与可视化图表评估模型预测性能的完整流程。

## 3. 环境准备

### 3.1 安装依赖

本案例依赖以下 Python 库，可通过 `pip` 统一安装：

```bash
pip install "torch>=2.5" torch_npu "akshare>=1.12" "pandas>=2.0" "matplotlib>=3.7" "numpy>=1.24" "scikit-learn>=1.3"
```

| 库名 | 用途 |
| --- | --- |
| `torch` | PyTorch 深度学习框架 |
| `torch_npu` | 昇腾 NPU 适配层，使 PyTorch 能调用 NPU 算力 |
| `akshare` | 获取 A 股历史交易数据（需要可访问外网） |
| `pandas` | 数据处理与日期解析 |
| `matplotlib` | 数据可视化与预测结果绘图 |
| `numpy` | 数值计算与指标计算 |
| `scikit-learn` | 数据归一化（MinMaxScaler）与 RMSE 评估 |

> 若使用 CANNLab 环境或魔搭（ModelScope）在线环境，PyTorch 与常用数据处理库通常已预装，可跳过安装步骤。

## 4. 前置知识

学习本案例前，建议具备以下知识储备：

- **RNN 与 LSTM 原理**：了解循环神经网络的基本结构，理解 LSTM 的门控机制（输入门、遗忘门、输出门）如何解决梯度消失问题。推荐阅读 [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)。
- **时间序列预测基础**：了解时间序列预测的基本概念，理解如何将时间序列转化为监督学习问题（滑动窗口法）。
- **PyTorch 基础**：了解 PyTorch 的 `nn.Module`、`nn.LSTM`、`DataLoader`、`TensorDataset` 等基本用法。
- **数据归一化**：了解 MinMaxScaler 的作用与原理，理解为何在训练前需要对价格数据进行归一化。

## 5. 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上 |
| Python | 3.9+ |
| PyTorch | `torch` + `torch_npu` |

## 6. 已验证的在线体验环境

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| CANNLab 云开发环境 | cann_9.0.0_py3.11-A2-arm | Python 3.11.4 | 参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md) 创建环境并运行 Notebook |

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件与 PyTorch + torch_npu，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)，选择对应 CANN 版本文档。

## 7. 课程章节目录

打开 `rnn_stock_prediction.ipynb`，按照 notebook 中的章节顺序逐步学习：

| 小节 | 标题 | 内容说明 |
| --- | --- | --- |
| 1 | 内容概述 | 案例背景、LSTM 原理简介与本案例涵盖的知识点 |
| 2 | 整体流程概览 | 端到端流程图与各环节简介 |
| 3 | 环境准备与依赖安装 | CANN 环境加载、依赖安装、NPU 可用性验证 |
| 4 | 模型介绍 | RNN/LSTM 门控机制、模型结构设计与代码实现 |
| 5 | 数据集介绍与下载 | akshare 获取股价数据、归一化、时间窗口构建 |
| 6 | 训练模型与代码介绍 | 损失函数、优化器、训练超参数、手写训练循环、模型训练与评估 |
