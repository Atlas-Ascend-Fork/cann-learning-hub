# 第11章 pypto 优化算法

## 章节概述
本章内容改编自 [Dive into Deep Learning](https://zh.d2l.ai/) 第 11 章「优化算法」。在本章中，我们深入探讨常见的深度学习优化算法，包括凸优化的入门，以及凸目标函数上非常简单的随机梯度下降算法的证明。优化算法的**更新公式**（11.5 起参数为向量的部分）使用 PyPTO 逐元素内核在昇腾 NPU 上实现（`src/pypto_optimizers.py`），11.11 的训练优化器亦替换为 PyPTO 内核驱动的 `PyPTOSGD`。

## 在线阅读
| 文档 | Link | 状态 |
|--|--|--|
| 11.00 章节介绍 | [在线阅读](./11.00_chapter_intro.ipynb) | ✅ 已发布 |
| 11.01 优化和深度学习 | [在线阅读](./11.01_optimization_intro.ipynb) | ✅ 已发布 |
| 11.02 凸性 | [在线阅读](./11.02_convexity.ipynb) | ✅ 已发布 |
| 11.03 梯度下降 | [在线阅读](./11.03_gd.ipynb) | ✅ 已发布 |
| 11.04 随机梯度下降 | [在线阅读](./11.04_sgd.ipynb) | ✅ 已发布 |
| 11.05 小批量随机梯度下降 | [在线阅读](./11.05_minibatch_sgd.ipynb) | ✅ 已发布 |
| 11.06 动量法 | [在线阅读](./11.06_momentum.ipynb) | ✅ 已发布 |
| 11.07 AdaGrad算法 | [在线阅读](./11.07_adagrad.ipynb) | ✅ 已发布 |
| 11.08 RMSProp算法 | [在线阅读](./11.08_rmsprop.ipynb) | ✅ 已发布 |
| 11.09 Adadelta | [在线阅读](./11.09_adadelta.ipynb) | ✅ 已发布 |
| 11.10 Adam算法 | [在线阅读](./11.10_adam.ipynb) | ✅ 已发布 |
| 11.11 学习率调度器 | [在线阅读](./11.11_lr_scheduler.ipynb) | ✅ 已发布 |
