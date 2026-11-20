# CANN 品牌课程体系规划

> **定位**：将 CANN/昇腾技术体系外挂式融入本科四年 + 研究生培养全链路，按年级递进，覆盖从"认知算力"到"驾驭算力"到"创造算力"的完整能力曲线。
> **融入策略**：换芯不换课——课程名与学分不动，CANN 以单元植入 / 实验替换 / 算子案例三种模式进入。

---

## 〇、CANN 可教学知识地图

> 三级难度梯度，从入门到精通：**初级 → 中级 → 高级**

| 层级 | 包含技术 | 说明 |
|------|---------|------|
| **L4 · AI 应用** | 企业应用案例 · 深度学习 · 推荐系统 · 生成式 AI · 科学计算 | 顶层场景：模型跑在什么业务上 |
| **L3 · AI 框架** | PyTorch 框架基础 · 模型训练 · 模型推理 · 模型部署与优化案例 | 中间层：框架适配与训推全流程 |
| **L2 · 加速库** | Aclnn 算子库 · 通信库（HCCL）· Thrust/C++ 并行算法库 · Python 加速库 · 领域加速库（ops-nn / ops-math / ops-transformer / MathDx）· 图引擎（GE） | CANN 核心：算子与通信加速 |
| **L1 · 算子编程 / NPU 架构** | Ascend C · PyPTO · PyAsc · CANNBot DSL · Ascend C 通信编程 · MathDx · NPU 架构 | 底层：直接操作硬件的编程能力 |

> **难度定义**：
> - **初级 · 认知入门**：跑通 HelloWorld，建立基本概念
> - **中级 · 系统实践**：独立完成模块开发，能解决实际问题
> - **高级 · 深度优化**：追求极致性能，能做架构级创新

---

## 一、课程体系总表（按年级 × 学时 × 知识地图层级 × 技术深度）

| 年级 | 课程名称 | 理论 | 实践 | 知识地图层级 | 涉及的 CANN 技术栈与深度 |
|------|---------|:---:|:---:|:---:|----------------------|
| **大一** | 人工智能基本原理与工程实践 I（智能导论） | 32 | 0 | L4→L3→L1 认知 | **L4 AI 应用**：企业应用案例、深度学习（初级）· **L3 AI 框架**：PyTorch 框架基础（初级）· **L1 NPU 架构**：NPU 芯片概览（初级） |
| **大二** | 人工智能基本原理与工程实践 II（机器学习、深度学习、强化学习） | 48 | 0 | L3→L2 | **L3 AI 框架**：PyTorch 训练与推理（中级）· **L2 加速库**：Aclnn 算子库（ops-nn / ops-math）、Thrust/C++ 并行算法库、通信库 HCCL（中级）· **L1 算子编程**：Ascend C / PyPTO（初级） |
| **大二** | 人工智能基本原理与工程实践 III（形式语言、自然语言、语音技术） | 80 | 0 | L2→L3 | **L2 加速库**：领域加速库 ops-transformer（中级）· Aclnn 算子库（Embedding/Attention/FFT）（中级）· **L3 AI 框架**：模型推理（中级）· **L1 算子编程**：Ascend C / PyPTO（初级） |
| **大二** | 计算机体系结构 | 48 | 0 | L1（● 主讲） | **L1 NPU 架构**：达芬奇架构（三核异构 / 软件管理存储 / 数据搬运 / 多级编程体系）、大模型负载画像与 Roofline、SIMD vs SIMT 架构辩论、超节点演进（中高级）· **AI 训推案例**：Matmul 950 病历诊断（10742us→2800us）（中高级）· **L3 AI 框架**：Qwen3-1.7B 部署大作业（初级） |
| **大三** | 人工智能基本原理与工程实践 IV（视觉与生成式 AI、大模型与开源、数据工程） | 80 | 0 | L2→L1 | **L1 算子编程**：Ascend C / PyPTO（中级）· 通算融合算子编程（中高级）· **L2 加速库**：ops-transformer（中级）· **图引擎 GE**（中高级）· **L3 AI 框架**：PyTorch 训推含量化（中高级）· 模型部署与优化案例（中高级） |
| **大三** | 人工智能基本原理与工程实践 V（系统与实践） | 0 | 48 | L1→社区 | **L1 算子编程**：Ascend C / PyPTO / CANNBot DSL / Triton（中级）· **AI 训练和推理优化**（中高级）· **CANN 社区任务**（算子交付 + Skill 反哺 + Issue→PR） |
| **大三** | 并行计算与算子编程 | 48 | 0 | L1 | **L1 算子编程**：Ascend C（中高级）· PyPTO（中高级）· PyAsc · CANNBot DSL · **L1 NPU 架构**：多级内存 / SIMD/SIMT · **算子最佳案例** · **CANN 社区任务** |
| **大三** | 人工智能芯片设计 | 48 | 0 | L1 | **L1 NPU 架构**（中高级）——达芬奇架构、SIMD/SIMT 新同构、多级存储、片上互联 |

---

## 二、知识地图层级覆盖矩阵

| 知识地图层级 | 大一 | 大二 | 大三 | 研究生 |
|-------------|:---:|:---:|:---:|:---:|
| **L4 AI 应用** | **初级**（I 课：企业案例、深度学习、生成式 AI、科学计算等场景认知） | **中级**（II 课：深度学习/ML/DL/RL 实践；III 课：NLP 语音应用实践） | **中高级**（IV 课：视觉与生成式 AI、大模型与开源、数据工程综合实践；V 课：社区任务真实场景） | **高级**（训推优化课：大模型端到端案例） |
| **L3 AI 框架**（PyTorch/训练/推理/部署优化） | **初级**（I 课：PyTorch 框架基础，.npu() 体验） | **中级**（II 课：torch_npu 训练；III 课：模型推理） | **中高级**（IV 课：量化部署+MindIE 推理；V 课：训练与推理优化实战） | 高级 |
| **L2 加速库**（Aclnn/HCCL/Thrust/Python 加速库/领域加速库/图引擎 GE） | — | **中级**（II 课：ops-nn/ops-math/HCCL/Thrust；III 课：ops-transformer） | **中高级**（IV 课：GE 图引擎；V 课：HCCL 多卡 + GE 入图） | 高级 |
| **L1 算子编程**（Ascend C/PyPTO/PyAsc/CANNBot DSL/Triton/MathDx/Ascend C 通信编程） | — | 初级（II/III 课：Ascend C 入门） | **中级→中高级**（V 课：Ascend C/PyPTO/CANNBot DSL/Triton 四前端实战；并行计算课：Ascend C 中高级） | 高级 |
| **L1 NPU 架构**（达芬奇/多级存储/SIMD/SIMT/片上互联） | 初级（I 课：架构一瞥） | **中高级**（计算机体系结构：达芬奇四维度 + Roofline + 超节点；II 课：NPU 芯片基础） | **中高级**（芯片设计课：达芬奇架构深度解析 + Ascend 950 新架构） | 中高级 |
| CANN 社区任务 | — | — | ✓（V 课 + 并行计算课） | ✓ |

> **深度定义**：初级 = 跑通 HelloWorld、建立基本概念；中级 = 独立完成模块开发、解决实际问题；中高级 = 能写算子/优化性能/参与社区；高级 = 追求极致性能、能做架构级创新。

---

## 三、年级递进路线图

```
大一              大二                        大三                              研究生
─────            ────────                  ──────────                      ──────────
认知算力          驾驭算力                    创造算力                          引领算力
(L4→L3 认知)     (L3 框架 + L2 加速库)      (L1 算子编程 + NPU 架构)          (L1/L2/L3 高级)
  │                 │                          │                                │
  ▼                 ▼                          ▼                                ▼
人工智能基本       人工智能基本原理与        人工智能基本原理与              深化方向课
原理与工程实践I   工程实践II/III           工程实践IV/V
(L4 AI应用认知    (L3 torch_npu训推        (L2 GE图引擎+L1融合算子          (极致优化/社区贡献)
 +L1 NPU架构一瞥)  +L2 ops-nn/ops-math      +L3 量化部署)
                    +L2 hccl/thrust)        工程实践V(L1社区任务)
                                           并行计算与算子编程(L1中高级)
                                           人工智能芯片设计(L1 NPU中高级)
  │                 │                          │                                │
  ▼                 ▼                          ▼                                ▼
CANNBot体验      torch_npu跑模型            Ascend C写算子                   社区贡献PR
CANNLab入门      Aclnn/HCCL调用             CANNJudge判题                    cann-bench打榜
                 ops-transformer推理        社区任务(真实贡献)               
```

---

## 四、各课程大纲

### 4.1 大一 · 人工智能基本原理与工程实践 I（智能导论）

> **学时**：32 理论 + 0 实践　|　**对标**：Stanford CS221 + AI 通识三讲
> **CANN 技术深度**：CANN（初级）· NPU 芯片（初级）· PyTorch（初级）

| 周 | 主题 | 内容要点 | CANN/昇腾融入点 |
|---|------|---------|--------------|
| 1 | AI 发展史与范式 | 符号主义→连接主义→深度学习→大模型；图灵测试到 GPT | 昇腾 NPU 发展史（910→910B→950 四代演进） |
| 2 | AI 生态全景 | 算力底座（CPU/GPU/NPU）→框架层→应用层 | CANN 六仓生态概览（cannbot/cannbot-skills/cannbot-knowledge/cannbot-dsl/cann-bench/cannbot-sentry） |
| 3 | NPU 与达芬奇架构一瞥 | 为什么 AI 需要 NPU；Cube/Vector/Scalar 三单元 | CANNLab 云上环境入门；Ascend C 编程模型一瞥 |
| 4 | 搜索与问题求解 | BFS/DFS/A*、约束满足、博弈搜索 | — |
| 5 | 机器学习初步 | 监督/无监督/强化学习三范式；过拟合与泛化 | PyTorch `.npu()` 初体验（torch_npu 一行 import） |
| 6 | 神经网络直觉 | 感知机→MLP→激活函数→损失函数 | NPU vs CPU 前向传播速度对比（CANNLab 实验） |
| 7 | AI 伦理与国产化 | 偏见/公平/可解释；科技自立与生态共建 | CANN Bot AI 辅助开发体验（Flash 版 11 分钟交付 add 算子） |
| 8 | 结业展示 | 小组项目：用 AI 解决一个真实问题 | 提交含 NPU 运行截图的项目报告 |

---

### 4.2 大二 · 人工智能基本原理与工程实践 II（机器学习、深度学习、强化学习）

> **学时**：48 理论 + 0 实践　|　**对标**：Stanford CS229/CS230 + CMU 10-701
> **CANN 技术深度**：NPU（初级）· Ascend C/PyPTO（初级）· CANN 加速库 ops-nn/ops-math/thrust/hccl（中级）· PyTorch（中级）

| 周 | 主题 | 内容要点 | CANN/昇腾融入点 |
|---|------|---------|--------------|
| 1 | 线性回归与梯度下降 | 最小二乘、SGD/Mini-batch、学习率调度 | CPU vs NPU 梯度下降速度对比（torch_npu） |
| 2 | 逻辑回归与正则化 | 交叉熵、L1/L2、偏差-方差分解 | — |
| 3 | SVM 与核方法 | 硬/软间隔、核技巧、SMO | — |
| 4 | 决策树与集成 | CART / Random Forest / GBDT / XGBoost | — |
| 5 | 无监督学习 | K-Means / PCA / t-SNE / 自编码器 | NPU 上跑 t-SNE 加速可视化 |
| 6 | 前馈神经网络 | 反向传播、激活函数、BatchNorm/Dropout | torch_npu `.npu()` 全流程训练 MLP |
| 7 | 卷积神经网络 | Conv / Pooling / ResNet / 迁移学习 | **ops-nn 算子库**：Conv/Pooling/BN 在昇腾上的加速调用 |
| 8 | 序列模型 | RNN / LSTM / GRU / Seq2Seq / Attention | — |
| 9 | Transformer | 自注意力 / 位置编码 / BERT / GPT | NPU 显存估算（参数量 vs FP16 字节数）；**ops-math 算子库**（GEMM/MatMul） |
| 10 | 分布式训练 | 数据并行 / 模型并行 / 流水线并行 | **HCCL 通信库**：DDP 多卡训练 ResNet（AllReduce 梯度同步）；thrust 并行算法库 |
| 11 | 强化学习 I | MDP / 值迭代 / Q-Learning / DQN | — |
| 12 | 强化学习 II | Policy Gradient / Actor-Critic / PPO | NPU 并行环境模拟实验 |
| 13 | 深度强化学习 | TRPO / SAC / 模仿学习 / 离线 RL | — |
| 14 | 项目答辩 | 训练完整模型（分类/检测/游戏 AI） | 提交训练曲线 + NPU 利用率截图 |

---

### 4.3 大二 · 人工智能基本原理与工程实践 III（形式语言、自然语言、语音技术）

> **学时**：80 理论 + 0 实践　|　**对标**：Stanford CS224n + MIT 6.806
> **CANN 技术深度**：Ascend C/PyPTO（初级）· CANN 加速库 ops-transformer（中级）· PyTorch（中级）

| 周 | 主题 | 内容要点 | CANN/昇腾融入点 |
|---|------|---------|--------------|
| 1 | 形式语言基础 | 有限自动机 DFA/NFA、正则表达式、泵引理 | — |
| 2 | 上下文无关文法 | CFG / CNF / CYK 算法 / Chomsky 层级 | — |
| 3 | 图灵机与可计算性 | 通用图灵机、停机问题、归约 | — |
| 4 | NLP 基础 | 分词 / 词性标注 / NER / 句法分析 | — |
| 5 | 词向量与语言模型 | Word2Vec / GloVe / n-gram / 困惑度 | PyPTO 并行化 Word2Vec 训练体验 |
| 6 | 序列标注 | HMM / CRF / BiLSTM-CRF | — |
| 7 | 注意力机制与 Transformer | Self-Attention / Multi-Head / 位置编码 | **ops-transformer 算子库**：Attention/Softmax/LayerNorm 在昇腾上的加速调用 |
| 8 | 预训练语言模型 | BERT / GPT / T5 / 预训练-微调范式 | HuggingFace `device_map='npu'` 微调 BERT |
| 9 | 大语言模型 | Scaling Law / RLHF / 涌现能力 / Prompt 工程 | MindIE LLM 推理加速实验（ops-transformer 调用） |
| 10 | 语音基础 | 声学信号 / FFT / MFCC / 声学特征提取 | NPU 加速特征提取实验 |
| 11 | 语音识别 | HMM-GMM → DNN-HMM → E2E（CTC/RNN-T） | CTC loss 在昇腾上的调用 |
| 12 | 语音合成与大模型 | Tacotron / FastSpeech / VALL-E / Whisper | Whisper NPU 推理部署 |
| 13 | 多模态与前沿 | CLIP / SpeechGPT / 语音+文本联合建模 | — |
| 14 | 项目答辩 | NLP 或语音任务 | NPU 推理 + 性能分析报告 |

---

### 4.4 大三 · 人工智能基本原理与工程实践 IV（视觉与生成式 AI、大模型与开源、数据工程）

> **学时**：80 理论 + 0 实践　|　**对标**：Stanford CS231n + Berkeley CS182 + CMU 11-785
> **CANN 技术深度**：Ascend C/PyPTO（中级）· ops-transformer（中级）· GE 图引擎（中高级）· PyTorch 训推含量化（中高级）· 通算融合算子编程（中高级）

| 周 | 主题 | 内容要点 | CANN/昇腾融入点 |
|---|------|---------|--------------|
| 1 | 计算机视觉基础 | 图像表示 / 滤波 / 边缘检测 / SIFT | — |
| 2 | CNN 与视觉算子 | Conv / Pooling / AlexNet→ResNet / ViT | **Ascend C/PyPTO 中级**：Conv 算子的 Tiling 与并行实现 |
| 3 | 目标检测与分割 | R-CNN / YOLO / Mask R-CNN / DETR | — |
| 4 | 生成模型 I：GAN | 生成器/判别器 / DCGAN / StyleGAN | NPU GAN 训练 + AMP 混合精度 |
| 5 | 生成模型 II：扩散模型 | DDPM / DDIM / Stable Diffusion | **GE 图引擎**：Diffusers 模型入图加速（AclGraph 捕获/复用） |
| 6 | 大模型 I：架构与训练 | Transformer 变体 / MoE / 长上下文 | **通算融合算子编程**：FlashAttention 的算子融合实现原理 |
| 7 | 大模型 II：对齐与微调 | RLHF / DPO / LoRA / QLoRA / PEFT | **PyTorch 训推含量化（中高级）**：QLoRA 量化微调实验（W8A16/W4A16） |
| 8 | 大模型 III：推理与部署 | KV Cache / PagedAttention / 投机解码 / PD 分离 | MindIE 大模型推理部署（PD 分离 + 量化推理） |
| 9 | 模型压缩与量化 | GPTQ / AWQ / SmoothQuant / FP8 / 剪枝/蒸馏 | 量化前后的 NPU 推理性能对比报告 |
| 10 | 数据工程 I | 数据湖 / ETL / 特征工程 / 数据质量 | — |
| 11 | 数据工程 II：GPU 加速流水线 | 多模态数据 / 合成数据生成 | **GE 图引擎**：端到端数据预处理+训练图融合 |
| 12 | 开源生态 | HuggingFace / Model Zoo / License / 社区贡献 | torch.compile 图模式 vs 手写 Ascend C 性能对比 |
| 13 | 综合项目开发 | 视觉/生成式/大模型/数据工程方向选题 | 完整 NPU 训推性能报告 |
| 14 | 项目答辩 | 性能优化路径陈述 | Roofline 分析 + 与原生性能对标 |

---

### 4.5 大三 · 人工智能基本原理与工程实践 V（系统与实践）

> **学时**：0 理论 + 48 实践　|　**对标**：CANN Bootcamp 一天/两天营 + CANN 社区任务
> **CANN 技术深度**：**L1 算子编程**：Ascend C / PyPTO / CANNBot DSL / Triton（中级）· **AI 训练和推理优化**（中高级）· **CANN 社区任务**

| 周 | 主题 | 实践内容 | CANN/昇腾融入点 |
|---|------|---------|--------------|
| 1 | Ascend C / PyPTO 入门 | CANN Bot Flash 版交付第一个算子（add/sigmoid）；PyPTO Python 前端快速开发 | CANN Bot 辅助开发：需求五要素 + 四步安装 + 八阶段推进 |
| 2 | 矢量算子开发 | Ascend C Memory/Reg 矢量算子（Add → Softmax）；CANNBot DSL 领域 DSL 快速生成 | CANNJudge 判题：多维 shape + 多 dtype + 精度门禁 |
| 3 | 矩阵与融合算子开发 | MatMul（Mmad 五步法）+ MatMul+LeakyReLU 融合（__mix__ + C/V 分离） | asc-devkit 样例参考 + 融合收益分析：消除中间 GM 写读 |
| 4 | Triton 算子开发 | Triton-Ascend 后端：Block-level 编程 + 自动调优 | Triton kernel 开发与 Ascend C 性能对比实验 |
| 5 | AI 训练与推理优化 | 模型训练优化（AMP 混合精度 + HCCL 多卡通信）；推理优化（量化 + torch.compile 入图） | **AI 训推优化**：torch_npu + MindIE + GE 图引擎 + HCCL |
| 6 | 调试与调优 | 精度调试（ascendc-precision-debug）+ 性能调优（msProf 剖析 + Roofline 分析） | msProf 剖析 + 调优三原则（数据先于观点 / 分档读法 / 流量思维） |
| 7 | **CANN 社区任务** | 从 gitcode.com/cann/discussions/211 认领真实社区算子任务，完成开发→测试→PR→评审→Skill 反哺 | Issue→PR 八步五条红线 + CANNJudge 判题 + cannbot-skills 仓贡献（三段式策略条目） |
| 8 | 结业答辩 | 算子交付 + 训推优化报告 + 社区贡献陈述 | PR 合入 / 判题通过 / Skill 贡献 即课程成绩 |

> **课程定位**：本课程不新授知识——它是人工智能基本原理与工程实践 I~IV + 并行计算与算子编程的综合实训出口，学生以真实社区任务驱动学习，产出（PR 合入 / 判题通过 / Skill 贡献）即课程成绩。

---

### 4.6 大三 · 并行计算与算子编程

> **学时**：48 理论 + 0 实践　|　**对标**：CMU 15-418 + Stanford CS149（昇腾版）
> **CANN 技术深度**：Ascend C / PyPTO / Python DSL（中高级）· Thrust 并行算法库 · 算子最佳案例 · **CANN 社区任务**

#### 第一部分　并行计算基础

| 周 | 主题 | 内容要点 | CANN/昇腾融入点 |
|---|------|---------|--------------|
| 1 | 并行计算概论 | 为什么要并行 / Flynn 分类（SIMD/SIMT/MIMD）/ NPU vs CPU 哲学 | 昇腾 NPU 与达芬奇架构总览 |
| 2 | 基本并行算法模式 | Map-Reduce / Scan-Prefix / Stencil-Convolution / Recursion 四大范式 | 四类范式在 NPU 矢量/矩阵单元上的映射 |
| 3 | 基于 Thrust 并行算法实现 | Thrust C++ 并行算法库：transform / reduce / sort / scan / inner_product | **Thrust 并行算法库**：在昇腾 NPU 上跑 Thrust 矢量运算（替代 CUDA Thrust，同一套 C++ 代码映射 NPU 后端） |

#### 第二部分　算子编程模型

| 周 | 主题 | 内容要点 | CANN/昇腾融入点 |
|---|------|---------|--------------|
| 4 | Ascend C SIMD 编程 | SPMD 异构编程（核函数 `__global__` / `<<<>>>` 启动 / block_idx 分片 / 执行空间限定符 `__vector__` / `__cube__` / `__mix__`）；多级内存层级（GM / UB / L1 / L0A/L0B/L0C / DataCopy 编排）；Reg 矢量算子编程（Memory 矢量 TPipe/TQue → Reg 矢量 `__simd_vf__`）；矩阵算子编程（Mmad 五步法）；融合算子编程（AIC/AIV 协同 / `__mix__` + SetFlag/WaitFlag 同步） | Ascend C 编程模型全体系：Add 入门→MatMul Tiling→MatMul+LeakyReLU 融合；UB 256K 布局与搬运流水 |
| 5 | Ascend C SIMT 编程 | Grid/Thread Block/Thread 三级架构 / 指针直访（无显式搬运）/ `__launch_bounds__` / gather 实战 | SIMT 一行搞定离散访存：`output[idx]=input[index[idx]]` |
| 6 | Ascend C SIMD & SIMT 混合编程 | `__simd_vf__` / `__simt_vf__` 两种 VF / `asc_vf_call` 编排 / 一个 Kernel 混合两种执行模式 | **算子最佳案例**：gather_and_adds（规整段 SIMD + 离散段 SIMT） |
| 7 | Python DSL 算子编程 | CANNBot DSL / PyAsc / PyPTO 三套 Python 前端：Tensor 编程 / SIMT 编程 / 领域 DSL；与 C++ Ascend C 的对比与互调 | **Python DSL**：用 PyAsc 一行声明 Tensor、用 PyPTO 快速开发 SIMT 算子——"Python 写得快，C++ 跑得快"的双前端范式 |

#### 第三部分　并行算法实践

| 周 | 主题 | 内容要点 | CANN/昇腾融入点 |
|---|------|---------|--------------|
| 8 | Elementwise / Reduce 类算子实践 | Add / GELU / Sigmoid 逐元素算子开发；Softmax 归约算子的分级归约与掩码尾块；毕昇编译器端到端编译流程（.asc→CMake→可执行） | **算子最佳案例**：Softmax v0→v6 逐版本优化（访存受限→分档读法→流水重叠）；**毕昇编译器**：编译各阶段产物分析 |
| 9 | 矩阵类算子实践 | MatMul 算子：Tiling 分块 / 双缓冲流水 / 与 cuBLAS 对标 / CANNJudge 判题 | **算子最佳案例**：MatMul+LeakyReLU 融合（v1→v8 优化史：消除中间 GM 写读） |
| 10 | Attention 融合算子实践 | FlashAttention：分块注意力 + online softmax + IO 感知优化 / AIC+AIV 协同 / SIMD+SIMT 混合实现 | **通算融合算子编程**：FlashAttention 的算子融合原理与昇腾实现 |
| 11 | 调试调优 + **CANN 社区任务** | msProf 剖析 / Roofline 分析；从 gitcode.com/cann/discussions/211 认领真实算子任务并完成开发→测试→提交 | Issue→PR 八步五条红线 + CANNJudge 判题 |
| 12 | 结业答辩 | 算子开发全流程展示 + 性能分析报告 + 社区贡献 | CANNJudge 判题 + 社区 PR + Roofline 分析 |

> **详细大纲**：见 [02_traditional_courses/parallel_computing.md](../02_traditional_courses/parallel_computing.md)

---

### 4.7 大三 · 人工智能芯片设计

> **学时**：48 理论 + 0 实践　|　**对标**：CMU 18-447 + Berkeley CS152（NPU 实例版）
> **CANN 技术深度**：NPU 芯片架构（中高级）

| 周 | 主题 | 内容要点 | CANN/昇腾融入点 |
|---|------|---------|--------------|
| 1 | AI 芯片全景 | GPU vs NPU vs ASIC / 市场格局 / 设计约束 | 昇腾 910→950 四代演进主线 |
| 2 | 达芬奇计算单元 | Cube（16×16×16 Mmad）/ Vector（256B VL）/ Scalar | AI Core 三单元的指令发射与独立调度 |
| 3 | 存储层次设计 | UB 256K / L1 / L0A/L0B/L0C / GM 多级金字塔 | 与 CPU Cache 层次对比：手动挡 vs 自动挡 |
| 4 | 数据通路与 DMA | GM→UB / L1→L0 / MTE 引擎 / NDDMA 多维搬运 | Ascend C 编程模型中的存储映射 |
| 5 | Ascend 950 新架构 | SIMD/SIMT 新同构 / Reg 矢量 / 低比特 FP4/6/8/HiF8 | VF 粒度切换机制与混合编程支撑 |
| 6 | SIMT 执行单元 | Thread Block≤2048 / 128K RegFile / Warp 调度 | 与 GPU SIMT 对比（TB 绑核 / UB 作 Shared Memory） |
| 7 | 片上互联 | 灵衢 2.0 / Clink / CCU 集合通信加速器 / HCCS | 超节点（384 卡）的通信拓扑与带宽 |
| 8 | MX 低比特计算 | FP4/FP8 / MXFP8 / HiF8 / 分块共享缩放 | CANN 软硬协同设计——算子如何利用低比特 |
| 9 | 芯片验证与流片 | RTL 设计 / 验证方法 / 性能评估 | CANN 仿真环境（无硬件代码验证） |
| 10 | 与 CUDA 架构对比 | SM vs AI Core / HBM vs UB / Cache vs 显式管理 | 两种设计哲学的取舍分析 |
| 11 | 超节点与集群 | Atlas 900 / 集合通信 / PD 分离 / 大模型训练集群 | HCCL 在超节点上的性能优化 |
| 12 | AI 芯片未来 | 存算一体 / 光计算 / 量子 AI / Neuromorphic | 昇腾路线图展望 |
| 13 | 综合案例 | 设计一个简化版 AI Core（指令集/数据通路/存储规划） | 期末设计项目 |
| 14 | 结业答辩 | 架构设计报告 + NPU 性能建模 | 与 Ascend 950 对标分析 |

> **详细大纲**：见 [02_traditional_courses/computer_architecture.md](../02_traditional_courses/computer_architecture.md)

---

### 4.8 大二 · 计算机体系结构

> **学时**：48 理论 + 0 实践（13 次大课，含 9 课时"AI 时代的体系结构挑战"主讲单元）　|　**对标**：CMU 18-447 / Berkeley CS252 + NPU 主讲单元
> **CANN 技术深度**：**L1 NPU 架构**（中高级）· **AI 训推案例**（中高级）· **大作业**（初级）

#### 第一部分　CPU 传统体系结构（课次 1~8、12~13，沿用现有讲义）

| 课次 | 模块 | 主题 | CANN/昇腾融入点 |
|------|------|------|--------------|
| 1 | Preliminaries | Intro + 性能指标（CPI/MIPS/功耗，○ TOPS/W 一瞥） | ○ 提及 |
| 2~4 | Processor | In-order / SMT / O3·Superscalar | ○ 埋"NPU 为什么不需要 O3"伏笔 |
| 5~7 | Memory | Cache ×2 / VM·TLB / RAID·Cloud | ◐ 课次 6：硬件预取 vs NPU 显式 Tiling 搬运 |
| 8 | 综合复习 | 第一部分收束 | — |
| 12 | Interconnect | NoC / CXL / RDMA / PCIe / HCCS | ◐ 收割课次 11 超节点框架 |
| 13 | Misc | Security + 其他领域 + 课程总结 | ○ DSA 叙事收束 |

#### 第二部分　AI 时代的体系结构挑战（课次 9~11，● 主讲 9 课时）

| 课次 | 主题 | 内容要点 | CANN/昇腾融入点 | 产出物 |
|------|------|---------|--------------|-------|
| 9 | **AI 时代大模型对体系结构的挑战**（需求评审会） | 大模型负载画像（算子 DAG 四类 / 存储特征 / Roofline 手算）；CPU 困境（苏格拉底三连问 / 功耗墙 / Amdahl）；CPU→GPU SIMT 演进与"SIMT 是终点吗"结构化辩论 | NPU 作为应答方案登场：需求清单逐条映射达芬奇四维度 | 《甲方需求清单》 |
| 10 | **面向大模型负载的 NPU 架构设计**（方案评审会） | 维度一·三核异构（Cube=磨盘 / Vector=排刀 / Scalar=主厨）；维度二·存储层次（CPU Cache 自动挡 vs UB/L1/L0 手动挡）；维度三·数据搬运（NDDMA 多维一次指令）；维度四·编程模型（多级 API = 同一底盘换变速箱） | **达芬奇架构四维度完整应答甲方需求**（逐条对照课） | 《乙方答辩表》 |
| 11 | **NPU 架构的性能优化与超节点扩展**（验收调优会） | 优化四维方法论（任务编排 / 流水并行 / 访存 / 计算）；Matmul 950 病历实战（10742us→2800us + 283.6%）；超节点（HCCS / 模型切分 / AllReduce 与 HCCL） | **通算融合 + PD 分离 + KV 池化**优化实践；HCCL 集合通信 | 《终章设计清单》 |

#### 第三部分　结业作业

| 类型 | 内容 | CANN/昇腾融入点 |
|------|------|--------------|
| 小作业 | 8 道问答题（对齐第 4 讲课件） | 含 CANN/NPU 综合题 |
| 大作业（必选） | Qwen3-1.7B 部署跑通 | **torch_npu + MindIE 部署推理**（CANNJudge 判题） |
| 大作业（可选加分） | add_rms_norm 三档（功能跑通 / 集成 / 优化） | **Ascend C 算子开发**（CANNJudge 承载） |

> **详细大纲**：见 [02_traditional_courses/computer_architecture.md](../02_traditional_courses/computer_architecture.md)

---

### 4.9 研究生 · 深化方向

| 方向 | 学时 | 技术栈 | 产出要求 |
|------|------|-------|---------|
| 大模型训练与推理优化 | 32 理论 + 16 实践 | MindIE / HCCL / GE 融合优化 / MSProfiler / 超节点 / cann-bench | 毕业设计或论文（性能对比数据） |
| 自定义算子开发 | 32 理论 + 16 实践 | Ascend C SIMT/混合编程 / CANN Bot Skill 反哺 / 算子极致优化案例 | cann-samples PR 合入 / cann-bench 上榜 |

> **驱动方式**：以 CANN 社区真实任务（gitcode.com/cann/discussions/211）驱动科研训练。

---

## 五、配套资源索引

| 资源 | 位置 | 用途 |
|------|------|------|
| CANN 社区主站 | https://gitcode.com/cann | 开源社区入口 |
| CANN 社区任务 | https://gitcode.com/org/cann/discussions/211 | 算子任务认领（人工智能基本原理与工程实践 V + 并行计算与算子编程 + 研究生） |
| CANN Learning Hub | https://gitcode.com/cann/cann-learning-hub/ | 全栈教程与 Notebook（各课实验载体） |
| Ascend C API 与样例 | https://gitcode.com/cann/asc-devkit | 算子开发参考实现 |
| CANN Bot | https://gitcode.com/cann/cannbot | AI 辅助算子开发（I 课体验→V 课全流程） |
| cannbot-skills | https://gitcode.com/cann/cannbot-skills | Skill 贡献目标仓（V 课社区实践） |
| cann-bench 评测 | https://gitcode.com/cann/cann-bench | 算子性能排行榜（IV/研究生课） |
| 教材《Ascend C 异构并行程序设计》 | https://gitcode.com/HIT1920/AscendCBook | 并行计算课系统化教材 |
