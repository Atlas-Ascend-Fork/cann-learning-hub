# 《AI 模型部署与工程实践》培训课大纲

---

## 一、课程简要介绍

| 项目               | 内容                                                                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **课程名称** | AI 模型部署与工程实践（四次课）                                                                                                                     |
| **课程类型** | 短期培训课（高校学生培训 / 高校夏令营 / 竞赛备赛）                                                                                                   |
| **学时安排** | 4 次课 × 4 小时 ≈ 16 小时（每次理论 2h + 实践 2h，1:1 配对；结业作业课后线上完成，不占课时）                                                          |
| **面向对象** | 高校学生（计算机 / 人工智能 / 软件工程等相关专业），具备 Python 基础、希望入门大模型推理优化与昇腾 NPU 开发的学习者                                 |
| **先修要求** | Python 基础语法；Linux 命令行基本操作；了解 GIT 基本操作；建议具备深度学习 / PyTorch 基本概念（非硬性要求）                                         |
| **配套讲义** | 5 份 PPT，由[cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/blob/master) 教程仓在线承载（获取链接见第四章教学安排）                     |
| **实践平台** | CANN Lab 云端开发环境 ＋ cann-learning-hub（教程型跟练 Notebook）＋ cann-recipes-infer（推理优化工程实践参考仓）                                    |
| **结业标准** | 按时线上提交结业作业：两项必选作业（Qwen3-8B 基线推理与 Profiling 瓶颈分析简报、W8A8 量化部署与对比报告）考核通过；可选综合优化报告作为优秀评级依据 |

---

## 二、课程目标和内容规划

### 2.1 课程目标（三维目标）

| 目标维度           | 达成要求                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **知识目标** | ① 理解大语言模型基本原理（文本表征、Transformer/Qwen 结构、自回归生成、Prefill/Decode 与 KV Cache）；<br />② 掌握推理性能评价指标（TTFT/TPOT/TPS）与主要优化路线（融合算子、图模式、KV Cache 优化、量化、多卡并行）的原理与适用场景；<br />③ 掌握量化基础概念（数值格式、量化位置/粒度、量化映射）与 W8A8、SmoothQuant 等主流量化方案；<br />④ 了解 MTP 投机算法、多流并行、权重预取等进阶优化技术；<br />⑤ 了解 vLLM/SGLang 主流推理服务框架的架构与核心优化机制 |
| **能力目标** | ① 能在 NPU 环境完成 Qwen3-8B 大模型部署与基线推理；② 能使用 Profiling 工具采集性能数据并定位推理瓶颈；③ 能完成 W8A8 量化部署并评估精度损失与性能收益；④ 能运用融合算子、图模式等手段优化推理性能并基于数据量化收益                                                                                                                                                                                                                                                 |
| **素养目标** | 建立「测量 → 分析 → 优化 → 验证」的数据驱动性能优化思维与系统观，具备独立查阅 CANN 官方文档、cann-learning-hub 与开源仓库资源解决问题的能力                                                                                                                                                                                                                                                                                                                         |

### 2.2 课程内容规划

| 模块                 | 讲次         | 核心内容                                                                                                                | 关键产出                                 |
| -------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 大语言模型基础       | 第 1 讲      | 文本表征（token/Embedding）与自回归生成；Transformer 与 Qwen 类结构；Prefill/Decode 与 KV Cache；TTFT/TPOT/TPS 性能指标 | Qwen3-8B 基线推理跑通 + Profiling 初体验 |
| 大语言模型调优       | 第 2 讲      | 推理优化评价目标与开销来源；融合算子与 AscendC；图模式；KV Cache 优化；多卡并行                                         | 融合算子 / 图模式优化 A/B 验证           |
| 大语言模型量化基础   | 第 3 讲      | 量化动机与基础概念；W8A8 与 SmoothQuant；PTQ vs QAT；主流量化路线对比；CANN 量化工具链                                  | Qwen3-8B W8A8 量化部署与精度/性能对比    |
| 进阶优化与服务化部署 | 第 4 讲（合并） | MTP 投机算法；多流并行；权重预取；vLLM/SGLang 框架概览；Paged/Radix Attention 与 HiCache；负载均衡与调度优化、PD 分离   | 进阶优化与服务化认知 + 综合优化实践     |
| 结业作业             | 课后线上完成 | 基线推理 + Profiling 瓶颈分析；W8A8 量化对比；综合优化（可选）                                                          | 结业优化报告                             |

### 2.3 能力提升画像

完成本课程后，学员将能够：

- **说清楚**：大语言模型推理完整流程（文本 → token → 向量 → Prefill/Decode → 生成输出）与 KV Cache、TTFT/TPOT/TPS 等核心概念，以及融合算子、图模式、KV Cache 优化、量化、多卡并行等优化原理与适用场景；
- **跑起来**：在昇腾NPU 环境中独立完成 Qwen3-8B 大模型的部署与基线推理，并使用 Profiling 工具采集与分析性能数据；
- **调得优**：运用融合算子、图模式等优化手段提升模型推理性能，并基于测量数据说明每次优化的收益；
- **量得准**：掌握 W8A8 量化的完整流程（量化配置 → 量化导出 → 部署运行 → 对比验证），能独立完成精度与性能的权衡评估；
- **看得懂**：理解 MTP、多流并行、权重预取等进阶优化技术与 vLLM/SGLang 等主流服务框架的核心机制，为后续深入学习推理服务化方向建立认知基础。

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别     | 要求                                                       |
| -------- | ---------------------------------------------------------- |
| 编程语言 | Python 基础语法（变量、循环、函数；能读懂并运行代码）      |
| 操作系统 | Windows或Linux 命令行基本操作（目录 / 文件 / 权限 / 进程） |
| 工程工具 | Git Clone 等基本操作；Jupyter Notebook 基本使用            |

### 3.2 领域前置（建议完成）

- 建议课前预习 [cann-learning-hub 快速入门](https://gitcode.com/cann/cann-learning-hub/blob/master/quick_start) 的 `cann_basics` 章节（AI 基础概念、NPU 硬件架构、CANN 软件栈）；
- 有 PyTorch 模型推理使用经验更佳（第 1~3 讲实践将直接涉及）；
- 了解神经网络 / Transformer 的基本概念更佳（第 1 讲理论直接涉及，课上将系统讲解）。

### 3.3 环境前置

- **在线环境（推荐）**：CANN Lab 云端开发环境（cann_9.0.0 py3.11-A3-arm），浏览器访问即可；
- **本地环境（可选）**：已部署 CANN 9.0.0+ 的昇腾开发环境（Atlas A3 训练/推理系列）。

---

## 四、课程教学安排（4 次课，每次理论 2h + 实践 2h）

### 4.1 总体安排

| 时段              | 讲次    | 课程主题                 | 内容要点                                                                              | 学时              | 教材 PPT（cann-learning-hub）                                                                                                                                                                                                                  |
| ----------------- | ------- | ------------------------ | ------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **第 1 次** | 第 1 讲 | 大语言模型基础           | 文本表征与自回归生成；Transformer/Qwen 结构；Prefill/Decode 与 KV Cache；推理性能指标 | 理论 2h + 实践 2h | [04_llm_deployment_and_inference_with_cann_2h.pptx](https://gitcode.com/cann/cann-learning-hub/blob/master/standard_course/01_bootcamp/two_weeks_course/04_llm_deployment_and_inference_with_cann_2h.pptx)                                        |
| **第 2 次** | 第 2 讲 | 大语言模型调优           | 优化目标与开销来源；融合算子；图模式；KV Cache 优化；多卡并行                         | 理论 2h + 实践 2h | [03_llm_training_and_inference_optimization_overview_with_cann_2h.pptx](https://gitcode.com/cann/cann-learning-hub/blob/master/standard_course/01_bootcamp/two_days_course/03_llm_training_and_inference_optimization_overview_with_cann_2h.pptx) |
| **第 3 次** | 第 3 讲 | 大语言模型量化基础       | 量化概念与映射；W8A8 与 SmoothQuant；主流量化路线；CANN 量化工具链                    | 理论 2h + 实践 2h | [02_04_llm_quantization_with_cann_2h.pptx](https://gitcode.com/cann/cann-learning-hub/blob/master/standard_course/01_bootcamp/two_weeks_course/02_04_llm_quantization_with_cann_2h.pptx)                                                          |
| **第 4 次** | 第 4 讲 | 进阶优化与服务化部署     | MTP 投机算法；多流并行；权重预取；vLLM/SGLang 概览；Paged/Radix Attention 与 HiCache；负载均衡与调度优化、PD 分离     | 理论 2h + 实践 2h | [PPT1](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/llm_inference/slides/advance/03_core_inference_optimization_mtp_multi_stream_prefetch.pptx)、[PPT2](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/llm_inference/slides/advance/05_cann_optimization_for_mainstream_inference_frameworks.pptx)（两份精编为 2h） |

> **关键节点说明：**

> - 每次课均为「理论 2h + 实践 2h」1:1 配对结构，理论课后立即实操加深理解，实践内容见第五章；
> - 第 4 次课理论为原第 4~5 讲内容精编（进阶优化 + 服务化框架概览，两份 PPT 压缩为 2h 重点讲授），实践为综合优化实践；
> - 结业作业为课后线上提交，不额外安排课时授课，详见第六章。

---

## 五、课程实践内容（CANN-Learning-Hub 承载）

> 每讲实践由 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 的跟练 Notebook 承载，与理论课 1:1 配对（各 2h），在 CANN Lab 云端环境完成。

### 5.1 实践安排总览

| 讲次      | 实践主题                        | 实践内容                                                                           | 实践路径（cann-learning-hub）                                                                       |
| --------- | ------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 第 1 讲   | 大模型基线推理与 Profiling 分析 | Qwen3-8B 部署与基线推理跑通；Profiling 数据采集，观测 TTFT/TPOT/TPS 与算子耗时分布 | `tutorials/llm_inference/qwen3_8b/02_baseline_inference.ipynb`、`03_profiling_analysis.ipynb`   |
| 第 2 讲   | 推理优化实践                    | Dense RMSNorm 融合算子优化 A/B 验证；图模式优化验证                                | `tutorials/llm_inference/qwen3_8b/04_npu_optimization.ipynb`、`07_npu_graph_optimization.ipynb` |
| 第 3 讲   | 量化实践                        | Qwen3-8B W8A8 量化全流程：读懂配置 → 量化导出 → 部署运行 → 精度/性能对比        | `tutorials/llm_inference/qwen3_8b/05_quantization_qwen3_8b.ipynb`                                 |
| 第 4 讲   | 进阶优化综合实践                | 权重预取与 GE 图优化验证；组合融合算子 / 图模式 / 量化等手段的端到端优化对比      | `tutorials/llm_inference/qwen3_8b/08_ge_o_proj_prefetch.ipynb`、`04_npu_optimization.ipynb`（组合优化） |

## 六、课程结业作业

### 6.1 作业要求

> * 实践 Notebook 运行结果（导出 HTML / PDF 或关键步骤截图，含运行环境信息）；
> * 结业报告（Markdown / Word / PDF），须包含：环境信息、关键指标数据表、优化前后对比图表、分析与结论。
>
> 结业作业为**线上提交**形式，**不额外安排课时授课**。学员于课程结束后 5 个工作日内，通过课程指定渠道（学习平台 / 邮箱）提交。

### 6.2 作业列表

| 作业             | 档位           | 内容                                                                                                      | 考核点                                                                                         |
| ---------------- | -------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **作业 1** | **必选** | **Qwen3-8B 基线推理与 Profiling 瓶颈分析简报**：跑通基线推理，采集 Profiling 数据，完成推理瓶颈分析 | 基线推理跑通；TTFT / TPOT / TPS 与 Top 耗时算子数据完整；瓶颈分析结论合理有据                  |
| **作业 2** | **必选** | **Qwen3-8B W8A8 量化部署与对比报告**：完成 W8A8 量化部署，对比量化前后精度与性能                    | 量化部署跑通；量化前后显存 / 时延 / 吞吐对比数据完整；输出量化性能对比与权衡结论               |
| **作业 3** | 可选           | **综合推理优化实践报告**：组合应用融合算子 / 图模式 / 量化等手段优化 Qwen3-8B 推理性能              | 端到端性能提升幅度 + 优化路径说明（基于 Profiling 数据的「测量 → 分析 → 优化 → 验证」环节） |

## 七、学习建议

- **第 1~2 次「听懂 + 跑通」**：第 1 次课重点建立大模型推理全流程认知，亲手跑通 Qwen3-8B 基线并完成首次 Profiling 观测；第 2 次课围绕优化路线完成融合算子与图模式的 A/B 验证，体会「用数据说话」；
- **第 3~4 次「深化 + 拓展」**：第 3 次课动手完成 W8A8 量化全流程，理解收益与代价的权衡艺术；第 4 次课纵览 MTP / 多流 / 权重预取等进阶技术与 vLLM / SGLang 服务框架，并动手完成组合优化的端到端对比，建立后续学习地图；
- **成功要素**：动手优先（每讲实践必须亲手完成）；数据驱动（每次优化前后记录指标变化）；文档查阅（养成查阅 CANN 官方文档与 cann-recipes-infer 仓的习惯）；同伴互助（邻座讨论 + 结果互评）；
- **后续方向**：结业后可深入学习推理服务化部署（vLLM / SGLang + PD 分离）、量化算法（GPTQ / AWQ / 低比特）、Ascend C 算子开发或分布式训练等方向，cann-learning-hub 与 cann-recipes-infer 提供相关进阶教程。

---

## 八、进一步学习参考

| 资源                              | 链接                                                                                                                  | 用途                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| CANN Learning Hub                 | [https://gitcode.com/cann/cann-learning-hub/](https://gitcode.com/cann/cann-learning-hub/)                               | 全栈教程与 Notebook 练习（本课程理论与实践载体） |
| CANN 推理优化实践仓               | [https://gitcode.com/cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)                               | 大模型推理部署、量化与优化工程实践               |
| CANN 算子领域样例仓库             | [https://gitcode.com/cann/cann-samples/tree/master/Samples/](https://gitcode.com/cann/cann-samples/tree/master/Samples/) | 各领域算子实现样例（对标 / 参考实现）            |
| Ascend C API 实现与样例           | [https://gitcode.com/cann/asc-devkit](https://gitcode.com/cann/asc-devkit)                                               | API 源码、API 使用示例、算子参考实现             |
| Ascend C 编程指南                 | [https://asc.gitcode.com](https://asc.gitcode.com)                                                                       | 编程模型、API 用法与最佳实践                     |
| 教材《Ascend C 异构并行程序设计》 | [https://gitcode.com/HIT1920/AscendCBook](https://gitcode.com/HIT1920/AscendCBook)                                       | 异构并行程序设计教材                             |
