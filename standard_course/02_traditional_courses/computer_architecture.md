# 《计算机体系结构》课程大纲

> 本文件为计算机体系结构课**单课程落地文件**（传统课程清单·课程一）；总规划见本目录 [README.md](./README.md)。
> 课程安排对齐参考条目（`traditional_courses.md`·课程一）：**三部分结构**——第一部分 CPU 传统体系结构介绍 + 第二部分"AI 时代的体系结构挑战"**9 课时（每次 3 课时 × 3 次）** + 第三部分结业作业；理论讲义复用 [05_npu_architecture](../00_atomic_courses/04_ops_programming/05_npu_architecture/) 讲义 PPT，课后作业对齐第 4 讲《小作业 & 大作业》课件——小作业 8 道问答题 + 大作业实践题（Qwen3-1.7B 部署跑通必选、`add_rms_norm` 三档可选加分，CANNJudge 承载）。
> **讲义就绪**：三份讲义 PPT 已完成（94/97/105 页，含 15 个内嵌互动页，华为课程模板版）；原稿（清华模板版）存于 `ref/` 供溯源。

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 计算机体系结构（AI 时代的体系结构挑战融入版） |
| **课程类型** | 传统课程 · 本科高年级 / 研究生学位课 |
| **学时安排** | 16 周学期扣减后 **13 次大课 × 135 分钟**（45'+5'+45'+5'+35' 三段式）：第一部分 CPU 传统体系结构（课次 1–8、12–13）＋ 第二部分 AI 时代的体系结构挑战 **9 课时**（课次 9–11，每次 3 课时）＋ 第三部分结业作业 |
| **授课对象** | 本科高年级 / 研究生，已修计算机组成原理 |
| **先修要求** | 计算机组成原理（流水线 / 存储层次 / 数据通路）；C 语言基础；矩阵乘法 |
| **配套讲义** | [05_npu_architecture/](../00_atomic_courses/04_ops_programming/05_npu_architecture/) 下对应 PPT：[1_ai_era_challenges_to_computer_architecture](../00_atomic_courses/04_ops_programming/05_npu_architecture/1_ai_era_challenges_to_computer_architecture_V1.0.pptx)（94 页）、[2_npu_architecture_design_for_large_model_workloads](../00_atomic_courses/04_ops_programming/05_npu_architecture/2_npu_architecture_design_for_large_model_workloads_V1.0.pptx)（97 页）、[3_npu_performance_optimization_and_supernode_expansion](../00_atomic_courses/04_ops_programming/05_npu_architecture/3_npu_performance_optimization_and_supernode_expansion_V1.0.pptx)（105 页） |
| **实践平台** | 暂无专门实践（Learning-Hub 暂缺）；课堂互动由讲义内嵌 15 个小测验 / 小讨论页承载 |
| **课后作业平台** | CANNJudge ＋ CANN-LearningHub（**对齐第 4 讲《小作业 & 大作业》课件**：小作业问答题 + 大作业实践题，见第五章） |
| **结业标准** | 三张课堂产出物（《甲方需求清单》《乙方答辩表》《终章设计清单》）＋ 小作业（问答题 8 题）＋ 大作业（**Qwen3-1.7B 部署跑通必选**、`add_rms_norm` 三档可选加分）＋ 期末考试（含 CANN 综合题 5~10 分） |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 掌握 CPU 传统体系结构核心（处理器 / 存储层次 / 互联 / 评估方法）；② 理解大模型负载的计算 / 存储 / 访存特征与量化方法（Roofline）；③ 掌握达芬奇架构四维度设计逻辑（三核异构 / 软件管理存储 / 数据搬运 / 多级编程体系）；④ 掌握性能优化四维方法论（任务编排 / 流水并行 / 访存 / 计算）与超节点演进 |
| **能力目标** | ① 能用 Roofline 对给定负载做 bound 判定与定量分析；② 能按负载特征对 CPU / GPU / NPU 三类架构做适配性比较与选择；③ 能基于剖析报告完成"诊断 → 药方 → 预后"的结构化性能分析（病历诊断法）；④ 能读懂 Ascend C Tiling 代码中的存储层次与硬件单元映射 |
| **素养目标** | 建立甲乙方工程闭环思维（需求 → 方案 → 验收 → 调优）；理解"隐式硬件管理 vs 显式软件管理"两种体系结构哲学；具备证据卡式辩论与结构化表达习惯 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **对得上**：把大模型负载的每条诉求映射到 NPU 架构决策（算子族 → 三核、KV Cache → 软件管理存储、访存墙 → 数据搬运体系）；
- **算得出**：Roofline 手算判 bound、Tiling 搬运量估算、环形 AllReduce 通信量推导；
- **辩得赢**：用证据卡与裁决框架完成"SIMT 是否终态"类开放架构辩论；
- **诊得准**：拿到一份剖析报告（如 Matmul 950 案例 10742us → 2800us，+283.6%），能开出"瓶颈定位 + 优化路径 + 预期收益"的三段式诊断。

### 2.3 课程内容规划（三部分）

| 部分 | 课次 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| **第一部分：CPU 传统体系结构介绍** | 1–8、12–13 | Preliminaries（Intro + 指标）；Processor（In-order / SMT / O3·Superscalar）；Memory（Cache / VM·TLB / RAID·Cloud·Cache-Free）；Interconnect（NoC / CXL / RDMA / PCIe / HCCS）；Misc（Security + 其他领域） | 传统主线作业与考试 |
| **第二部分：AI 时代的体系结构挑战**（9 课时） | **9–11** | **2.1 AI 时代大模型对体系结构的挑战**（3 课时，需求评审）→ **2.2 面向大模型负载的 NPU 架构设计**（3 课时，方案评审）→ **2.3 NPU 架构的性能优化与超节点扩展**（3 课时，验收调优） | 三张产出物（见 5.2） |
| **第三部分：结业作业** | 期末 | 小作业（问答 8 题）＋ 大作业（Qwen3-1.7B 部署必选 + `add_rms_norm` 三档加分）＋ 期末综合论述 | 结业成绩 |

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 体系结构基础 | 计算机组成原理：五级流水线与冒险、存储层次（寄存器/Cache/主存）、DMA 概念 |
| 编程语言 | C 语言基本语法（能读指针与内存布局即可，课后算子作业需 C/C++ 基础） |
| 数学基础 | 矩阵乘法与分块思想 |

### 3.2 领域前置（无硬性要求）

- 无 AI / 大模型背景要求——第二部分 2.1 讲义 Part 1 从 ChatGPT 到 DeepSeek 完整讲起；
- 建议课次 9 前预习 [cann-learning-hub 快速入门](../../quick_start) 的 `cann_basics` 章节（`02_what_is_npu`、`03_what_is_cann`，约 30 分钟），建立 NPU/CANN 基本词汇；
- 结业算子作业需 Ascend C 基础——课后习题已按"跟练 → 独立"梯度设计（复用启航营两天素材）。

### 3.3 环境前置

- 课堂讲授 + 互动为主，无上机硬性要求；
- 完成 CANNJudge 课后作业需：CANNLab 云端开发环境（浏览器访问，推荐）或本地已部署 CANN 8.5.0+ 的昇腾开发环境（Atlas A2/A3 系列）。

---

## 四、教学安排与理论讲义（PPT）（13 次大课）

| 课次 | 模块（部分） | 主题 | 内容要点 | 教材 PPT |
|------|------|------|---------|---------|
| 1 | 第一部分 | Intro + Evaluation metrics | 什么是体系结构；性能指标（CPI/MIPS/功耗，○ TOPS/W 一瞥） | 沿用课程现有讲义 |
| 2–4 | 第一部分 | Processor：In-order / Multithreading / O3·Superscalar | 五级流水回顾、SMT、乱序与超标量（○ 埋"NPU 为什么不需要 O3"伏笔） | 沿用课程现有讲义 |
| 5–8 | 第一部分 | Memory：Cache×2 / VM·TLB / RAID·Cloud | VIPT、替换策略、prefetcher/predictor、虚存、RAID、Cloud Storage、(Cache-Free)（◐ 课次 6 预取 vs Tiling 对照） | 沿用课程现有讲义 |
| **9** | **第二部分 2.1** | **AI 时代大模型对体系结构的挑战**（需求评审会） | 大模型负载画像 + Roofline + CPU→GPU SIMT + 结构化辩论 | 3 课时, [1_ai_era_challenges_to_computer_architecture](../00_atomic_courses/04_ops_programming/05_npu_architecture/1_ai_era_challenges_to_computer_architecture_V1.0.pptx)（94 页 / 6 Part） |
| **10** | **第二部分 2.2** | **面向大模型负载的 NPU 架构设计**（方案评审会） | 三核异构 + 软件管理存储 + 数据搬运 + 编程体系 + 乙方答辩 | 3 课时, [2_npu_architecture_design_for_large_model_workloads](../00_atomic_courses/04_ops_programming/05_npu_architecture/2_npu_architecture_design_for_large_model_workloads_V1.0.pptx)（97 页 / 5 Part） |
| **11** | **第二部分 2.3** | **NPU 架构的性能优化与超节点扩展**（验收调优会） | 优化四维方法论 + Matmul 950 病历实战 + 超节点演进 | 3 课时, [3_npu_performance_optimization_and_supernode_expansion](../00_atomic_courses/04_ops_programming/05_npu_architecture/3_npu_performance_optimization_and_supernode_expansion_V1.0.pptx)（105 页 / 7 Part） |
| 12 | 第一部分 | Interconnect：NoC / CXL / RDMA / PCIe / HCCS | 互联谱系与案例（收割课次 11 超节点框架） | 沿用课程现有讲义（◐） |
| 13 | 第一部分 | Misc：Security + Other fields + 总结 | 安全、Cloud/quantum/unary computing + 全课收束 | 沿用课程现有讲义（○） |

> **关键节点说明：**
> - 课次 1–8、12–13 为传统主体（10 次大课，沿用现有讲义），CANN 以 ○ 提及 / ◐ 对照轻量融入——每次亮相都先给体系结构问题、再给 CANN 应答；
> - 课次 9–11 为三次主讲课（各 3 课时），讲义即 `05_npu_architecture` 三份 PPT 一一对应；课次 11 收束的超节点框架在课次 12 Interconnect 收割；
> - 三次课产出物当堂完成——后一次课的输入是前一次课的产出（见 5.2）。

---

## 五、每节课的实践内容（课堂互动由讲义内嵌互动页承载）

> 本课无独立上机环节，实践由三部分构成：**课前预习**（quick_start Notebook）＋ **主讲课课堂互动**（讲义内嵌 15 个小测验 / 小讨论页）＋ **结业作业实践**（CANNJudge 承载，见第六章）。

### 5.1 课前预习（课次 9 前）

| 环节 | 实践内容 | 实践路径 |
|------|---------|---------|
| 课前预习 | NPU/CANN 基本词汇（硬件架构、软件栈，约 30 分钟） | [quick_start/cann_basics](../../quick_start/cann_basics/)（`02_what_is_npu`、`03_what_is_cann`） |

### 5.2 课堂互动（课次 9–11，随讲义内嵌互动页）

| 课次 | 互动形式 | 实践内容 | 产出物 |
|------|---------|---------|--------|
| 9 | 拆机举手 + 结构化辩论 | A100 拆机认知冲突（开场必答）；"SIMT 是不是终点"正反方证据卡辩论 | 《甲方需求清单》 |
| 10 | 乙方答辩一分钟纸 | 需求-方案逐条应答，各维度一句"最大风险与代价" | 《乙方答辩表》 |
| 11 | 病历诊断 | Matmul 950 案例（10742us → 2800us）"诊断 → 药方 → 预后"三段式小组诊断 | 《终章设计清单》 |

> 三张产出物形成前后衔接的对照链（需求 → 方案 → 调优），保留至结业，作为平时成绩依据；结业算子作业需 Ascend C 基础——课后习题已按"跟练 → 独立"梯度设计（复用启航营两天素材）。

---

## 六、课后习题与作业（CANNJudge 承载）

> 作业体系对齐三次主讲课配套的第 4 讲《小作业 & 大作业》课件（`4_4 大小作业.pptx`）：**小作业**为 8 道问答题（对应三讲内容，随课布置）；**大作业**为实践题——必选档为 **Qwen3-1.7B 部署跑通**，可选加分档构成 `add_rms_norm`"功能跑通 → 性能优化 → 接入模型"进阶链（CANNJudge 承载，与启航营两天课程作业 2–4 同源）。

### 6.1 小作业（问答题 · 8 题，对应三讲）

| # | 题目要点 | 对应讲义（课次） |
|---|---------|----------------|
| 1 | **架构与算子特征**：以 DeepSeek V4.1 长文本 Prefill 为例，列出三个核心算子及其张量形状，并结合算术强度（Arithmetic Intensity）区分其计算 / 访存密集型特征 | 讲义 1（课次 9） |
| 2 | **SIMT 局限与架构演进**：分析 SIMT 在大模型动态负载下的局限性；梳理 NVIDIA Tensor Core 从 Volta 到 Hopper 的演进，说明 TMA 等新特性对传统 SIMT 编程范式的挑战及应对 | 讲义 1（课次 9） |
| 3 | **FlashAttention 软硬协同**：梳理 v1 至 v3 的核心演进；从体系结构角度剖析各版本改动（如 IO 切块、异步流水）的硬件动因，论述对其他访存密集型算子优化的借鉴意义 | 讲义 1（课次 9） |
| 4 | **异构计算交互**：以向量 Add 算子为例，描述 CPU＋NPU/GPU 异构计算中任务下发的完整交互时序；明确 Host 与 Device 职责边界，指出性能瓶颈（如 Launch 开销）及优化方向 | 讲义 1/2（课次 9–10） |
| 5 | **SIMD 与 SIMT 编程对比**：基于 Add 算子，对比 SIMD 与 SIMT 编程模型的差异 | 讲义 1/2（课次 9–10） |
| 6 | **AI Core 架构演进分析**：以 Add＋Gather（非连续访存）算子为例，对比 Ascend A2/A3 与 Ascend 950 的 AI Core 架构变化；从大模型动态负载特征论述引入 SIMT 的必要性，阐明"SIMD 为主、SIMT 为辅"混合方案的设计优势 | 讲义 2（课次 10） |
| 7 | **算子性能优化策略**：以 NPU 上的 FlashAttention 算子为例，从数据分块规划、多级流水并行（Pipeline）、显式访存优化、计算指令融合四个维度，系统梳理 AI Core 算子性能优化的常见方法与实操策略 | 讲义 3（课次 11） |
| 8 | **芯片软硬协同设计（论述）**：论述如何设计一款兼顾大模型极致性能诉求与良好可编程性的 AI 芯片——大模型核心计算诉求剖析、性能与易用性的矛盾分析、软硬协同设计案例及未来架构演进方向 | 三讲综合 |

### 6.2 大作业（实践题 · CANNJudge 承载）

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| 作业 1 · 部署跑通 Qwen3-1.7B | **必选** | 部署跑通 qwen3-1.7b 模型，可参考 [tutorials/llm_inference/qwen3_1.7B](../../tutorials/llm_inference/qwen3_1.7B/)（章节导学 → 基线推理 → 剖析分析 → NPU 优化 Notebook） | 模型部署端到端跑通，推理结果正确 |
| 作业 2 · `add_rms_norm` 功能跑通 | 可选加分 | 用基础 API 或 C API 编写 `add_rms_norm`（shape 与 Qwen3-1.7B 模型匹配），功能跑通；判题链接：<https://cannjudge.cn/public/vector/addrmsnormqwen3> | CANNJudge 判题通过（正确性） |
| 作业 3 · `add_rms_norm` 性能优化 | 可选加分 | 分析当前算子的性能瓶颈点，结合 Atlas A2/A3 架构或 Ascend 950 架构**列出每个硬件的流水利用率**，给出优化方案，并提供优化后的算子性能数据与算子代码 | 瓶颈定位准确性 + 流水利用率证据 + 优化前后性能对比 |
| 作业 4 · `add_rms_norm` 接入模型 | 可选加分 | 将新写的 `add_rms_norm` 接入 Qwen3-1.7B 模型，对比与原来模型推理性能差异 | 集成端到端跑通 + 性能对比数据与分析 |

> 大作业可自学 [CANN-LearningHub](https://gitcode.com/cann/cann-learning-hub/) 与 CANNJudge 完成（入口 <https://gitcode.com/cann>）；可选三档构成"功能跑通 → 性能优化 → 接入模型"进阶链。

### 6.3 期末考试 CANN 综合题（必选）

期末综合类题目 5~10 分：概念辨析（SIMT vs SIMD vs Cube / 自动挡 vs 手动挡）、定量计算（AI 值判 bound / Tiling 搬运量 / AllReduce 通信量）、代码分析（读不写）、**病历诊断题**（素材即 4_3 Part 6 真实案例）、对比论述（两种管理哲学）、终章综合论述（超节点与 AMAT 扩展）——考查体系结构回归与 CANN 知识整合。

---

## 七、学习建议

- **第一部分（课次 1–8、12–13）「筑基」**：Cache/流水线/互联知识是第二部分的"回归锚点"——每次 CANN 亮相都先给体系结构问题、再给 CANN 应答，建议带着"NPU 会怎么回答"的预期学传统内容；
- **第二部分三次主讲课（课次 9–11）「递进」**：三次课层层深入（需求评审 → 方案应答 → 性能调优），产出物当堂完成，勿欠账——后一次课的输入是前一次课的产出；
- **第三部分结业「收口」**：必选大作业从部署跑通 Qwen3-1.7B 起步（[tutorials/llm_inference](../../tutorials/llm_inference/) 教程跟练即可），可选档沿"功能跑通 → 性能优化 → 接入模型"进阶链挑战——无算子开发基础者按时完成必选档即可；
- **成功要素**：敢于竞猜（数量级错了不扣分，参与即得分）；证据优先（辩论与诊断都要求"有数据、有出处"）；回归意识（每个 CANN 概念都要能翻译回一条体系结构原理）；
- **学有余力**：完成可选算子作业后，可衔接启航营完整体系（[01_bootcamp](../01_bootcamp/README.md)：两天 / 两周 / 四周营）进入 Ascend C 算子开发实战。

---

## 八、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现（课后算子作业素材） |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（对标/参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（课前预习载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |
