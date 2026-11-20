# 《启航营（一周 · 两天启航营进阶版）》课程大纲

> 本课程为 **[两天启航营](./syllabus_two_days_bootcamp.md) 的进阶课程**：在两天营完成「生态认知 + 大模型部署推理 + 矢量算子入门 + PyTorch 接入」的基础上，进阶到大模型量化基础与优化、调试调优与矢量 / 矩阵 / 融合算子极致性能，并引入 Ascend 950 SIMD/SIMT 新同构编程，实现从「会写算子」到「写得快、调得优」的跨越。

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 启航营（一周 · 两天启航营进阶版）：CANN 算子开发进阶与极致性能 |
| **课程类型** | 短期进阶集训营（两天启航营进阶课程 / 高校夏令营进阶班 / 竞赛备赛强化） |
| **学时安排** | **5 天（10 个半天）≈ 30 小时**（理论 10h + 实践 17h，上午理论 / 下午实践与案例；另含结业作业 2~3h） |
| **授课对象** | 已完成两天启航营（或具备同等基础）的学员 |
| **先修要求** | 完成两天启航营或具备同等基础（Ascend C 矢量算子开发 + 大模型部署推理）；Python/C/C++ 基础；了解Makefile/GIT操作等 |
| **配套讲义** | [tutorials/llm_inference](../../tutorials/llm_inference/)（量化讲义 1 份 + Notebook）＋ [a2a3 原子课程](../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/)（5 份）＋ [Ascend 950 算子编程概述](../00_atomic_courses/04_ops_programming/01_ascendc/ascend950_ascend_c_operator_programming_overview.pptx)（D5 第 9 讲）＋ [ascend950 原子课程](../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/)（第 10 讲参考） |
| **实践平台** | cann-learning-hub（教程型跟练 Notebook）＋ CANNJudge（判题型自动评测）＋ 仿真环境 |
| **结业标准** | 独立完成量化矩阵算子 `quant_matmul` 开发（Qwen3 模型匹配）并通过判题，可选集成到 Qwen3 网络模型、持续优化算子与整网性能 |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 掌握大模型量化基础与优化方法（PTQ / QAT、INT8 / FP8 低比特量化）；② 掌握 Ascend C 调试调优方法论（仿真 / 板调 / msprof 性能分析）；③ 掌握矩阵算子与融合算子的编程模型与开发流程；④ 理解矢量 / 矩阵 / 融合算子的极致性能优化方法；⑤ 了解 Ascend 950 SIMD/SIMT 新同构编程模型 |
| **能力目标** | ① 能独立完成大模型量化优化实践（Qwen3-8B 量化）；② 能对矢量算子（Softmax）做极致性能优化并输出数据报告；③ 能独立开发量化矩阵算子 `quant_matmul` 并通过判题；④ 能实现融合算子（Matmul+LeakyReLU）并完成优化；⑤ 能在仿真环境完成 Ascend 950 算子编程 |
| **素养目标** | 形成「测量 → 分析 → 优化 → 验证」性能优化闭环，建立极致性能调优意识与数据驱动优化习惯 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **调得优**：运用 profiling 与仿真工具定位性能瓶颈，借鉴最佳实践逐步优化并产出数据报告
- **写得快**：独立开发矩阵算子与融合算子，完成从编码、编译到判题的全流程
- **极致化**：将 Softmax / Matmul 等算子优化至接近理论性能，掌握分块、双缓冲、流水并行等优化手段
- **跨架构**：理解 A2/A3 与 Ascend 950 的架构差异，能在仿真环境完成 950 算子编程

### 2.3 课程内容规划

| 模块 | 讲次 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| 大模型量化基础与优化 | 第 1~2 讲 | 量化基础（PTQ / QAT）；低比特量化优化；Qwen3-8B 量化实践 | 量化优化实践与精度 / 性能对比 |
| 调试调优与矢量极致性能 | 第 3~4 讲 | 调试调优方法论；Softmax 逐步实现极致性能 | Softmax 极致性能优化 + 数据报告 |
| 矩阵算子 | 第 5~6 讲 | 矩阵算子编程；Matmul 逐步实现极致性能 | Matmul 独立实现 + 优化 |
| 融合算子 | 第 7~8 讲 | 融合算子编程；Matmul+LeakyReLU 案例与优化 | 融合算子实现与优化 |
| 950 新架构 | 第 9~10 讲 | SIMD/SIMT 新同构编程概述；950 算子编程（仿真） | 950 算子编程实践 |

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 课程前置 | **完成两天启航营**或具备同等基础：Ascend C 矢量算子开发全流程、大模型部署推理流程、算子接入 PyTorch |
| 编程语言 | Python / C / C++ 基础语法（指针与内存概念；本课程算子开发以 C/C++ 为主） |
| 系统操作 | Linux 命令行操作（目录 / 文件 / 权限 / 进程 / 环境变量） |
| 工程工具 | Makefile / CMake 基本规则；GIT clone / commit / push 基本操作 |

### 3.2 领域前置（建议完成）

- 复习两天营的 Ascend C SIMD 编程模型（核函数 / 内存层级 / 同步机制）与大模型推理优化基础（量化 / 算子融合 / KV Cache）；
- 建议课前预习 [cann-learning-hub](../../quick_start) 的 `tutorials/ascendc_operator_development_light` 与 `tutorials/llm_inference/qwen3_1.7B` 章节；
- 矩阵运算基础（矩阵乘法维度规则、分块乘法思想，第 5~6 讲直接涉及）。

### 3.3 环境前置

- **在线环境（推荐）**：CANNLab 云端开发环境（cann_9.0.0 py3.11-A3-arm）；推理优化实践需 A3 系列环境；
- **本地环境（可选）**：已部署 CANN 9.0.0+ 的昇腾开发环境（Atlas A3 训练/推理系列）；
- **仿真环境**：Ascend 950 算子编程实践（第 9~10 讲）可先使用仿真运行验证代码正确性，性能数据需在真实硬件上获取。

---

## 四、教学安排与理论讲义（PPT）（5 天，上午理论 + 下午实践）

| 时段 | 讲次 | 主题 | 内容要点 | 教材PPT |
|------|------|------|---------|------|
| **D1 上午** | 第 1 讲 | 大模型量化基础与优化（理论） | 量化基础概念（PTQ / QAT）；INT8 / FP8 低比特量化；量化优化实践方法 | 2h, [PPT](../../tutorials/llm_inference/slides/advance/04_llm_quantization_optimization_and_practice.pptx) |
| **D1 下午** | 第 2 讲 | 大模型量化基础与优化（实践） | Qwen3-8B 量化实践；量化前后精度 / 性能对比；低比特量化验证 | 3h 实践, [Notebook](../../tutorials/llm_inference/qwen3_8b/05_quantization_qwen3_8b.ipynb) |
| **D2 上午** | 第 3 讲 | Ascend C A2/A3 调试调优与最佳实践（理论） | 调试工具链（仿真 / 板调）；msprof 性能分析；Roofline；常见瓶颈定位 | 2h, [PPT](../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/06_a2a3_ascend_c_operator_debugging_tuning_and_best_practices.pptx) |
| **D2 下午** | 第 4 讲 | Ascend C A2/A3 矢量算子 Softmax 如何逐步实现极致性能（理论 + 实践） | Softmax 分块 / 双缓冲 / 访存优化逐步剖析；性能数据采集与对比 | 3h（理论 + 实践）, [PPT](../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/a2a3_softmax_membase_optimization_2h.pptx) |
| **D3 上午** | 第 5 讲 | Ascend C A2/A3 矩阵算子编程（理论） | Cube 单元与 L0A/L0B/L0C；Matmul 分块 Tiling；NZ 数据布局 | 2h, [PPT](../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/04_a2a3_ascend_c_simd_matrix_operator_programming.pptx) |
| **D3 下午** | 第 6 讲 | Ascend C A2/A3 矩阵算子 Matmul 如何逐步实现极致性能（实践） | Matmul 分块 / 流水 / 访存优化逐步剖析；性能对比与判题 | 3h 实践, 讲义建设中（待补充） |
| **D4 上午** | 第 7 讲 | Ascend C A2/A3 融合算子编程（理论） | 融合算子模式；AIC/AIV 分离；Cube+Vector 协同 | 2h, [PPT](../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/05_a2a3_ascend_c_simd_fused_operator_programming.pptx) |
| **D4 下午** | 第 8 讲 | Ascend C A2/A3 融合算子（Matmul+LeakyRelu）案例与优化实践 | Matmul+LeakyReLU 融合案例剖析；访存 / 计算优化实践 | 3h 实践, [PPT](../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/a2a3_matmul_leakyrelu_fusion_optimization.pptx) |
| **D5 上午** | 第 9 讲 | Ascend C 950 SIMD/SIMT 新同构编程概述 | 950 架构升级；SIMD 与 SIMT 双执行模型；VF 粒度切换 | 2h, [PPT](../00_atomic_courses/04_ops_programming/01_ascendc/ascend950_ascend_c_operator_programming_overview.pptx) |
| **D5 下午** | 第 10 讲 | Ascend C 950 算子编程实践（基于仿真环境） | Reg 矢量算子编程；SIMT 算子；仿真运行验证 | 3h 实践, 讲义待定（参考 [ascend950](../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/)） |

> **关键节点说明：**
> - 节奏为「上午理论 2h + 下午实践 3h」，实践占比约 60%，与进阶课程「重实践、重极致性能」的定位一致；
> - **D2 / D3 / D4 下午为极致性能实践**，沿「逐步实现极致性能」主线，从 Softmax → Matmul → 融合算子层层递进，是区别于两天营的核心进阶内容；
> - **D5 为 Ascend 950 新架构日**，第 9 讲概述讲义已确定（[950 算子编程概述 PPT](../00_atomic_courses/04_ops_programming/01_ascendc/ascend950_ascend_c_operator_programming_overview.pptx)），第 10 讲仿真实践讲义与课后判题资源**待定更新**；
> - **D3 下午 Matmul 极致性能讲义建设中**，实践以 `04_matmul_basic` + `08_performance_optimization` 跟练 Notebook 与判题承载，进阶可参考 `qwen3_8b/06_custom_matmul`（自定义 Matmul 集成 Qwen3-8B），讲义上线后同步补充。

---

## 五、每节课的实践内容（CANN-Learning-Hub 承载）

> 每讲实践由 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 的跟练 Notebook 承载，下午实践为主（各 3h）；Ascend 950 实践基于仿真环境。

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 1 讲 | 量化基础认知 | 量化原理（PTQ / QAT）演示 + 量化方法梳理 | `tutorials/llm_inference/qwen3_8b` |
| 第 2 讲 | 量化优化实践 | Qwen3-8B 量化跟练 + 量化前后精度 / 性能对比 | `tutorials/llm_inference/qwen3_8b`（05_quantization_qwen3_8b） |
| 第 3 讲 | 调试调优认知 | 仿真 / 板调流程演示 + msprof 采集分析 | `tutorials/ascendc_operator_development/07_Troubleshooting` |
| 第 4 讲 | Softmax 极致性能 | Softmax 逐步优化 → 性能对比 → 数据报告 | `tutorials/ascendc_operator_development/08_performance_optimization` |
| 第 5 讲 | 矩阵算子认知 | Matmul 分块 / 布局演示 | `tutorials/ascendc_operator_development/04_matmul_basic` |
| 第 6 讲 | Matmul 极致性能 | Matmul 逐步优化 → 独立实现 → 判题（进阶：集成 Qwen3-8B） | `tutorials/ascendc_operator_development/04_matmul_basic`、`08_performance_optimization`、`tutorials/llm_inference/qwen3_8b`（06_custom_matmul） |
| 第 7 讲 | 融合算子认知 | 融合模式演示 + AIC/AIV 分离 | `tutorials/ascendc_operator_development/05_fused_operator_development` |
| 第 8 讲 | 融合算子优化 | Matmul+LeakyReLU 跟练 → 优化 → 判题 | `tutorials/ascendc_operator_development/05_fused_operator_development`（05.04_cv_fused） |
| 第 9 讲 | 950 新同构认知 | SIMD/SIMT 双执行模型演示 | 待补充（仿真环境） |
| 第 10 讲 | 950 算子编程（仿真） | Reg 矢量算子 / SIMT 算子仿真运行验证 | 待补充（仿真环境） |

> **实践目录说明：** 「实践路径」列为 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 教程仓（本仓库）`quick_start/`、`tutorials/` 下的实际教程目录，进入对应目录即可跟练 Notebook；Ascend 950 实践路径待仿真教程补充后同步。

---

## 六、课后习题与作业（CANNJudge 承载）

> 课后习题用于每讲结束后的即时巩固（在线题库 / 判题），小作业用于实践产出验收（判题型，易=跟练跑通 / 中=变体改造 / 难=独立实现+性能要求）；结业大作业为课程最终产出。**当前各讲课后习题、小作业及结业判题作业的 CANNJudge 题目链接均缺失**，待平台上线后补充。

### 6.1 每讲课后习题与小作业

> 「作业参考资源」列基于各讲内容标注具体参考来源，**并非全部位于同一仓库**：无前缀相对路径位于 CANN asc-devkit 仓 `examples/` 目录（<https://gitcode.com/cann/asc-devkit/tree/master/examples>）；`learning-hub:` 前缀路径位于 cann-learning-hub 教程仓；无参考资源的作业标 `—`。

| 讲次 | 课后习题（CANNJudge 在线题库） | 小作业（CANNJudge 判题） | 作业参考资源 | CANNJudge 链接 |
|------|------------------------------|------------------------|-------------|---------------|
| 第 1~2 讲 | 量化概念题（PTQ / QAT、低比特量化） | Qwen3-8B 量化实践 + 精度 / 性能对比简报 | learning-hub: `tutorials/llm_inference/qwen3_8b` | 待补充 |
| 第 3 讲 | 调试工具链操作题（仿真 / 板调 / msprof） | Profiling 采集 + 瓶颈定位简报 | `01_simd_cpp_api/01_utilities/04_profiling`、`01_simd_cpp_api/01_utilities/08_simulator` | 待补充 |
| 第 4 讲 | 矢量算子极致性能 | Softmax 优化前后性能对比 + 优化路径说明 | `01_simd_cpp_api/05_best_practices` | 待补充 |
| 第 5~6 讲 | Ascend C 矩阵算子开发 | Matmul 算子判题（性能达基线） | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 7~8 讲 | Ascend C 融合算子开发 | Matmul+LeakyReLU 算子判题 | `01_simd_cpp_api/00_introduction/03_fusion_operation/matmul_leakyrelu_basic_api` | 待补充 |
| 第 9~10 讲 | 950 SIMD/SIMT 编程概念题 | Reg 矢量算子仿真运行验证（仿真环境） | 待补充 | 待补充 |

### 6.2 结业大作业（三档）

> 结业参考 [两周启航营](./syllabus_two_weeks_bootcamp.md) 双算子体系，聚焦**量化矩阵算子 `quant_matmul`**（与 Qwen3 大模型匹配）：必选功能跑通 → 推荐集成 Qwen3 网络模型 → 可选持续优化算子与整网性能；与两天营「矢量单算子」、两周营「双算子 + 集成 + 优化」形成递进。

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发与 **Qwen3 大模型匹配**的量化矩阵算子 **`quant_matmul`** —— 第一部分：功能跑通 | CANNJudge 判题通过（正确性） |
| **作业 2** | **推荐** | 将 `quant_matmul` **集成到 Qwen3 网络模型**，端到端跑通验证 | 单算子调用正确性 + 整网端到端验证 |
| **作业 3** | 可选 | **持续优化** `quant_matmul` 算子和整网性能 —— 第二部分：持续优化性能 | 性能提升幅度 + 优化路径剖析证据（profiling 数据 + 改动说明） |

---

## 七、学习建议

### 7.1 一周学习节奏

- **D1「量化基础与优化」**：从量化原理（PTQ / QAT）出发，完成 Qwen3-8B 量化实践，掌握大模型低比特量化的完整方法，并通过精度 / 性能对比建立量化取舍直觉
- **D2「调优 + 矢量极致性能」**：掌握调试调优工具链，随后以 Softmax 为切入，体验「逐步实现极致性能」的完整优化过程
- **D3–D4「矩阵 + 融合」**：从矩阵算子编程到融合算子优化，是算力利用率提升的核心进阶内容，D3/D4 下午的极致性能实践是关键
- **D5「950 新架构」**：认识 Ascend 950 的 SIMD/SIMT 新同构，通过仿真完成算子编程，为后续 950 真板开发做准备

### 7.2 成功要素

1. **极致性能实践是核心**：D2/D3/D4 下午的「逐步实现极致性能」实践，务必独立完成优化并记录每次改动的性能变化
2. **性能优化要有数据支撑**：所有优化都要基于 profiling 数据，形成「测量 → 分析 → 优化 → 验证」的闭环
3. **与两天营形成对照**：将本课程的矩阵 / 融合算子与两天营的矢量算子对照，理解不同算子类型的优化思路差异
4. **跨架构视角**：D5 的 950 新架构学习重在理解「为什么变」（架构动因），而非死记 API
5. **仿真与真板结合**：950 实践先用仿真验证正确性，性能结论需真板数据支撑

### 7.3 后续学习路径

完成一周进阶启航营后，可根据兴趣选择深入方向：

| 方向 | 推荐路径 | 目标 |
|------|---------|------|
| **系统进阶** | 两周启航营（双算子 + 框架集成）→ 四周启航营（训推全链路 + 全类型算子 + 社区实践） | 系统掌握算子开发与优化全流程 |
| **算子极致性能** | Ascend C 高级原子课程（SIMD&SIMT 混合编程）→ Aclnn 工程化开发 | 独立开发达到理论峰值 90%+ 的高性能算子 |
| **框架深度开发** | TorchNPU 深度实践 → TorchAir / AclGraph → 图融合 PASS 开发 | 深度参与 AI 框架层开发与优化 |
| **大模型系统优化** | 大模型推理优化高级 → KV Cache / 量化 / 推理服务 → 分布式训练 | 成为大模型系统级性能优化专家 |

---

## 八、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现 |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（对标 / 参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（本课程实践作业载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |
