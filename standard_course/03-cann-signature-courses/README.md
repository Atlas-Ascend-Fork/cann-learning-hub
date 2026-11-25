# CANN 品牌课程体系

> 基于 CANN 可教学知识地图（L1 算子编程/NPU 架构 → L2 加速库 → L3 AI 框架 → L4 AI 应用），将 CANN/昇腾技术体系外挂式融入本科四年 + 研究生培养全链路。课程名与学分不动，CANN 以单元植入 / 实验替换 / 算子案例三种模式进入——"换芯不换课"。

---

## 一、品牌课程体系总表

| 年级 | 课程名称 | 理论 | 实践 | 知识地图层级 | 涉及的 CANN 技术栈与深度 |
|------|---------|:---:|:---:|:---:|----------------------|
| **大一** | 人工智能基本原理与工程实践 I（智能导论） | 32 | 0 | L4→L3→L1 认知 | **L4 AI 应用**：企业应用案例、深度学习（初级）· **L3 AI 框架**：PyTorch 框架基础（初级）· **L1 NPU 架构**：NPU 芯片概览（初级） |
| **大二** | 人工智能基本原理与工程实践 II（机器学习、深度学习、强化学习） | 48 | 0 | L3→L2 | **L3 AI 框架**：PyTorch 训练与推理（中级）· **L2 加速库**：Aclnn 算子库（ops-nn / ops-math）、Thrust/C++ 并行算法库、通信库 HCCL（中级）· **L1 算子编程**：Ascend C / PyPTO（初级） |
| **大二** | 人工智能基本原理与工程实践 III（形式语言、自然语言、语音技术） | 80 | 0 | L2→L3 | **L2 加速库**：领域加速库 ops-transformer（中级）· Aclnn 算子库（Embedding/Attention/FFT）（中级）· **L3 AI 框架**：模型推理（中级）· **L1 算子编程**：Ascend C / PyPTO（初级） |
| **大三** | 人工智能基本原理与工程实践 IV（视觉与生成式 AI、大模型与开源、数据工程） | 80 | 0 | L2→L1 | **L1 算子编程**：Ascend C / PyPTO（中级）· 通算融合算子编程（中高级）· **L2 加速库**：ops-transformer（中级）· **图引擎 GE**（中高级）· **L3 AI 框架**：PyTorch 训推含量化（中高级）· 模型部署与优化案例（中高级） |
| **大三** | 人工智能基本原理与工程实践 V（系统与实践） | 0 | 48 | L1→社区 | **L1 算子编程**：Ascend C / PyPTO / CANNBot DSL / Triton（中级）· **AI 训练和推理优化**（中高级）· **CANN 社区任务**（算子交付 + Skill 反哺 + Issue→PR） |
| **大三** | 并行计算与算子编程 | 48 | 0 | L1 | **L1 算子编程**：Ascend C（中高级）· PyPTO（中高级）· PyAsc · CANNBot DSL · **L1 NPU 架构**：多级内存 / SIMD/SIMT · **算子最佳案例** · **CANN 社区任务** |
| **大三** | 人工智能芯片设计 | 48 | 0 | L1 | **L1 NPU 架构**（中高级）——达芬奇架构、SIMD/SIMT 新同构、多级存储、片上互联 |
| **大三** | 人工智能芯片与系统（AI Chips and Systems）🆕 | 20 | 20 | L1→L2→L3 | **L1 NPU 架构**（高级）：达芬奇架构、SIMD/SIMT、多级存储、灵衢总线、超节点 · **L1 算子编程**：Ascend C SIMD/SIMT/融合/通信编程（中高级→高级）· **L2 加速库**：HCCL、GE 图引擎（中级）· **L3 AI 框架**：算子入图/图模式（中级）· **软芯协同**：CANN 四层垂直栈 |
| **大三** | 昇腾 AI 与复杂系统数据智能 ✅ 已排课 | 32 | 0 | L3→L4 | **L3 AI 框架**：torch_npu 大模型训推、MindIE 部署推理（中高级）· **L2 加速库**：ATC 离线编译、入图优化、PD 分离 & KV 池化（中高级）· **L1 算子编程**：Ascend C 算子编程实践（中级） |
| **研究生** | AI 计算与神经网络计算架构实践 ✅ 已排课 | 28 | 36 | L1→L2→L3 全栈 | **L3 AI 框架**：大模型训推全链路（预训练/SFT/RL→Qwen3 部署）（高级）· **L2 加速库**：入图/PD 分离/KV 池化/自动融合（高级）· **L1 算子编程**：Ascend C 双路线全体系（A2/A3 + 950）+ PyPTO + CANN Bot（中高级）· **社区**：开源贡献（PR 合入） |

> 完整课程大纲：[cann_brand_curriculum_plan.md](./cann_brand_curriculum_plan.md)

---

## 二、知识地图层级覆盖矩阵

| 知识地图层级 | 大一 | 大二 | 大三 | 研究生 |
|-------------|:---:|:---:|:---:|:---:|
| **L4 AI 应用** | **初级**（I 课：企业案例、深度学习、生成式 AI、科学计算等场景认知） | **中级**（II 课：深度学习/ML/DL/RL 实践；III 课：NLP 语音应用实践） | **中高级**（IV 课：视觉与生成式 AI、大模型与开源、数据工程综合实践；昇腾 AI 与复杂系统数据智能：行业综合实践；V 课：社区任务真实场景） | **高级**（AI 计算与神经网络架构实践：大模型训推端到端） |
| **L3 AI 框架**（PyTorch/训练/推理/部署优化） | **初级**（I 课：PyTorch 框架基础，.npu() 体验） | **中级**（II 课：torch_npu 训练；III 课：模型推理） | **中高级**（IV 课：量化部署+MindIE 推理；昇腾 AI 与复杂系统数据智能：大模型训推+部署；V 课：训练与推理优化实战） | **高级**（AI 计算与神经网络架构实践：预训练/SFT/RL→Qwen3 部署全链路） |
| **L2 加速库**（Aclnn/HCCL/Thrust/Python 加速库/领域加速库/图引擎 GE） | — | **中级**（II 课：ops-nn/ops-math/HCCL/Thrust；III 课：ops-transformer） | **中高级**（IV 课：GE 图引擎；昇腾 AI 与复杂系统数据智能：ATC 离线编译+入图优化+PD 分离；V 课：HCCL 多卡 + GE 入图） | **高级**（AI 计算与神经网络架构实践：入图/PD 分离/KV 池化/自动融合） |
| **L1 算子编程**（Ascend C/PyPTO/PyAsc/CANNBot DSL/Triton/MathDx/Ascend C 通信编程） | — | 初级（II/III 课：Ascend C 入门） | **中级→中高级**（V 课：Ascend C/PyPTO/CANNBot DSL/Triton 四前端实战；并行计算课：Ascend C 中高级；昇腾 AI 与复杂系统数据智能：矢量算子实践） | **高级**（AI 计算与神经网络架构实践：A2/A3+950 双路线全体系+社区 PR） |
| **L1 NPU 架构**（达芬奇/多级存储/SIMD/SIMT/片上互联） | 初级（I 课：架构一瞥） | 初级（II 课：NPU 芯片基础） | **中高级**（芯片设计课：达芬奇架构深度解析 + Ascend 950 新架构） | 中高级（AI 计算与神经网络架构实践：双路线架构对比） |
| CANN 社区任务 | — | — | ✓（V 课 + 并行计算课） | ✓（AI 计算与神经网络架构实践：开源贡献 PR） |

---

## 三、课程体系衔接

### 3.1 与品牌课程规划的衔接

已排课的两门课程（AI 计算与神经网络计算架构实践 / 昇腾 AI 与复杂系统数据智能）对应品牌课程体系中的大三阶段——是"并行计算与算子编程"+"人工智能芯片设计"课程的实训出口。修完这两门课程的学生可直接进入研究生深化方向。

### 3.2 与启航营的衔接

| 启航营 | 结业算子 | 对应课程 | 衔接点 |
|--------|---------|---------|--------|
| 一天营 | add_rms_norm（矢量单算子，功能跑通） | 课程一（轻量体验） | 课程浓缩体验版 |
| 两天营 | add_rms_norm（矢量单算子） | 课程一第三部分 | 算子集成与端到端优化起点 |
| 两周营 | add_rms_norm + quant-matmul（双算子）+ PyTorch 集成 | 课程一第二、三部分 | Ascend C 算子编程与集成进阶 |
| 四周营 | add_rms_norm + 整网集成 + 性能优化 + 社区任务 | 课程一全程 | 训推全链路 + 算子编程 + 集成优化集训形态 |

---

## 四、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| CANN 社区主站 | <https://gitcode.com/cann> | CANN 开源社区入口 |
| CANN 社区任务 | <https://gitcode.com/org/cann/discussions/211> | 算子任务认领 |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook |
| Ascend C API 与样例 | <https://gitcode.com/cann/asc-devkit> | 算子开发参考实现 |
| CANN Bot | <https://gitcode.com/cann/cannbot> | AI 辅助算子开发 |
| cannbot-skills | <https://gitcode.com/cann/cannbot-skills> | Skill 贡献目标仓 |
| cann-bench 评测 | <https://gitcode.com/cann/cann-bench> | 算子性能排行榜 |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 系统化教材 |
| CANNBot Tutorial | `tutorials/CANNBot/README.md` | 三阶段自学课程 |
