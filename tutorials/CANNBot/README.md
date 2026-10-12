# CANNBot 算子开发课程

本系列教程教授开发者使用 CANNBot（面向昇腾 NPU 开发的 AI Agent 团队）完成算子开发：从交付第一个算子，到理解工作流机制，再到独立指挥完整版工作流实战并反哺生态；算子代码由 AI 团队编写与修复，开发者聚焦指挥与决策。面授场景可按需取用为半天课（阶段一~二精讲）或全天基础课（阶段一~二扩展实操），本 README 以自学路径为主线。

- **阶段一 · 零基础**（`01_basics/`，约 1h）：认识编程 Agent，用 Flash 轻量版交付第一个算子 add_custom。
- **阶段二 · 中级**（`02_core_workflows/`，核心约 1.5h + 专题选学）：理解完整版工作流机制，一个 xlog1py 算子贯穿开发、调优、上库全链路。
- **阶段三 · 高级**（`03_advanced_path/`，约 6.5h 自学）：指挥完整版工作流交付 gelu_approx（8 步计算链），走完生成、调试、优化、反哺 Skill 的闭环。

## 前置条件

- 无须具备 Agent 或算子开发经验，阶段一从零讲起。
- 建议了解 Python 基础；希望系统学习 Ascend C 手动开发者，可结合 [Ascend C 算子开发系列教程（Kernel 直调）](../ascendc_operator_development_light/README.md)。
- 阶段三需先完成阶段一~二（或同等经验），并另有 **Node.js ≥ 20.11 与 tmux** 要求（完整版 harness 依赖）。

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2 / A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上（课程实测基于 CANN 9.1） |
| CLI 工具 | OpenCode / Claude Code / Codex / Trae 任选其一（课程以 OpenCode 为例）；Linux / macOS，Windows 可经 WSL2 使用（[原生安装脚本](https://gitcode.com/cann/cannbot-skills/blob/master/install.ps1)） |
| Python | 3.11 |
| Node.js / tmux | 阶段三需要（Node.js ≥ 20.11） |

## 在线体验环境

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| cann-learning-hub 在线体验 notebook | cann_9.0.0_py3.11-A2-arm | Python 3.11.15 | Notebook 可直接打开运行（[示例：01.01](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/CANNBot/01_basics&scanFilePath=tutorials/CANNBot/01_basics/01.01_cannbot_basics.ipynb)） |
| CANNLab 云开发环境 | cann_9.0.0_py3.11-A2-arm | Python 3.11 | 参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md) 创建环境运行 notebook |

> **注意：** 本地环境需自行安装配套的 CANN 软件与 CLI 工具，参考 [CANN 安装指南](https://www.hiascend.com/document)。

## 教程结构

### 阶段一：认识 CANNBot，交付第一个算子（`01_basics/`）

| Notebook | 内容 |
|---|---|
| [01.01 初识 CANNBot](./01_basics/01.01_cannbot_basics.ipynb) | 编程 Agent 与聊天 AI 的区别；三层架构与工作流；六仓生态 |
| [01.02 第一个算子](./01_basics/01.02_flash_workflow.ipynb) | Flash 版四步安装；需求五要素；add_custom 实操与精度验收 |
| [01.03 玩转与共建](./01_basics/01.03_capabilities_and_community.ipynb) | 调试与性能入口；五条开发路径；cann-bench 评测；贡献四入口 |

### 阶段二：理解机制，深入路径（`02_core_workflows/`）

| Notebook | 内容 |
|---|---|
| [02.01 完整版工作流](./02_core_workflows/02.01_complete_workflow.ipynb) | 两套工作流选型与三大设计思想；xlog1py 3.6 小时实录精编 |
| [02.02 性能调优](./02_core_workflows/02.02_performance_optimization.ipynb) | 调优三原则与行动模板；16 用例实录；需求演进与停止条件 |
| [02.03 协作上库](./02_core_workflows/02.03_gitcode_collaboration.ipynb) | PR 八步与五条红线；AI 边界框架；课程总结 |

### 阶段三：进阶自学实战（`03_advanced_path/`）

主线算子 gelu_approx（LLM FFN 核心激活，8 步计算链）。本阶段以指挥为主：算子代码由 AI 团队编写与修复，开发者负责指挥团队、判读结果与决策。实操素材统一在 `src/`（gelu 缺陷注入工程、性能采集脚本与真机基线、Skill 增补样例、需求卡）；环境自检与已知环境坑见 03.01 开篇。

| Notebook | 内容 |
|---|---|
| [03.01 衔接与生成](./03_advanced_path/03.01_operator_generation.ipynb) | 三算子对照；精度预算；Lab A gelu 交付 |
| [03.02 多维度调试](./03_advanced_path/03.02_operator_debug.ipynb) | 症状分诊 × 根因分层；三坑修复 |
| [03.03 性能工程](./03_advanced_path/03.03_performance_engineering.ipynb) | 计算密度；四步流水分析法；优化闭环 |
| [03.04 反哺与结业](./03_advanced_path/03.04_skill_and_assessment.ipynb) | Skill 工程；四件产物验收；能力地图 |

## 课件（`slides/`）

| 序号 | 主题 | 课件 |
|---|---|---|
| 01 | CANNBot 入门宣讲（面授 2~2.5h，含 abs 算子实操） | [01_cannbot_start.pdf](./slides/01_cannbot_start.pdf) |
| 02 | Ascend C 算子开发实操（四角色协同、七步流水线、Softmax 实操） | [02_ascend_c_operator.pdf](./slides/02_ascend_c_operator.pdf) |
| 03 | PyPTO 算子开发实操（编排器 + 8 子 Agent、调优 5.9× 实录） | [03_pypto_operator.pdf](./slides/03_pypto_operator.pdf) |
| 04 | TileLang-Ascend 算子开发（DSL、方案设计/代码生成/代码 review） | [04_tilelang_operator.pdf](./slides/04_tilelang_operator.pdf) |
| 05 | Vector 算子自动生成之 RegBase（910→950 代际迁移、GELU 实操） | [05_vector_regbase.pdf](./slides/05_vector_regbase.pdf) |
| 06 | Vector 算子排序性能优化（msProf 瓶颈分析、v1→v8 优化史） | [06_vector_sort_opt.pdf](./slides/06_vector_sort_opt.pdf) |
| 07 | SIMT 单指令多线程算子（Warp 调度、四层拦截交付体系） | [07_multi_thread_operator.pdf](./slides/07_multi_thread_operator.pdf) |
| 08 | Harness 工程建设（CP 门控、spec 机器校验、证据对账） | [08_harness_engineering.pdf](./slides/08_harness_engineering.pdf) |
| 09 | 算子测试全流程（测试五流程、golden 生成、三层问题定界） | [09_operator_testing.pdf](./slides/09_operator_testing.pdf) |

## 参考链接

- [cannbot-skills 仓库](https://gitcode.com/cann/cannbot-skills)：Skills 与 Agents 托管，[安装指南](https://gitcode.com/cann/cannbot-skills/blob/master/docs/installation-guide.md)
- [cannbot 主仓](https://gitcode.com/cann/cannbot)：完整版插件 v3.0.0（阶段三实操对象）；版本锚定 `cann/cannbot` @ `7cf1845` · `cann/cannbot-skills` @ `94216ba2`（2026-09-30）
- [Ascend C 算子开发系列教程](../ascendc_operator_development/README.md) · [Kernel 直调版](../ascendc_operator_development_light/README.md)：手动开发与零基础入门
- [PyPTO 算子开发系列教程](../pypto_development/README.md)
