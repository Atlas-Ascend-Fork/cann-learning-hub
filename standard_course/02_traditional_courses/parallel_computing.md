# 《并行计算》课程大纲

> 本文件为《并行计算》课**单课程落地文件**（传统课程清单·并行与异构计算学科，学科覆盖总表 P1"昇腾版主线课·对标 ECE408"）；总规划见本目录 [README.md](./README.md)。
> 融入定位：**M2 单元植入 + 实验融入**——传统并行主体（并行基础 / 并行算法 / OpenMP 共享存储 / MPI 消息传递）不动，第四部分为 **● 昇腾/CANN 融入单元**（第 6 章，暂定 2 课时：**CPU 与 NPU 异构计算 → NPU 架构 → Ascend C 算子编程**；上机深度由实验 3/4 承载）。
> 课程主线：**并行思想的范式迁移**——数据划分（PCAM 域分解 → NPU Tiling 分块）、通信与计算重叠（MPI 非阻塞 → NPU 双缓冲流水）、归约与集合通信（MPI Allreduce → 超节点集合通信）；CPU 三种范式讲"怎么并行"，第 6 章用 NPU 讲"AI 负载怎么并行"，前后互为镜像。

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 并行计算（Parallel Computing，昇腾异构融入版） |
| **课程类型** | 传统课程 · 计算机类本科专业课（3 学分 / 42 学时，暂定） |
| **学时安排** | 授课 26 学时（暂定）＋ 实验 16 学时（4 项 × 4 学时）；四部分结构，实验按章节穿插；**第 6 章暂定 2 课时**，缩减学时的重排待核定 |
| **授课对象** | 已修数据结构与计算机组成原理的本科高年级学生 |
| **先修要求** | C/C++（指针 / 结构体 / Makefile）；计算机组成原理（存储层次 / 流水线 / CPI）；数据结构与算法（复杂度分析）；Linux 命令行基本操作 |
| **配套讲义** | 第 1—5 章沿用课程现有讲义（Pthreads/OpenMP/MPI 参考 Pacheco 教材配套体系）；第 6 章（● 融入单元，暂定 2 课时）用 [npu_architecture_and_ascend_c_programming_introduction](../00_atomic_courses/04_ops_programming/01_ascendc/npu_architecture_and_ascend_c_programming_introduction.pptx)（NPU 架构与 Ascend C 异构编程导论） |
| **主教材（拟）** | Peter Pacheco《An Introduction to Parallel Programming》（Pthreads / OpenMP / MPI 编程主线） |
| **参考教材（拟）** | 陈国良《并行计算：结构 · 算法 · 编程》（体系结构与算法体系）；Hennessy & Patterson《计算机体系结构：量化研究方法》（DLP / GPU / WSC 章节） |
| **昇腾资料** | 华为昇腾社区《CANN 文档》《Ascend C 编程指南》及 AscendCL 接口参考（第 6 章与实验用，入口 <https://www.hiascend.com/cann/document> ） |
| **实践平台** | 多核 CPU 服务器 / 集群（OpenMP：单机多核；MPI：`mpicc` 多进程，可单机模拟多节点）；CANNLab 云端开发环境（推荐）或本地昇腾环境（Atlas A2/A3 系列，第 6 章与实验 3—4）；cann-learning-hub Notebook 跟练；CANNJudge 判题 |
| **结业标准** | 平时习题与课堂（20%）＋ 4 项实验（20%）＋ 结业大作业（20%：**"同一负载、四种范式"对比报告 ＋ `sigmoid` 功能跑通双必选**，`add_rms_norm` 三档可选加分）＋ 期末闭卷考试（40%，含跨范式综合题）——比例为拟定方案，开课前核定 |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 掌握并行体系结构与性能模型（Flynn 分类 / Amdahl / Gustafson / 强弱扩展 / 互联成本）；② 掌握 PCAM 并行算法设计方法与典型并行算法（归约 / 扫描 / 矩阵乘 / 排序）；③ 掌握共享存储（Pthreads/OpenMP）与消息传递（MPI）两种编程模型；④ 掌握 NPU 异构体系结构与 Ascend C 算子编程模型 |
| **能力目标** | ① 能对给定问题完成 PCAM 设计并定量估算通信开销；② 能用 OpenMP/MPI 编写正确高效的并行程序，诊断数据竞争 / 死锁 / 伪共享；③ 能用 Ascend C 开发矢量算子并用剖析数据完成 Tiling / 双缓冲优化；④ 能跨范式（串行 / OpenMP / MPI / Ascend C）评测、对比与取舍 |
| **素养目标** | 建立"划分—通信—映射"的并行思维；理解并行思想的范式迁移（数据划分→Tiling、通信计算重叠→双缓冲、归约→集合通信）；养成定量工程习惯（一切结论用加速比 / 效率 / 利用率说话） |

### 2.2 结业能力画像

完成本课程后，学生将能够：

- **划得开**：对问题做域分解 / 功能分解，画出任务依赖图与通信模式，给出映射方案；
- **算得清**：用 Amdahl/Gustafson、α-β 模型估算加速比上界与通信量（含环形 Allreduce 推导）；
- **写得出**：完成 OpenMP、MPI、Ascend C 三种范式的可运行实现（含非阻塞通信与 Tiling 分块）；
- **调得优**：定位并消除数据竞争 / 死锁 / 伪共享；用 msprof 定位 NPU 瓶颈并给出优化前后对比数据；
- **比得明**：面对一个负载能回答"选哪种范式、为什么"——共享存储 / 消息传递 / 异构各自的天花板与代价。

### 2.3 课程内容规划（四部分）

| 部分 | 章次 | 授课学时 | 内容主线 | 穿插实验 |
|------|------|---------|---------|---------|
| **第一部分：并行基础** | 第 1—2 章 | 8 | 为什么并行 → 平台谱系与编程模型概览 → 性能定律 → 硬件与互联（一致性 / NUMA / 拓扑） | — |
| **第二部分：并行算法** | 第 3 章 | 6 | PCAM 四步法 → 通信模式与成本 → 归约 / 扫描 / 矩阵乘 / 排序 → 负载均衡 | — |
| **第三部分：并行编程** | 第 4—5 章 | 10 | 共享存储（Pthreads/OpenMP）→ 消息传递（MPI 点对点 / 集合 / 分块实战） | 实验 1（第 4 章）、实验 2（第 5 章） |
| **第四部分：面向 AI 的并行计算** | 第 6 章 | 2（暂定） | **● 昇腾/CANN 融入单元**：CPU 与 NPU 异构计算 → NPU 架构 → Ascend C 算子编程；并行思想范式迁移收束 | 实验 3、实验 4（第 6 章） |

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 编程语言 | C/C++ 基础（指针、结构体、函数指针、Makefile；MPI/OpenMP 均为 C/Fortran 接口） |
| 体系结构 | 计算机组成原理：存储层次（寄存器 / Cache / 主存）、流水线、CPI 与性能分解 |
| 算法基础 | 数据结构与算法：时间复杂度分析、分治思想（归并 / 快排） |
| 系统操作 | Linux 命令行（ssh / 编译 / 作业提交基础） |

### 3.2 领域前置（无硬性要求）

- 无 AI / 大模型背景要求——第 6 章从 AI 负载画像讲起；
- 建议第 6 章开课前预习 [quick_start/cann_basics](../../quick_start/cann_basics/)（`02_what_is_npu`、`03_what_is_cann`，约 30 分钟），建立 NPU/CANN 基本词汇。

### 3.3 环境前置

- **CPU 环境（第 4—5 章 / 实验 1—2）**：多核 Linux 服务器（`gcc -fopenmp`）或小型集群（`mpicc` / `mpirun`）；MPI 可单机多进程模拟多节点，集群资源不足时不影响教学；
- **NPU 环境（第 6 章 / 实验 3—4）**：CANNLab 云端开发环境（浏览器访问，推荐）或本地已部署 CANN 8.5.0+ 的昇腾环境（Atlas A2/A3 系列）；教师开课前试做并固定 CANN 及驱动版本、授权算力与可用时段；
- **测量纪律**：所有性能对比固定输入、预热与重复测量规则，保留原始记录；加速比 / 效率必须注明测量环境（核数 / 节点数 / 并行度）；CPU 与 NPU 结果分别报告，不直接跨平台排名。

---

## 四、教学安排与理论讲义（PPT）（授课暂定 26 学时 · 6 章）

| 章 | 主题 | 内容要点 | 教材 PPT |
|----|------|---------|---------|
| 第 1 章 | 并行计算概论 | 为什么并行；Flynn 分类与平台谱系；编程模型概览；性能度量与 Amdahl/Gustafson | 4 学时，沿用课程现有讲义 |
| 第 2 章 | 并行硬件与互联网络 | Cache 一致性与伪共享；NUMA；互联拓扑与成本模型；加速器一瞥 | 4 学时，沿用课程现有讲义 |
| 第 3 章 | 并行算法设计 | PCAM 方法；通信模式与 α-β 模型；归约 / 扫描 / 矩阵乘 / 排序；负载均衡 | 6 学时，沿用课程现有讲义 |
| 第 4 章 | 共享存储并行编程 | Pthreads 线程与同步；OpenMP 增量并行化；数据作用域与调度；伪共享实测 | 4 学时，沿用课程现有讲义 |
| 第 5 章 | 消息传递并行编程（MPI） | SPMD 与六函数；点对点 / 非阻塞；集合通信与 Allreduce；数据划分与虚拟拓扑；矩阵乘实战 | 6 学时，沿用课程现有讲义 |
| **第 6 章** | **面向 AI 的并行计算：CPU 与 NPU 异构计算、NPU 架构与 Ascend C 算子编程** | **● 昇腾/CANN 融入单元（暂定 2 课时）**：① CPU 与 NPU 异构计算（为什么异构 / Host—Device 协同）；② NPU 架构（计算 / 存储 / 搬运单元；Cache 自动管理 vs 软件显式管理）；③ Ascend C 算子编程（SIMD 为主、SIMT 为辅；Tiling 分块） | 2 学时（暂定），[NPU 架构与 Ascend C 编程导论 PPT](../00_atomic_courses/04_ops_programming/01_ascendc/npu_architecture_and_ascend_c_programming_introduction.pptx)（Part 1 算力之问 → ①；Part 2 芯片之芯 → ②；Part 3 编程之道 → ③） |

> **关键节点说明：**
> - 教学总要求：能对问题做 PCAM 设计并定量分析；能编写与调试 OpenMP/MPI/Ascend C 程序；能用性能数据（加速比 / 效率 / 利用率）支撑结论；
> - 第 1—5 章为传统并行主体（24 学时），沿用课程现有讲义（Pthreads/OpenMP/MPI 参考 Pacheco 教材配套体系）；CANN 轻量融入：第 1—3 章 ○（性能定律配 NPU 例题 / 为第 6 章铺垫 / 分块与归约思想将在 NPU 重现）、第 5 章 ◐（Allreduce ↔ 超节点集合通信）；
> - **第 6 章暂定 2 课时**（● 昇腾/CANN 融入单元）：CPU 与 NPU 异构计算 → NPU 架构 → Ascend C 算子编程；理论讲义用 [NPU 架构与 Ascend C 编程导论 PPT](../00_atomic_courses/04_ops_programming/01_ascendc/npu_architecture_and_ascend_c_programming_introduction.pptx)，课时实践见 5.2，CANNJudge 作业见 6.2；单算子 API 调用、msprof 调优等上机深度由实验 3/4 承载；
> - 并行思想范式迁移主线贯穿全课：数据划分（第 3 章 → NPU Tiling）、通信与计算重叠（第 5 章非阻塞通信 → NPU 双缓冲）、归约（MPI Allreduce → 集合通信）——CPU 三种范式与 NPU 范式前后互为镜像。

---

## 五、每节课的实践内容（实验 16 学时 ＋ cann-learning-hub 跟练）

> 课程实践由两部分构成：**四项课程实验**（16 学时，CPU 集群 + 昇腾环境，计入成绩）与**第 6 章课时级跟练**（cann-learning-hub Notebook，随课配对）。实验 1—4 构成一条"同一负载、逐范式递进"的证据链，为结业大作业直接积累素材。

### 5.1 课程实验总览（实验 1—4，16 学时）

| 实验 | 学时 | 对应章 | 内容 | 范式 / 平台 |
|------|------|--------|------|------------|
| 实验 1 共享存储并行编程 | 4 | 第 4 章 | OpenMP 矩阵乘并行化：static / dynamic / guided 调度对比；并行归约（reduction 子句）；伪共享实测（时间差异复现）与 cache line 对齐消除；提交不同核数的加速比曲线与分析 | OpenMP / 多核 CPU |
| 实验 2 MPI 分布式并行编程 | 4 | 第 5 章 | 点对点与集合通信练习（含死锁复现与规避）；MPI 矩阵乘（行列分块）；非阻塞 Isend/Irecv 通信—计算重叠优化；强 / 弱扩展测试与效率曲线 | MPI / CPU 集群或单机多进程 |
| 实验 3 NPU 认知与昇腾运行基线 | 4 | 第 6 章 | cann_basics Notebook 走读（NPU 架构与 CANN 组件认知）→ 单算子 API 调用全流程（ACL 初始化 → 内存 → 传输 → 执行 → 同步 → 释放，观测 Host—Device 数据流）→ 首个 Ascend C 自定义算子（Add）跑通；记录输入形状 / 类型 / 误差规则与环境版本 | 昇腾 / CANNLab |
| 实验 4 Ascend C 算子开发与调优 | 4 | 第 6 章 | 矢量算子变体实现（Tensor API / C API 各一）→ msprof 剖析（单元利用率 / 流水图）→ Tiling 分块与双缓冲优化 → 优化前后性能对比；与实验 1/2 的 CPU 结果汇成跨范式对比素材（分别报告，不跨平台排名） | Ascend C / 昇腾 |

### 5.2 第 6 章课时实践（cann-learning-hub 承载，随 2 课时配对）

> 课时实践由 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 的跟练 Notebook 承载，与 2 个课时配对；单算子 API 调用与调优初体验等上机深度由实验 3/4 承载（见 5.1）；实机资源不足时以"分析替代"处理并明示。

| 课时 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 1 | CPU 与 NPU 异构计算 · NPU 架构认知 | 课堂交互式走读：为什么需要 NPU / CPU vs NPU 算力差距 / AI Core 三要素与多级存储层次；课后实测 CPU vs NPU 矩阵乘性能对比，与实验 1 的 OpenMP 加速比曲线并读 | 课堂：[quick_start/cann_basics](../../quick_start/cann_basics/) `02_what_is_npu`；课后：`04_npu_practice` 第 1—2 题 |
| 2 | 首个 Ascend C 自定义算子 | Add 类矢量算子全流程跟练：tiling 结构体 → 核函数类 → CopyIn/Compute/CopyOut → 编译运行与结果比对；进阶：静态 Tensor 矢量算子、SIMT 一瞥 | 跟练：[quick_start/first_custom_operator](../../quick_start/first_custom_operator/)；进阶：[tutorials/ascendc_operator_development_light/02_AscendC_basic](../../tutorials/ascendc_operator_development_light/02_AscendC_basic/) |

### 5.3 实验要点与约束

- **实验 1**：加速比曲线必须注明核数与调度策略；伪共享实验保留修复前后的对照数据；
- **实验 2**：死锁复现须附成因分析与规避方案；通信重叠优化给出通信时间占比的前后对比；
- **实验 3**：重点为建立可重复基线，难点为区分计算与传输阶段；提交运行记录与 Host—Device 数据流图；
- **实验 4**：Tiling 改动须给出 UB 容量约束下的分块依据；优化结论以剖析数据（利用率 / 流水图）与优化前后耗时支撑，不以预期替代实测；
- **共性纪律**：固定输入与测量条件，保留原始记录；CPU 与 NPU 结果分别报告，不直接跨平台排名。

---

## 六、课后习题与作业

> 作业体系分两层：**每章课后习题**巩固方法与定量分析（计入平时成绩）；**结业大作业**为全课程最终产出——"同一负载、四种范式"综合对比报告（必选）＋ CANNJudge 四档算子作业（`sigmoid` 必选入门 ＋ `add_rms_norm` 三档可选加分，对齐启航营一天大作业设计）。

### 6.1 每章课后习题与小作业

| 章 | 课后习题 / 小作业要点 |
|----|---------------------|
| 第 1 章 | Amdahl / Gustafson 定量计算（含"CPU+NPU 异构系统中 NPU 占比与加速上界"例题）；Flynn 分类辨析（SIMD vs SIMT vs MIMD） |
| 第 2 章 | 伪共享场景分析（给出时间差异的成因与消除方案）；给定拓扑下广播 / 归约的 α-β 成本估算 |
| 第 3 章 | 给定问题的 PCAM 设计（任务依赖图 + 通信模式 + 映射方案）；Cannon 算法通信量推导；Blelloch 扫描手算 |
| 第 4 章 | OpenMP 代码分析（找数据竞争、判数据作用域）；调度子句选择论述（static/dynamic/guided 适用负载） |
| 第 5 章 | MPI 代码分析（死锁诊断与改写）；环形 Allreduce 通信量推导；行列分块的通信量计算 |
| 第 6 章 | 概念辨析（SIMD vs SIMT vs Cube / 自动挡 vs 手动挡）；Tiling 搬运量计算（给定 UB 容量与数据宽度）；**跨范式论述**：数据划分、通信计算重叠、归约三种思想在 CPU 范式（OpenMP/MPI）与 NPU 范式（Ascend C）中的形态对照 |

### 6.2 结业大作业

> **CANNJudge 算子作业采用四档进阶设计（对齐启航营[一天课程大纲](../01_bootcamp/syllabus_one_day_bootcamp.md)大作业）**：`sigmoid` 为**必选入门档**（正确性门禁，承接实验 3/4 的 Add 跟练，作为 Ascend C 独立开发的第一道判题）；**`add_rms_norm` 结业算子作业由两部分构成**——**第一部分 · 功能跑通**（作业 2，CANNJudge 判题通过）→ **第二部分 · 持续优化性能**（作业 4，性能提升数据 + 优化路径剖析）；作业 3 PyTorch 集成为独立作业维度。`add_rms_norm` 部分与《计算机体系结构》课第 4 讲作业课件同源，两课互认不重复选做。

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **"同一负载、四种范式"综合报告** | **必选**（课程自设档） | 选定负载（推荐矩阵乘或实验系列中的算子级任务），完成 **串行 → OpenMP → MPI → Ascend C** 四种实现：每种给出正确性验证、性能测量（固定输入与测量条件）、加速比 / 效率 / 强扩展曲线；最后回答"该负载选哪种范式、为什么"（含天花板与代价分析） | 四种实现正确性 ＋ 测量规范（环境注明 / 重复测量）＋ 跨范式对比与取舍分析的定量依据 |
| **作业 1 · 矢量算子 `sigmoid` 功能跑通** | **必选**（CANNJudge 入门档） | 开发矢量算子 `sigmoid`，功能跑通——从"跟练 Add"到"独立写一个算子"的第一步 | CANNJudge 判题通过（正确性门禁：多形状 / 多类型 / 精度达标） |
| 作业 2 · `add_rms_norm` 功能跑通 | 可选加分 | 用基础 API 或 C API 编写矢量算子 `add_rms_norm`（shape 与 Qwen3-1.7B 模型匹配），功能跑通；判题链接：<https://cannjudge.cn/public/vector/addrmsnormqwen3> | CANNJudge 判题通过（正确性） |
| 作业 3 · `add_rms_norm` 接入模型 | 可选加分 | 将 `add_rms_norm` 算子接入 **PyTorch Qwen3-1.7B**，对比与原来模型推理性能差异 | 整网集成端到端跑通，推理结果正确 ＋ 性能对比数据与分析 |
| 作业 4 · `add_rms_norm` 持续优化 | 可选加分 | **持续优化** `add_rms_norm` 算子性能：分析性能瓶颈点，结合 Atlas A2/A3 或 Ascend 950 架构列出每个硬件的流水利用率，给出优化方案，提供优化后性能数据与算子代码 | 性能提升数据 ＋ 优化路径剖析（profiling 证据 ＋ 改动说明）＋ 流水利用率对照 |

> 大作业可自学 [CANN-LearningHub](https://gitcode.com/cann/cann-learning-hub/) 与 CANNJudge 完成（入口 <https://gitcode.com/cann>）；四档与启航营梯度一致（跟练 → sigmoid → add_rms_norm 三档），无算子开发基础者完成两项必选档即可。

---

## 七、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| 华为昇腾 CANN 文档 | <https://www.hiascend.com/cann/document> | 第 6 章与实验 3—4 指定参考资料 |
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现 |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（实验 4 参考） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（第 6 章课时实践载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材（第 6 章延伸阅读） |
