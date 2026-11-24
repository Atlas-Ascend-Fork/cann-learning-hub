![CANN 编程错误调试](images/readme_cover.png)

# CANN 编程错误调试

用一个可切换故障的 Add 算子学习 msSanitizer 和 msDebug。先观察报错、定位源码，再修复并用同一个检查工具复测。

## 开始前

具备 C++、Ascend C、CMake 和基本算子构建经验。环境需提供 `mssanitizer` 和 `msdebug`。

## 学习目标

- 识别越界、数据竞争、未初始化读取和同步不匹配。
- 给独立 ASC 程序和标准算子工程添加检查选项。
- 用断点、调用栈和源码查看定位问题。
- 保存修复前后日志，解释问题是否解决。

## 学习顺序

按下表打开 Notebook。每个实验已包含相关原理、编译运行和结果分析；做完实验后再完成章节练习。

| 顺序 | Notebook | 内容 |
|---|---|---|
| 1 | [CANN 调试实验](08_debugging.ipynb) | 准备环境、理解原理、运行并分析结果 |
| 2 | [第 08 章练习](08_practice.ipynb) | 检查理解、完成编程与分析题 |

## 实验环境

课程使用 CANNLab Atlas A3（SoC `Ascend910_9362`）、CANN 9.0，镜像为 `cann_9.0.0-py3.11-A3-arm-20260829`，Notebook 内核为 `Python 3.11.4`。平台使用方法见[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。

msDebug 需要 `/proc/debug_switch` 的状态为 `1`，并允许当前用户读写 `/dev/drv_debug`。Notebook 会先检查这些条件；条件不足时记录 `BLOCKED` 和原因，跳过后续调试。请联系平台管理员准备调试环境后重测。故障程序得到匹配的诊断时记为 `EXPECTED_DIAGNOSTIC`；修复后校验通过记为 `PASS`。

## 故障与检查工具

<img src="images/debug_fault_tool_matrix.svg" width="760" style="display:block; margin-left:0;" />

<table style="text-align:left; margin-left:0;">
<tr><th>工具</th><th>代码问题</th><th>关键诊断</th><th>实验状态</th></tr>
<tr><td>memcheck</td><td>DataCopy 长度超过 LocalTensor</td><td>illegal read/write</td><td>EXPECTED_DIAGNOSTIC</td></tr>
<tr><td>racecheck</td><td>MTE2 到 Vector 缺少 WaitFlag</td><td>RAW hazard</td><td>EXPECTED_DIAGNOSTIC</td></tr>
<tr><td>initcheck</td><td>未初始化 LocalTensor 被搬出</td><td>uninitialized read</td><td>EXPECTED_DIAGNOSTIC</td></tr>
<tr><td>synccheck</td><td>SetFlag 没有匹配 WaitFlag</td><td>unpaired set_flag</td><td>EXPECTED_DIAGNOSTIC</td></tr>
<tr><td>msDebug</td><td>观察 Kernel 执行位置</td><td>breakpoint/backtrace/frame/source</td><td>PASS、BLOCKED 或 FAIL</td></tr>
</table>

调试权限由平台管理员配置；实验不修改 `/proc/debug_switch` 或设备节点权限。临时安装目录由当前用户创建和清理。
