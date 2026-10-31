# AI 应用开发实践

本目录汇总基于 CANNLab 和昇腾 NPU 的 AI 应用开发参考实践，以具体应用场景串联数据准备、本地模型部署、应用调用和结果验证。每个场景提供一个完整的 Notebook，开发者可以按需选择并独立运行。

当前提供 RAG 知识库问答实践：导入技术文档，使用 bge-m3 和 FAISS 构建向量索引，在本地 NPU 部署 Qwen3-8B，并完成带来源的回答和拒答验证。后续应用场景将在本目录持续补充。

## 适用人群

- 具备基础 Python 编程和 Jupyter Notebook 使用经验的开发者。
- 希望在 CANNLab 中完成本地模型部署与应用开发的开发者。
- 需要将自己的文档接入知识库问答流程的应用开发者。

## 实践列表

<table>
<tr><th align="left">实践名称</th><th align="left">实践内容</th><th align="left">在线体验</th><th align="left">状态</th></tr>
<tr><td><a href="./rag_knowledge_base.ipynb">RAG 知识库问答</a></td><td>文档处理、向量检索、Qwen3-8B 本地服务、来源引用与问答验证</td><td>在 CANNLab 中运行（<a href="../../docs/CANNLab_env_experience_guide.md">环境体验指南</a>）</td><td>✅ 已验证（A3）</td></tr>
</table>

## 软硬件配套说明

以下配置适用于当前的 RAG 知识库问答实践，详细依赖版本见 Notebook 中的“已验证的软件基线”。

| 项目 | 要求 |
| --- | --- |
| 已验证硬件环境 | CANNLab A3，1 张 Ascend 910C NPU（2 个芯片、128 GiB HBM），实践使用逻辑设备 `npu:0` |
| 操作系统 / 架构 | Ubuntu 24.04.3 LTS / aarch64 |
| CANN 版本 | 9.1.0 |
| Python | 3.12.9 |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0.post4 |
| 磁盘空间 | `/mnt/workspace` 至少有 35 GB 可用空间 |
| 网络 | 首次准备需访问 GitCode、ModelScope；缺少 Python 依赖时需访问包源 |

## 在线体验环境

| 体验环境 | 镜像模板 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| CANNLab 云开发环境 | `py3.12-A3-arm-vllm-20260928` | Python 3.12.9，预装 CANN 9.1.0、PyTorch 2.10.0+cpu 和 torch-npu 2.10.0.post4 | 按照 [CANNLab 环境体验指南](../../docs/CANNLab_env_experience_guide.md)创建并连接环境 |

以上镜像名和版本是 2026-09-29 实际验证环境的快照，不代表最低兼容版本。PyTorch 包版本中的 `+cpu` 是该预装包的构建标识，NPU 能力由配套的 `torch-npu` 提供。

1. 在 CANNLab 中创建上述 A3 环境，通过 WebIDE 或 VS Code 打开本仓库。
2. 打开 [rag_knowledge_base.ipynb](./rag_knowledge_base.ipynb)，选择对应的 Python 内核。
3. 从环境检查开始，依次执行 Notebook 中的代码单元，完成建库、模型部署、问答和验证。
4. 需要使用自己的资料时，参考 Notebook 中“换成自己的知识文档”的说明。
5. 结束体验时，执行 Notebook 末尾的停止服务单元，释放模型服务占用的 NPU 显存。

模型权重、示例文档、索引和日志保存在 `/mnt/workspace/rag_knowledge_base`。运行过程中的数据处理和模型推理均在当前实例内完成，不调用外部在线大模型 API。

## 目录结构

```text
ai_application_development/
├── README.md                     # 实践介绍、环境说明和 Notebook 入口
└── rag_knowledge_base.ipynb       # RAG 知识库问答完整实践
```

## 反馈与建议

若在实践过程中遇到问题，请在本仓库提交 Issue，并说明实践名称、运行环境、出错单元和错误信息，便于复现与定位。
