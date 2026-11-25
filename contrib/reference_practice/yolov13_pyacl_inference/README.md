# YOLOv13 pyACL 免编译离线推理

本案例为官方案例 [reference_practice/yolov13_offline_inference](../../../reference_practice/yolov13_offline_inference/README.md) 第 3 章 ACL 离线推理 C++ 程序（`yolov13_main`）的**免编译备选**，由社区开发者贡献。在缺少 C++ 编译环境（如精简镜像未预装 g++/gcc-c++）或希望快速验证推理结果的场景下，用 pyACL 完成与 C++ 程序相同的推理流程。

> 本案例不改变官方 C++ 推理路径，仅为其提供低成本替代。

## 使用方式

**推荐**：运行端到端 Notebook [yolov13_pyacl_inference.ipynb](yolov13_pyacl_inference.ipynb)——从权重下载、ATC 模型编译、输入预处理到 pyACL 推理、结果可视化全链路可独立执行，无需预先准备官方案例产物。

**整脚本方式**：如仅需推理已有产物，可直接运行 `scripts/infer_pyacl.py`（用法与常见问题见 Notebook §4–§5）：

```bash
source "$ASCEND_HOME_PATH/set_env.sh"
python3 scripts/infer_pyacl.py --model=model/yolov13.om --input=data/bus.bin
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `model/yolov13.om` | OM 模型路径 |
| `--input` | `data/bus.bin` | 预处理 BIN 输入路径 |
| `--conf` | `0.25` | 置信度阈值（与 C++ 样例一致） |
| `--iou` | `0.45` | NMS IoU 阈值（与 C++ 样例一致） |

## 目录结构

```
yolov13_pyacl_inference/
├── yolov13_pyacl_inference.ipynb    # 端到端 Notebook（环境检查 → 模型准备 → 推理 → 可视化 → FAQ）
├── README.md                        # 本文件（索引）
└── scripts/
    └── infer_pyacl.py               # pyACL 免编译推理整脚本
```

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 硬件 | 昇腾 Atlas 推理系列（已在 Atlas 300I Duo / 310P 实机验证） |
| CANN | 8.0.0（Toolkit，含 pyACL）及以上 |
| Python | 3.x（官方案例推荐 3.11），依赖 numpy |

> 验证环境说明：本案例在 Atlas 300I Duo（310P）+ CANN 8.0.0 上完成全链路实测；与官方案例标注的 A2 系列验证环境不同，请按实际设备调整 atc 的 `soc_version`（Notebook 2.3 节已自动识别）。

## 反馈

问题与建议欢迎在本仓库提 Issue 并 @ 贡献者，或在官方案例对应 Issue 中讨论。
