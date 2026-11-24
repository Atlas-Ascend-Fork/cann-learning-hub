# MuduoXinyu 源码与课程版本

原项目来自 [BNU-SYS/muduoXinyu](https://gitee.com/bnu-sys/muduoXinyu)，采用 MIT 许可证，原署名见本目录的 `LICENSE-MuduoXinyu`。

课程使用[公开教学副本](https://gitcode.com/bnujwx/muduoXinyu)的 `course-attention-ab` 标签。下载后直接编译，默认运行 Path A；加 `--useNpuFlashAttention` 运行 Path B。

## 源码阅读位置

以下路径相对于下载的 MuduoXinyu 目录：

| 文件 | 阅读内容 |
|---|---|
| `src/backend/npuBackend.cpp` | `prepareFlashAttention`、`attentionAllHeadsFlashDevice`，FP16 KV cache、4 次 Cast 和调用计数 |
| `src/backend/npuBackend.hpp` | 后端接口与成员 |
| `src/infer/infer.cpp` | 初始化与参数检查 |
| `src/model/model.cpp`、`src/model/model.hpp` | `useNpuFlashAttention` 开关，默认关闭 |
| `src/model/modelForwardOpt.cpp` | forward 中的 A/B 分支 |

源码、stories110M 模型和配套 tokenizer 的下载、构建命令见[实验 Notebook](../09_muduo.ipynb)。构建后用 `run_ab_benchmark.sh` 比较两种实现。
