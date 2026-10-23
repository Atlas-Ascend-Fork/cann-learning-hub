# 章节源码与运行资源

本目录集中保存章节的全部工程资源；课程总览、学习目标、硬件和在线环境说明见上级课程 [README](../../README.md)。

## 内容

```text
src/
├── requirements.txt
├── data/                         # GTSRB 小规模演示数据
├── assets/                       # 正式 checkpoint、指标与检测报告
├── onsite_demo/                  # Notebook 共用运行库、配置和脚本
├── final_summary.json            # 正式攻击结果摘要
├── dataset.py
├── evaluate.py
├── experiment.py
├── model.py
├── poison.py
├── train.py
└── utils.py
```

Notebook 从章节根目录向上定位 `src/onsite_demo/`，运行库再统一解析 `src/data/`、`src/assets/` 和 `src/onsite_demo/outputs/`，无需修改工作目录或设置额外的项目绝对路径。

## 安装普通依赖

在章节根目录执行：

```bash
python -m pip install -r src/requirements.txt
```

MindSpore、CANN 与 TBE 应由 CANNLab 镜像提供，不建议在普通 CPU 环境中单独安装替代版本。

## 现场演示与云端验收

在章节根目录打开 `02.01_badnets_attack.ipynb`、`02.02_neural_cleanse_detection.ipynb`
或 `02.03_bnp_defend.ipynb`，按各 Notebook 的说明执行。Notebook 的 `RUN_PROFILE`
通过环境变量 `ONSITE_DEMO_PROFILE` 选择，默认使用 `cloud_live`。
攻击与检测的现场训练流程可通过以下命令执行：

```bash
python src/onsite_demo/scripts/run_full_onsite_demo.py --mode cloud_live --epochs 10 --batch-size 8
python src/onsite_demo/scripts/run_cloud_live_10epoch_acceptance.py --mode cloud_live
```

十轮验收针对攻击与检测流程，不替代 BNP 防御 Notebook 的实验验证。
运行前需要准备 `src/data/train/`、`src/data/test/`、正式 checkpoint 和正式结果，
并使用匹配的 MindSpore、CANN 与设备环境。

交互控件依赖 `ipywidgets` 和 `jupyterlab_widgets`。需要控件时，在 Notebook
所用环境安装这两个包；控件不可用时按 Notebook 中的 fallback 单元执行。
检测结果不代表后门已消除，需结合具体实验指标解释。

## 完整课程打包

打包要求保留仓库中的课程布局：
`contrib/tutorials/ai_security_nuist/README.md` 与同级的
`02_model_attack_defense/`。单独下载的章节目录不能直接作为完整课程打包。

准备好数据、checkpoint 和正式结果后，在章节根目录执行：

```bash
python src/onsite_demo/scripts/build_onsite_demo_package.py
python src/onsite_demo/scripts/final_preupload_audit.py
```

ZIP 顶层为 `ai_security_nuist/`，三个 Notebook 位于
`ai_security_nuist/02_model_attack_defense/` 下。为兼容现有流程，默认 ZIP 文件名
仍为 `ai_model_backdoor_attack_and_detection_cloud_live.zip`；该文件名不决定包内路径。
演示输出、缓存、`kernel_meta/` 和 `offload/` 不进入打包结果。

## 验证

```bash
python src/onsite_demo/scripts/validate_onsite_demo.py
python src/onsite_demo/scripts/final_preupload_audit.py
```


