# HiDevLab 首次真机验证 2026-10-28

通过 Chrome 的 WebIDE 在 DevEnv_193623 上完成。环境标注为 A3、CANN 9.0.0 Release、Ubuntu 22.04。课程提交为 054c9bfb4397ad9ac505f5d1cfa1685147087e05。

GMRES 真实后端构建成功，已有测试 1/1 通过。U1、单 rank、逻辑 device 0、MGS、restart 30、max-iterations 300、tolerance 1e-6：

- 单次 solve 4693.852051 ms。
- repeat=3 平均 solve 4742.979980 ms，最后一次迭代数 33。
- 相对残差 4.749471e-7，相对解误差 5.452064e-7。
- 后端为 ACL + HCCL / Ascend C RTC Device GMRES。
- msprof 采集和显式导出退出码均为 0，已生成 op_statistic、task_time、api_statistic、op_summary 等 CSV。

远端结果在 /workspace/cann-demo/：gmres-build.log、gmres-u1.log/csv、gmres-repeat3.log/csv、profile.log、export.log、profile-u1/。尚未下载原始产物到本地，本记录根据 WebIDE 可见输出整理。

首次尝试时，MuduoXinyu 上游访问触发 GitCode 用户名认证，未取得源码。随后已发布课程基线到新仓库、取得并校验真实模型和 tokenizer，完成完整 A/B。详见 [MuduoXinyu 实测与复现](./muduo_hidevlab_run.md)。
