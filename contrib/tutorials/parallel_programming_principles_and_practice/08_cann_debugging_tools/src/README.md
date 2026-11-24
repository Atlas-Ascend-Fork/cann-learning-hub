# 调试实验源码

- `add_sanitizer.asc` + `CMakeLists.txt`：独立 ASC 可执行程序；
- `operator_project_sanitizer/`：标准 `op_host + op_kernel + 算子包 + ACLNN Host` 工程；
- `run_sanitizer.sh`：构建故障程序、运行检查并保存日志与退出码。
- `UPSTREAM.md`、`LICENSE.asc-devkit`：样例来源与许可证。

Notebook 中直接列出了独立 ASC 的编译、检查和 msDebug 命令。若在本 `src/` 目录用终端运行独立 ASC：

```bash
MODE=memcheck
cmake -S . -B build-$MODE -DLAB05_FAULT_MODE=$MODE -DLAB05_ENABLE_SANITIZER=ON
cmake --build build-$MODE -j
mssanitizer --tool=$MODE ./build-$MODE/lab05_add_sanitizer
```

`MODE` 可替换为 `racecheck`、`initcheck`、`synccheck`。故障程序可能返回非零，查看日志中的对应诊断。标准算子工程的完整命令见 `operator_project_sanitizer/README.md`。

msDebug 只读检查 `/proc/debug_switch` 和 `/dev/drv_debug`。系统设备权限和调试开关由平台配置；权限不足时记录 `BLOCKED`。临时实验目录可以正常清理。
