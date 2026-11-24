# ACL C/C++ GEMM 与 SpMV 实验手册

## 1. 实验目标和准备

本章分别运行 GEMM 和稀疏 SpMV。开始前准备 CANN、可用设备和 ops-sparse；下面的命令使用 `-DACL_C_STUB=OFF` 构建设备版本。

资源生命周期为：`aclInit` → `aclrtSetDevice` → `aclrtCreateStream` → Device 内存/H2D → Tensor 或稀疏描述符 →（仅 GEMM）Workspace 查询与分配 → 算子执行 → `aclrtSynchronizeStream` → D2H → 销毁描述符、内存、Stream、Device、ACL。

GEMM 的关键链是 `aclnnGemmGetWorkspaceSize` → workspace allocation → `aclnnGemm` → synchronize。SpMV 使用 `aclsparseCreateCsr`、`aclsparseCreateDnVec`、`aclsparseSpMV`，随后同步、D2H 并与 `cpu_spmv` 比较。注意区分：只有 GEMM 需要 `aclnnGemmGetWorkspaceSize` 查询并分配 workspace；SpMV 不查询 workspace，`aclsparseSpMV` 的最后一个参数直接传 `nullptr`（API 默认 workspace），对应输出 `Workspace bytes=0 (API default workspace)`。

下面从第 04 章目录开始，依次在同一个终端执行。

## 2. 安装 ops-sparse

本章在 CANNLab A3、CANN 9.0 环境中使用 ops-sparse 的固定提交 [e0015bb](https://gitcode.com/cann/ops-sparse/commit/e0015bb76090487165f43dbaf2aae37648887a4b)。A3 的构建参数为 `ascend910_93`。

使用 Notebook 时，依次运行 4.4 节的源码下载、构建安装和路径设置单元格。默认安装目录为 `~/Ascend/cann`；安装过课程版本后，从路径设置继续。使用其他目录时，`OPS_SPARSE_ROOT` 填写 `<安装位置>/cann`，并在 4.4 节和章节练习中保持一致。

使用终端首次安装时，从第 04 章目录执行：

```bash
git clone https://gitcode.com/cann/ops-sparse.git ops-sparse
git -C ops-sparse checkout e0015bb76090487165f43dbaf2aae37648887a4b
source "$ASCEND_HOME_PATH/set_env.sh"
cd ops-sparse
bash build.sh --pkg --soc=ascend910_93 --ops=spmv
bash build_out/cann-A3-ops-sparse-1.0.0_linux-aarch64.run --install --quiet --install-path="$HOME/Ascend"
export OPS_SPARSE_ROOT="$HOME/Ascend/cann"
export LD_LIBRARY_PATH="$OPS_SPARSE_ROOT/lib64:$LD_LIBRARY_PATH"
test -f "$OPS_SPARSE_ROOT/include/cann_ops_sparse.h"
test -f "$OPS_SPARSE_ROOT/lib64/libops_sparse.so"
cd ..
```

`--quiet` 让安装过程无需输入确认。安装失败后，保留已下载的源码，从构建命令继续；Notebook 可直接重跑构建和安装单元格。

## 3. 直接编译与独立运行

```bash
source "$ASCEND_HOME_PATH/set_env.sh"

cd src/acl_operator_calls/GEMM-acl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DACL_C_STUB=OFF
cmake --build build -j
./build/bin/gemm_acl --m 512 --n 512 --k 512 --warmup 3 --repeat 10 --device 0

cd ../SpMV-acl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DACL_C_STUB=OFF
cmake --build build -j
./build/bin/spmv_acl --rows 4096 --cols 4096 --nnz 65536 --warmup 3 --repeat 10 --device 0
```

## 4. 验证、性能与练习

每次运行记录 `Actual Backend`（SpMV 为 `ACL/CANN ops-sparse`，GEMM 为 `ACL/CANN`）、Device ID、时间字段（SpMV 为 `ACL SpMV time`，GEMM 为 `ACL GEMM time`，二者都包含 Stream 同步；GEMM 还包含 workspace 查询、分配与释放）和 CPU reference relative L2 error；以程序退出码为首要判据，并使用 `1e-6` 作为实验记录阈值（与程序 Correctness 门槛一致）。

| Operator | Actual backend | Device | ACL time (ms) | CPU reference error | Pass |
|---|---:|---:|---:|:---:|:---:|
| GEMM | | | | | |
| SpMV | | | | | |

练习：说明为何异步算子计时必须包含 Stream 同步，以及为何只有 GEMM 需要 `aclnnGemmGetWorkspaceSize` 在执行前查询并分配 workspace，而 SpMV 直接传 `nullptr` 使用 API 默认 workspace。参考答案见 `answer/`。
