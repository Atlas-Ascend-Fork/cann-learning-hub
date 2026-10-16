"""PyPTO NPU 算子库（第 8 章共享模块）。

本模块把第 8 章后续各节需要用到的 pypto Kernel 与
``torch.autograd.Function`` 封装收集到一处，供本章后续各节通过
``from src.pypto_ops import ...`` 直接复用，避免每节都把同一份 Kernel 代码
原样复制一遍。

提供的算子：

| 算子 | 前向 PyPTO API | 反向 PyPTO API | tiling |
|------|---------------|----------------|--------|
| matmul | pypto.matmul | pypto.matmul (transpose variants) | cube [32,32] x3 |
| bias_add | pypto.add (broadcast) | sum over batch dim | vec (128,128) |
| add | pypto.add (same shape) | identity | vec (128,128) |
| tanh | pypto.tanh (v0.2.1+ 原生) | (1-y²)*grad | vec (128,64) |
| relu | pypto.maximum | pypto.where | vec (128,128) |
| softmax+CE | pypto.softmax + gather/log | softmax - one_hot | vec (rows 自适应, aligned) |
| linear 融合 | pypto.matmul + add | 三梯度单 kernel | cube+vec (0.2.1+) |
| rnn_hidden 融合 | tanh(add(add(a,b),bias)) | 两路梯度 + sum | vec (128,64) (0.2.1+) |
| rnn 整段展开 | 投影 1 次 + 每步融合 cell kernel | 每步 dz/dh 单 kernel + 权梯度大 GEMM | cube+vec (0.2.1+) |

深度优化说明（0.2.1+ 路径，实测 batch=32/num_steps=35/hidden=256
训练 step 约 3 倍提速，前向 logits 与逐算子路径逐位一致）：
1. 整序列输入投影：``X @ W_xh`` 对全部时间步一次算完（reshape 零拷贝），
   35 次发射 → 1 次（8.4 节恒等式 ``[X,H] @ [W_xh;W_hh]`` 的投影半边）；
2. 融合 RNN cell：``h = tanh(matmul(h,W_hh) + xw + b)`` 单 kernel，
   每时间步 3 次发射 → 1 次；
3. 整段展开 autograd Function + 手写 BPTT：反向每步单 kernel 求
   ``dz = dh·(1-h²)`` 与 ``dh_prev = dz @ W_hhᵀ``，全部权梯度用"大 GEMM"
   事后一次计算（``Σ_t h_{t-1}ᵀ dz_t = hprev_flatᵀ @ dz_flat`` 的矩阵化），
   消除逐步 autograd 的 105 次梯度累积发射；
4. softmax+CE 前向融合为单发射（反向保持 host 链式化简——小 logits
   上 3 个 aten 算子比一次 pypto 发射更快，数值逐位一致）。

每个算子采用工厂函数模式：
1. 定义 @pypto.frontend.jit 内核 (fwd + bwd)
2. 包装为 torch.autograd.Function
3. 导出为 PyPTO<Op> 类，通过 .apply() 调用
4. 对有需要的算子提供 nn.Module 封装

导出清单：
- PyPTOMatmul / PyPTOBiasAdd / PyPTOAdd / PyPTOTanh / PyPTOReLUOp
- PyPTOLinearFused / PyPTORNNHidden（融合算子，0.2.1+ 启用）
- PyPTOReLU / PyPTOLinear (nn.Module 封装)
- loss_fn (softmax + cross-entropy 损失入口)
"""

import importlib.metadata

import pypto
import torch
from torch import nn
from torch.nn import functional as F

try:
    _PYPTO_VERSION = tuple(
        int(x) for x in importlib.metadata.version("pypto").split(".")[:3])
except Exception:
    _PYPTO_VERSION = (0, 2, 0)
# pypto 0.2.1+ 提供原生 pypto.tanh，且 matmul+bias 融合路径数值稳定
# （0.2.0 下该融合存在调用序列相关的数值不稳定，见第 10 章 KNOWN_ISSUES）
_PYPTO_GE_021 = _PYPTO_VERSION >= (0, 2, 1)


# ---------------------------------------------------------------------------
# matmul：前向 C = A @ B，反向 grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    # cube tile 16→32/64（第 4 章实测：tile 过小导致切分次数过多，-45%）
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    c.move(pypto.matmul(a, b, pypto.DT_FP32))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_bwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    grad_a.move(
        pypto.matmul(grad_c, b, pypto.DT_FP32, b_trans=True))
    grad_b.move(
        pypto.matmul(a, grad_c, pypto.DT_FP32, a_trans=True))


def make_pypto_matmul(fwd_kernel, bwd_kernel):
    """创建 PyPTO 矩阵乘法算子。"""

    class PyPTOMatmulImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty(a.shape[0], b.shape[1],
                            device=a.device, dtype=a.dtype)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            a, b = ctx.saved_tensors
            need_a = ctx.needs_input_grad[0]
            need_b = ctx.needs_input_grad[1]
            grad_a = torch.empty_like(a) if need_a else None
            grad_b = torch.empty_like(b) if need_b else None
            tmp_a = grad_a if need_a else torch.empty_like(a)
            tmp_b = grad_b if need_b else torch.empty_like(b)
            bwd_kernel(a.contiguous(), b.contiguous(),
                       grad_c.contiguous(), tmp_a, tmp_b)
            return grad_a, grad_b

    return PyPTOMatmulImpl


# ---------------------------------------------------------------------------
# linear 融合版：c = matmul(a, w) + bias，单 kernel（少一次 kernel 启动）。
#    实测 (640,512)x(512,512) 从分两次的 0.32ms 降到 0.14ms（约 2.3x）。
#    注意：pypto 0.2.0 下该融合存在调用序列相关的数值不稳定
#    （详见第 10 章 KNOWN_ISSUES 问题 D），仅在 0.2.1+ 启用。
#    反向复用 matmul_bwd_kernel + bias_add_bwd_kernel。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def linear_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(64, 128)
    c.move(pypto.add(pypto.matmul(a, w, pypto.DT_FP32), bias))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def linear_bias_bwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_w: pypto.Tensor([], pypto.DT_FP32),
    grad_bias: pypto.Tensor([], pypto.DT_FP32),
):
    # dX/dW/db 三个梯度同 kernel（少两次发射，第 4 章调优报告先例）
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    grad_a.move(pypto.matmul(grad_c, w, pypto.DT_FP32, b_trans=True))
    grad_w.move(pypto.matmul(a, grad_c, pypto.DT_FP32, a_trans=True))
    pypto.set_vec_tile_shapes(16, 16)
    grad_bias.move(pypto.sum(grad_c, 0, True))


def make_pypto_linear_fused(fwd_kernel, bwd_kernel):
    """创建融合 matmul+bias 的线性层算子（y = x @ W + b）。
    前向/反向均为单 kernel 发射。"""

    class PyPTOLinearFusedImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias):
            ctx.save_for_backward(x, weight)
            c = torch.empty(x.shape[0], weight.shape[1],
                            device=x.device, dtype=x.dtype)
            fwd_kernel(x, weight, bias, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            x, weight = ctx.saved_tensors
            need_x = ctx.needs_input_grad[0]
            need_w = ctx.needs_input_grad[1]
            need_b = ctx.needs_input_grad[2]
            if not (need_x or need_w or need_b):
                return None, None, None
            grad_x = torch.empty_like(x) if need_x else None
            grad_w = torch.empty_like(weight) if need_w else None
            grad_bias = torch.empty(1, weight.shape[1], device=x.device,
                                    dtype=x.dtype) if need_b else None
            tmp_x = grad_x if need_x else torch.empty_like(x)
            tmp_w = grad_w if need_w else torch.empty_like(weight)
            tmp_b = grad_bias if need_b else torch.empty(
                1, weight.shape[1], device=x.device, dtype=x.dtype)
            bwd_kernel(x.contiguous(), weight.contiguous(),
                       grad_c.contiguous(), tmp_x, tmp_w, tmp_b)
            if need_b:
                grad_bias = grad_bias.view(-1)
            return grad_x, grad_w, grad_bias

    return PyPTOLinearFusedImpl


# ---------------------------------------------------------------------------
# rnn_hidden 融合版：c = tanh(a + b + bias)，单 kernel（替代 add+bias_add+tanh
#    三次启动）。反向：grad_a = grad_b = grad_c·(1-y²)（逐元素），
#    grad_bias = sum 行归约。
#    依赖 pypto.tanh（0.2.1+），仅在 0.2.1+ 启用融合路径。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rnn_hidden_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    # pypto.tanh 内部临时 workspace 约 4 倍 tile 数据量，行 tile 收窄为 64
    pypto.set_vec_tile_shapes(128, 64)
    summed = pypto.add(a, b)
    biased = pypto.add(summed, bias)
    c.move(pypto.tanh(biased))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rnn_hidden_bwd_kernel(
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_bias: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 64)
    # move 会消费中间张量（move 后原 tensor 为 nullptr），
    # 三个输出各计算一份 dy = grad_c·(1-y²)，避免复用
    one_minus_y2 = pypto.neg(pypto.sub(pypto.mul(y, y), 1.0))
    grad_a.move(pypto.mul(grad_c, one_minus_y2))
    one_minus_y2_2 = pypto.neg(pypto.sub(pypto.mul(y, y), 1.0))
    grad_b.move(pypto.mul(grad_c, one_minus_y2_2))
    one_minus_y2_3 = pypto.neg(pypto.sub(pypto.mul(y, y), 1.0))
    grad_bias.move(pypto.sum(pypto.mul(grad_c, one_minus_y2_3), 0))


def make_pypto_rnn_hidden(fwd_kernel, bwd_kernel):
    """创建 RNN 隐层融合算子：c = tanh(a + b + bias)。"""

    class PyPTORNNHiddenImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b, bias):
            c = torch.empty_like(a)
            fwd_kernel(a, b, bias, c)
            ctx.save_for_backward(c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            (c,) = ctx.saved_tensors
            need_a = ctx.needs_input_grad[0]
            need_b = ctx.needs_input_grad[1]
            need_bias = ctx.needs_input_grad[2]
            grad_a = torch.empty_like(c) if need_a else None
            grad_b = torch.empty_like(c) if need_b else None
            grad_bias = torch.empty(c.shape[1], device=c.device,
                                    dtype=c.dtype) if need_bias else None
            tmp_a = grad_a if need_a else torch.empty_like(c)
            tmp_b = grad_b if need_b else torch.empty_like(c)
            tmp_bias = grad_bias if need_bias else torch.empty(
                c.shape[1], device=c.device, dtype=c.dtype)
            bwd_kernel(c.contiguous(), grad_c.contiguous(),
                       tmp_a, tmp_b, tmp_bias)
            return grad_a, grad_b, grad_bias

    return PyPTORNNHiddenImpl


# ---------------------------------------------------------------------------
# RNN 整段展开（0.2.1+ 深度优化路径）：
#   前向：整序列投影 X@W_xh 一次算完（复用 matmul_fwd_kernel，reshape 为
#         零拷贝视图）+ 每时间步单个融合 cell kernel
#         h_t = tanh(matmul(h_{t-1}, W_hh) + xw_t + b_h)
#   反向：每时间步单个 kernel 求 dz_t = dh_t·(1-h_t²) 与
#         dh_{t-1} = dz_t @ W_hhᵀ，并把 dz_t 写入整序列缓冲；随后权梯度
#         用"大 GEMM"各一次算完：
#         dW_xh = Xᵀ @ dz、dW_hh = hprevᵀ @ dz（Σ_t h_{t-1}ᵀdz_t 的矩阵化）、
#         db_h = sum(dz, dim=0)（复用 bias_add_bwd_kernel）
#   相比逐步 autograd（每步 3 次发射 + 逐步梯度累积），发射次数约 220 → 78/step。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rnn_cell_fwd_kernel(
    h_prev: pypto.Tensor([], pypto.DT_FP32),
    w_hh: pypto.Tensor([], pypto.DT_FP32),
    xw: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    h_out: pypto.Tensor([], pypto.DT_FP32),
):
    # 融合 matmul + 双加法 + tanh 单 kernel（先例：linear_fwd_kernel 的
    # cube+vec 融合与 rnn_hidden 的 add/add/tanh 融合）
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    hw = pypto.matmul(h_prev, w_hh, pypto.DT_FP32)
    pypto.set_vec_tile_shapes(128, 64)
    h_out.move(pypto.tanh((hw + xw) + b))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rnn_cell_bwd_kernel(
    h_out: pypto.Tensor([], pypto.DT_FP32),
    dy: pypto.Tensor([], pypto.DT_FP32),
    dh_in: pypto.Tensor([], pypto.DT_FP32),
    w_hh: pypto.Tensor([], pypto.DT_FP32),
    dz_out: pypto.Tensor([], pypto.DT_FP32),
    dh_out: pypto.Tensor([], pypto.DT_FP32),
):
    # dh_total = dy + dh_in；dz = dh_total·(1-h²)
    # dh_out = dz @ W_hhᵀ；dz_out 写入整序列缓冲（供事后大 GEMM 求权梯度）
    # 注意：move 会消费中间量，每个消费者各算一份 dz
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(128, 64)
    dh_total = (dy + dh_in)
    om1 = pypto.neg(pypto.sub(pypto.mul(h_out, h_out), 1.0))
    dz_a = pypto.mul(dh_total, om1)
    dh_out.move(pypto.matmul(dz_a, w_hh, pypto.DT_FP32, b_trans=True))
    om1_b = pypto.neg(pypto.sub(pypto.mul(h_out, h_out), 1.0))
    dz_b = pypto.mul(dh_total, om1_b)
    dz_out.move(dz_b)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def wgrad_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    # out = aᵀ @ b（权梯度：dW = 输入ᵀ @ 上游梯度）
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out.move(pypto.matmul(a, b, pypto.DT_FP32, a_trans=True))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def btrans_matmul_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    # out = x @ wᵀ（仅在需要 grad wrt X 时用于 dX = dz @ W_xhᵀ）
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out.move(pypto.matmul(x, w, pypto.DT_FP32, b_trans=True))


class PyPTORNNUnrollFunction(torch.autograd.Function):
    """整段展开的 RNN 前向/反向（单自定义 Function，内部手写 BPTT）。

    forward: X (seq, batch, input_size) one-hot 或任意浮点输入，
             h0 (batch, hidden) → Y (seq, batch, hidden)
    backward: dY → (dX, dW_xh, dW_hh, db_h, dh0)
    """

    @staticmethod
    def forward(ctx, X, W_xh, W_hh, b_h, h0):
        T_, B_, V_ = X.shape
        H_ = W_hh.shape[1]
        # 整序列投影：X (T,B,V) → (T*B,V) @ (V,H)，一次发射
        x_flat = X.reshape(T_ * B_, V_)
        xw = torch.empty(T_ * B_, H_, device=X.device, dtype=X.dtype)
        matmul_fwd_kernel(x_flat, W_xh, xw)
        # 按时间步递推：注意平铺缓冲必须经 view(T,B,H) 取时间步块，
        # 直接 xw[t] 取的是单行（kernel 不报错但读到越界数据块）
        xw3 = xw.view(T_, B_, H_)
        Y = torch.empty(T_, B_, H_, device=X.device, dtype=X.dtype)
        h = h0
        for t in range(T_):
            rnn_cell_fwd_kernel(h.contiguous(), W_hh, xw3[t], b_h, Y[t])
            h = Y[t]
        ctx.save_for_backward(X, W_xh, W_hh, Y, h0)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        X, W_xh, W_hh, Y, h0 = ctx.saved_tensors
        T_, B_, V_ = X.shape
        H_ = W_hh.shape[1]
        need_x = ctx.needs_input_grad[0]
        need_wxh = ctx.needs_input_grad[1]
        need_whh = ctx.needs_input_grad[2]
        need_bh = ctx.needs_input_grad[3]
        need_h0 = ctx.needs_input_grad[4]

        dY = grad_Y.contiguous()
        dz = torch.empty(T_, B_, H_, device=X.device, dtype=X.dtype)
        dh = torch.zeros(B_, H_, device=X.device, dtype=X.dtype)
        dh0 = None
        for t in range(T_ - 1, -1, -1):
            dh_new = torch.empty(B_, H_, device=X.device, dtype=X.dtype)
            rnn_cell_bwd_kernel(Y[t], dY[t], dh, W_hh, dz[t], dh_new)
            dh = dh_new
            if t == 0:
                dh0 = dh_new  # 传到初始隐状态的梯度

        dz_flat = dz.reshape(T_ * B_, H_)
        dW_xh = dW_hh = db_h = None
        if need_wxh:
            dW_xh = torch.empty(V_, H_, device=X.device, dtype=X.dtype)
            wgrad_kernel(X.reshape(T_ * B_, V_), dz_flat, dW_xh)
        if need_whh:
            hprev = torch.cat([h0, Y[:-1].reshape(-1, H_)], dim=0)
            dW_hh = torch.empty(H_, H_, device=X.device, dtype=X.dtype)
            wgrad_kernel(hprev, dz_flat, dW_hh)
        if need_bh:
            db_h = torch.empty(H_, device=X.device, dtype=X.dtype)
            bias_add_bwd_kernel(dz_flat, dz_flat, db_h)
        dX = None
        if need_x:
            dX = torch.empty(T_ * B_, V_, device=X.device, dtype=X.dtype)
            btrans_matmul_kernel(dz_flat, W_xh, dX)
            dX = dX.view(T_, B_, V_)
        return dX, dW_xh, dW_hh, db_h, dh0


# ---------------------------------------------------------------------------
# bias_add：MxN + N → MxN，广播
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def bias_add_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.add(a, b))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def bias_add_bwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_b.move(pypto.sum(grad_c, 0))


def make_pypto_bias_add(fwd_kernel, bwd_kernel):
    """创建 PyPTO 偏置加法算子 (MxN + N -> MxN)。"""

    class PyPTOBiasAddImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty_like(a)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            a, _ = ctx.saved_tensors
            need_grad_a = ctx.needs_input_grad[0]
            need_grad_b = ctx.needs_input_grad[1]
            grad_a = grad_c if need_grad_a else None
            grad_b = torch.empty(a.shape[1], device=a.device,
                                 dtype=a.dtype) if need_grad_b else None
            if need_grad_b:
                bwd_kernel(a.contiguous(), grad_c.contiguous(), grad_b)
            return grad_a, grad_b

    return PyPTOBiasAddImpl


# ---------------------------------------------------------------------------
# add：MxN + MxN → MxN，同形状逐元素加法（反向恒等映射，无需 NPU 内核）
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.add(a, b))


def make_pypto_add(fwd_kernel):
    """创建 PyPTO 同形状逐元素加法算子 (MxN + MxN -> MxN)。
       add(a,b) 的反向：grad_a = grad_c, grad_b = grad_c。"""

    class PyPTOAddImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty_like(a)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            grad_a = grad_c if ctx.needs_input_grad[0] else None
            grad_b = grad_c if ctx.needs_input_grad[1] else None
            return grad_a, grad_b

    return PyPTOAddImpl


# ---------------------------------------------------------------------------
# tanh：原生 pypto.tanh（v0.2.1+），早期版本用 2·sigmoid(2x)-1 组合
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def tanh_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    # pypto.tanh 内部临时 workspace 约 4 倍 tile 数据量，
    # 行 tile 收窄为 64 控制 UB 占用（同第 4 章实测结论）
    pypto.set_vec_tile_shapes(128, 64)
    b.move(pypto.tanh(a))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def tanh_bwd_kernel(
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 64)
    b_sq = pypto.mul(b, b)
    one_minus_b2 = pypto.neg(pypto.sub(b_sq, 1.0))
    grad_a.move(pypto.mul(grad_b, one_minus_b2))


def make_pypto_tanh(fwd_kernel, bwd_kernel):
    """创建 PyPTO tanh 激活算子。"""

    class PyPTOTanhImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            b = torch.empty_like(a)
            fwd_kernel(a, b)
            ctx.save_for_backward(b)
            return b

        @staticmethod
        def backward(ctx, grad_b):
            (b,) = ctx.saved_tensors
            grad_a = torch.empty_like(b) if ctx.needs_input_grad[0] else None
            if grad_a is not None:
                bwd_kernel(b.contiguous(), grad_b.contiguous(), grad_a)
            return grad_a

    return PyPTOTanhImpl


# ---------------------------------------------------------------------------
# relu：y = max(x, 0)，反向梯度 = grad_out if y > 0 else 0
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def relu_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    y.move(pypto.maximum(x, 0.0))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def relu_bwd_kernel(
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    grad_in: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_in.move(pypto.where(y > 0.0, grad_out, 0.0))


def make_pypto_relu(fwd_kernel, bwd_kernel):
    """创建 PyPTO ReLU 激活算子。"""

    class PyPTOReLUOpImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            y = torch.empty_like(x)
            fwd_kernel(x, y)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            (y,) = ctx.saved_tensors
            grad_in = torch.empty_like(y) if ctx.needs_input_grad[0] else None
            if grad_in is not None:
                bwd_kernel(y.contiguous(), grad_out.contiguous(), grad_in)
            return grad_in

    return PyPTOReLUOpImpl


# ---------------------------------------------------------------------------
# softmax + cross_entropy：组合损失函数
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    # 行 tile 8→128（第 4/10 章实测 4.9x）；UB 护栏（0.2.0/0.2.1 实测）：
    # softmax 内部 DIV 约需 2×rows×cols×4B，
    # cols ≤ 160 → 128 行，cols ≤ 320 → 64 行，否则退回 8 行
    cols = ((num_classes + 7) // 8) * 8
    rows = 128 if cols <= 160 else (64 if cols <= 320 else 8)
    pypto.set_vec_tile_shapes(rows, cols)
    y.move(pypto.softmax(x, dim=-1))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def cross_entropy_fwd_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    indices: pypto.Tensor([], pypto.DT_INT32),
    out: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    cols = ((num_classes + 7) // 8) * 8
    rows = 128 if cols <= 160 else (64 if cols <= 320 else 8)
    pypto.set_vec_tile_shapes(rows, cols)
    gathered = pypto.gather(y_hat, 1, pypto.reshape(indices, (-1, 1)))
    gathered = pypto.reshape(gathered, (-1,))
    out.move(pypto.neg(pypto.log(pypto.maximum(gathered, 1e-12))))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_ce_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    y_1d: pypto.Tensor([], pypto.DT_INT32),
    softmax_out: pypto.Tensor([], pypto.DT_FP32),
    ce_out: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    # softmax → gather → -log 单发射（0.2.1+ 融合路径）。
    # 注意顺序：必须先 gather 消费中间量、再 move 写回同一中间量，
    # 顺序颠倒会触发 GatherElements nullptr（第 4 章调优报告坑）
    cols = ((num_classes + 7) // 8) * 8
    rows = 128 if cols <= 160 else (64 if cols <= 320 else 8)
    pypto.set_vec_tile_shapes(rows, cols)
    s = pypto.softmax(x, dim=-1)
    y_2d = pypto.reshape(y_1d, [-1, 1])
    selected = pypto.gather(s, -1, y_2d)
    softmax_out.move(s)
    ce_out.move(pypto.neg(pypto.log(pypto.maximum(selected, 1e-12))))


class PyPTOSoftmaxCrossEntropyLossFunction(torch.autograd.Function):
    """组合 Softmax + CrossEntropy 损失的自定义 autograd 函数。

    前向（0.2.1+）：softmax+gather+log 融合为单 kernel 发射；
    反向：链式法则代入化简为 ``(softmax - one_hot) * grad``，走 host
    实现——语言模型场景的 logits 形如 (num_steps*batch, vocab)（如
    1120×28），3 个小 aten 算子（one_hot/sub/mul，实测 ~0.08ms）比一次
    pypto kernel 发射（~0.13ms，含 ~94μs 固定启动开销）更快，且两者
    数值逐位一致。
    """

    @staticmethod
    def forward(ctx, x, y, num_classes):
        ctx.num_classes = num_classes
        y_i32 = y.to(torch.int32).reshape(-1)

        if _PYPTO_GE_021:
            softmax_out = torch.empty_like(x)
            ce_out = torch.empty(x.shape[0], 1, device=x.device,
                                 dtype=x.dtype)
            softmax_ce_fwd_kernel(x, y_i32, softmax_out, ce_out, num_classes)
        else:
            softmax_out = torch.empty_like(x)
            softmax_fwd_kernel(x, softmax_out, num_classes)
            ce_out = torch.empty(y_i32.shape[0], device=x.device,
                                 dtype=x.dtype)
            cross_entropy_fwd_kernel(softmax_out, y_i32, ce_out, num_classes)
            ce_out = ce_out.view(-1, 1)

        ctx.save_for_backward(softmax_out, y_i32)
        return ce_out

    @staticmethod
    def backward(ctx, grad_out):
        softmax_out, y_i32 = ctx.saved_tensors
        num_classes = ctx.num_classes

        one_hot = F.one_hot(
            y_i32.long(), num_classes).to(dtype=softmax_out.dtype)
        grad_x = grad_out * (softmax_out - one_hot)

        return grad_x, None, None


# =========================================================================
# 实例化所有算子（工厂函数调用）
# =========================================================================

PyPTOMatmul = make_pypto_matmul(matmul_fwd_kernel, matmul_bwd_kernel)
PyPTOBiasAdd = make_pypto_bias_add(bias_add_fwd_kernel, bias_add_bwd_kernel)
PyPTOAdd = make_pypto_add(add_fwd_kernel)
PyPTOTanh = make_pypto_tanh(tanh_fwd_kernel, tanh_bwd_kernel)
PyPTOReLUOp = make_pypto_relu(relu_fwd_kernel, relu_bwd_kernel)
PyPTOSoftmaxCrossEntropyLoss = PyPTOSoftmaxCrossEntropyLossFunction
# 融合算子（0.2.1+ 启用：原生 tanh + 数值稳定；0.2.0 回退分解路径）
PyPTOLinearFused = make_pypto_linear_fused(
    linear_fwd_kernel, linear_bias_bwd_kernel)
PyPTORNNHidden = make_pypto_rnn_hidden(
    rnn_hidden_fwd_kernel, rnn_hidden_bwd_kernel)


# =========================================================================
# nn.Module 封装
# =========================================================================

class PyPTOLinear(nn.Module):
    """用 PyPTO 算子实现的线性层 (与 nn.Linear 接口兼容)。"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if _PYPTO_GE_021 and self.bias is not None:
            # 融合路径：matmul+bias 单 kernel、三梯度反向单 kernel（0.2.1+）
            return PyPTOLinearFused.apply(x, self.weight, self.bias)
        y = PyPTOMatmul.apply(x, self.weight)
        if self.bias is not None:
            y = PyPTOBiasAdd.apply(y, self.bias)
        return y


class PyPTOReLU(nn.Module):
    """用 PyPTO ReLU 算子实现的 ReLU 激活层。"""

    def forward(self, x):
        return PyPTOReLUOp.apply(x)


class PyPTORNN(nn.Module):
    """PyPTO RNN 隐层（单层单向）。

    仅做隐状态递推：h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)。
    不含输出投影，与 nn.RNN 的隐藏层接口兼容，
    搭配 PyPTOLinear 即可组成完整语言模型。
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.bidirectional = False
        self.W_xh = nn.Parameter(
            torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, X, state=None):
        """前向传播。

        Args:
            X: 输入，shape (seq_len, batch, input_size)
            state: 初始隐状态 (1, batch, hidden_size) 或 None

        Returns:
            output: (seq_len, batch, hidden_size) 各时间步隐状态
            h_n: (1, batch, hidden_size) 最终隐状态
        """
        if _PYPTO_GE_021:
            # 整段展开路径（0.2.1+）：整序列投影 1 次 + 每时间步 1 个融合
            # cell kernel + 手写 BPTT（大 GEMM 求权梯度），实测训练 step ~3x
            if state is None:
                h0 = torch.zeros(X.shape[1], self.hidden_size,
                                 device=X.device, dtype=X.dtype)
            else:
                h0 = state.squeeze(0)
            Y = PyPTORNNUnrollFunction.apply(
                X.contiguous(), self.W_xh, self.W_hh, self.b_h, h0)
            return Y, Y[-1:]

        # 0.2.0 分解路径：matmul×2 + add + bias_add + tanh = 每步 5 次启动
        batch_size = X.shape[1]
        if state is None:
            h_flat = torch.zeros(batch_size, self.hidden_size,
                                 device=X.device, dtype=X.dtype)
        else:
            h_flat = state.squeeze(0)

        outputs = []
        for t in range(X.shape[0]):
            xw = PyPTOMatmul.apply(X[t], self.W_xh)
            hw = PyPTOMatmul.apply(h_flat, self.W_hh)
            summed = PyPTOAdd.apply(xw, hw)
            biased = PyPTOBiasAdd.apply(summed, self.b_h)
            h_flat = PyPTOTanh.apply(biased)
            outputs.append(h_flat)

        Y = torch.stack(outputs, dim=0)
        h_n = h_flat.unsqueeze(0)
        return Y, h_n


# =========================================================================
# 便利函数
# =========================================================================

def loss_fn(logits, y, num_classes=10):
    """计算 softmax cross-entropy 损失。

    Args:
        logits: shape [N, num_classes]
        y: shape [N] 整数标签
        num_classes: 类别数

    Returns:
        标量损失
    """
    return PyPTOSoftmaxCrossEntropyLossFunction.apply(
        logits, y, num_classes).mean()


__all__ = [
    "PyPTOMatmul", "PyPTOBiasAdd", "PyPTOAdd",
    "PyPTOTanh", "PyPTOReLUOp", "PyPTOSoftmaxCrossEntropyLoss",
    "PyPTOLinearFused", "PyPTORNNHidden",
    "PyPTOLinear", "PyPTOReLU", "PyPTORNN",
    "loss_fn",
]
