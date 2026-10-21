"""PyPTO NPU 算子库（第 9 章共享模块）。

本模块继承第 8 章的算子（matmul / bias_add / add / tanh / relu / softmax+CE /
PyPTOLinear / PyPTORNN），并新增第 9 章所需的算子：sigmoid / mul / sub /
PyPTOGRU / PyPTOLSTM。

提供的算子：

**基础算子（与第 8 章同构，每个算子独立 kernel）：**

| 算子 | 前向 PyPTO API | 反向 | tiling |
|------|---------------|------|--------|
| matmul | pypto.matmul | transpose variants | cube [32,32]×[64,64]×[64,64] |
| bias_add | pypto.add (broadcast) | sum over batch dim | vec (128,128) |
| add | pypto.add (same shape) | identity | vec (128,128) |
| tanh | pypto.sigmoid 组合 2σ(2x)-1 | (1-y²)*grad | vec (128,128) |
| relu | pypto.maximum | pypto.where | vec (128,128) |
| softmax+CE | pypto.softmax + gather/log | softmax - one_hot | vec (8, aligned) |
| sigmoid | pypto.sigmoid | y*(1-y)*grad | vec (128,128) |
| mul | pypto.mul | swap operands | vec (128,128) |
| sub | pypto.sub | identity / neg | vec (128,128) |
| linear_fused | pypto.matmul + add | matmul bwd + sum | cube [32,32]×[64,64]×[64,64] + vec |

**融合门控算子（GRU/LSTM 专用，性能调优产物）：**

| 算子 | 融合内容（等价公式） | 一次 kernel 替代的独立调用 |
|------|--------------------|--------------------------|
| PyPTOFusedSigmoidGate | y = σ(a + b + bias) | add + bias_add + sigmoid |
| PyPTOFusedTanhGate | y = tanh(a + b + bias) | add + bias_add + tanh |
| PyPTOFusedUpdate | z·h + (1-z)·h~ | mul + sub + mul + add |
| PyPTODualSigmoidGate | (Z, R) = (σ(...), σ(...)) 双门 | 2× 单门（kernel 内 slice） |
| PyPTOGateRH | (Z, R, R⊙H) 双门+重置乘法 | dual_gate + mul |
| PyPTOQuadGate | (I, F, O, C~) 四门 | 4× 单门（kernel 内 slice） |
| PyPTOFusedCellUpdate | f·c + i·c~ | mul + mul + add |
| PyPTOFusedCellTanhMul | (c_new, o·tanh(c_new)) | cell_update + tanh + mul |
| PyPTOFusedTanhMul | o·tanh(c) | tanh + mul |
| PyPTOLinearFused | x@w + bias | matmul + bias_add |

**融合动机（性能调优结论）**：第 9 章 workload 的瓶颈是 host 侧 kernel 启动开销
（单次调用 ~100-190us，无 sync 流水 99.6us/call），设备端 e2e 仅 ~50us。
将时间步内多次独立 kernel 调用合并为单个 kernel，使 GRU/LSTM 每时间步的
kernel 调用从 ~16 次降至 5/3 次，训练 step 性能提升 2.7~3.8 倍。

每个算子采用工厂函数模式：
1. 定义 @pypto.frontend.jit 内核 (fwd + bwd)
2. 包装为 torch.autograd.Function
3. 导出为 PyPTO<Op> 类，通过 .apply() 调用
4. 对有需要的算子提供 nn.Module 封装

导出清单：
- 基础算子：PyPTOMatmul / PyPTOBiasAdd / PyPTOAdd / PyPTOTanh / PyPTOReLUOp
  / PyPTOSigmoid / PyPTOMul / PyPTOSub / PyPTOLinearFused
- 融合门控算子：PyPTOFusedSigmoidGate / PyPTOFusedTanhGate / PyPTOFusedUpdate
  / PyPTODualSigmoidGate / PyPTOGateRH / PyPTOQuadGate / PyPTOFusedCellUpdate
  / PyPTOFusedCellTanhMul / PyPTOFusedTanhMul
- 模型封装：PyPTOReLU / PyPTOLinear / PyPTORNN / PyPTOGRU / PyPTOLSTM
- 损失：loss_fn / PyPTOSoftmaxCrossEntropyLoss
"""

import pypto
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# matmul：前向 C = A @ B，反向 grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
# ---------------------------------------------------------------------------

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def matmul_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    c.move(pypto.matmul(a, b, pypto.DT_FP32))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
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
# bias_add：MxN + N → MxN，广播
# ---------------------------------------------------------------------------

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def bias_add_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.add(a, b))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
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

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
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
# tanh：tanh(x) = 2·sigmoid(2x) - 1
# ---------------------------------------------------------------------------

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def tanh_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    two_a = pypto.mul(a, 2.0)
    s = pypto.sigmoid(two_a)
    two_s = pypto.mul(s, 2.0)
    b.move(pypto.sub(two_s, 1.0))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def tanh_bwd_kernel(
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
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

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def relu_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    y.move(pypto.maximum(x, 0.0))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
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
# sigmoid：y = 1 / (1 + e^(-x))，反向：grad_x = y · (1-y) · grad_y
# ---------------------------------------------------------------------------

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def sigmoid_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    b.move(pypto.sigmoid(a))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def sigmoid_bwd_kernel(
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    one_minus_b = pypto.neg(pypto.sub(b, 1.0))
    g_mul_b = pypto.mul(grad_b, b)
    grad_a.move(pypto.mul(g_mul_b, one_minus_b))


def make_pypto_sigmoid(fwd_kernel, bwd_kernel):
    """创建 PyPTO sigmoid 激活算子。"""

    class PyPTOSigmoidImpl(torch.autograd.Function):
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

    return PyPTOSigmoidImpl


# ---------------------------------------------------------------------------
# mul：逐元素乘法 c = a * b，反向：grad_a = grad_c * b, grad_b = grad_c * a
# ---------------------------------------------------------------------------

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def mul_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.mul(a, b))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def mul_bwd_kernel(
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_a.move(pypto.mul(grad_c, b))
    grad_b.move(pypto.mul(grad_c, a))


def make_pypto_mul(fwd_kernel, bwd_kernel):
    """创建 PyPTO 逐元素乘法算子 c = a * b。"""

    class PyPTOMulImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty_like(a)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            a, b = ctx.saved_tensors
            need_grad_a = ctx.needs_input_grad[0]
            need_grad_b = ctx.needs_input_grad[1]
            grad_a = torch.empty_like(a) if need_grad_a else None
            grad_b = torch.empty_like(b) if need_grad_b else None
            tmp_a = grad_a if need_grad_a else torch.empty_like(a)
            tmp_b = grad_b if need_grad_b else torch.empty_like(b)
            if need_grad_a or need_grad_b:
                bwd_kernel(grad_c.contiguous(), a.contiguous(),
                           b.contiguous(), tmp_a, tmp_b)
            return grad_a, grad_b

    return PyPTOMulImpl


# ---------------------------------------------------------------------------
# sub：逐元素减法 c = a - b，反向：grad_a = grad_c, grad_b = -grad_c
# ---------------------------------------------------------------------------

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def sub_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.sub(a, b))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def sub_bwd_grad_b_kernel(
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_b.move(pypto.neg(grad_c))


def make_pypto_sub(fwd_kernel, bwd_grad_b_kernel):
    """创建 PyPTO 逐元素减法算子 c = a - b。

       反向：grad_a = grad_c (恒等), grad_b = -grad_c。
    """

    class PyPTOSubImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty_like(a)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            need_grad_a = ctx.needs_input_grad[0]
            need_grad_b = ctx.needs_input_grad[1]
            grad_a = grad_c if need_grad_a else None
            grad_b = torch.empty_like(grad_c) if need_grad_b else None
            if need_grad_b:
                bwd_grad_b_kernel(grad_c.contiguous(), grad_b)
            return grad_a, grad_b

    return PyPTOSubImpl


# ---------------------------------------------------------------------------
# 融合门控算子（第 9 章 GRU/LSTM 专用）：
#   y = σ(a + b + bias)             → PyPTOFusedSigmoidGate
#   y = tanh(a + b + bias)          → PyPTOFusedTanhGate
#   out = z·h + (1-z)·h_tilde       → PyPTOFusedUpdate
#   out = f·c + i·c_tilde           → PyPTOFusedCellUpdate
#   out = o·tanh(c)                 → PyPTOFusedTanhMul
# 将 RNN 时间步内多个独立 kernel 调用合并为单个 kernel，减少 host launch 开销。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def gate_sigmoid_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    s = pypto.add(pypto.add(a, b), bias)
    y.move(pypto.sigmoid(s))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def gate_sigmoid_bwd_kernel(
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_y: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_bias: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    one_minus_y = pypto.neg(pypto.sub(y, 1.0))
    g = pypto.mul(pypto.mul(grad_y, y), one_minus_y)
    grad_bias.move(pypto.sum(g, 0))
    grad_a.move(g)


def make_pypto_gate_sigmoid(fwd_kernel, bwd_kernel):
    """创建融合 sigmoid 门控算子 y = σ(a + b + bias)。

    反向：grad_a = grad_b = g, grad_bias = sum(g, 0)
    其中 g = y·(1-y)·grad_y。

    ⚠️ 中间优化版本：当前 GRU 使用 PyPTOGateRH（双门+重置乘法合一），
    本算子保留作融合思路的教学对照（单门融合的最小示例）。
    """

    class PyPTOFusedSigmoidGateImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b, bias):
            y = torch.empty_like(a)
            fwd_kernel(a, b, bias, y)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            (y,) = ctx.saved_tensors
            need_a = ctx.needs_input_grad[0]
            need_b = ctx.needs_input_grad[1]
            need_bias = ctx.needs_input_grad[2]
            grad_a = torch.empty_like(y) if need_a else None
            grad_bias = torch.empty(y.shape[1], device=y.device,
                                    dtype=y.dtype) if need_bias else None
            # PyPTO kernel 要求全部输出参数存在：不需要梯度的输入用临时缓冲占位；
            # grad_a 与 grad_b 数学上相等，故 grad_b 直接复用 tmp_a（零开销）
            tmp_a = grad_a if need_a else torch.empty_like(y)
            tmp_bias = grad_bias if need_bias else torch.empty(
                y.shape[1], device=y.device, dtype=y.dtype)
            if need_a or need_b or need_bias:
                bwd_kernel(y.contiguous(), grad_y.contiguous(), tmp_a, tmp_bias)
            grad_b = tmp_a if need_b else None
            return grad_a, grad_b, grad_bias

    return PyPTOFusedSigmoidGateImpl


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def gate_tanh_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    two_s = pypto.mul(pypto.add(pypto.add(a, b), bias), 2.0)
    s = pypto.sigmoid(two_s)
    y.move(pypto.sub(pypto.mul(s, 2.0), 1.0))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def gate_tanh_bwd_kernel(
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_y: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_bias: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    y_sq = pypto.mul(y, y)
    one_minus = pypto.neg(pypto.sub(y_sq, 1.0))
    g = pypto.mul(grad_y, one_minus)
    grad_bias.move(pypto.sum(g, 0))
    grad_a.move(g)


def make_pypto_gate_tanh(fwd_kernel, bwd_kernel):
    """创建融合 tanh 门控算子 y = tanh(a + b + bias)。

    反向：grad_a = grad_b = g, grad_bias = sum(g, 0)
    其中 g = (1-y²)·grad_y。
    """

    class PyPTOFusedTanhGateImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b, bias):
            y = torch.empty_like(a)
            fwd_kernel(a, b, bias, y)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            (y,) = ctx.saved_tensors
            need_a = ctx.needs_input_grad[0]
            need_b = ctx.needs_input_grad[1]
            need_bias = ctx.needs_input_grad[2]
            grad_a = torch.empty_like(y) if need_a else None
            grad_bias = torch.empty(y.shape[1], device=y.device,
                                    dtype=y.dtype) if need_bias else None
            tmp_a = grad_a if need_a else torch.empty_like(y)
            tmp_bias = grad_bias if need_bias else torch.empty(
                y.shape[1], device=y.device, dtype=y.dtype)
            if need_a or need_b or need_bias:
                bwd_kernel(y.contiguous(), grad_y.contiguous(), tmp_a, tmp_bias)
            grad_b = tmp_a if need_b else None
            return grad_a, grad_b, grad_bias

    return PyPTOFusedTanhGateImpl


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def update_fused_fwd_kernel(
    z: pypto.Tensor([], pypto.DT_FP32),
    h: pypto.Tensor([], pypto.DT_FP32),
    ht: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    one_minus_z = pypto.neg(pypto.sub(z, 1.0))
    out.move(pypto.add(pypto.mul(z, h), pypto.mul(one_minus_z, ht)))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def update_fused_bwd_kernel(
    z: pypto.Tensor([], pypto.DT_FP32),
    h: pypto.Tensor([], pypto.DT_FP32),
    ht: pypto.Tensor([], pypto.DT_FP32),
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    grad_z: pypto.Tensor([], pypto.DT_FP32),
    grad_h: pypto.Tensor([], pypto.DT_FP32),
    grad_ht: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    one_minus_z = pypto.neg(pypto.sub(z, 1.0))
    grad_h.move(pypto.mul(z, grad_out))
    grad_ht.move(pypto.mul(one_minus_z, grad_out))
    grad_z.move(pypto.mul(grad_out, pypto.sub(h, ht)))


def make_pypto_update(fwd_kernel, bwd_kernel):
    """创建融合状态更新算子 out = z·h + (1-z)·h_tilde。

    反向：
      grad_z = (h - h_tilde)·grad_out
      grad_h = z·grad_out
      grad_h_tilde = (1-z)·grad_out
    """

    class PyPTOFusedUpdateImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z, h, ht):
            out = torch.empty_like(z)
            fwd_kernel(z, h, ht, out)
            ctx.save_for_backward(z, h, ht)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            z, h, ht = ctx.saved_tensors
            grad_z = torch.empty_like(z) if ctx.needs_input_grad[0] else None
            grad_h = torch.empty_like(h) if ctx.needs_input_grad[1] else None
            grad_ht = torch.empty_like(ht) if ctx.needs_input_grad[2] else None
            tmp_z = grad_z if grad_z is not None else torch.empty_like(z)
            tmp_h = grad_h if grad_h is not None else torch.empty_like(h)
            tmp_ht = grad_ht if grad_ht is not None else torch.empty_like(ht)
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                bwd_kernel(z.contiguous(), h.contiguous(), ht.contiguous(),
                           grad_out.contiguous(), tmp_z, tmp_h, tmp_ht)
            return grad_z, grad_h, grad_ht

    return PyPTOFusedUpdateImpl


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def dual_gate_fwd_kernel(
    xw_z: pypto.Tensor([], pypto.DT_FP32),
    xw_r: pypto.Tensor([], pypto.DT_FP32),
    hw_zr: pypto.Tensor([], pypto.DT_FP32),
    b_z: pypto.Tensor([], pypto.DT_FP32),
    b_r: pypto.Tensor([], pypto.DT_FP32),
    z: pypto.Tensor([], pypto.DT_FP32),
    r: pypto.Tensor([], pypto.DT_FP32),
    H: int,
):
    pypto.set_vec_tile_shapes(128, 128)
    z.move(pypto.sigmoid(pypto.add(pypto.add(xw_z, hw_zr[:, :H]), b_z)))
    r.move(pypto.sigmoid(pypto.add(pypto.add(xw_r, hw_zr[:, H:2 * H]), b_r)))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def dual_gate_bwd_kernel(
    z: pypto.Tensor([], pypto.DT_FP32),
    r: pypto.Tensor([], pypto.DT_FP32),
    grad_z: pypto.Tensor([], pypto.DT_FP32),
    grad_r: pypto.Tensor([], pypto.DT_FP32),
    g_z: pypto.Tensor([], pypto.DT_FP32),
    g_r: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_hw: pypto.Tensor([], pypto.DT_FP32),
    H: int,
):
    pypto.set_vec_tile_shapes(128, 128)
    one_minus_z = pypto.neg(pypto.sub(z, 1.0))
    gz = pypto.mul(pypto.mul(grad_z, z), one_minus_z)
    one_minus_r = pypto.neg(pypto.sub(r, 1.0))
    gr = pypto.mul(pypto.mul(grad_r, r), one_minus_r)
    g_full = pypto.concat([gz, gr], 1)
    grad_b.move(pypto.sum(g_full, 0))
    grad_hw.move(g_full)
    g_z.move(gz)
    g_r.move(gr)


def make_pypto_dual_gate(fwd_kernel, bwd_kernel):
    """创建 GRU 合并双门算子 (Z, R) = (σ(xw_z+hw_z+b_z), σ(xw_r+hw_r+b_r))。

    ⚠️ 中间优化版本：当前 GRU 使用 PyPTOGateRH（双门+重置乘法合一），
    本算子保留作融合思路的教学对照（多输出 kernel 的示例）。

    hw_zr 为 (batch, 2H) 整块，kernel 内按列 slice，避免 host 侧两次 copy。
    反向：
      g_z = z·(1-z)·grad_z, g_r = r·(1-r)·grad_r
      grad_xw_z = g_z, grad_xw_r = g_r
      grad_hw_zr = concat(g_z, g_r, 1)
      grad_b_z = sum(g_z,0), grad_b_r = sum(g_r,0)
    """

    class PyPTODualSigmoidGateImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xw_z, xw_r, hw_zr, b_z, b_r):
            H = xw_z.shape[1]
            z = torch.empty_like(xw_z)
            r = torch.empty_like(xw_r)
            fwd_kernel(xw_z, xw_r, hw_zr, b_z, b_r, z, r, H)
            ctx.save_for_backward(z, r, hw_zr)
            return z, r

        @staticmethod
        def backward(ctx, grad_z, grad_r):
            z, r, hw_zr = ctx.saved_tensors
            H = z.shape[1]
            need_xw_z = ctx.needs_input_grad[0]
            need_xw_r = ctx.needs_input_grad[1]
            need_hw = ctx.needs_input_grad[2]
            need_bz = ctx.needs_input_grad[3]
            need_br = ctx.needs_input_grad[4]
            g_z = torch.empty_like(z)
            g_r = torch.empty_like(r)
            grad_hw = torch.empty_like(hw_zr) if need_hw else None
            tmp_hw = grad_hw if need_hw else torch.empty_like(hw_zr)
            grad_b = torch.empty(2 * H, device=z.device, dtype=z.dtype)
            if need_xw_z or need_xw_r or need_hw or need_bz or need_br:
                bwd_kernel(z.contiguous(), r.contiguous(),
                           grad_z.contiguous(), grad_r.contiguous(),
                           g_z, g_r, grad_b, tmp_hw, H)
            grad_xw_z = g_z if need_xw_z else None
            grad_xw_r = g_r if need_xw_r else None
            grad_b_z = grad_b[:H] if need_bz else None
            grad_b_r = grad_b[H:] if need_br else None
            return grad_xw_z, grad_xw_r, grad_hw, grad_b_z, grad_b_r

    return PyPTODualSigmoidGateImpl


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def quad_gate_fwd_kernel(
    xw_i: pypto.Tensor([], pypto.DT_FP32),
    xw_f: pypto.Tensor([], pypto.DT_FP32),
    xw_o: pypto.Tensor([], pypto.DT_FP32),
    xw_c: pypto.Tensor([], pypto.DT_FP32),
    hw: pypto.Tensor([], pypto.DT_FP32),          # (batch, 4H)
    b_i: pypto.Tensor([], pypto.DT_FP32),
    b_f: pypto.Tensor([], pypto.DT_FP32),
    b_o: pypto.Tensor([], pypto.DT_FP32),
    b_c: pypto.Tensor([], pypto.DT_FP32),
    i_out: pypto.Tensor([], pypto.DT_FP32),
    f_out: pypto.Tensor([], pypto.DT_FP32),
    o_out: pypto.Tensor([], pypto.DT_FP32),
    c_out: pypto.Tensor([], pypto.DT_FP32),
    H: int,
):
    pypto.set_vec_tile_shapes(128, 128)
    i_out.move(pypto.sigmoid(
        pypto.add(pypto.add(xw_i, hw[:, :H]), b_i)))
    f_out.move(pypto.sigmoid(
        pypto.add(pypto.add(xw_f, hw[:, H:2 * H]), b_f)))
    o_out.move(pypto.sigmoid(
        pypto.add(pypto.add(xw_o, hw[:, 2 * H:3 * H]), b_o)))
    two_s = pypto.mul(pypto.add(pypto.add(xw_c, hw[:, 3 * H:4 * H]), b_c), 2.0)
    s = pypto.sigmoid(two_s)
    c_out.move(pypto.sub(pypto.mul(s, 2.0), 1.0))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def quad_gate_bwd_kernel(
    i: pypto.Tensor([], pypto.DT_FP32),
    f: pypto.Tensor([], pypto.DT_FP32),
    o: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
    grad_i: pypto.Tensor([], pypto.DT_FP32),
    grad_f: pypto.Tensor([], pypto.DT_FP32),
    grad_o: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    g_i: pypto.Tensor([], pypto.DT_FP32),
    g_f: pypto.Tensor([], pypto.DT_FP32),
    g_o: pypto.Tensor([], pypto.DT_FP32),
    g_c: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),       # (4H,)
    grad_hw: pypto.Tensor([], pypto.DT_FP32),      # (batch, 4H)
    H: int,
):
    pypto.set_vec_tile_shapes(128, 128)
    one_minus_i = pypto.neg(pypto.sub(i, 1.0))
    gi = pypto.mul(pypto.mul(grad_i, i), one_minus_i)
    one_minus_f = pypto.neg(pypto.sub(f, 1.0))
    gf = pypto.mul(pypto.mul(grad_f, f), one_minus_f)
    one_minus_o = pypto.neg(pypto.sub(o, 1.0))
    go = pypto.mul(pypto.mul(grad_o, o), one_minus_o)
    c_sq = pypto.mul(c, c)
    one_minus_c2 = pypto.neg(pypto.sub(c_sq, 1.0))
    gc = pypto.mul(grad_c, one_minus_c2)
    g_full = pypto.concat([gi, gf, go, gc], 1)
    grad_b.move(pypto.sum(g_full, 0))
    grad_hw.move(g_full)
    g_i.move(gi)
    g_f.move(gf)
    g_o.move(go)
    g_c.move(gc)


def make_pypto_quad_gate(fwd_kernel, bwd_kernel):
    """创建 LSTM 合并四门算子 (I,F,O,C~) 一次 kernel 计算。

    hw 为 (batch, 4H) 整块，kernel 内按列 slice，避免 host 侧四次 copy。
    反向：
      g_i = i(1-i)grad_i, g_f = f(1-f)grad_f, g_o = o(1-o)grad_o
      g_c = (1-c²)grad_c
      grad_xw_* = g_*；grad_hw = concat(g_i,g_f,g_o,g_c,1)；grad_b_* = sum(g_*,0)
    """

    class PyPTOQuadGateImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xw_i, xw_f, xw_o, xw_c, hw, b_i, b_f, b_o, b_c):
            H = xw_i.shape[1]
            i_out = torch.empty_like(xw_i)
            f_out = torch.empty_like(xw_f)
            o_out = torch.empty_like(xw_o)
            c_out = torch.empty_like(xw_c)
            fwd_kernel(xw_i, xw_f, xw_o, xw_c, hw, b_i, b_f, b_o, b_c,
                       i_out, f_out, o_out, c_out, H)
            ctx.save_for_backward(i_out, f_out, o_out, c_out, hw)
            return i_out, f_out, o_out, c_out

        @staticmethod
        def backward(ctx, grad_i, grad_f, grad_o, grad_c):
            i, f, o, c, hw = ctx.saved_tensors
            H = i.shape[1]
            needs = ctx.needs_input_grad
            g_i = torch.empty_like(i)
            g_f = torch.empty_like(f)
            g_o = torch.empty_like(o)
            g_c = torch.empty_like(c)
            grad_hw = torch.empty_like(hw) if needs[4] else None
            tmp_hw = grad_hw if needs[4] else torch.empty_like(hw)
            grad_b = torch.empty(4 * H, device=i.device, dtype=i.dtype)
            if any(needs):
                bwd_kernel(i.contiguous(), f.contiguous(), o.contiguous(),
                           c.contiguous(), grad_i.contiguous(),
                           grad_f.contiguous(), grad_o.contiguous(),
                           grad_c.contiguous(), g_i, g_f, g_o, g_c,
                           grad_b, tmp_hw, H)
            grad_xw_i = g_i if needs[0] else None
            grad_xw_f = g_f if needs[1] else None
            grad_xw_o = g_o if needs[2] else None
            grad_xw_c = g_c if needs[3] else None
            grad_b_i = grad_b[:H] if needs[5] else None
            grad_b_f = grad_b[H:2 * H] if needs[6] else None
            grad_b_o = grad_b[2 * H:3 * H] if needs[7] else None
            grad_b_c = grad_b[3 * H:] if needs[8] else None
            return (grad_xw_i, grad_xw_f, grad_xw_o, grad_xw_c, grad_hw,
                    grad_b_i, grad_b_f, grad_b_o, grad_b_c)

    return PyPTOQuadGateImpl


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def gate_rh_fwd_kernel(
    xw_z: pypto.Tensor([], pypto.DT_FP32),
    xw_r: pypto.Tensor([], pypto.DT_FP32),
    hw_zr: pypto.Tensor([], pypto.DT_FP32),
    b_z: pypto.Tensor([], pypto.DT_FP32),
    b_r: pypto.Tensor([], pypto.DT_FP32),
    h_state: pypto.Tensor([], pypto.DT_FP32),
    z: pypto.Tensor([], pypto.DT_FP32),
    r: pypto.Tensor([], pypto.DT_FP32),
    rh: pypto.Tensor([], pypto.DT_FP32),
    H: int,
):
    pypto.set_vec_tile_shapes(128, 128)
    z.move(pypto.sigmoid(pypto.add(pypto.add(xw_z, hw_zr[:, :H]), b_z)))
    r.move(pypto.sigmoid(pypto.add(pypto.add(xw_r, hw_zr[:, H:2 * H]), b_r)))
    rh.move(pypto.mul(r, h_state))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def gate_rh_bwd_kernel(
    z: pypto.Tensor([], pypto.DT_FP32),
    r: pypto.Tensor([], pypto.DT_FP32),
    h_state: pypto.Tensor([], pypto.DT_FP32),
    grad_z: pypto.Tensor([], pypto.DT_FP32),
    grad_r: pypto.Tensor([], pypto.DT_FP32),
    grad_rh: pypto.Tensor([], pypto.DT_FP32),
    g_z: pypto.Tensor([], pypto.DT_FP32),
    g_r: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_hw: pypto.Tensor([], pypto.DT_FP32),
    grad_h_state: pypto.Tensor([], pypto.DT_FP32),
    H: int,
):
    pypto.set_vec_tile_shapes(128, 128)
    one_minus_z = pypto.neg(pypto.sub(z, 1.0))
    gz = pypto.mul(pypto.mul(grad_z, z), one_minus_z)
    gr_in = pypto.add(grad_r, pypto.mul(h_state, grad_rh))
    one_minus_r = pypto.neg(pypto.sub(r, 1.0))
    gr = pypto.mul(pypto.mul(gr_in, r), one_minus_r)
    g_full = pypto.concat([gz, gr], 1)
    grad_b.move(pypto.sum(g_full, 0))
    grad_hw.move(g_full)
    g_z.move(gz)
    g_r.move(gr)
    grad_h_state.move(pypto.mul(r, grad_rh))


def make_pypto_gate_rh(fwd_kernel, bwd_kernel):
    """创建 GRU 融合门+重置乘法算子 (Z, R, rh) = (σ(...), σ(...), R·H)。

    一次 kernel 完成双门计算和重置门乘法，减少 host launch 次数。
    反向：
      gz = z(1-z)·grad_z；gr_in = grad_r + H·grad_rh；gr = r(1-r)·gr_in
      grad_xw_z = gz, grad_xw_r = gr, grad_hw_zr = concat(gz,gr,1)
      grad_b_z = sum(gz,0), grad_b_r = sum(gr,0)
      grad_h_state = r·grad_rh
    """

    class PyPTOGateRHImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xw_z, xw_r, hw_zr, b_z, b_r, h_state):
            H = xw_z.shape[1]
            z = torch.empty_like(xw_z)
            r = torch.empty_like(xw_r)
            rh = torch.empty_like(h_state)
            fwd_kernel(xw_z, xw_r, hw_zr, b_z, b_r, h_state, z, r, rh, H)
            ctx.save_for_backward(z, r, h_state, hw_zr)
            return z, r, rh

        @staticmethod
        def backward(ctx, grad_z, grad_r, grad_rh):
            z, r, h_state, hw_zr = ctx.saved_tensors
            H = z.shape[1]
            needs = ctx.needs_input_grad
            g_z = torch.empty_like(z)
            g_r = torch.empty_like(r)
            grad_hw = torch.empty_like(hw_zr) if needs[2] else None
            tmp_hw = grad_hw if needs[2] else torch.empty_like(hw_zr)
            grad_b = torch.empty(2 * H, device=z.device, dtype=z.dtype)
            grad_h_state = torch.empty_like(h_state) if needs[5] else None
            tmp_hs = grad_h_state if needs[5] else torch.empty_like(h_state)
            if any(needs):
                bwd_kernel(z.contiguous(), r.contiguous(),
                           h_state.contiguous(), grad_z.contiguous(),
                           grad_r.contiguous(), grad_rh.contiguous(),
                           g_z, g_r, grad_b, tmp_hw, tmp_hs, H)
            grad_xw_z = g_z if needs[0] else None
            grad_xw_r = g_r if needs[1] else None
            grad_b_z = grad_b[:H] if needs[3] else None
            grad_b_r = grad_b[H:] if needs[4] else None
            return grad_xw_z, grad_xw_r, grad_hw, grad_b_z, grad_b_r, grad_h_state

    return PyPTOGateRHImpl


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def cell_tanh_mul_fwd_kernel(
    f: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
    i: pypto.Tensor([], pypto.DT_FP32),
    ct: pypto.Tensor([], pypto.DT_FP32),
    o: pypto.Tensor([], pypto.DT_FP32),
    c_out: pypto.Tensor([], pypto.DT_FP32),
    h_out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c_new = pypto.add(pypto.mul(f, c), pypto.mul(i, ct))
    two_c = pypto.mul(c_new, 2.0)
    s = pypto.sigmoid(two_c)
    tanh_c = pypto.sub(pypto.mul(s, 2.0), 1.0)
    h_out.move(pypto.mul(o, tanh_c))
    c_out.move(c_new)


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def cell_tanh_mul_bwd_kernel(
    f: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
    i: pypto.Tensor([], pypto.DT_FP32),
    ct: pypto.Tensor([], pypto.DT_FP32),
    o: pypto.Tensor([], pypto.DT_FP32),
    c_new: pypto.Tensor([], pypto.DT_FP32),
    grad_c_out: pypto.Tensor([], pypto.DT_FP32),
    grad_h_out: pypto.Tensor([], pypto.DT_FP32),
    grad_f: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_i: pypto.Tensor([], pypto.DT_FP32),
    grad_ct: pypto.Tensor([], pypto.DT_FP32),
    grad_o: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    two_c = pypto.mul(c_new, 2.0)
    s = pypto.sigmoid(two_c)
    tanh_c = pypto.sub(pypto.mul(s, 2.0), 1.0)
    one_minus_t2 = pypto.neg(pypto.sub(pypto.mul(tanh_c, tanh_c), 1.0))
    # grad_c_new = grad_c_out + o·(1-tanh²)·grad_h_out
    grad_c_new = pypto.add(grad_c_out, pypto.mul(pypto.mul(o, one_minus_t2), grad_h_out))
    grad_f.move(pypto.mul(c, grad_c_new))
    grad_c.move(pypto.mul(f, grad_c_new))
    grad_i.move(pypto.mul(ct, grad_c_new))
    grad_ct.move(pypto.mul(i, grad_c_new))
    grad_o.move(pypto.mul(tanh_c, grad_h_out))


def make_pypto_cell_tanh_mul(fwd_kernel, bwd_kernel):
    """创建 LSTM 融合记忆元更新+隐状态输出算子
    (C_new, H_new) = (f·c + i·c_tilde, o·tanh(C_new))。

    一次 kernel 完成记忆元更新与隐状态输出，减少 host launch 次数。
    反向：
      grad_c_new = grad_c_out + o·(1-tanh(c_new)²)·grad_h_out
      grad_f = c·grad_c_new, grad_c = f·grad_c_new
      grad_i = c_tilde·grad_c_new, grad_c_tilde = i·grad_c_new
      grad_o = tanh(c_new)·grad_h_out
    """

    class PyPTOFusedCellTanhMulImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, f, c, i, ct, o):
            c_out = torch.empty_like(f)
            h_out = torch.empty_like(o)
            fwd_kernel(f, c, i, ct, o, c_out, h_out)
            ctx.save_for_backward(f, c, i, ct, o, c_out)
            return c_out, h_out

        @staticmethod
        def backward(ctx, grad_c_out, grad_h_out):
            f, c, i, ct, o, c_new = ctx.saved_tensors
            needs = ctx.needs_input_grad
            grad_f = torch.empty_like(f) if needs[0] else None
            grad_c = torch.empty_like(c) if needs[1] else None
            grad_i = torch.empty_like(i) if needs[2] else None
            grad_ct = torch.empty_like(ct) if needs[3] else None
            grad_o = torch.empty_like(o) if needs[4] else None
            tmp_f = grad_f if grad_f is not None else torch.empty_like(f)
            tmp_c = grad_c if grad_c is not None else torch.empty_like(c)
            tmp_i = grad_i if grad_i is not None else torch.empty_like(i)
            tmp_ct = grad_ct if grad_ct is not None else torch.empty_like(ct)
            tmp_o = grad_o if grad_o is not None else torch.empty_like(o)
            if any(needs):
                bwd_kernel(f.contiguous(), c.contiguous(), i.contiguous(),
                           ct.contiguous(), o.contiguous(), c_new.contiguous(),
                           grad_c_out.contiguous(), grad_h_out.contiguous(),
                           tmp_f, tmp_c, tmp_i, tmp_ct, tmp_o)
            return grad_f, grad_c, grad_i, grad_ct, grad_o

    return PyPTOFusedCellTanhMulImpl


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def cell_update_fwd_kernel(
    f: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
    i: pypto.Tensor([], pypto.DT_FP32),
    ct: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    out.move(pypto.add(pypto.mul(f, c), pypto.mul(i, ct)))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def cell_update_bwd_kernel(
    f: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
    i: pypto.Tensor([], pypto.DT_FP32),
    ct: pypto.Tensor([], pypto.DT_FP32),
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    grad_f: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_i: pypto.Tensor([], pypto.DT_FP32),
    grad_ct: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_f.move(pypto.mul(c, grad_out))
    grad_i.move(pypto.mul(ct, grad_out))
    grad_ct.move(pypto.mul(i, grad_out))
    grad_c.move(pypto.mul(f, grad_out))


def make_pypto_cell_update(fwd_kernel, bwd_kernel):
    """创建融合 LSTM 记忆元更新算子 out = f·c + i·c_tilde。

    ⚠️ 中间优化版本：当前 LSTM 使用 PyPTOFusedCellTanhMul（记忆元更新+
    隐状态输出合一），本算子保留作融合思路的教学对照。

    反向：
      grad_f = c·grad_out, grad_c = f·grad_out
      grad_i = c_tilde·grad_out, grad_c_tilde = i·grad_out
    """

    class PyPTOFusedCellUpdateImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, f, c, i, ct):
            out = torch.empty_like(f)
            fwd_kernel(f, c, i, ct, out)
            ctx.save_for_backward(f, c, i, ct)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            f, c, i, ct = ctx.saved_tensors
            grad_f = torch.empty_like(f) if ctx.needs_input_grad[0] else None
            grad_c = torch.empty_like(c) if ctx.needs_input_grad[1] else None
            grad_i = torch.empty_like(i) if ctx.needs_input_grad[2] else None
            grad_ct = torch.empty_like(ct) if ctx.needs_input_grad[3] else None
            tmp_f = grad_f if grad_f is not None else torch.empty_like(f)
            tmp_c = grad_c if grad_c is not None else torch.empty_like(c)
            tmp_i = grad_i if grad_i is not None else torch.empty_like(i)
            tmp_ct = grad_ct if grad_ct is not None else torch.empty_like(ct)
            if any(ctx.needs_input_grad):
                bwd_kernel(f.contiguous(), c.contiguous(), i.contiguous(),
                           ct.contiguous(), grad_out.contiguous(),
                           tmp_f, tmp_c, tmp_i, tmp_ct)
            return grad_f, grad_c, grad_i, grad_ct

    return PyPTOFusedCellUpdateImpl


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def tanh_mul_fwd_kernel(
    o: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    two_c = pypto.mul(c, 2.0)
    s = pypto.sigmoid(two_c)
    tanh_c = pypto.sub(pypto.mul(s, 2.0), 1.0)
    out.move(pypto.mul(o, tanh_c))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def tanh_mul_bwd_kernel(
    o: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    grad_o: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    two_c = pypto.mul(c, 2.0)
    s = pypto.sigmoid(two_c)
    tanh_c = pypto.sub(pypto.mul(s, 2.0), 1.0)
    grad_o.move(pypto.mul(tanh_c, grad_out))
    one_minus_t2 = pypto.neg(pypto.sub(pypto.mul(tanh_c, tanh_c), 1.0))
    grad_c.move(pypto.mul(pypto.mul(o, one_minus_t2), grad_out))


def make_pypto_tanh_mul(fwd_kernel, bwd_kernel):
    """创建融合 tanh·mul 算子 out = o·tanh(c)。

    ⚠️ 中间优化版本：当前 LSTM 使用 PyPTOFusedCellTanhMul（记忆元更新+
    隐状态输出合一），本算子保留作融合思路的教学对照。

    反向：
      grad_o = tanh(c)·grad_out
      grad_c = o·(1-tanh(c)²)·grad_out
    """

    class PyPTOFusedTanhMulImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, o, c):
            out = torch.empty_like(o)
            fwd_kernel(o, c, out)
            ctx.save_for_backward(o, c)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            o, c = ctx.saved_tensors
            grad_o = torch.empty_like(o) if ctx.needs_input_grad[0] else None
            grad_c = torch.empty_like(c) if ctx.needs_input_grad[1] else None
            tmp_o = grad_o if grad_o is not None else torch.empty_like(o)
            tmp_c = grad_c if grad_c is not None else torch.empty_like(c)
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                bwd_kernel(o.contiguous(), c.contiguous(),
                           grad_out.contiguous(), tmp_o, tmp_c)
            return grad_o, grad_c

    return PyPTOFusedTanhMulImpl


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def linear_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(128, 128)
    y.move(pypto.add(pypto.matmul(x, w, pypto.DT_FP32), bias))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def linear_bwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    grad_y: pypto.Tensor([], pypto.DT_FP32),
    grad_x: pypto.Tensor([], pypto.DT_FP32),
    grad_w: pypto.Tensor([], pypto.DT_FP32),
    grad_bias: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(128, 128)
    grad_x.move(
        pypto.matmul(grad_y, w, pypto.DT_FP32, b_trans=True))
    grad_w.move(
        pypto.matmul(x, grad_y, pypto.DT_FP32, a_trans=True))
    grad_bias.move(pypto.sum(grad_y, 0))


def make_pypto_linear(fwd_kernel, bwd_kernel):
    """创建融合线性层算子 y = x@w + bias（matmul+bias_add 一次 kernel）。"""

    class PyPTOLinearFusedImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, bias):
            ctx.save_for_backward(x, w)
            y = torch.empty(x.shape[0], w.shape[1],
                            device=x.device, dtype=x.dtype)
            fwd_kernel(x, w, bias, y)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            x, w = ctx.saved_tensors
            need_x = ctx.needs_input_grad[0]
            need_w = ctx.needs_input_grad[1]
            need_bias = ctx.needs_input_grad[2]
            grad_x = torch.empty_like(x) if need_x else None
            grad_w = torch.empty_like(w) if need_w else None
            grad_bias = torch.empty(w.shape[1], device=w.device,
                                    dtype=w.dtype) if need_bias else None
            tmp_x = grad_x if need_x else torch.empty_like(x)
            tmp_w = grad_w if need_w else torch.empty_like(w)
            tmp_bias = grad_bias if need_bias else torch.empty(
                w.shape[1], device=w.device, dtype=w.dtype)
            if need_x or need_w or need_bias:
                bwd_kernel(x.contiguous(), w.contiguous(),
                           grad_y.contiguous(), tmp_x, tmp_w, tmp_bias)
            return grad_x, grad_w, grad_bias

    return PyPTOLinearFusedImpl


# ---------------------------------------------------------------------------
# softmax + cross_entropy：组合损失函数
# ---------------------------------------------------------------------------

@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def softmax_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    y.move(pypto.softmax(x, dim=-1))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def softmax_bwd_kernel(
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_y: pypto.Tensor([], pypto.DT_FP32),
    grad_x: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    diag = pypto.mul(y, grad_y)
    s = pypto.sum(diag, 1, True)
    grad_x.move(pypto.sub(diag, pypto.mul(y, s)))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def cross_entropy_fwd_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    indices: pypto.Tensor([], pypto.DT_INT32),
    out: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    gathered = pypto.gather(y_hat, 1, pypto.reshape(indices, (-1, 1)))
    gathered = pypto.reshape(gathered, (-1,))
    out.move(pypto.neg(pypto.log(pypto.maximum(gathered, 1e-12))))


@pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU, "launch_early_mode": 1, "device_sched_mode": 1})
def cross_entropy_bwd_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    indices: pypto.Tensor([], pypto.DT_INT32),
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    grad_y_hat: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    one_hot_vec = pypto.one_hot(indices, ((num_classes + 7) // 8) * 8)
    one_hot_vec = pypto.cast(one_hot_vec, pypto.DT_FP32)
    neg_grad = pypto.mul(pypto.reshape(grad_out, (-1, 1)), one_hot_vec)
    grad_y_hat.move(pypto.div(neg_grad, y_hat))


class PyPTOSoftmaxCrossEntropyLossFunction(torch.autograd.Function):
    """组合 Softmax + CrossEntropy 损失的自定义 autograd 函数。"""

    @staticmethod
    def forward(ctx, x, y, num_classes):
        ctx.num_classes = num_classes

        softmax_out = torch.empty_like(x)
        softmax_fwd_kernel(x, softmax_out, num_classes)

        y_i32 = y.to(torch.int32).reshape(-1)
        ce_out = torch.empty(y_i32.shape[0], device=x.device, dtype=x.dtype)
        cross_entropy_fwd_kernel(softmax_out, y_i32, ce_out, num_classes)

        ctx.save_for_backward(softmax_out, y_i32)
        return ce_out.view(-1, 1)

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
PyPTOSigmoid = make_pypto_sigmoid(sigmoid_fwd_kernel, sigmoid_bwd_kernel)
PyPTOMul = make_pypto_mul(mul_fwd_kernel, mul_bwd_kernel)
PyPTOSub = make_pypto_sub(sub_fwd_kernel, sub_bwd_grad_b_kernel)
PyPTOFusedSigmoidGate = make_pypto_gate_sigmoid(
    gate_sigmoid_fwd_kernel, gate_sigmoid_bwd_kernel)
PyPTOFusedTanhGate = make_pypto_gate_tanh(
    gate_tanh_fwd_kernel, gate_tanh_bwd_kernel)
PyPTODualSigmoidGate = make_pypto_dual_gate(
    dual_gate_fwd_kernel, dual_gate_bwd_kernel)
PyPTOGateRH = make_pypto_gate_rh(
    gate_rh_fwd_kernel, gate_rh_bwd_kernel)
PyPTOFusedUpdate = make_pypto_update(
    update_fused_fwd_kernel, update_fused_bwd_kernel)
PyPTOFusedCellUpdate = make_pypto_cell_update(
    cell_update_fwd_kernel, cell_update_bwd_kernel)
PyPTOFusedCellTanhMul = make_pypto_cell_tanh_mul(
    cell_tanh_mul_fwd_kernel, cell_tanh_mul_bwd_kernel)
PyPTOQuadGate = make_pypto_quad_gate(
    quad_gate_fwd_kernel, quad_gate_bwd_kernel)
PyPTOFusedTanhMul = make_pypto_tanh_mul(
    tanh_mul_fwd_kernel, tanh_mul_bwd_kernel)
PyPTOLinearFused = make_pypto_linear(linear_fwd_kernel, linear_bwd_kernel)
PyPTOSoftmaxCrossEntropyLoss = PyPTOSoftmaxCrossEntropyLossFunction


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
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        orig_m = x.shape[0]
        padded = False
        if orig_m % 16 != 0:
            pad_m = 16 - orig_m % 16
            x = F.pad(x, (0, 0, 0, pad_m))
            padded = True
        if self.bias is not None:
            y = PyPTOLinearFused.apply(x, self.weight, self.bias)
        else:
            y = PyPTOMatmul.apply(x, self.weight)
        if padded:
            y = y[:orig_m]
        if len(orig_shape) > 2:
            y = y.reshape(orig_shape[:-1] + (y.shape[-1],))
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
        orig_batch = X.shape[1]
        batch_size = orig_batch
        if orig_batch % 16 != 0:
            batch_size += 16 - orig_batch % 16
            X = F.pad(X, (0, 0, 0, batch_size - orig_batch, 0, 0))
        H = hidden_size = self.hidden_size
        if state is None:
            h_flat = torch.zeros(batch_size, H,
                                 device=X.device, dtype=X.dtype)
        else:
            h_flat = state.squeeze(0)
            if batch_size != orig_batch:
                h_flat = F.pad(h_flat, (0, 0, 0, batch_size - orig_batch))

        outputs = []
        for t in range(X.shape[0]):
            xw = PyPTOMatmul.apply(X[t], self.W_xh)
            hw = PyPTOMatmul.apply(h_flat, self.W_hh)
            h_flat = PyPTOTanh.apply(PyPTOBiasAdd.apply(PyPTOAdd.apply(xw, hw), self.b_h))
            outputs.append(h_flat)

        Y = torch.stack(outputs, dim=0)
        if batch_size != orig_batch:
            Y = Y[:, :orig_batch]
            h_flat = h_flat[:orig_batch]
        return Y, h_flat.unsqueeze(0)


class PyPTOGRU(nn.Module):
    """PyPTO GRU 隐层（支持多层单向）。

    单层隐状态递推公式：
      R_t = σ(X_t @ W_xr + H_{t-1} @ W_hr + b_r)
      Z_t = σ(X_t @ W_xz + H_{t-1} @ W_hz + b_z)
      H~_t = tanh(X_t @ W_xh + (R_t ⊙ H_{t-1}) @ W_hh + b_h)
      H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ H~_t

    多层时逐层堆叠：第 l 层的输入是第 l-1 层全部时间步的输出，
    层间 dropout 仅在训练时生效（与 nn.GRU 语义一致，不对输出层施加）。
    输出为最后一层全部时间步的隐状态 (seq, batch, hidden)，
    state 为逐层最终隐状态堆叠 (num_layers, batch, hidden)。

    不含输出投影，与 nn.GRU 的隐藏层接口兼容。

    性能优化（等价于上述公式，仅 kernel 边界不同）：
    - X 投影外提：全部时间步的 X @ (W_xz|W_xr|W_xh) 在层循环外一次大 matmul，
      并按门切分为连续块（XW_z / XW_r / XW_h），时间步循环内零拷贝取切片；
    - 时间步内 kernel 调用 5 次：H@(W_hz|W_hr) → GateRH(Z,R,R⊙H) → (R⊙H)@W_hh
      → FusedTanhGate(H~) → FusedUpdate(H)。
    融合算子与公式 9.1.1-9.1.3 的对照见模块头 docstring。
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = False
        # 逐层参数：第 0 层命名不带后缀（W_x 等），后续层带 _l 后缀，
        # 参数名均含 "W_" 前缀，兼容按 "W_" 匹配的 Xavier 初始化逻辑。
        for l in range(num_layers):
            suffix = '' if l == 0 else f'_{l}'
            in_size = input_size if l == 0 else hidden_size
            # 合并 X 权重: [W_xz | W_xr | W_xh] shape (in_size, 3*hidden_size)
            setattr(self, f'W_x{suffix}',
                    nn.Parameter(torch.randn(in_size, hidden_size * 3) * 0.01))
            # 合并 H 权重 (z, r): [W_hz | W_hr] shape (hidden_size, 2*hidden_size)
            setattr(self, f'W_hzr{suffix}',
                    nn.Parameter(torch.randn(hidden_size, hidden_size * 2) * 0.01))
            # 候选隐状态 H 权重（保持独立，因为乘的是 R_t ⊙ H）
            setattr(self, f'W_hh{suffix}',
                    nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            setattr(self, f'b_z{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_r{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_h{suffix}', nn.Parameter(torch.zeros(hidden_size)))

    def _step(self, xw_z, xw_r, xw_h, H_state, l):
        """单层单时间步递推（层 l）。

        Args:
            xw_z/xw_r/xw_h: X_t @ W_x 的预切分切片，各 (batch, hidden)，
                在 forward 的层循环外一次 matmul + 一次 contiguous() 得到；
                循环内通过 XW_z[t] 零拷贝取用（PyPTO kernel 要求输入连续，
                故切分在循环外完成，循环内不再有拷贝）。
            H_state: 当前隐状态 (batch, hidden)
            l: 层索引（0 为第一层，参数名带 _l 后缀）
        """
        suffix = '' if l == 0 else f'_{l}'
        H = self.hidden_size
        W_hzr = getattr(self, f'W_hzr{suffix}')
        W_hh = getattr(self, f'W_hh{suffix}')

        # 一次 matmul 计算 H @ [W_hz|W_hr]
        hw_zr = PyPTOMatmul.apply(H_state, W_hzr)

        # 融合双门+重置乘法：(Z, R, rh) 一次 kernel（原 dual_gate+mul 2 次 → 1 次）
        Z, R, rh = PyPTOGateRH.apply(
            xw_z, xw_r, hw_zr,
            getattr(self, f'b_z{suffix}'), getattr(self, f'b_r{suffix}'),
            H_state)

        # 候选隐状态 (R ⊙ H) @ W_hh，tanh 门控融合
        rhw = PyPTOMatmul.apply(rh, W_hh)
        H_tilde = PyPTOFusedTanhGate.apply(
            xw_h, rhw, getattr(self, f'b_h{suffix}'))

        # 融合状态更新：H = Z⊙H + (1-Z)⊙H~（原 mul+sub+mul+add 4 次调用 → 1 次）
        return PyPTOFusedUpdate.apply(Z, H_state, H_tilde)

    def forward(self, X, state=None):
        orig_batch = X.shape[1]
        batch_size = orig_batch
        if orig_batch % 16 != 0:
            batch_size += 16 - orig_batch % 16
            X = F.pad(X, (0, 0, 0, batch_size - orig_batch, 0, 0))
        H = hidden_size = self.hidden_size
        num_layers = self.num_layers
        if state is None:
            states = [torch.zeros(batch_size, H,
                                  device=X.device, dtype=X.dtype)
                      for _ in range(num_layers)]
        else:
            states = [state[l] for l in range(num_layers)]
            if batch_size != orig_batch:
                states = [F.pad(s, (0, 0, 0, batch_size - orig_batch))
                          for s in states]

        inputs = X
        Y = None
        final_states = []
        for l in range(num_layers):
            suffix = '' if l == 0 else f'_{l}'
            in_size = self.input_size if l == 0 else hidden_size
            W_x = getattr(self, f'W_x{suffix}')
            # 一次 matmul 计算全部时间步的 X @ [W_xz|W_xr|W_xh]
            XW = PyPTOMatmul.apply(inputs.reshape(-1, in_size), W_x)
            XW = XW.reshape(inputs.shape[0], batch_size, hidden_size * 3)
            # 循环外切分 Z/R/H 三块：contiguous() 仅在此做一次（保证每块连续），
            # 时间步循环内 XW_z[t] 等为零拷贝 view
            XW_z = XW[:, :, :H].contiguous()
            XW_r = XW[:, :, H:2 * H].contiguous()
            XW_h = XW[:, :, 2 * H:].contiguous()
            H_state = states[l]
            outputs = []
            for t in range(inputs.shape[0]):
                H_state = self._step(XW_z[t], XW_r[t], XW_h[t], H_state, l)
                outputs.append(H_state)
            layer_out = torch.stack(outputs, dim=0)
            Y = layer_out
            final_states.append(H_state)
            # 层间 dropout：仅训练时、仅在层之间施加（与 nn.GRU 一致）
            if l < num_layers - 1:
                if self.dropout > 0 and self.training:
                    inputs = F.dropout(layer_out, p=self.dropout)
                else:
                    inputs = layer_out

        if batch_size != orig_batch:
            Y = Y[:, :orig_batch]
            final_states = [s[:orig_batch] for s in final_states]
        return Y, torch.stack(final_states, dim=0)


class PyPTOLSTM(nn.Module):
    """PyPTO LSTM 隐层（支持多层单向）。

    单层隐状态递推公式：
      I_t = σ(X_t @ W_xi + H_{t-1} @ W_hi + b_i)
      F_t = σ(X_t @ W_xf + H_{t-1} @ W_hf + b_f)
      O_t = σ(X_t @ W_xo + H_{t-1} @ W_ho + b_o)
      C~_t = tanh(X_t @ W_xc + H_{t-1} @ W_hc + b_c)
      C_t = F_t ⊙ C_{t-1} + I_t ⊙ C~_t
      H_t = O_t ⊙ tanh(C_t)

    多层时逐层堆叠：第 l 层的输入是第 l-1 层全部时间步的输出，
    层间 dropout 仅在训练时生效（与 nn.LSTM 语义一致，不对输出层施加）。
    输出为最后一层全部时间步的隐状态 (seq, batch, hidden)，
    state 为 (H, C) 二元组，H/C 各为逐层最终状态堆叠 (num_layers, batch, hidden)。

    不含输出投影，与 nn.LSTM 的隐藏层接口兼容。

    性能优化（等价于上述公式，仅 kernel 边界不同）：
    - X 投影外提：全部时间步的 X @ (W_xi|W_xf|W_xo|W_xc) 在层循环外一次大 matmul，
      并按门切分为连续块（XW_i / XW_f / XW_o / XW_c），时间步循环内零拷贝取切片；
    - 时间步内 kernel 调用 3 次：H@(W_hi|W_hf|W_ho|W_hc) → QuadGate(I,F,O,C~)
      → FusedCellTanhMul(C_new, H)。
    融合算子与公式 9.2.1-9.2.4 的对照见模块头 docstring。
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = False
        # 逐层参数：第 0 层命名不带后缀（W_x 等），后续层带 _l 后缀，
        # 参数名均含 "W_" 前缀，兼容按 "W_" 匹配的 Xavier 初始化逻辑。
        for l in range(num_layers):
            suffix = '' if l == 0 else f'_{l}'
            in_size = input_size if l == 0 else hidden_size
            # 合并 X 权重: [W_xi | W_xf | W_xo | W_xc] shape (in_size, 4*hidden_size)
            setattr(self, f'W_x{suffix}',
                    nn.Parameter(torch.randn(in_size, hidden_size * 4) * 0.01))
            # 合并 H 权重: [W_hi | W_hf | W_ho | W_hc] shape (hidden_size, 4*hidden_size)
            setattr(self, f'W_h{suffix}',
                    nn.Parameter(torch.randn(hidden_size, hidden_size * 4) * 0.01))
            setattr(self, f'b_i{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_f{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_o{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_c{suffix}', nn.Parameter(torch.zeros(hidden_size)))

    def _step(self, xw_i, xw_f, xw_o, xw_c, H_state, C, l):
        """单层单时间步递推（层 l）。

        Args:
            xw_i/xw_f/xw_o/xw_c: X_t @ W_x 的预切分切片，各 (batch, hidden)，
                在 forward 的层循环外一次 matmul + 一次 contiguous() 得到；
                循环内通过 XW_i[t] 等零拷贝取用（PyPTO kernel 要求输入连续，
                故切分在循环外完成，循环内不再有拷贝）。
            H_state: 当前隐状态 (batch, hidden)
            C: 当前记忆元 (batch, hidden)
            l: 层索引（0 为第一层，参数名带 _l 后缀）
        """
        suffix = '' if l == 0 else f'_{l}'
        H = self.hidden_size
        W_h = getattr(self, f'W_h{suffix}')

        # 一次 matmul：H @ [W_hi|W_hf|W_ho|W_hc]
        hw = PyPTOMatmul.apply(H_state, W_h)

        # 融合四门：I/F/O/C~ 一次 kernel（原 4 次 gate 调用 + 4 次 slice copy → 1 次）
        I, F, O, C_tilde = PyPTOQuadGate.apply(
            xw_i, xw_f, xw_o, xw_c, hw,
            getattr(self, f'b_i{suffix}'), getattr(self, f'b_f{suffix}'),
            getattr(self, f'b_o{suffix}'), getattr(self, f'b_c{suffix}'))

        # 融合记忆元更新+隐状态输出：(C, H) 一次 kernel（原 2 次调用 → 1 次）
        C, H_state = PyPTOFusedCellTanhMul.apply(F, C, I, C_tilde, O)
        return H_state, C
    def forward(self, X, state=None):
        orig_batch = X.shape[1]
        batch_size = orig_batch
        if orig_batch % 16 != 0:
            batch_size += 16 - orig_batch % 16
            X = F.pad(X, (0, 0, 0, batch_size - orig_batch, 0, 0))
        num_layers = self.num_layers
        if state is None:
            H_states = [torch.zeros(batch_size, self.hidden_size,
                                    device=X.device, dtype=X.dtype)
                        for _ in range(num_layers)]
            C_states = [torch.zeros(batch_size, self.hidden_size,
                                    device=X.device, dtype=X.dtype)
                        for _ in range(num_layers)]
        else:
            H_stack, C_stack = state
            if batch_size != orig_batch:
                H_stack = F.pad(H_stack, (0, 0, 0, batch_size - orig_batch, 0, 0))
                C_stack = F.pad(C_stack, (0, 0, 0, batch_size - orig_batch, 0, 0))
            H_states = [H_stack[l] for l in range(num_layers)]
            C_states = [C_stack[l] for l in range(num_layers)]

        inputs = X
        final_H, final_C = [], []
        for l in range(num_layers):
            suffix = '' if l == 0 else f'_{l}'
            in_size = self.input_size if l == 0 else self.hidden_size
            W_x = getattr(self, f'W_x{suffix}')
            # 一次 matmul 计算全部时间步的 X @ [W_xi|W_xf|W_xo|W_xc]
            XW = PyPTOMatmul.apply(inputs.reshape(-1, in_size), W_x)
            XW = XW.reshape(inputs.shape[0], batch_size, self.hidden_size * 4)
            # 循环外切分 I/F/O/C 四块：contiguous() 仅在此做一次（保证每块连续），
            # 时间步循环内 XW_i[t] 等为零拷贝 view
            XW_i = XW[:, :, :self.hidden_size].contiguous()
            XW_f = XW[:, :, self.hidden_size:2 * self.hidden_size].contiguous()
            XW_o = XW[:, :, 2 * self.hidden_size:3 * self.hidden_size].contiguous()
            XW_c = XW[:, :, 3 * self.hidden_size:].contiguous()
            H_state, C = H_states[l], C_states[l]
            outputs = []
            for t in range(inputs.shape[0]):
                H_state, C = self._step(XW_i[t], XW_f[t], XW_o[t], XW_c[t],
                                        H_state, C, l)
                outputs.append(H_state)
            layer_out = torch.stack(outputs, dim=0)
            final_H.append(H_state)
            final_C.append(C)
            # 层间 dropout：仅训练时、仅在层之间施加（与 nn.LSTM 一致）
            if l < num_layers - 1:
                if self.dropout > 0 and self.training:
                    inputs = F.dropout(layer_out, p=self.dropout)
                else:
                    inputs = layer_out

        if batch_size != orig_batch:
            layer_out = layer_out[:, :orig_batch]
            final_H = [h[:orig_batch] for h in final_H]
            final_C = [c[:orig_batch] for c in final_C]
        return layer_out, (torch.stack(final_H, dim=0),
                           torch.stack(final_C, dim=0))


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
    "PyPTOTanh", "PyPTOReLUOp",
    "PyPTOSigmoid", "PyPTOMul", "PyPTOSub",
    "PyPTOSoftmaxCrossEntropyLoss",
    "PyPTOLinear", "PyPTOReLU",
    "PyPTORNN", "PyPTOGRU", "PyPTOLSTM",
    # 当前使用的融合算子（GateRH / FusedCellTanhMul 为最新版本；
    # FusedSigmoidGate / DualSigmoidGate / FusedCellUpdate / FusedTanhMul
    # 为中间版本，仅作教学对照，未导出）
    "PyPTOFusedTanhGate", "PyPTOFusedUpdate",
    "PyPTOGateRH", "PyPTOQuadGate", "PyPTOFusedCellTanhMul",
    "PyPTOLinearFused",
    "loss_fn",
]
