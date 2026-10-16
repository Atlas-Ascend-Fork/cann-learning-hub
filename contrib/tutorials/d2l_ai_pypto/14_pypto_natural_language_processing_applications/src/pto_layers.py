"""PyPTO NPU 算子库（第 9 章共享模块）。

本模块继承第 8 章的算子（matmul / bias_add / add / tanh / relu / softmax+CE /
PyPTOLinear / PyPTORNN），并新增第 9 章所需的算子：sigmoid / mul / sub /
PyPTOGRU / PyPTOLSTM。

提供的算子：

| 算子 | 前向 PyPTO API | 反向 | tiling |
|------|---------------|------|--------|
| matmul | pypto.matmul | transpose variants | cube [128,128],[64,256],[256,256] |
| matmul+bias | pypto.matmul + add (融合) | matmul 反向 + sum | cube [128,128],[64,256],[256,256] |
| bias_add | pypto.add (broadcast) | sum over batch dim | vec (128,128) |
| add | pypto.add (same shape) | identity | vec (128,128) |
| tanh | pypto.sigmoid 组合 2σ(2x)-1 | (1-y²)*grad | vec (128,128) |
| relu | pypto.maximum | pypto.where | vec (128,128) |
| softmax+CE | pypto.softmax + gather/log | softmax - one_hot | vec (8, aligned) |
| sigmoid | pypto.sigmoid | y*(1-y)*grad | vec (128,128) |
| mul | pypto.mul | swap operands | vec (128,128) |
| sub | pypto.sub | identity / neg | vec (128,128) |

每个算子采用工厂函数模式：
1. 定义 @pypto.frontend.jit 内核 (fwd + bwd)
2. 包装为 torch.autograd.Function
3. 导出为 PyPTO<Op> 类，通过 .apply() 调用
4. 对有需要的算子提供 nn.Module 封装

导出清单：
- PyPTOMatmul / PyPTOBiasAdd / PyPTOAdd / PyPTOTanh / PyPTOReLUOp
- PyPTOSigmoid / PyPTOMul / PyPTOSub
- PyPTOSoftmaxCrossEntropyLoss
- PyPTOLinear / PyPTOReLU
- PyPTORNN / PyPTOGRU / PyPTOLSTM
- loss_fn / PyPTOSoftmaxCrossEntropyLoss
"""

import pypto
import torch
from torch import nn
from torch.nn import functional as F

# 初始化 pypto NPU 运行时（进程内仅执行一次）。
# pypto 的首个 kernel 若未经 DeviceInit，卷积等算子的编译耗时会成倍增长
# 甚至卡死（参见 fast_pypto_lstm._warmup 的说明）。
try:
    pypto.pypto_impl.DeviceInit()
except Exception:
    pass


# ---------------------------------------------------------------------------
# matmul：前向 C = A @ B，反向 grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
    c.move(pypto.matmul(a, b, pypto.DT_FP32))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_bwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
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
# matmul + bias：融合前向 C = A @ B + bias（一次 kernel 调用）
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_bias_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.add(pypto.matmul(a, b, pypto.DT_FP32), bias))


def make_pypto_matmul_bias(fwd_kernel, mm_bwd_kernel, bias_bwd_kernel):
    """创建融合 matmul+bias 算子，前向仅一次 kernel 调用。"""

    class PyPTOMatmulBiasImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b, bias):
            ctx.save_for_backward(a, b)
            c = torch.empty(a.shape[0], b.shape[1],
                            device=a.device, dtype=a.dtype)
            fwd_kernel(a.contiguous(), b.contiguous(), bias.contiguous(), c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            a, b = ctx.saved_tensors
            need_a = ctx.needs_input_grad[0]
            need_b = ctx.needs_input_grad[1]
            need_bias = ctx.needs_input_grad[2]
            grad_a = torch.empty_like(a) if need_a else None
            grad_b = torch.empty_like(b) if need_b else None
            tmp_a = grad_a if need_a else torch.empty_like(a)
            tmp_b = grad_b if need_b else torch.empty_like(b)
            mm_bwd_kernel(a.contiguous(), b.contiguous(),
                          grad_c.contiguous(), tmp_a, tmp_b)
            grad_bias = (torch.empty(b.shape[1], device=b.device, dtype=b.dtype)
                         if need_bias else None)
            if need_bias:
                bias_bwd_kernel(a.contiguous(), grad_c.contiguous(), grad_bias)
            return grad_a, grad_b, grad_bias

    return PyPTOMatmulBiasImpl


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
# tanh：tanh(x) = 2·sigmoid(2x) - 1
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def tanh_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    two_a = pypto.mul(a, 2.0)
    s = pypto.sigmoid(two_a)
    two_s = pypto.mul(s, 2.0)
    b.move(pypto.sub(two_s, 1.0))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
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
# sigmoid：y = 1 / (1 + e^(-x))，反向：grad_x = y · (1-y) · grad_y
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def sigmoid_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    b.move(pypto.sigmoid(a))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
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

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def mul_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.mul(a, b))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
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

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def sub_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.sub(a, b))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
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
# softmax + cross_entropy：组合损失函数
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    y.move(pypto.softmax(x, dim=-1))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
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


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
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


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
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
PyPTOMatmulBias = make_pypto_matmul_bias(
    matmul_bias_fwd_kernel, matmul_bwd_kernel, bias_add_bwd_kernel)
PyPTOBiasAdd = make_pypto_bias_add(bias_add_fwd_kernel, bias_add_bwd_kernel)
PyPTOAdd = make_pypto_add(add_fwd_kernel)
PyPTOTanh = make_pypto_tanh(tanh_fwd_kernel, tanh_bwd_kernel)
PyPTOReLUOp = make_pypto_relu(relu_fwd_kernel, relu_bwd_kernel)
PyPTOSigmoid = make_pypto_sigmoid(sigmoid_fwd_kernel, sigmoid_bwd_kernel)
PyPTOMul = make_pypto_mul(mul_fwd_kernel, mul_bwd_kernel)
PyPTOSub = make_pypto_sub(sub_fwd_kernel, sub_bwd_grad_b_kernel)
PyPTOSoftmaxCrossEntropyLoss = PyPTOSoftmaxCrossEntropyLossFunction


# =========================================================================
# nn.Module 封装
# =========================================================================

# M 维分块大小。pypto matmul 内核按 16x16 cube tile 将全部 tile 展开生成 C++，
# M 过大时（如 SNLI 训练 256x50=12800 行，800 个 M-tile）单次编译可达数小时
# 且每种新 M 形状都要重新编译。将 M 按 PYPTO_M_CHUNK 分块、尾块补齐到同一
# 块大小后，整个训练只需编译一个 (CHUNK, K, N) 形状的内核并复用（与
# PyPTOConv1d 的 CONV_BATCH_CHUNK 分块思路一致）。
PYPTO_M_CHUNK = 512


def _apply_chunked(x, fn):
    """对 2D 张量按 M 维分块调用 fn，避免 pypto 内核对超大 M 的编译时间爆炸。

    - x 行数超过 PYPTO_M_CHUNK 时按块切分，尾块补齐到完整块大小，
      保证所有内核调用的 M 维形状唯一（只需编译一次）；
    - x 行数不超过 PYPTO_M_CHUNK 时保持原逻辑，仅按 16 对齐补齐。
    """
    orig_m = x.shape[0]
    outs = []
    for i in range(0, orig_m, PYPTO_M_CHUNK):
        chunk = x[i:i + PYPTO_M_CHUNK]
        actual = chunk.shape[0]
        if actual < PYPTO_M_CHUNK:
            if orig_m > PYPTO_M_CHUNK:
                pad_m = PYPTO_M_CHUNK - actual
            else:
                pad_m = 0 if actual % 16 == 0 else 16 - actual % 16
            if pad_m:
                chunk = F.pad(chunk, (0, 0, 0, pad_m))
        outs.append(fn(chunk)[:actual])
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


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

        def _linear(chunk):
            if self.bias is not None:
                return PyPTOMatmulBias.apply(chunk, self.weight, self.bias)
            return PyPTOMatmul.apply(chunk, self.weight)

        y = _apply_chunked(x, _linear)
        if len(orig_shape) > 2:
            y = y.reshape(orig_shape[:-1] + (y.shape[-1],))
        return y


class PyPTOReLU(nn.Module):
    """用 PyPTO ReLU 算子实现的 ReLU 激活层。

    底层 relu 内核使用 2D vec tile (128, 128)，因此对高于 2 维的输入
    先展平为 2D 再计算，最后还原形状（ReLU 是逐元素运算，展平不影响结果）。
    同样按 M 维分块（_apply_chunked），避免大形状触发漫长的内核编译。
    """

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        y = _apply_chunked(x, PyPTOReLUOp.apply)
        if len(orig_shape) > 2:
            y = y.reshape(orig_shape)
        return y


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

    优化：将 X_t 对 (W_xz|W_xr|W_xh) 三次 matmul 合并为一次大 matmul，
    将 H 对 (W_hz|W_hr) 两次 matmul 合并为一次，核函数调用 6→3。
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

    def _step(self, x_t, H_state, l):
        """单层单时间步递推（层 l）。"""
        suffix = '' if l == 0 else f'_{l}'
        H = self.hidden_size
        W_x = getattr(self, f'W_x{suffix}')
        W_hzr = getattr(self, f'W_hzr{suffix}')
        W_hh = getattr(self, f'W_hh{suffix}')

        # 一次 matmul 计算 X_t @ [W_xz|W_xr|W_xh]
        xw = PyPTOMatmul.apply(x_t, W_x)
        xw_z = xw[:, :H].contiguous()
        xw_r = xw[:, H:2 * H].contiguous()
        xw_h = xw[:, 2 * H:].contiguous()

        # 一次 matmul 计算 H @ [W_hz|W_hr]
        hw_zr = PyPTOMatmul.apply(H_state, W_hzr)
        hw_z = hw_zr[:, :H].contiguous()
        hw_r = hw_zr[:, H:].contiguous()

        Z = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_z, hw_z),
                               getattr(self, f'b_z{suffix}')))
        R = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_r, hw_r),
                               getattr(self, f'b_r{suffix}')))

        # 候选隐状态 (R ⊙ H) @ W_hh
        rh = PyPTOMul.apply(R, H_state)
        rhw = PyPTOMatmul.apply(rh, W_hh)
        H_tilde = PyPTOTanh.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_h, rhw),
                               getattr(self, f'b_h{suffix}')))

        # 最终状态更新
        z_h = PyPTOMul.apply(Z, H_state)
        inv_z_h_tilde = PyPTOMul.apply(
            PyPTOSub.apply(torch.ones_like(Z), Z), H_tilde)
        return PyPTOAdd.apply(z_h, inv_z_h_tilde)

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
            H_state = states[l]
            outputs = []
            for t in range(inputs.shape[0]):
                H_state = self._step(inputs[t], H_state, l)
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
    """PyPTO LSTM 隐层（支持多层、单向/双向）。

    单层隐状态递推公式：
      I_t = σ(X_t @ W_xi + H_{t-1} @ W_hi + b_i)
      F_t = σ(X_t @ W_xf + H_{t-1} @ W_hf + b_f)
      O_t = σ(X_t @ W_xo + H_{t-1} @ W_ho + b_o)
      C~_t = tanh(X_t @ W_xc + H_{t-1} @ W_hc + b_c)
      C_t = F_t ⊙ C_{t-1} + I_t ⊙ C~_t
      H_t = O_t ⊙ tanh(C_t)

    多层时逐层堆叠：第 l 层的输入是第 l-1 层全部时间步的输出，
    bidirectional=True 时该输出为前向/反向隐状态在特征维的拼接（2*hidden_size），
    与 nn.LSTM(bidirectional=True) 语义一致；层间 dropout 仅训练时施加。
    输出为最后一层全部时间步的隐状态 (seq, batch, hidden*num_directions)，
    state 为 (H, C) 二元组，各形状 (num_layers*num_directions, batch, hidden)，
    方向与层交错排列（layer0_fwd, layer0_rev, layer1_fwd, ...）。

    参数命名：前向方向无后缀（第 0 层 W_x，第 l 层 W_x_{l}），
    反向方向加 _rev 后缀（W_x_rev、W_x_rev_{l}），参数名均含 "W_" 前缀。
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0,
                 bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # 逐方向逐层参数：方向后缀（前向 ''，反向 '_rev'）+ 层后缀（第 0 层 ''，第 l 层 '_{l}'）
        self._flat_weights_names = []
        for direction in range(self.num_directions):
            d = '_rev' if direction == 1 else ''
            for l in range(num_layers):
                lsuf = '' if l == 0 else f'_{l}'
                suffix = f'{d}{lsuf}'
                # 第 0 层输入为 input_size；后续层输入为上一层输出 hidden_size*num_directions
                in_size = input_size if l == 0 else hidden_size * self.num_directions
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
                self._flat_weights_names.append(f'W_x{suffix}')
                self._flat_weights_names.append(f'W_h{suffix}')

    def flatten_parameters(self):
        """与 nn.LSTM 接口兼容（权重已是独立 Parameter，无需展平）。"""
        pass

    def _step(self, x_t, H_state, C, l, rev=False):
        """单层单时间步递推（层 l，方向由 rev 指定）。"""
        d = '_rev' if rev else ''
        lsuf = '' if l == 0 else f'_{l}'
        suffix = f'{d}{lsuf}'
        H = self.hidden_size
        W_x = getattr(self, f'W_x{suffix}')
        W_h = getattr(self, f'W_h{suffix}')

        # 一次 matmul：X_t @ [W_xi|W_xf|W_xo|W_xc]
        xw = PyPTOMatmul.apply(x_t, W_x)
        xw_i = xw[:, :H].contiguous()
        xw_f = xw[:, H:2 * H].contiguous()
        xw_o = xw[:, 2 * H:3 * H].contiguous()
        xw_c = xw[:, 3 * H:].contiguous()

        # 一次 matmul：H @ [W_hi|W_hf|W_ho|W_hc]
        hw = PyPTOMatmul.apply(H_state, W_h)
        hw_i = hw[:, :H].contiguous()
        hw_f = hw[:, H:2 * H].contiguous()
        hw_o = hw[:, 2 * H:3 * H].contiguous()
        hw_c = hw[:, 3 * H:].contiguous()

        I = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_i, hw_i),
                               getattr(self, f'b_i{suffix}')))
        F = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_f, hw_f),
                               getattr(self, f'b_f{suffix}')))
        O = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_o, hw_o),
                               getattr(self, f'b_o{suffix}')))
        C_tilde = PyPTOTanh.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_c, hw_c),
                               getattr(self, f'b_c{suffix}')))

        f_c = PyPTOMul.apply(F, C)
        i_c_tilde = PyPTOMul.apply(I, C_tilde)
        C = PyPTOAdd.apply(f_c, i_c_tilde)

        H_state = PyPTOMul.apply(O, PyPTOTanh.apply(C))
        return H_state, C

    def forward(self, X, state=None):
        # X: (seq, batch, input)
        orig_batch = X.shape[1]
        batch_size = orig_batch
        if orig_batch % 16 != 0:
            batch_size += 16 - orig_batch % 16
            X = F.pad(X, (0, 0, 0, batch_size - orig_batch, 0, 0))
        H = self.hidden_size
        num_layers = self.num_layers
        nd = self.num_directions
        seq_len = X.shape[0]

        # 初始状态按 (层, 方向) 组织，索引 l*num_directions + d
        if state is None:
            H_stack = torch.zeros(num_layers * nd, batch_size, H,
                                  device=X.device, dtype=X.dtype)
            C_stack = torch.zeros(num_layers * nd, batch_size, H,
                                  device=X.device, dtype=X.dtype)
        else:
            H_stack, C_stack = state
            if batch_size != orig_batch:
                pad = batch_size - orig_batch
                H_stack = F.pad(H_stack, (0, 0, 0, pad, 0, 0))
                C_stack = F.pad(C_stack, (0, 0, 0, pad, 0, 0))

        inputs = X
        final_H, final_C = [], []
        for l in range(num_layers):
            # 前向方向
            H_f, C_f = H_stack[l * nd], C_stack[l * nd]
            out_f = []
            for t in range(seq_len):
                H_f, C_f = self._step(inputs[t], H_f, C_f, l, rev=False)
                out_f.append(H_f)
            out_f = torch.stack(out_f, dim=0)
            final_H.append(H_f)
            final_C.append(C_f)
            if nd == 2:
                # 反向方向：时间步从后向前递推，输出再反转为正序
                H_r, C_r = H_stack[l * nd + 1], C_stack[l * nd + 1]
                out_r = []
                for t in range(seq_len - 1, -1, -1):
                    H_r, C_r = self._step(inputs[t], H_r, C_r, l, rev=True)
                    out_r.append(H_r)
                out_r = torch.stack(out_r[::-1], dim=0)
                final_H.append(H_r)
                final_C.append(C_r)
                layer_out = torch.cat([out_f, out_r], dim=-1)
            else:
                layer_out = out_f
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


# ---------------------------------------------------------------------------
# Conv1d：PyPTO Conv1d封装，底层调用 pypto.conv
# 输入：(N, C_in, L)，输出：(N, C_out, L_out)
# pypto.conv 为2D卷积，将1D序列映射为 [N,Cin,L,1] 的伪2D张量
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def conv1d_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),    # [N, Cin, Win]
    w: pypto.Tensor([], pypto.DT_FP32),    # [Cout, Cin, K]
    bias: pypto.Tensor([], pypto.DT_FP32), # [Cout]
    y: pypto.Tensor([], pypto.DT_FP32),    # [N, Cout, Wout]
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    win: int,
):
    # Atlas A2(昇腾910) 1D 卷积必须设置 3D vec_tile_shapes {n,c,w}：
    #   n 为 16 整数倍, c 为 C0(=32/sizeof(dtype)=8) 整数倍, w 为 32B 对齐(=8 元素)。
    # 若不设置，tile 框架用默认 2D (128,128) 给 3D 张量做 TransData 会段错误。
    pypto.set_vec_tile_shapes(16, 8, 8)
    # tileWin/tileWout 不能超过实际输入/输出宽度，否则编译报 OUT_OF_RANGE；
    # 训练时 Win=500 用 64，预测短句 Win=5 时自适应缩小。
    wout = (win + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    tw_in = min(64, win)
    tw_out = min(64, wout)
    pypto.set_conv_tile_shapes(
        pypto.pypto_impl.TileL1Info(
            tileHin=1, tileHout=1,
            tileWin=tw_in, tileWout=tw_out,
            tileCinFmap=8, tileCinWeight=8,
            tileN=16, tileBatch=1,
        ),
        pypto.pypto_impl.TileL0Info(
            tileH=1, tileW=tw_out, tileK=8, tileN=16,
        ),
    )
    strides = [stride]
    paddings = [padding, padding]
    dilations = [dilation]
    conv_out = pypto.conv(x, w, pypto.DT_FP32, strides, paddings, dilations, extend_params={"bias_tensor": bias})
    y.move(conv_out)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def conv1d_bwd_data_kernel(
    dy: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    dx: pypto.Tensor([], pypto.DT_FP32),
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    """输入梯度：反卷积(转置卷积)求 dx"""
    pypto.set_vec_tile_shapes(128, 128)
    strides = [stride, 1]
    paddings = [padding, padding, 0, 0]
    dilations = [dilation, 1]
    dx.move(pypto.conv_backward_data(dy, w, strides, paddings, dilations))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def conv1d_bwd_weight_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    dy: pypto.Tensor([], pypto.DT_FP32),
    dw: pypto.Tensor([], pypto.DT_FP32),
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    """权重梯度"""
    pypto.set_vec_tile_shapes(128, 128)
    strides = [stride, 1]
    paddings = [padding, padding, 0, 0]
    dilations = [dilation, 1]
    dw.move(pypto.conv_backward_filter(x, dy, strides, paddings, dilations))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def conv1d_bwd_bias_kernel(
    dy: pypto.Tensor([], pypto.DT_FP32),
    dbias: pypto.Tensor([], pypto.DT_FP32),
):
    """bias梯度：对N、L维度求和"""
    pypto.set_vec_tile_shapes(128, 128)
    # dy shape [N, Cout, Lout, 1]; sum dim=[0,2,3]
    sum0 = pypto.sum(dy, dim=0)
    sum2 = pypto.sum(sum0, dim=2)
    sum3 = pypto.sum(sum2, dim=2)
    dbias.move(sum3)


def make_pypto_conv1d(fwd_kernel):
    """工厂函数：生成 PyPTOConv1d autograd Function。

    前向走 pypto.conv (原生 1D, NPU 加速); pypto 未提供 conv 反向算子,
    故反向用 torch.nn.grad 的 conv1d_input / conv1d_weight (torch_npu 加速)。
    """
    from torch.nn.grad import conv1d_input, conv1d_weight

    class PyPTOConv1dImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, bias, kernel_size, stride, padding, dilation):
            # x: [N, Cin, Win], w: [Cout, Cin, K], bias: [Cout]
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.save_for_backward(x, w)

            N, Cin, Win = x.shape
            Cout, Cin_, k = w.shape
            assert Cin == Cin_ and k == kernel_size
            Wout = (Win + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            # pypto.conv 的 tileWout 要求 16 对齐，Wout < 16（如预测短句）时无法编译，
            # 退化为 torch 原生 conv1d（反向已由 conv1d_input/conv1d_weight 覆盖）。
            if Wout < 16:
                return F.conv1d(x, w, bias, stride=stride,
                                padding=padding, dilation=dilation)
            # pypto.conv 的 tileBatch 硬件约束为 1，batch 过大时单次编译耗时会爆炸
            # （N=64,W=500 实测 >10min）。按 CONV_BATCH_CHUNK 分块调用：
            # 仅首个 chunk 触发编译，后续复用；不足一个 chunk 的尾部补齐后截取。
            CHUNK = 4
            wc = w.contiguous()
            bc = bias.contiguous()
            outputs = []
            for i in range(0, N, CHUNK):
                chunk = x[i:i + CHUNK].contiguous()
                actual = chunk.shape[0]
                if actual < CHUNK:
                    chunk = F.pad(chunk, (0, 0, 0, 0, 0, CHUNK - actual))
                y_chunk = torch.empty((CHUNK, Cout, Wout),
                                      device=x.device, dtype=x.dtype)
                fwd_kernel(chunk, wc, bc, y_chunk,
                           kernel_size, stride, padding, dilation, Win)
                outputs.append(y_chunk[:actual])
            y = torch.cat(outputs, dim=0)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            x, w = ctx.saved_tensors
            k = ctx.kernel_size
            s = ctx.stride
            p = ctx.padding
            d = ctx.dilation
            need_dx = ctx.needs_input_grad[0]
            need_dw = ctx.needs_input_grad[1]
            need_db = ctx.needs_input_grad[2]

            dx = dw = dbias = None
            gy = grad_y.contiguous()
            if need_dx:
                dx = conv1d_input(x.shape, w, gy, stride=s, padding=p, dilation=d)
            if need_dw:
                dw = conv1d_weight(x, w.shape, gy, stride=s, padding=p, dilation=d)
            if need_db:
                dbias = gy.sum(dim=(0, 2))
            return dx, dw, dbias, None, None, None, None
    return PyPTOConv1dImpl


# 实例化算子
PyPTOConv1dOp = make_pypto_conv1d(conv1d_fwd_kernel)


class PyPTOConv1d(nn.Module):
    """PyPTO Conv1d: 前向 pypto.conv (原生 1D, NPU), 反向由 torch autograd 计算。

    约束: out_channels 需为 8 的倍数 (pypto.conv 要求 Cout/groups 为 C0=8 的整数倍,
    FP32 下 C0 = 32/4 = 8)。权重布局 [out_channels, in_channels, kernel_size] 与 nn.Conv1d 一致。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return PyPTOConv1dOp.apply(
            x, self.weight, self.bias,
            self.kernel_size, self.stride, self.padding, self.dilation,
        )


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
    "PyPTOMatmul", "PyPTOMatmulBias", "PyPTOBiasAdd", "PyPTOAdd",
    "PyPTOTanh", "PyPTOReLUOp",
    "PyPTOSigmoid", "PyPTOMul", "PyPTOSub",
    "PyPTOSoftmaxCrossEntropyLoss",
    "PyPTOLinear", "PyPTOReLU",
    "PyPTORNN", "PyPTOGRU", "PyPTOLSTM",
    "PyPTOConv1d",
    "loss_fn",
]
