"""PyPTO 卷积原语模块

包含原始的（非优化的）二维卷积算子、kernel 和 API，
从 06.02_conv_layer、06.03_padding_and_strides、06.04_channels 三个 notebook 中提取。

本模块提供:
  - _pad2d / _pad4d: 零填充辅助函数 (普通 Python 函数)
  - corr2d_forward_kernel / corr2d_backward_k_kernel / corr2d_backward_x_kernel:
      06.02 中的基础互相关 kernel (2D, stride=1, padding=0)
  - PyPTOCorr2DFunction / corr2d / PyPTOConv2d (06.02 原始版):
      基于 PyPTO 互相关 kernel 的 autograd Function 和 nn.Module
  - corr2d_kernel_strided / conv2d (06.03 版):
      06.03 中支持 stride/padding 的互相关
  - corr2d_multi_in / corr2d_multi_in_out / corr2d_multi_in_out_1x1:
      06.04 中的多通道卷积

说明：PyPTO 新 IR 路径不再支持在 @pypto.frontend.jit 内核中调用
@pypto.frontend.function 装饰的嵌套函数（NestedFunctionMarker）；本模块的
kernel 均直接内联互相关循环，padding 画布由宿主侧分配后传入。
"""

import pypto
import torch
from torch import nn


# ── 从 PyPTOConv2DModule.py 提取的 2D/4D 零填充原语 ──


# 普通 Python 辅助函数：新 IR 路径会直接展开，不再使用 NestedFunctionMarker。
def _pad2d(
    X:   pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
):
    """对 2D 张量 X(H,W) 的 H/W 维度零填充，写入 out。"""
    H = X.shape[0]
    W = X.shape[1]
    pypto.set_vec_tile_shapes(8, 8)
    cur = X.clone()
    cur_h = H
    cur_w = W
    if pad_top > 0:
        zero_t = pypto.zeros(pad_top, cur_w, dtype=pypto.DT_FP32)
        cur = pypto.concat([zero_t, cur], dim=0)
        cur_h = cur_h + pad_top
    if pad_bottom > 0:
        zero_b = pypto.zeros(pad_bottom, cur_w, dtype=pypto.DT_FP32)
        cur = pypto.concat([cur, zero_b], dim=0)
        cur_h = cur_h + pad_bottom
    if pad_left > 0:
        zero_l = pypto.zeros(cur_h, pad_left, dtype=pypto.DT_FP32)
        cur = pypto.concat([zero_l, cur], dim=1)
        cur_w = cur_w + pad_left
    if pad_right > 0:
        zero_r = pypto.zeros(cur_h, pad_right, dtype=pypto.DT_FP32)
        cur = pypto.concat([cur, zero_r], dim=1)
        cur_w = cur_w + pad_right
    out.move(cur)


def _pad4d(
    X:   pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
):
    """对 4D 张量 X(N,C,H,W) 的 H/W 维度零填充，写入 out。"""
    N = X.shape[0]
    C = X.shape[1]
    H = X.shape[2]
    W = X.shape[3]
    pypto.set_vec_tile_shapes(1, 1, 8, 8)
    cur = X.clone()
    cur_h = H
    cur_w = W
    if pad_top > 0:
        zero_t = pypto.zeros(N, C, pad_top, cur_w, dtype=pypto.DT_FP32)
        cur = pypto.concat([zero_t, cur], dim=2)
        cur_h = cur_h + pad_top
    if pad_bottom > 0:
        zero_b = pypto.zeros(N, C, pad_bottom, cur_w, dtype=pypto.DT_FP32)
        cur = pypto.concat([cur, zero_b], dim=2)
        cur_h = cur_h + pad_bottom
    if pad_left > 0:
        zero_l = pypto.zeros(N, C, cur_h, pad_left, dtype=pypto.DT_FP32)
        cur = pypto.concat([zero_l, cur], dim=3)
        cur_w = cur_w + pad_left
    if pad_right > 0:
        zero_r = pypto.zeros(N, C, cur_h, pad_right, dtype=pypto.DT_FP32)
        cur = pypto.concat([cur, zero_r], dim=3)
        cur_w = cur_w + pad_right
    out.move(cur)


# ── 从 06.02_conv_layer.ipynb 提取的基础互相关 kernel (2D, stride=1, padding=0) ──


@pypto.frontend.jit()
def corr2d_forward_kernel(
    X: pypto.Tensor([], pypto.DT_FP32),
    K: pypto.Tensor([], pypto.DT_FP32),
    Y: pypto.Tensor([], pypto.DT_FP32),
):
    """基础二维互相关前向 kernel (stride=1, padding=0)。"""
    kH, kW = K.shape[0], K.shape[1]
    H_out, W_out = X.shape[0] - kH + 1, X.shape[1] - kW + 1
    pypto.set_vec_tile_shapes(8, 8)
    for i_idx in pypto.loop(H_out, name="LOOP_L0_i", idx_name="i_idx"):
        for j_idx in pypto.loop(W_out, name="LOOP_L1_j", idx_name="j_idx"):
            pypto.set_vec_tile_shapes(8, 8)
            window = X[i_idx:i_idx + kH, j_idx:j_idx + kW]
            mul_result = pypto.mul(window, K)
            sum_h = pypto.sum(mul_result, dim=0, keepdim=False)
            scalar = pypto.sum(sum_h, dim=0, keepdim=False)
            result_1x1 = pypto.reshape(scalar, [1, 1])
            pypto.assemble(result_1x1, [i_idx, j_idx], Y)


@pypto.frontend.jit()
def corr2d_backward_k_kernel(
    grad_Y: pypto.Tensor([], pypto.DT_FP32),
    X:      pypto.Tensor([], pypto.DT_FP32),
    grad_K: pypto.Tensor([], pypto.DT_FP32),
):
    """权重梯度：把 grad_Y 当作卷积核，对 X 做互相关。"""
    kH, kW = grad_Y.shape[0], grad_Y.shape[1]
    H_out, W_out = X.shape[0] - kH + 1, X.shape[1] - kW + 1
    pypto.set_vec_tile_shapes(8, 8)
    for i_idx in pypto.loop(H_out, name="LOOP_L0_i", idx_name="i_idx"):
        for j_idx in pypto.loop(W_out, name="LOOP_L1_j", idx_name="j_idx"):
            pypto.set_vec_tile_shapes(8, 8)
            window = X[i_idx:i_idx + kH, j_idx:j_idx + kW]
            mul_result = pypto.mul(window, grad_Y)
            sum_h = pypto.sum(mul_result, dim=0, keepdim=False)
            scalar = pypto.sum(sum_h, dim=0, keepdim=False)
            result_1x1 = pypto.reshape(scalar, [1, 1])
            pypto.assemble(result_1x1, [i_idx, j_idx], grad_K)


@pypto.frontend.jit()
def corr2d_backward_x_kernel(
    grad_Y_pad: pypto.Tensor([], pypto.DT_FP32),
    K:          pypto.Tensor([], pypto.DT_FP32),
    grad_X:     pypto.Tensor([], pypto.DT_FP32),
):
    """输入梯度：对宿主侧已 padding 的 grad_Y 与 K_rot180 做互相关。"""
    kH, kW = K.shape[0], K.shape[1]
    pypto.set_vec_tile_shapes(8, 8)

    idx_h = pypto.arange(kH - 1, -1, -1)
    idx_w = pypto.arange(kW - 1, -1, -1)
    idx_h_2d = pypto.expand_clone(pypto.reshape(idx_h, [kH, 1]), [kH, kW])
    idx_w_2d = pypto.expand_clone(pypto.reshape(idx_w, [1, kW]), [kH, kW])
    K_rot = pypto.gather(pypto.gather(K, 0, idx_h_2d), 1, idx_w_2d)

    H_out = grad_Y_pad.shape[0] - kH + 1
    W_out = grad_Y_pad.shape[1] - kW + 1
    for i_idx in pypto.loop(H_out, name="LOOP_L0_i", idx_name="i_idx"):
        for j_idx in pypto.loop(W_out, name="LOOP_L1_j", idx_name="j_idx"):
            pypto.set_vec_tile_shapes(8, 8)
            window = grad_Y_pad[i_idx:i_idx + kH, j_idx:j_idx + kW]
            mul_result = pypto.mul(window, K_rot)
            sum_h = pypto.sum(mul_result, dim=0, keepdim=False)
            scalar = pypto.sum(sum_h, dim=0, keepdim=False)
            result_1x1 = pypto.reshape(scalar, [1, 1])
            pypto.assemble(result_1x1, [i_idx, j_idx], grad_X)


class PyPTOCorr2DFunction(torch.autograd.Function):
    """基于 PyPTO 互相关 kernel 的 autograd Function (4D, 单通道)。"""

    @staticmethod
    def forward(ctx, X, K):
        X_c = X.contiguous()
        K_c = K.contiguous()

        n, _, h, w = X_c.shape
        _, _, kH, kW = K_c.shape
        out_h = h - kH + 1
        out_w = w - kW + 1

        Y = torch.zeros((n, 1, out_h, out_w), dtype=X_c.dtype, device=X_c.device)
        K_2d = K_c[0, 0]

        for batch_idx in range(n):
            corr2d_forward_kernel(X_c[batch_idx, 0], K_2d, Y[batch_idx, 0])

        ctx.save_for_backward(X_c, K_c)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, K = ctx.saved_tensors
        grad_output_c = grad_output.contiguous()

        n, _, h, w = X.shape
        _, _, kH, kW = K.shape

        grad_X = torch.zeros_like(X)
        grad_K = torch.zeros_like(K)
        K_2d = K[0, 0]

        grad_K_acc = torch.zeros_like(K_2d)

        for batch_idx in range(n):
            gY_pad = torch.nn.functional.pad(
                grad_output_c[batch_idx, 0],
                (kW - 1, kW - 1, kH - 1, kH - 1),
            )
            corr2d_backward_x_kernel(
                gY_pad,
                K_2d,
                grad_X[batch_idx, 0],
            )

            grad_K_one = torch.zeros_like(K_2d)
            corr2d_backward_k_kernel(
                grad_output_c[batch_idx, 0],
                X[batch_idx, 0],
                grad_K_one,
            )
            grad_K_acc += grad_K_one

        grad_K[0, 0] = grad_K_acc
        return grad_X, grad_K


def corr2d(X, K):
    """二维互相关函数式 API，接受 (H,W) 张量。"""
    X_4d = X.unsqueeze(0).unsqueeze(0)
    K_4d = K.unsqueeze(0).unsqueeze(0)

    Y_4d = PyPTOCorr2DFunction.apply(X_4d, K_4d)
    return Y_4d[0, 0]


class PyPTOConv2d(nn.Module):
    """原始二维卷积层 (基于 PyPTO 互相关 kernel, 单通道, stride=1, padding=0)。"""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=(1, 2),
        bias=False,
        device="npu:0",
        dtype=torch.float32,
    ):
        super().__init__()

        if bias:
            raise NotImplementedError(
                "当前实现不支持 bias，请使用 bias=False。"
            )
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"当前实现仅支持单通道输入输出（in_channels=1, out_channels=1），"
                f"实际传入 in_channels={in_channels}, out_channels={out_channels}。"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1], device=device, dtype=dtype)
        )

    def forward(self, x):
        return PyPTOCorr2DFunction.apply(x, self.weight)


# ── 从 06.03_padding_and_strides.ipynb 提取的支持 stride/padding 的互相关 ──


@pypto.frontend.jit()
def corr2d_kernel_strided(
    input:  pypto.Tensor([], pypto.DT_FP32),
    kernel: pypto.Tensor([], pypto.DT_FP32),
    padded: pypto.Tensor([], pypto.DT_FP32),
    output: pypto.Tensor([], pypto.DT_FP32),
    stride_h: int, stride_w: int,
    pad_top: int, pad_left: int,
):
    """支持 stride 和 padding 的互相关 kernel (06.03 版)。

    padded 为宿主持有的零画布；kernel 只负责把 input 写入画布内部，
    避免在 kernel 内创建局部临时张量。
    """
    pypto.set_vec_tile_shapes(16, 16)
    pypto.assemble(input, [pad_top, pad_left], padded)

    kH, kW = kernel.shape[0], kernel.shape[1]
    H_out = (padded.shape[0] - kH) // stride_h + 1
    W_out = (padded.shape[1] - kW) // stride_w + 1
    for i_idx in pypto.loop(H_out, name="LOOP_L0_i", idx_name="i_idx"):
        for j_idx in pypto.loop(W_out, name="LOOP_L1_j", idx_name="j_idx"):
            pypto.set_vec_tile_shapes(16, 16)
            i_start = i_idx * stride_h
            j_start = j_idx * stride_w
            window = pypto.view(padded, [kH, kW], [i_start, j_start])
            mul_result = pypto.mul(window, kernel)
            sum_h = pypto.sum(mul_result, dim=0, keepdim=True)
            result_1x1 = pypto.sum(sum_h, dim=1, keepdim=True)
            pypto.assemble(result_1x1, [i_idx, j_idx], output)


def conv2d(X, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), K=None):
    """支持 stride/padding 的二维卷积函数式 API (06.03 版)。

    接受 2D 张量 X(H,W)，返回 2D 输出。
    可选参数 K: 自定义卷积核，为 None 时随机生成。
    """
    kH, kW = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    if K is None:
        K = torch.rand(kH, kW, device=X.device)
    H, W = X.shape
    X_c, K_c = X.contiguous(), K.contiguous()

    H_padded = H + 2 * p_h
    W_padded = W + 2 * p_w
    H_out = (H_padded - kH) // s_h + 1
    W_out = (W_padded - kW) // s_w + 1
    if H_out <= 0 or W_out <= 0:
        return torch.zeros((0, 0), dtype=X_c.dtype, device=X_c.device)

    Y = torch.zeros((H_out, W_out), dtype=X_c.dtype, device=X_c.device)
    # 零画布由宿主持有，避免 kernel 内的局部临时张量在重复调用时被复用
    padded = torch.zeros((H_padded, W_padded), dtype=X_c.dtype, device=X_c.device)
    corr2d_kernel_strided(X_c, K_c, padded, Y, s_h, s_w, p_h, p_w)
    return Y


# ── 从 06.04_channels.ipynb 提取的多通道卷积 ──


def corr2d_multi_in(X, K):
    """多输入通道互相关。"""
    return sum(corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    """多输入多输出通道互相关。"""
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


@pypto.frontend.jit()
def corr2d_multi_in_out_1x1(
    X: pypto.Tensor([], pypto.DT_FP32),
    K: pypto.Tensor([], pypto.DT_FP32),
    Y: pypto.Tensor([], pypto.DT_FP32),
):
    """1x1 卷积: 等价于全连接层的矩阵乘法。"""
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X_2d = pypto.reshape(X, (c_i, h * w))
    K_2d = pypto.reshape(K, (c_o, c_i))
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out = pypto.matmul(K_2d, X_2d, out_dtype=X.dtype)
    Y[:] = pypto.reshape(out, (c_o, h, w))
