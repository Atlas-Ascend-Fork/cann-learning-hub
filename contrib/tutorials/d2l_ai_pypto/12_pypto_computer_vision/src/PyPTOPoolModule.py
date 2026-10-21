"""PyPTO Pool2d — AvgPool2d / MaxPool2d (fwd + bwd)

按参数走两条路径:

- **快路径 (stride==kernel 且无 padding 且整除)**: PyPTO kernel, 把窗口
  维拆出来做 reshape + sum/amax 归约, 并合并 N*C 维 (减少 N 倍调用):
    avg fwd: (NC,H,W)→(NC,H/kh,kh,W)→sum(dim=2)→(NC,H/kh,W/kw,kw)→sum(dim=3)
    avg bwd: expand_clone 上采样 kh×kw 后除面积
    max fwd: 同 avg fwd 但 amax
  整个 kernel 无循环, 固定 ~7 个 op。
- **通用路径 (重叠/padding)**: 直接派发 torch 原生 kernel (与 autograd
  逐位一致): max 用 ``aten.max_pool2d_with_indices(_backward)``, avg 用
  ``F.avg_pool2d`` + ``aten.avg_pool2d_backward``。

> 调优变更 (2026-09-26): 通用路径由旧版 host im2col + argmax + col2im 散射
> 改为原生 aten kernel。旧实现需在 host 物化整张 cols 并逐窗口散射, 大
> batch 下慢 1~2 个数量级 (b32 64ch 128×128 k3s2p1 实测 ~134ms, 原生
> ~1.6ms)。逐位一致性由 ``src/selftest`` 的 pool 用例保证。
> 另: 快路径增加空间上限 ``_FAST_MAX_HW`` —— PyPTO 快路径 kernel 按 shape
> 特化, 编译耗时 ∝ H×W, 大特征图 (如 VGG 300×450) 单 kernel 编译可达数分钟,
> 超过上限改走原生 (快路径仅用于小 spatial)。
"""

import torch
import torch.nn.functional as F
from torch import nn

import pypto

# 单次 pool kernel 处理的元素上限 (codegen 体积 ∝ 张量规模, 全展开)
_POOL_CAP_ELEMS = 4 * 1024 * 1024
# 快路径空间上限: pypto 快路径 kernel 按 shape 特化, 编译体积/耗时 ∝ H×W
# (实测 maxpool k2s2: 128×128 约 10s, 300×450 约 292s)。超过此值时快路径
# 的 PyPTO kernel (reshape+sum/amax 全展开) 编译不可接受, 改走原生 aten
# (零编译, 逐位一致)。小 spatial 仍走 PyPTO 快路径。
_FAST_MAX_HW = 4096        # (64x64) 以上的输入空间走原生通用路径


# ── PyPTO kernels ──


@pypto.frontend.jit()
def _avgpool_fwd_fast_kernel(
    X:   pypto.Tensor([], pypto.DT_FP32),   # (N, C, H, W)
    out: pypto.Tensor([], pypto.DT_FP32),   # (N, C, H//kh, W//kw)
    p_h: int,
    p_w: int,
    area: float,
):
    """非重叠 avg pool fwd: reshape + 两次 sum + 除面积 (合并 N*C)."""
    N, C, H, W = X.shape
    NC = N * C
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    v = pypto.reshape(X, [NC, H, W])
    r1 = pypto.reshape(v, [NC, H // p_h, p_h, W])
    s1 = pypto.sum(r1, dim=2, keepdim=True)
    r2 = pypto.reshape(s1, [NC, H // p_h, W // p_w, p_w])
    s2 = pypto.sum(r2, dim=3, keepdim=True)
    r3 = pypto.reshape(s2, [NC, H // p_h, W // p_w])
    g = pypto.reshape(r3, [N, C, H // p_h, W // p_w])
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    out.move(pypto.div(g, area))


@pypto.frontend.jit()
def _avgpool_bwd_fast_kernel(
    grad_Y: pypto.Tensor([], pypto.DT_FP32),   # (N, C, H_out, W_out)
    out:    pypto.Tensor([], pypto.DT_FP32),   # (N, C, H_out*kh, W_out*kw)
    p_h: int,
    p_w: int,
    area: float,
):
    """非重叠 avg pool bwd: grad_Y 上采样 kh×kw 后除面积 (合并 N*C)."""
    N, C, H_out, W_out = grad_Y.shape
    NC = N * C
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    v = pypto.reshape(grad_Y, [NC, H_out, W_out])
    r1 = pypto.reshape(v, [NC, H_out, W_out, 1])
    e1 = pypto.expand_clone(r1, [NC, H_out, W_out, p_w])
    r2 = pypto.reshape(e1, [NC, H_out, W_out * p_w])
    e2 = pypto.expand_clone(
        pypto.reshape(r2, [NC, H_out, 1, W_out * p_w]),
        [NC, H_out, p_h, W_out * p_w])
    r3 = pypto.reshape(e2, [NC, H_out * p_h, W_out * p_w])
    g_up = pypto.reshape(r3, [N, C, H_out * p_h, W_out * p_w])
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    out.move(pypto.div(g_up, area))


@pypto.frontend.jit()
def _maxpool_fwd_fast_kernel(
    X:   pypto.Tensor([], pypto.DT_FP32),   # (N, C, H, W)
    out: pypto.Tensor([], pypto.DT_FP32),   # (N, C, H//kh, W//kw)
    p_h: int,
    p_w: int,
):
    """非重叠 max pool fwd: reshape + 两次 amax (合并 N*C)."""
    N, C, H, W = X.shape
    NC = N * C
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    v = pypto.reshape(X, [NC, H, W])
    r1 = pypto.reshape(v, [NC, H // p_h, p_h, W])
    s1 = pypto.amax(r1, dim=2, keepdim=True)
    r2 = pypto.reshape(s1, [NC, H // p_h, W // p_w, p_w])
    s2 = pypto.amax(r2, dim=3, keepdim=True)
    r3 = pypto.reshape(s2, [NC, H // p_h, W // p_w])
    g = pypto.reshape(r3, [N, C, H // p_h, W // p_w])
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    out.move(g)


# ── host 辅助 ──


def _to_pair(v):
    """标量/二元组 → (int, int)。"""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _pool_geom(H, W, kH, kW, s_h, s_w, pad):
    """汇聚输出尺寸 (H_out, W_out)。"""
    return (H + 2 * pad - kH) // s_h + 1, (W + 2 * pad - kW) // s_w + 1


def _is_fast(kH, kW, s_h, s_w, pad, H, W):
    """快路径判定: 非重叠 (stride==kernel)、无 padding、空间整除, 且空间
    不超过 _FAST_MAX_HW (否则 PyPTO kernel 编译代价不可接受, 见常量注释)。"""
    return (s_h == kH and s_w == kW and pad == 0
            and H % kH == 0 and W % kW == 0 and H * W <= _FAST_MAX_HW)


# ── avg pool ──


def _avg_pool_forward(X, kH, kW, s_h, s_w, pad):
    N, C, H, W = X.shape
    H_out, W_out = _pool_geom(H, W, kH, kW, s_h, s_w, pad)
    Xc = X.contiguous()
    if _is_fast(kH, kW, s_h, s_w, pad, H, W):
        Y = torch.empty(N, C, H_out, W_out, device=X.device, dtype=X.dtype)
        _avgpool_fwd_fast_kernel(Xc, Y, kH, kW, float(kH * kW))
        return Y
    # 通用路径: 原生 F.avg_pool2d (count_include_pad=True, 语义等价于
    # 逐位求和后除以 kH*kW)
    return F.avg_pool2d(Xc, [kH, kW], [s_h, s_w], pad)


def _avg_pool_backward(grad_Y, input_shape, kH, kW, s_h, s_w, pad):
    N, C, H, W = input_shape
    gY = grad_Y.contiguous()
    if _is_fast(kH, kW, s_h, s_w, pad, H, W):
        grad_X = torch.empty(N, C, H, W, device=gY.device, dtype=gY.dtype)
        _avgpool_bwd_fast_kernel(gY, grad_X, kH, kW, float(kH * kW))
        return grad_X
    # 通用路径: 原生 aten 反向 (与 torch autograd 逐位一致, maxdiff=0);
    # self 仅用于形状推导, 传空张量即可
    X_dummy = torch.empty(N, C, H, W, device=gY.device, dtype=gY.dtype)
    return torch.ops.aten.avg_pool2d_backward(
        gY, X_dummy, [kH, kW], [s_h, s_w], pad, False, True, None)


# ── max pool ──


def _max_pool_forward(X, kH, kW, s_h, s_w, pad):
    """返回 (Y, idx): 通用路径的 idx 供 bwd 复用 (快路径为 None)。"""
    N, C, H, W = X.shape
    H_out, W_out = _pool_geom(H, W, kH, kW, s_h, s_w, pad)
    Xc = X.contiguous()
    if _is_fast(kH, kW, s_h, s_w, pad, H, W):
        # NC 分块: codegen 体积 ∝ 单次 kernel 的张量规模, 大输入按 batch 拆分
        # (必须保持 4 维, kernel 内部按 N,C,H,W 解包)
        Y = torch.empty(N, C, H_out, W_out, device=X.device, dtype=X.dtype)
        n_chunk = max(1, _POOL_CAP_ELEMS // max(1, C * H * W))
        if n_chunk >= N:
            _maxpool_fwd_fast_kernel(Xc, Y, kH, kW)
        else:
            for n0 in range(0, N, n_chunk):
                n1 = min(n0 + n_chunk, N)
                _maxpool_fwd_fast_kernel(Xc[n0:n1], Y[n0:n1], kH, kW)
        return Y, None
    # 通用路径: 原生带索引前向 (idx 供 bwd 复用)
    return torch.ops.aten.max_pool2d_with_indices(
        Xc, [kH, kW], [s_h, s_w], [pad, pad], [1, 1], False)


def _max_pool_backward(grad_Y, X, kH, kW, s_h, s_w, pad, idx):
    """原生带索引反向 (与 torch autograd 逐位一致); idx 缺失时原生重算。"""
    Xc = X.contiguous()
    gY = grad_Y.contiguous()
    if idx is None:
        _, idx = torch.ops.aten.max_pool2d_with_indices(
            Xc, [kH, kW], [s_h, s_w], [pad, pad], [1, 1], False)
    return torch.ops.aten.max_pool2d_with_indices_backward(
        gY, Xc, [kH, kW], [s_h, s_w], [pad, pad], [1, 1], False, idx)


# ── autograd Function + nn.Module ──


class PyPTOAvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, kH, kW, s_h, s_w, pad):
        ctx.input_shape = X.shape
        ctx.params = (kH, kW, s_h, s_w, pad)
        return _avg_pool_forward(X, kH, kW, s_h, s_w, pad)

    @staticmethod
    def backward(ctx, grad_output):
        return (_avg_pool_backward(grad_output, ctx.input_shape, *ctx.params),
                None, None, None, None, None)


class PyPTOMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, kH, kW, s_h, s_w, pad):
        Y, idx = _max_pool_forward(X, kH, kW, s_h, s_w, pad)
        ctx.params = (kH, kW, s_h, s_w, pad)
        if idx is not None:
            ctx.save_for_backward(X, idx)       # 通用路径: bwd 复用原生索引
        else:
            ctx.save_for_backward(X)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X = ctx.saved_tensors[0]
        idx = ctx.saved_tensors[1] if len(ctx.saved_tensors) > 1 else None
        return (_max_pool_backward(grad_output, X, *ctx.params, idx),
                None, None, None, None, None)


class PyPTOAvgPool2d(nn.Module):
    """PyPTO 二维平均汇聚: 非重叠整除走 PyPTO 快路径 kernel,
    重叠/带 padding 走 torch 原生 kernel (语义逐位一致)。"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_pair(kernel_size)
        self.stride = _to_pair(stride) if stride is not None else self.kernel_size
        self.padding = _to_pair(padding)

    def forward(self, x):
        kH, kW = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        if p_h != p_w:
            raise NotImplementedError("对称 padding 仅支持单值")
        return PyPTOAvgPool2dFunction.apply(x, kH, kW, s_h, s_w, p_h)


class PyPTOMaxPool2d(nn.Module):
    """PyPTO 二维最大汇聚: 非重叠整除走 PyPTO 快路径 kernel,
    重叠/带 padding 走 torch 原生带索引 kernel (fwd/bwd 逐位一致)。"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_pair(kernel_size)
        self.stride = _to_pair(stride) if stride is not None else self.kernel_size
        self.padding = _to_pair(padding)

    def forward(self, x):
        kH, kW = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        if p_h != p_w:
            raise NotImplementedError("对称 padding 仅支持单值")
        return PyPTOMaxPool2dFunction.apply(x, kH, kW, s_h, s_w, p_h)
