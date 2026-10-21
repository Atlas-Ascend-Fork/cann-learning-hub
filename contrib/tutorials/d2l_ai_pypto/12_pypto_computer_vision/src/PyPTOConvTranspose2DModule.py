"""PyPTO ConvTranspose2d 模块 (相位分解 + matmul 路线, 与 PyPTOConv2DModule 同风格)

fwd 走 PyPTO ``matmul`` (AIC/Cube, FP32, 相位分解); dX/dW 走原生 aten
(F.conv2d), 与第7章 conv 后向原生化一致。

转置卷积 (transposed conv / deconv):
    Y[oc, h, w] = Σ_ic Σ_kh Σ_kw X[ic, h_src, w_src] * K[oc, ic, kh, kw]
    h = h_src * s - p + kh,  w = w_src * s - p + kw

对大 stride (s) 采用**相位分解** fwd, 避免 64x64 大核下 im2col 内存爆炸:
    记 r = ceil(K / s)。输出像素 (h, w) 只依赖 r×r 个输入像素:
        h = h'*s + a - p,  a = (h + p) mod s,  h' = (h + p - a) / s
        kh = a + s*m,  m ∈ [0, r)
    ⇒ Y[oc, h'*s+a-p, w'*s+b-p] = Σ_ic Σ_{m,n} X[ic, h'-m, w'-n] * K[oc, ic, a+s*m, b+s*n]

- fwd:   cols = im2col_{r×r}(pad(X, r-1))              (M, Cin*r²)
         out_tmp = cols @ W_full^T                     (M, Cout*s²)  [单次大 matmul]
         Y = pixel_shuffle(out_tmp) 后裁剪有效区
         (W_full 展平为 2D: (Cout*s², Cin*r²), 满足 matmul 同维约束)
- dX:    dX = conv2d(dY, W, stride=s, padding=p)      原生 (转置卷积 backward 恒等式)
- dW:    dW = unfold(dY) @ X                          原生 (im2col+matmul)

限制: groups=1, dilation=1, output_padding=0。
"""

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .PyPTOConv2DModule import (
        _matmul_bt_capped, _im2col_strided, _to_pair, _pad4d)
except ImportError:
    from PyPTOConv2DModule import (
        _matmul_bt_capped, _im2col_strided, _to_pair, _pad4d)


def _build_W_full(K_conv, r_h, r_w, s_h, s_w):
    """K_conv (Cout, Cin, Kh, Kw) → W_full (Cout*s_h*s_w, Cin*r_h*r_w).

    核零 pad 到 r*s 再切成 (m, a; n, b) 相位块并翻转 m/n 序后展平,
    使单次 matmul 输出列即为 (Cout, a, b) pixel_shuffle 排布。
    """
    Cout, Cin, Kh, Kw = K_conv.shape
    Kpad = F.pad(K_conv, [0, r_w * s_w - Kw, 0, r_h * s_h - Kh])
    Kph = Kpad.reshape(Cout, Cin, r_h, s_h, r_w, s_w).permute(0, 1, 2, 4, 3, 5)
    # (Cout, Cin, m, n, a, b) --flip(m,n)--> 重排 (Cout, a, b, Cin, m, n) 后展平,
    # 行序 = out_tmp 列的 pixel_shuffle 分解 (Cout, a, b), 列序 = im2col 的 (Cin, m, n)
    return Kph.flip(2, 3).permute(0, 4, 5, 1, 2, 3).reshape(
        Cout * s_h * s_w, Cin * r_h * r_w).contiguous()


def _conv_transpose_forward(X, K_conv, bias, stride, padding):
    N, Cin, Hin, Win = X.shape
    Cout = K_conv.shape[0]
    Kh, Kw = K_conv.shape[2], K_conv.shape[3]
    s_h, s_w = _to_pair(stride)
    p_h, p_w = _to_pair(padding)
    r_h, r_w = -(-Kh // s_h), -(-Kw // s_w)
    Hout = (Hin - 1) * s_h - 2 * p_h + Kh
    Wout = (Win - 1) * s_w - 2 * p_w + Kw

    Xg = _pad4d(X, r_h - 1, r_w - 1)
    cols = _im2col_strided(Xg, r_h, r_w, (1, 1))
    W_full = _build_W_full(K_conv, r_h, r_w, s_h, s_w)
    out_tmp = torch.empty(cols.shape[0], Cout * s_h * s_w,
                          device=X.device, dtype=X.dtype)
    _matmul_bt_capped(cols, W_full, out_tmp)

    # (M, Cout*s²) → (N, Hg, Wg, Cout, s_h, s_w) → (N, Cout, Hg*s_h, Wg*s_w)
    Hg, Wg = Xg.shape[2] - r_h + 1, Xg.shape[3] - r_w + 1
    raw = out_tmp.reshape(N, Hg, Wg, Cout, s_h, s_w
                          ).permute(0, 3, 1, 4, 2, 5
                          ).reshape(N, Cout, Hg * s_h, Wg * s_w)
    Y = raw[:, :, p_h:p_h + Hout, p_w:p_w + Wout].contiguous()
    if bias is not None:
        Y = Y + bias.view(1, Cout, 1, 1)
    return Y


def _conv_transpose_dW(X, grad_Y, Kh, Kw, stride, padding):
    """dW 原生实现: 对 dY 做 unfold (im2col) 后与 X 做 matmul。

    dW[cin,cout,kh,kw] = Σ_{n,h,w} X[n,cin,h,w] * gY[n,cout,h*s+kh-p,w*s+kw-p]。

    不走 torch 的 ConvTranspose 权重反传 (``aclnnConvolutionBackward``) —— 实测
    该算子在 64×64 大核 + stride=32 (FCN) 形状下触发 NPU TransData vector-core
    out-of-bounds 异常, 且小 shape 结果亦有偏差。unfold+matmul 在 NPU 上稳定,
    且单次 matmul 无分块累加, fp32 精度好。
    """
    s_h, s_w = _to_pair(stride)
    p_h, p_w = _to_pair(padding)
    gY_pad = F.pad(grad_Y, (p_w, p_w, p_h, p_h))
    cols = F.unfold(gY_pad, (Kh, Kw), stride=(s_h, s_w))   # (N, Cout*Kh*Kw, L)
    N, Ck, L = cols.shape
    Cin = X.shape[1]
    X2 = X.permute(0, 2, 3, 1).reshape(N, L, Cin)           # (N, L, Cin)
    dW_flat = (cols @ X2).sum(0)                            # (Cout*Kh*Kw, Cin)
    return dW_flat.reshape(Ck // (Kh * Kw), Kh, Kw, Cin
                           ).permute(3, 0, 1, 2).contiguous()


class PyPTOConvTranspose2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, stride, padding, output_padding, dilation, bias):
        if _to_pair(dilation) != (1, 1):
            raise NotImplementedError("AIC matmul 路线不支持 dilation")
        if _to_pair(output_padding) != (0, 0):
            raise NotImplementedError("AIC matmul 路线不支持 output_padding")
        s_h, s_w = _to_pair(stride)
        X_c, W_c = X.contiguous(), W.contiguous()
        # ConvT weight (Cin, Cout, kH, kW) → conv 视角核 (Cout, Cin, kH, kW)
        K_conv = W_c.permute(1, 0, 2, 3).contiguous()
        Y = _conv_transpose_forward(X_c, K_conv, bias, stride, padding)
        ctx.save_for_backward(X_c, K_conv)
        ctx.params = (s_h, s_w, padding)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, K_conv = ctx.saved_tensors
        s_h, s_w, padding = ctx.params
        grad_Y = grad_output.contiguous()
        # ConvT weight (Cin, Cout, kH, kW) = conv 视角核的通道互换
        W = K_conv.permute(1, 0, 2, 3)

        # dX = conv(dY, W, s, p): 转置卷积 backward 恒等式, 原生 aten
        dX = (F.conv2d(grad_Y, W, stride=(s_h, s_w), padding=padding)
              if ctx.needs_input_grad[0] else None)
        # dW: dY unfold 后与 X matmul (避开 NPU ConvTranspose 权重反传的 TransData bug)
        dW = (_conv_transpose_dW(X, grad_Y, W.shape[2], W.shape[3],
                                 (s_h, s_w), padding)
              if ctx.needs_input_grad[1] else None)
        db = (grad_Y.sum(dim=(0, 2, 3)).contiguous()
              if ctx.needs_input_grad[6] else None)
        return dX, dW, None, None, None, None, db


class PyPTOConvTranspose2d(nn.Module):
    """PyPTO 二维转置卷积 (相位分解 + AIC matmul, FP32): fwd PyPTO, bwd 原生 aten."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, device="cpu", dtype=torch.float32):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("AIC matmul 路线不支持 groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(
            tuple([in_channels, out_channels]) + self.kernel_size,
            device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=2.23606797749979)  # sqrt(5)
        self.bias = (nn.Parameter(torch.zeros(out_channels, device=device,
                                              dtype=dtype))
                     if bias else None)

    def forward(self, x):
        return PyPTOConvTranspose2dFunction.apply(
            x, self.weight, self.stride, self.padding,
            self.output_padding, self.dilation, self.bias)
