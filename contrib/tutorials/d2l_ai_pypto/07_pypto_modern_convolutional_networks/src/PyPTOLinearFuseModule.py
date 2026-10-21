"""PyPTO Linear — 全连接层 (fwd matmul kernel + host 叠加 bias, bwd 原生)

- fwd: ``C = X @ W`` 单个 pypto matmul kernel。M 维向上取桶 (16/32/64/…)
  编译: 不足桶值的 M 行补零、输出裁回 (零行对结果无贡献, 逐位精确) ——
  逐层 shape 检查 (M=1) 与训练 (M=batch) 共用同一 kernel, 显著减少
  JIT 编译变体数。bias 不在 kernel 内融合 (extend_params 的 bias_tensor
  在多核切 K 时结果错误, 框架限制), 由 host 加法叠加。
- bwd: 原生 aten mm/sum (零 JIT 编译)。weight 为 (in, out) 转置约定
  (X @ W 直接相乘), 与 nn.Linear 的 (out, in) 相反, 故不走
  aten.linear_backward (需每步 W.t().contiguous() 整权重拷贝);
  mm 的 strided view 由原生 kernel 免拷贝处理。

cube tile 固定 [32,32]/[64,64]/[64,64], 适配教程网络的小批量全连接形状。
"""

import torch
from torch import nn

import pypto

_M_BUCKETS = (16, 32, 64, 128, 256)   # M 向上取桶, 减少 JIT 编译变体


def _bucket_m(M):
    """M 向上取桶值; 超过最大桶返回 None (用实际值编译)。"""
    for b in _M_BUCKETS:
        if M <= b:
            return b
    return None


@pypto.frontend.jit()
def linear_forward_kernel(
    X: pypto.Tensor([], pypto.DT_FP32),
    W: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out[:] = pypto.matmul(X, W, pypto.DT_FP32)


class PyPTOLinearFuseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b):
        X_c, W_c = X.contiguous(), W.contiguous()
        M = X_c.shape[0]
        Mp = _bucket_m(M)
        if Mp is not None and Mp != M:
            # 补零行参与计算, 裁回真实 M 行 (零行对结果无贡献, 逐位精确)
            Xp = torch.nn.functional.pad(X_c, (0, 0, 0, Mp - M))
            out_full = torch.empty(Mp, W_c.shape[1], dtype=X.dtype, device=X.device)
            linear_forward_kernel(Xp, W_c, out_full)
            out = out_full[:M]
        else:
            out = torch.empty(M, W_c.shape[1], dtype=X.dtype, device=X.device)
            linear_forward_kernel(X_c, W_c, out)
        out.add_(b.view(1, -1))   # bias host 叠加 (kernel 融合有框架限制, 见模块 docstring)
        ctx.save_for_backward(X_c, W_c)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X, W = ctx.saved_tensors
        g = grad_out.contiguous()
        grad_X = torch.mm(g, W.t())       # (M, in), strided view 免拷贝
        grad_W = torch.mm(X.t(), g)       # (in, out)
        grad_b = g.sum(dim=0)
        return grad_X, grad_W, grad_b


class PyPTOLinear(nn.Module):
    """PyPTO 全连接层, 权重布局与 nn.Linear 一致: (out_features, in_features)。"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重须显式初始化 (未初始化的 torch.empty 内存未定义, 会产生
        # 巨大值/NaN, 导致 softmax 饱和、训练不下降); bias 置零
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        return PyPTOLinearFuseFunction.apply(X, self.weight, self.bias)
