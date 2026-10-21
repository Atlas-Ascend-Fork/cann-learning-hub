"""PyPTO BatchNorm (fwd + bwd) — 7.5 节批量规范化层

把批量规范化转成"按行归约"问题, 归约方向统一为最后一维:

- **卷积形态 (num_dims=4)**: X (N,C,H,W) → (N*C, H*W) 纯 reshape 零拷贝 view。
  行 = (n,c) 组。注意: 行内统计只是"样本内"统计, 而 BN 需要整个 batch 的
  通道统计, 因此 kernel 内做两阶段归约: 每行先求局部和 (R,1), 再按 n 组
  求和得到全局通道统计 ((1,C)), 展开回 (R,1) 后做归一化。
- **全连接形态 (num_dims=2)**: X (N,F) → host 转置 (F,N), 行 = 特征、列 = batch
  (F 很小, 一次小拷贝)。n_groups=1 时两阶段归约自然退化为行内归约。
- **gamma/beta/moving 统计**: host 侧 reshape/expand 成 (R,1) 零拷贝 view,
  kernel 内用 ``pypto.expand_clone`` 展开到 (R,M)。

kernel 划分 (每层每步 fwd+bwd 各 1 个 kernel):
- ``_bn_fwd_train_kernel``: 两阶段 mean/var → ``rsqrt`` → x_hat → γ/β 仿射
  (结构同第 10 章 layer_norm_fwd); 另输出通道级 mean/var 供 host 做
  momentum 更新。
- ``_bn_fwd_eval_kernel``: 用 moving 统计 (训练过程中 evaluate 走 net.eval())。
- ``_bn_bwd_train_kernel``: 标准 BN 反向三式 (通道均值用同样的两阶段归约),
  mean/var/inv/x_hat 由 X 重算 (前向只需保存 X)。
- ``_bn_bwd_eval_kernel``: eval 态反向 (固定统计, 无去中心化项)。

大 R 路径 (R = 行数 > ``_BN_MAX_R``, 如 batch=64 的深层 BN):
- 触发原因: 归约 kernel 的累加器 tile (R×8×4B) 超 MEM_UB, R 超上限时
  bwd kernel 编译直接失败并中断进程。
- 实现: host torch 两遍法统计 (``_host_stats``) + torch 逐元素归一化/三式,
  公式与 kernel 完全一致 (先均值再算平方差)。大 R 下也不复用
  ``_bn_fwd_eval_kernel`` (行 tile 迭代在大 R 下开销过高)。

限制: FP32; 输入需连续 (函数内 ``.contiguous()`` 防御); eps 硬编码 1e-5;
动量 0.9 (与 7.5 节手写实现一致, 注意与 nn.BatchNorm 的 0.1 不同)。
vec tile 用 (8, aligned): 逐行归约的 UB 预算随 cols 增长 (cols≤1024 时
双缓冲约 64KB < 192KB)。
"""

import torch
from torch import nn

import pypto

# 单 kernel 行数上限: R 超过 4096 后 bwd 归约 kernel 编译失败 (会中断进程),
# 更大的 R 由 host torch 路径承接 (公式与 kernel 一致, 见 forward/backward)。
_BN_MAX_R = 4096


# ── PyPTO kernels ──
# 所有 kernel 输入统一为 2D (R, M): 行 = 统计组, 列 = 归约元素。
#   conv: R = N*C, M = H*W, n_groups = N, rows_per_group = C
#   FC:   R = F,  M = N,  n_groups = 1, rows_per_group = F
# 两阶段归约: 先每行局部和 (R,1), 再按 n_groups 分组求和 (1, C),
# 均值项再除以 n_groups·M (方差为两遍法: 用全局通道均值算平方差, 数值稳定)。


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def _bn_fwd_train_kernel(
    X:        pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    gamma:    pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    beta:     pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    out:      pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    mean_ch:  pypto.Tensor([], pypto.DT_FP32),   # (1, R/n_groups)
    var_ch:   pypto.Tensor([], pypto.DT_FP32),   # (1, R/n_groups)
    n_rows:   int,
    n_cols:   int,
    n_groups: int,
):
    """训练态 BN 前向: y = γ·(x-mean)/sqrt(var+eps) + β, 两阶段通道归约。"""
    # 行 tile 收窄为 8 (同 layer_norm): 逐行归约的 UB 占用随 cols 增长
    pypto.set_vec_tile_shapes(8, ((n_cols + 7) // 8) * 8)
    # 第一阶段: 每行局部和 → 第二阶段: 按组求和得全局通道统计
    mch = pypto.mul(pypto.sum(pypto.reshape(
        pypto.sum(X, 1, True), [n_groups, n_rows // n_groups]), 0, True),
        1.0 / (n_groups * n_cols))                        # (1, C) 通道均值
    mean_row = pypto.reshape(
        pypto.expand_clone(mch, [n_groups, n_rows // n_groups]),
        [n_rows, 1])                                      # (R,1)
    xc = pypto.sub(X, mean_row)                           # (R,M)
    vch = pypto.mul(pypto.sum(pypto.reshape(
        pypto.sum(pypto.mul(xc, xc), 1, True),
        [n_groups, n_rows // n_groups]), 0, True),
        1.0 / (n_groups * n_cols))                        # (1, C) 通道方差
    var_row = pypto.reshape(
        pypto.expand_clone(vch, [n_groups, n_rows // n_groups]),
        [n_rows, 1])                                      # (R,1)
    inv_row = pypto.rsqrt(pypto.add(var_row, 1e-5))       # (R,1)
    x_hat = pypto.mul(xc, inv_row)                        # (R,M)
    g_exp = pypto.expand_clone(gamma, [n_rows, n_cols])
    b_exp = pypto.expand_clone(beta, [n_rows, n_cols])
    out.move(pypto.add(pypto.mul(x_hat, g_exp), b_exp))
    # 通道级统计输出 (供 host 动量更新): 各输出走自己的表达式链
    mean_ch.move(pypto.mul(pypto.sum(pypto.reshape(
        pypto.sum(X, 1, True), [n_groups, n_rows // n_groups]), 0, True),
        1.0 / (n_groups * n_cols)))
    xc2 = pypto.sub(X, pypto.reshape(
        pypto.expand_clone(mch, [n_groups, n_rows // n_groups]),
        [n_rows, 1]))
    var_ch.move(pypto.mul(pypto.sum(pypto.reshape(
        pypto.sum(pypto.mul(xc2, xc2), 1, True),
        [n_groups, n_rows // n_groups]), 0, True),
        1.0 / (n_groups * n_cols)))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def _bn_fwd_eval_kernel(
    X:      pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    mmean:  pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    mvar:   pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    gamma:  pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    beta:   pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    out:    pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    n_rows: int,
    n_cols: int,
):
    """eval 态 BN 前向: y = γ·(x-moving_mean)/sqrt(moving_var+eps) + β。"""
    pypto.set_vec_tile_shapes(8, ((n_cols + 7) // 8) * 8)
    mm_exp = pypto.expand_clone(mmean, [n_rows, n_cols])
    mv_exp = pypto.expand_clone(mvar, [n_rows, n_cols])
    g_exp = pypto.expand_clone(gamma, [n_rows, n_cols])
    b_exp = pypto.expand_clone(beta, [n_rows, n_cols])
    x_hat = pypto.mul(pypto.sub(X, mm_exp),
                      pypto.rsqrt(pypto.add(mv_exp, 1e-5)))
    out.move(pypto.add(pypto.mul(x_hat, g_exp), b_exp))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def _bn_bwd_train_kernel(
    X:           pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    gamma:       pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    grad_Y:      pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    grad_X:      pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    grad_gamma:  pypto.Tensor([], pypto.DT_FP32),   # (1, R/n_groups)
    grad_beta:   pypto.Tensor([], pypto.DT_FP32),   # (1, R/n_groups)
    n_rows:      int,
    n_cols:      int,
    n_groups:    int,
):
    """训练态 BN 反向 (标准三式): mean/var/inv/x_hat 由 X 重算 (同 layer_norm_bwd)。"""
    pypto.set_vec_tile_shapes(8, ((n_cols + 7) // 8) * 8)
    mean_ch = pypto.mul(pypto.sum(pypto.reshape(
        pypto.sum(X, 1, True), [n_groups, n_rows // n_groups]), 0, True),
        1.0 / (n_groups * n_cols))
    mean_row = pypto.reshape(
        pypto.expand_clone(mean_ch, [n_groups, n_rows // n_groups]),
        [n_rows, 1])
    xc = pypto.sub(X, mean_row)
    var_ch = pypto.mul(pypto.sum(pypto.reshape(
        pypto.sum(pypto.mul(xc, xc), 1, True),
        [n_groups, n_rows // n_groups]), 0, True),
        1.0 / (n_groups * n_cols))
    var_row = pypto.reshape(
        pypto.expand_clone(var_ch, [n_groups, n_rows // n_groups]),
        [n_rows, 1])
    inv_row = pypto.rsqrt(pypto.add(var_row, 1e-5))
    x_hat = pypto.mul(xc, inv_row)
    g_exp = pypto.expand_clone(gamma, [n_rows, n_cols])
    inv_exp = pypto.expand_clone(inv_row, [n_rows, n_cols])
    gy_g = pypto.mul(grad_Y, g_exp)                          # γ·gY
    # 三式的通道均值也走两阶段归约: 局部 (R,1) → 按组求和 → 展开回 (R,1)
    gm_ch = pypto.mul(pypto.sum(pypto.reshape(
        pypto.sum(gy_g, 1, True), [n_groups, n_rows // n_groups]),
        0, True), 1.0 / (n_groups * n_cols))
    gm_exp = pypto.expand_clone(
        pypto.reshape(pypto.expand_clone(
            gm_ch, [n_groups, n_rows // n_groups]), [n_rows, 1]),
        [n_rows, n_cols])
    gh_ch = pypto.mul(pypto.sum(pypto.reshape(
        pypto.sum(pypto.mul(gy_g, x_hat), 1, True),
        [n_groups, n_rows // n_groups]), 0, True),
        1.0 / (n_groups * n_cols))
    gh_exp = pypto.expand_clone(
        pypto.reshape(pypto.expand_clone(
            gh_ch, [n_groups, n_rows // n_groups]), [n_rows, 1]),
        [n_rows, n_cols])
    grad_X.move(pypto.mul(inv_exp,
                          pypto.sub(pypto.sub(gy_g, gm_exp),
                                    pypto.mul(x_hat, gh_exp))))
    # grad_gamma/beta: 列向求和 (无平均), 再按组求和得通道梯度
    grad_gamma.move(pypto.sum(pypto.reshape(
        pypto.sum(pypto.mul(grad_Y, x_hat), 1, True),
        [n_groups, n_rows // n_groups]), 0, True))
    grad_beta.move(pypto.sum(pypto.reshape(
        pypto.sum(grad_Y, 1, True),
        [n_groups, n_rows // n_groups]), 0, True))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def _bn_bwd_eval_kernel(
    X:           pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    mmean:       pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    mvar:        pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    gamma:       pypto.Tensor([], pypto.DT_FP32),   # (R, 1)
    grad_Y:      pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    grad_X:      pypto.Tensor([], pypto.DT_FP32),   # (R, M)
    grad_gamma:  pypto.Tensor([], pypto.DT_FP32),   # (1, R/n_groups)
    grad_beta:   pypto.Tensor([], pypto.DT_FP32),   # (1, R/n_groups)
    n_rows:      int,
    n_cols:      int,
    n_groups:    int,
):
    """eval 态 BN 反向: 统计固定, grad_X = gY/sqrt(mvar+eps) (无去中心化项)。"""
    pypto.set_vec_tile_shapes(8, ((n_cols + 7) // 8) * 8)
    mv_exp = pypto.expand_clone(mvar, [n_rows, n_cols])
    inv_exp = pypto.rsqrt(pypto.add(mv_exp, 1e-5))
    grad_X.move(pypto.mul(grad_Y, inv_exp))
    mm_exp = pypto.expand_clone(mmean, [n_rows, n_cols])
    x_hat = pypto.mul(pypto.sub(X, mm_exp), inv_exp)
    grad_gamma.move(pypto.sum(pypto.reshape(
        pypto.sum(pypto.mul(grad_Y, x_hat), 1, True),
        [n_groups, n_rows // n_groups]), 0, True))
    grad_beta.move(pypto.sum(pypto.reshape(
        pypto.sum(grad_Y, 1, True),
        [n_groups, n_rows // n_groups]), 0, True))


# ── autograd Function ──


class PyPTOBatchNormFunction(torch.autograd.Function):
    """BN autograd 包装: 训练态返回 (Y, new_mm, new_mv) 供动量更新。"""

    @staticmethod
    def _host_stats(Xv):
        """host 通道统计 (大 R 路径, 公式与 kernel 一致, 两遍法)。

        ``Xv`` 为 (n_groups, rows_per_group, M) view; 直接在最后两维归约
        得到 (1, rows_per_group, 1) 的通道 mean/var —— 用 torch 的
        ``var_mean`` 单趟融合归约 (同时出 mean/var) 代替逐行 sum + 分组
        sum + expand 的多 kernel 组合, 大 R 下显著减少 host 发射。
        """
        var, mean = torch.var_mean(Xv, dim=(0, 2), unbiased=False,
                                   keepdim=True)               # (1,C,1)
        return mean, var

    @staticmethod
    def forward(ctx, X, gamma, beta, moving_mean, moving_var, num_dims,
                training):
        X = X.contiguous()
        if num_dims == 4:
            N, C, H, W = X.shape
            R, M, n_groups = N * C, H * W, N
            rows_per_group = C
            X2 = X.reshape(R, M)                       # 零拷贝 view
            # gamma (1,C,1,1) → (1,C) → expand(N,C) → (N*C,1): 全部零拷贝 view
            g2 = gamma.reshape(1, C).expand(N, C).reshape(R, 1)
            b2 = beta.reshape(1, C).expand(N, C).reshape(R, 1)
            mm2 = moving_mean.reshape(1, C).expand(N, C).reshape(R, 1)
            mv2 = moving_var.reshape(1, C).expand(N, C).reshape(R, 1)
            # 输出 buffer 按最终形状 (N,C,H,W) 分配, kernel 写 (R,M) 视图,
            # 返回 base 张量 (自定义 Function 不能返回 view, 否则后续
            # in-place 操作会触发 autograd 的 view 修改报错)
            out2 = torch.empty(N, C, H, W, device=X.device, dtype=X.dtype)
            out_view = out2.reshape(R, M)
        else:
            N, F = X.shape
            R, M, n_groups = F, N, 1
            rows_per_group = F
            X2 = X.t().contiguous()                    # (F, N), 一次小转置
            g2 = gamma.reshape(F, 1)
            b2 = beta.reshape(F, 1)
            mm2 = moving_mean.reshape(F, 1)
            mv2 = moving_var.reshape(F, 1)
            out2 = torch.empty(F, M, device=X.device, dtype=X.dtype)
            out_view = out2
        ctx.num_dims = num_dims
        ctx.N = N
        ctx.C = C if num_dims == 4 else F
        ctx.H = H if num_dims == 4 else None
        ctx.W = W if num_dims == 4 else None
        ctx.R, ctx.M, ctx.n_groups = R, M, n_groups
        ctx.rows_per_group = rows_per_group
        ctx.host_stats = None          # 仅大 R (host) 路径填充, 供反向复用

        if training:
            if R > _BN_MAX_R:
                # 大 R: 全 torch 路径 (广播式通道统计 + torch 归一化)。
                # 注意: 大 R 下不能用 _bn_fwd_eval_kernel (行 tile 迭代
                # 开销过高), 用 torch 逐元素归一化。
                Xv = X2.reshape(n_groups, rows_per_group, M)
                mean, var = PyPTOBatchNormFunction._host_stats(Xv)
                g3 = g2.reshape(n_groups, rows_per_group, 1)
                b3 = b2.reshape(n_groups, rows_per_group, 1)
                # 归一化+仿射合并为 per-channel scale/shift (张量运算减半):
                # y = (x-mean)*inv*g + b = x*(inv*g) + (b - mean*inv*g)
                scale = torch.rsqrt(var + 1e-5) * g3               # (1,C,1)
                shift = b3 - mean * scale
                # 写回预分配 base buffer (自定义 Function 返回多输出时,
                # 若某输出是 view 会被 autograd 的 inplace 检查拒绝)
                out_view.copy_((Xv * scale + shift).reshape(R, M))
                mean_ch = mean.reshape(1, rows_per_group)
                var_ch = var.reshape(1, rows_per_group)
                ctx.host_stats = (mean.detach(), var.detach())  # 反向复用 (常数)
            else:
                mean_ch = torch.empty(1, rows_per_group,
                                      device=X.device, dtype=X.dtype)
                var_ch = torch.empty_like(mean_ch)
                _bn_fwd_train_kernel(X2, g2, b2, out_view, mean_ch, var_ch,
                                     R, M, n_groups)
            # momentum 更新在 host 用小张量完成 (同 linear 的 host 叠加 bias 风格)
            new_mm = 0.9 * moving_mean + 0.1 * mean_ch.reshape(
                moving_mean.shape)
            new_mv = 0.9 * moving_var + 0.1 * var_ch.reshape(
                moving_var.shape)
            ctx.mode = "train"
            ctx.save_for_backward(X2, g2)
            return (out2 if num_dims == 4 else out2.t().contiguous()), \
                new_mm, new_mv
        else:
            if R > _BN_MAX_R:
                # 大 R eval: 同样绕开行 tile 迭代较慢的 eval kernel
                Xv = X2.reshape(n_groups, rows_per_group, M)
                mm3 = mm2.reshape(n_groups, rows_per_group, 1)
                mv3 = mv2.reshape(n_groups, rows_per_group, 1)
                g3 = g2.reshape(n_groups, rows_per_group, 1)
                b3 = b2.reshape(n_groups, rows_per_group, 1)
                scale = torch.rsqrt(mv3 + 1e-5) * g3               # (1,C,1)
                shift = b3 - mm3 * scale
                out_view.copy_((Xv * scale + shift).reshape(R, M))
            else:
                _bn_fwd_eval_kernel(X2, mm2, mv2, g2, b2, out_view, R, M)
            ctx.mode = "eval"
            ctx.save_for_backward(X2, mm2, mv2, g2)
            return (out2 if num_dims == 4 else out2.t().contiguous()), \
                moving_mean, moving_var

    @staticmethod
    def backward(ctx, grad_out, _grad_mm, _grad_mv):
        R, M, n_groups = ctx.R, ctx.M, ctx.n_groups
        num_dims = ctx.num_dims
        # grad_out 可能是 stride-0 广播视图 (如 .sum() 反向), reshape 不会拷贝,
        # 必须先 contiguous (kernel 要求连续张量)
        grad_Y = (grad_out.contiguous().reshape(R, M) if num_dims == 4
                  else grad_out.t().contiguous())
        # grad_X buffer 按最终形状分配, kernel 写 (R,M) 视图, 返回 base
        # (同 forward: 自定义 Function 不能返回 view)
        if num_dims == 4:
            N, C, H, W = ctx.N, ctx.C, ctx.H, ctx.W
            grad_X2 = torch.empty(N, C, H, W, device=grad_out.device,
                                  dtype=grad_out.dtype)
            grad_X_view = grad_X2.reshape(R, M)
        else:
            N, F = ctx.N, ctx.C
            grad_X2 = torch.empty(R, M, device=grad_out.device,
                                  dtype=grad_out.dtype)
            grad_X_view = grad_X2
        g_ch = torch.empty(1, ctx.rows_per_group, device=grad_out.device,
                           dtype=grad_out.dtype)
        b_ch = torch.empty_like(g_ch)
        if ctx.mode == "train":
            X2, g2 = ctx.saved_tensors
            if R > _BN_MAX_R:
                # 大 R: 广播式通道统计 + 三式 (同 kernel 公式, 无大归约 kernel)
                rpg = ctx.rows_per_group
                Xv = X2.reshape(n_groups, rpg, M)
                gyv = grad_Y.reshape(n_groups, rpg, M)
                if ctx.host_stats is not None:
                    mean, var = ctx.host_stats          # 复用前向统计
                else:
                    mean, var = PyPTOBatchNormFunction._host_stats(Xv)
                inv = torch.rsqrt(var + 1e-5)                 # (1,C,1)
                xhat = (Xv - mean) * inv                      # (N,C,M)
                g3 = g2.reshape(n_groups, rpg, 1)
                gy_g = gyv * g3                               # γ·gY
                # 三式的通道均值 = 对 (n, M) 维求均值 (单融合归约)
                gm = gy_g.mean(dim=(0, 2), keepdim=True)      # (1,C,1)
                gh = (gy_g * xhat).mean(dim=(0, 2), keepdim=True)
                # grad = inv*(gy*g - gm - xhat*gh) 展开成 2 个张量乘 + 常数:
                #   = gyv*(inv*g) + (Xv-mean)*(-inv^2*gh) - inv*gm
                ca = inv * g3                                  # (1,C,1)
                cb = -(inv * inv * gh)
                cc = -inv * gm
                xcent = Xv - mean
                grad3 = gyv * ca + xcent * cb + cc             # (N,C,M)
                grad_X_view.copy_(grad3.reshape(R, M))
                g_ch = (grad_Y.reshape(n_groups, rpg, M) * xhat).sum(
                    dim=(0, 2)).reshape(1, rpg)
                b_ch = grad_Y.reshape(n_groups, rpg, M).sum(
                    dim=(0, 2)).reshape(1, rpg)
            else:
                _bn_bwd_train_kernel(X2, g2, grad_Y, grad_X_view, g_ch, b_ch,
                                     R, M, n_groups)
        else:
            X2, mm2, mv2, g2 = ctx.saved_tensors
            _bn_bwd_eval_kernel(X2, mm2, mv2, g2, grad_Y, grad_X_view, g_ch,
                                b_ch, R, M, n_groups)
        if num_dims == 4:
            grad_gamma = g_ch.reshape(1, C, 1, 1).contiguous()
            grad_beta = b_ch.reshape(1, C, 1, 1).contiguous()
            return grad_X2, grad_gamma, grad_beta, None, None, None, None
        else:
            grad_gamma = g_ch.reshape(1, F).contiguous()
            grad_beta = b_ch.reshape(1, F).contiguous()
            return grad_X2.t().contiguous(), grad_gamma, grad_beta, \
                None, None, None, None


# ── nn.Module 封装 ──


class PyPTOBatchNorm(nn.Module):
    """PyPTO 批量规范化层 (训练态: 两阶段通道归约 + 动量更新; eval 态: moving 统计)。

    接口与 7.5 节手写 BatchNorm 一致: ``PyPTOBatchNorm(num_features, num_dims)``,
    gamma/beta 形状 (1, num_features) 或 (1, num_features, 1, 1)。
    与手写实现的差异: moving 统计注册为 buffer, 随 ``.to()`` 移动,
    免原实现 forward 内手动搬设备。
    """

    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成 1 和 0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的移动平均统计量初始化为 0 和 1 (buffer 随 .to() 移动)
        self.register_buffer("moving_mean", torch.zeros(shape))
        self.register_buffer("moving_var", torch.ones(shape))

    def forward(self, X):
        Y, new_mm, new_mv = PyPTOBatchNormFunction.apply(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            4 if X.dim() == 4 else 2, self.training)
        self.moving_mean = new_mm.detach()
        self.moving_var = new_mv.detach()
        return Y
