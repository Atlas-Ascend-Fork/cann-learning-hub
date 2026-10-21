"""PyPTO Conv2d — 12.x 章 CV 训练用的卷积层 (fwd pypto 引擎 + bwd 原生)

fwd 按 shape 分四条路线:

* **conv 引擎** (``pypto.conv``, 隐式 GEMM), 常规 spatial 的 k>1 走此路线:
  - batch 维用 ``pypto.DYNAMIC`` 编译 + 核内按 ``_BATCH_TILE`` 循环,
    缓存 key 不含 batch, 每个 shape 只编译一次;
  - Cin/Cout 按 C0=8 补零对齐;
  - 输出宽度 Wo 非 16 倍数时, 输入右侧补零使运行期输出宽度对齐 16,
    输出再裁回 —— 否则引擎 tileHout 被迫为 1, H 方向在编译期全展开,
    单个 shape 可生成多个十万行级的 _aic.cpp, 编译时间不可接受;
  - tile 推导采用联合优化: 遍历 tileCin 候选, 取 H_tiles × Cin_tiles
    最小的组合 (宁可多切 K 轴, 不让 H 方向逐行展开)。
* **1x1 (p=0, d=1, 任意 stride)**: ``X.view`` 零拷贝 + b_trans matmul
  (1x1 的 im2col 就是张量自身布局, 无需任何搬运), bias 核外加; stride>1
  时先做 stride 切片 (零拷贝视图) 再 matmul, 避免走引擎的补宽/裁回 host
  拷贝与 tileHout 编译开销。
* **极小 spatial 的 k>1**: 原生 ``F.unfold`` 物化 im2col + 同一个 b_trans
  matmul kernel —— 引擎隐含 GEMM 在 M=N*Ho*Wo 很小时 cube 利用率差且
  随 Cout 增长 (512->512 @6 实测 4.9ms vs unfold 0.6ms); 阈值内改用
  显式 matmul。@24 及以上仍走引擎 (host im2col 拷贝随 spatial 增长)。

* **大 spatial / 大通道的 k>1** (输出空间 > ``_ENGINE_MAX_HW`` 或单侧通道
  > ``_ENGINE_MAX_CH``, 如 SSD 256×256、VGG 256→512 @37×56): 直接派发原生
  ``F.conv2d``。引擎编译体积 ∝ 输出空间与通道 (H=64 约 10s、H=128 约 70s、
  H=256 >4min; VGG 256→512 @37×56 >8min), 大特征图/大通道编译不可接受;
  原生 kernel 零编译且更快 (见 ``_fwd_conv_native``)。

bwd: ``aten.convolution_backward`` 一次原生调用同时输出 dX/dW/db,
零 JIT 编译。stride/padding/dilation 任意原生透传; transposed 不支持。

> 调优变更 (2026-09-26): 旧版 fwd/dK/dX 全部走 host im2col + pypto matmul
> (v3 路线), host 侧 ``as_strided`` gather 是主要瓶颈 (b32 3ch 256×256
> 实测 153ms vs torch 1.8ms)。本版与 ch07 对齐: fwd 换 pypto.conv 引擎
> 路由, bwd 换原生 ``aten.convolution_backward``。以下辅助函数
> (``_matmul_bt_capped``/``_im2col_strided``/``_pad4d``/``_to_pair``) 保留,
> 供 ``PyPTOConvTranspose2DModule`` 复用。

注意: conv 引擎 kernel 按 shape 首次调用触发 JIT 编译, 耗时可达分钟级
(进程内缓存, 无磁盘缓存), 全形状预热较久, 首个 epoch 会明显偏慢。

限制: transposed=False, FP32。
"""

import torch
import torch.nn.functional as F
from torch import nn

import pypto
from pypto import pypto_impl

_C0 = 8                    # FP32 通道对齐 (conv 引擎要求 Cin/Cout % 8 == 0)
_L1_SIZE = 480 * 1024      # L1 tile 预算
_BATCH_TILE = 8            # 引擎核内 batch 循环步长 (transdata UB 随 batch 线性增长)
_KERNEL_CACHE = {}         # jit kernel 编译缓存, key = 全部定形参数 (不含 batch)
# fwd 路由阈值 (极小 spatial 改走 unfold+matmul, 见 _conv_forward):
#   输出 Ho*Wo <= _IM2COL_MAX_HW 且 (空间极小 或 Cout 大) 时用 unfold。
_IM2COL_MAX_HW = 144       # 输出空间上限 (12x12)
_IM2COL_TINY_HW = 36       # 无论 Cout 都走 unfold 的极小空间 (6x6)
_IM2COL_MIN_COUT = 64      # 大 Cout 阈值 (引擎墙钟随 Cout 增长)
# 引擎输出空间上限: 引擎编译体积 ∝ 输出空间 (H×W), 实测 3ch->16 单 shape
# 编译 H=32 4s / H=64 10s / H=128 70s / H=256 >4min (超线性)。大 spatial
# (SSD 256×256、FCN 320×480 等) 编译不可接受, 直接派发原生 conv (torch
# 原生 kernel, 零编译, 逐位一致); 小 spatial 仍走 PyPTO 引擎。
_ENGINE_MAX_HW = 4096      # (64x64) 以上的输出空间改走原生 conv
# 引擎单侧通道上限: 权重 tile (tileN*tileCin*kH*kW) 随通道增长, K 轴切分
# 变多, 编译体积/耗时显著上升 (VGG19 37×56 256->512 单 shape 编译 >8min
# 未完成)。通道超过此值同样改走原生。
_ENGINE_MAX_CH = 256
# 引擎单 tile 输出行上限。引擎的 program.json / aic.cpp 体积 ∝ tileHout
# (H 方向按 tileHout 展开), 大 spatial 下若 tileHout 取满 Ho (如 256/320),
# 单 kernel 编译可 >2h。故即使 L1 容得下也把 tileHout 截断, 以 H 方向多切
# 几个 tile 换取可控编译体积 (tileHout 只影响运行期 tile 循环次数)。
_MAX_TILE_HOUT = 32


def _cached_kernel(factory):
    """kernel 闭包工厂的编译缓存装饰器。"""
    def get(*key):
        kernel = _KERNEL_CACHE.get(key)
        if kernel is None:
            kernel = factory(*key)
            _KERNEL_CACHE[key] = kernel
        return kernel
    return get


def _to_pair(v):
    """标量/二元组 → (int, int)。"""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _out_size(size, k, s, p, d):
    """卷积输出尺寸: (size + 2p - d*(k-1) - 1) // s + 1。"""
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


def _align_w(W, kW, s_w, p_w, d_w):
    """输入宽度补零量, 使输出 Wo 对齐 16 (已对齐返回 0)。

    Wo 非 16 倍数时引擎 tileHout 被迫为 1 → H 方向编译期全展开 (见模块
    docstring); 右侧补零列对有效输出零贡献, 输出裁回即精确。
    """
    Wo = _out_size(W, kW, s_w, p_w, d_w)
    wo_pad = -Wo % 16
    if wo_pad == 0:
        return 0
    return max(0, (Wo + wo_pad - 1) * s_w + d_w * (kW - 1) + 1 - 2 * p_w - W)


def _pad_c0(X, C):
    """通道维补零到 C0 对齐 (FP32: 8), 供 conv 引擎使用。"""
    pad = ((C + _C0 - 1) // _C0) * _C0 - C
    if pad == 0:
        return X
    return F.pad(X, (0, 0, 0, 0, 0, pad, 0, 0))


# ════════════════════════════════════════════════════════════════════
# fwd 路线 1: pypto.conv 引擎 (k>1)
# ════════════════════════════════════════════════════════════════════


def _derive_conv_tiles(Cin, Cout, H, W, kH, kW, s_h, s_w, p_h, p_w, d_h, d_w):
    """L1/L0/vec tile 预算式推导 (按 DAV 硬件约束):

    - L1: 输入 tile (hin*win*tileCin*4B) + 权重 tile (tileN*tileCin*kH*kW*4B)
      <= L1 预算; tileCin 从 Cin 向下减半做 K 轴切分
    - L0: L0A tileH*tileW*tileK*4 <= 64KB, L0C tileH*tileW*tileN*4 <= 128KB,
      tileN <= 16, tileW 需 16 元素对齐
    - tileK 必须整除 tileCin*kH*kW (kAL1/kBL1)
    - Wo 已由调用方以 _align_w 保证 16 对齐, tileHout 可大于 1
    - vec: batch 16 * ceil8(Cout)(<=64) * 1 * C0 (C 维须为 C0 整数倍, 否则
      pypto 报 "tileShape C should be an integer multiple of C0"; 故按
      cout_pad 取, 小 Cout (如 LeNet 的 6) 也满足), 双缓冲受 UB 预算约束

    联合优化: 贪心取最大 tileCin 会把权重 tile 撑满 L1, 导致输入 tile
    空间不足、tileHout 被压到 1 (H 方向逐行静态展开, 编译爆炸); 故遍历
    tileCin 候选, 选 H_tiles × Cin_tiles 最小者 —— K 轴多切一次的运行
    时代价远小于 H 展开的编译代价。
    """
    Ho = _out_size(H, kH, s_h, p_h, d_h)
    Wo = _out_size(W, kW, s_w, p_w, d_w)
    cin_pad = ((Cin + _C0 - 1) // _C0) * _C0
    cout_pad = ((Cout + _C0 - 1) // _C0) * _C0
    kH_eff = d_h * (kH - 1) + 1
    kW_eff = d_w * (kW - 1) + 1
    tile_wout = 16
    tileN = 16
    win = (tile_wout - 1) * s_w + kW_eff
    tile_hout = 1 if Wo % 16 != 0 else None

    def fit(tc):
        return tc * kH * kW * tileN * 4 + win * kH * tc * 4 <= _L1_SIZE - 2048

    def _hout_for(tc):
        w_bytes = tc * kH * kW * tileN * 4
        hin_max = (_L1_SIZE - w_bytes - 2048) // (win * tc * 4)
        if s_h == 1:
            return max(1, min(hin_max - kH_eff + 1, Ho, _MAX_TILE_HOUT))
        return max(1, min((hin_max - kH_eff) // s_h + 1, Ho, _MAX_TILE_HOUT))

    if tile_hout is None:
        best_tiles, tile_cin, tile_hout = None, cin_pad, 1
        tc = cin_pad
        while tc >= _C0:
            if fit(tc):
                hout = _hout_for(tc)
                tiles = -(-Ho // hout) * -(-cin_pad // tc)
                if best_tiles is None or tiles < best_tiles:
                    best_tiles, tile_cin, tile_hout = tiles, tc, hout
            tc //= 2
    else:
        tile_cin = cin_pad
        while not fit(tile_cin) and tile_cin > _C0:
            tile_cin //= 2
    hi_al1 = min((tile_hout - 1) * s_h + kH_eff, H)
    wi_al1 = min((tile_wout - 1) * s_w + kW_eff, W)
    k_fact = tile_cin * kH * kW
    tile_k = 64
    while k_fact % tile_k != 0 and tile_k > _C0:
        tile_k //= 2
    l1 = pypto_impl.TileL1Info(tileHin=hi_al1, tileHout=tile_hout, tileWin=wi_al1,
                               tileWout=tile_wout, tileCinFmap=tile_cin,
                               tileCinWeight=tile_cin, tileN=tileN, tileBatch=1)
    l0 = pypto_impl.TileL0Info(tileH=min(tile_hout, 8), tileW=tile_wout,
                               tileK=tile_k, tileN=16)
    vec = (16, min(cout_pad, 64), 1, _C0)
    return l1, l0, vec


@_cached_kernel
def _conv_kernel(Cin, Cout, H, W, kH, kW, s_h, s_w, p_h, p_w, d_h, d_w):
    """按 shape 生成 conv fwd kernel (闭包工厂, 编译后缓存)。

    batch 维 pypto.DYNAMIC, 核内按 _BATCH_TILE 循环 (view + assemble),
    一次调用处理整个 batch; 缓存 key 不含 batch。
    """
    Ho = _out_size(H, kH, s_h, p_h, d_h)
    Wo = _out_size(W, kW, s_w, p_w, d_w)
    cin_pad = ((Cin + _C0 - 1) // _C0) * _C0
    cout_pad = ((Cout + _C0 - 1) // _C0) * _C0
    l1, l0, vec = _derive_conv_tiles(Cin, Cout, H, W, kH, kW, s_h, s_w, p_h, p_w, d_h, d_w)
    # pypto 0.2.1 动态维需用 status-shape 列表 [pypto.DYNAMIC, ...] (tuple 报
    # "Invalid value type", 见 Tensor.__init__ 的 _validate_status_shape)
    fmap_shape = [pypto.DYNAMIC, cin_pad, H, W]
    out_shape = [pypto.DYNAMIC, cout_pad, Ho, Wo]

    @pypto.frontend.jit()
    def conv_fwd_kernel(fmap: pypto.Tensor(fmap_shape, pypto.DT_FP32),
                        weight: pypto.Tensor((cout_pad, cin_pad, kH, kW), pypto.DT_FP32),
                        bias: pypto.Tensor([cout_pad], pypto.DT_FP32),
                        out: pypto.Tensor(out_shape, pypto.DT_FP32)):
        pypto.set_conv_tile_shapes(l1, l0)
        pypto.set_vec_tile_shapes(*vec)
        batch = fmap.shape[0]
        tile_batch = pypto.symbolic_scalar(_BATCH_TILE)
        batch_loop = (batch + tile_batch - 1) // tile_batch
        for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_batch"):
            batch_offset = batch_idx * tile_batch
            input_view = pypto.view(fmap, [tile_batch, cin_pad, H, W],
                                    [batch_offset, 0, 0, 0])
            output = pypto.conv(input_view, weight, pypto.DT_FP32, [s_h, s_w],
                                [p_h, p_w, p_h, p_w], [d_h, d_w],
                                extend_params={"bias_tensor": bias}, groups=1)
            pypto.assemble(output, [batch_offset, 0, 0, 0], out)
    return conv_fwd_kernel


def _fwd_conv(X, K, bias, s_h, s_w, p_h, p_w, d_h, d_w):
    """conv 引擎 fwd: 通道/宽度补零对齐 + 一次 kernel 调用处理整个 batch。"""
    N, Cin, H, W = X.shape
    Cout, _, kH, kW = K.shape
    Ho = _out_size(H, kH, s_h, p_h, d_h)
    Wo = _out_size(W, kW, s_w, p_w, d_w)
    cout_pad = ((Cout + _C0 - 1) // _C0) * _C0
    w_pad = _align_w(W, kW, s_w, p_w, d_w)
    Wo_run = _out_size(W + w_pad, kW, s_w, p_w, d_w)
    kernel = _conv_kernel(Cin, Cout, H, W + w_pad, kH, kW, s_h, s_w, p_h, p_w, d_h, d_w)
    Xp = _pad_c0(X, Cin)
    if w_pad:
        Xp = F.pad(Xp, (0, w_pad))
    Kp = _pad_c0(K, Cin)
    cout_tail = cout_pad - Cout
    if cout_tail:
        Kp = F.pad(Kp, (0, 0, 0, 0, 0, 0, 0, cout_tail))
    if bias is None:
        b = torch.zeros(cout_pad, device=X.device, dtype=X.dtype)
    elif cout_tail:
        b = F.pad(bias, (0, cout_tail))
    else:
        b = bias
    tail = (-N) % _BATCH_TILE
    if tail:
        Xp = F.pad(Xp, (0, 0, 0, 0, 0, 0, 0, tail))
    out = torch.empty(Xp.shape[0], cout_pad, Ho, Wo_run, device=X.device, dtype=X.dtype)
    kernel(Xp, Kp, b, out)
    out = out[:N]
    if w_pad:
        out = out[:, :, :, :Wo].contiguous()
    if cout_tail:
        out = out[:, :Cout]
    return out


# ════════════════════════════════════════════════════════════════════
# fwd 路线 2: view + b_trans matmul (1x1)
# ════════════════════════════════════════════════════════════════════


def _tiles_for(M, K, N):
    """按 M/K/N 分档选 cube tile: 小 shape 用小 tile, 其余 128-256
    (FP32 下 L1 预算需 mL1*kL1+nL1*kL1 <= 48K 元素)。

    返回 hashable 的嵌套 tuple —— matmul kernel 缓存以 tile 组合为键
    (M 只影响分档, 不进键), 故所有落到同一档的 1x1 形状共享一次编译。
    """
    if M <= 144 or N <= 64:
        return ((64, 64), (64, 128), (128, 128))
    return ((128, 128), (64, 256), (128, 128))


@_cached_kernel
def _matmul_kernel(mt, kt, nt):
    """C = A @ B^T (免 B 转置拷贝) 的 b_trans matmul kernel, 按 tile 组合缓存。

    kernel 张量形状全动态 (``pypto.Tensor([], ...)``), M/K/N 不参与编译,
    故同一 tile 档下任意 shape 复用同一 kernel —— 1x1 卷积形状随
    spatial/batch 变化时不再逐 shape 重编 (流程同 Linear 的 M 分桶)。
    """
    @pypto.frontend.jit()
    def matmul_kernel(A: pypto.Tensor([], pypto.DT_FP32),
                      B: pypto.Tensor([], pypto.DT_FP32),
                      out: pypto.Tensor([], pypto.DT_FP32)):
        pypto.set_cube_tile_shapes(list(mt), list(kt), list(nt))
        out[:] = pypto.matmul(A, B, pypto.DT_FP32, b_trans=True)
    return matmul_kernel


def _fwd_1x1(X, K, bias, s_h, s_w):
    """1x1 fwd (任意 stride, padding=0): C = X_sub_nhwc_2d @ K^T。

    X 是 NCHW 布局, 须先 permute 到 (N,H,W,Cin) 再展平 (直接 reshape
    NCHW 到 (M, Cin) 是非法重排 —— 通道维不在尾维); 一次 permute 拷贝
    即 1x1 的全部 "im2col" 开销。

    stride>1 时先做 ``X[:, :, ::s_h, ::s_w]`` 空间切片 (stride 切片是
    视图, 零拷贝), 等价于 1x1 卷积的稀疏采样 —— 避免走 conv 引擎:
    引擎对该形状需 ``_align_w`` 补宽 + 输出裁回 (多次 host 拷贝), 且
    tileHout 编译代价高; 切片 + matmul 的 M 更小, 实测 ~5x。
    输出 permute 视图不 ``.contiguous()`` (调用方模块入口均自行
    ``.contiguous()``), 省一次整张量拷贝。
    """
    N, Cin, H, W = X.shape
    Cout = K.shape[0]
    if s_h != 1 or s_w != 1:
        X = X[:, :, ::s_h, ::s_w]
    Ho, Wo = X.shape[2], X.shape[3]
    M = N * Ho * Wo
    mt, kt, nt = _tiles_for(M, Cin, Cout)
    kernel = _matmul_kernel(mt, kt, nt)
    cols = X.permute(0, 2, 3, 1).reshape(M, Cin).contiguous()
    out = torch.empty(M, Cout, device=X.device, dtype=X.dtype)
    kernel(cols, K.reshape(Cout, Cin), out)
    if bias is not None:
        out = out + bias
    return out.reshape(N, Ho, Wo, Cout).permute(0, 3, 1, 2)


def _fwd_conv_im2col(X, K, bias, s_h, s_w, p_h, p_w, d_h, d_w):
    """极小 spatial 的 conv fwd: 原生 ``F.unfold`` 物化 im2col + pypto matmul。

    引擎对极小输出效率低: 为满足 tileWout=16 对齐, ``_align_w`` 把输入
    宽度补到 16 的倍数, 无效输出列被大量计算 (如 6x6 输出按 6x16 计),
    且隐含 GEMM 的 M=N*Ho*Wo 很小时 cube 利用率差、墙钟随 Cin/Cout 增长
    —— 实测 512->512 @6 引擎 4.9ms, 本路线 0.6ms (逻辑 GEMM 相同, 显式
    matmul 效率高得多)。此处用原生 unfold 物化 cols (小 spatial 下拷贝
    量小、带宽友好), 再交给同一个 pypto b_trans matmul kernel, 保留
    PyPTO 计算路径。@24 及以上回归引擎 (host im2col 拷贝量随 spatial
    增长, 引擎更优; 见 ``_conv_forward`` 的路由条件)。
    """
    N, Cin, H, W = X.shape
    Cout, _, kH, kW = K.shape
    cols = F.unfold(X, (kH, kW), dilation=(d_h, d_w), padding=(p_h, p_w),
                    stride=(s_h, s_w))                      # (N, Cin*kH*kW, L)
    L = cols.shape[2]
    M = N * L
    Kd = Cin * kH * kW
    A = cols.transpose(1, 2).reshape(M, Kd).contiguous()
    mt, kt, nt = _tiles_for(M, Kd, Cout)
    out = torch.empty(M, Cout, device=X.device, dtype=X.dtype)
    _matmul_kernel(mt, kt, nt)(A, K.reshape(Cout, Kd), out)
    if bias is not None:
        out = out + bias
    Ho = _out_size(H, kH, s_h, p_h, d_h)
    Wo = _out_size(W, kW, s_w, p_w, d_w)
    return out.reshape(N, Ho, Wo, Cout).permute(0, 3, 1, 2)


def _fwd_conv_native(X, K, bias, s_h, s_w, p_h, p_w, d_h, d_w):
    """大 spatial 的 conv fwd: 直接派发 torch 原生 ``F.conv2d``。

    引擎编译体积 ∝ 输出空间 (见 ``_ENGINE_MAX_HW``), SSD/FCN 的大特征图
    编译需数分钟~数小时, 不可接受; 原生 kernel 零编译且更快 (SSD 3->16
    @256×256 fwd+bwd 1.7ms vs 引擎不可用)。bwd 本就走原生
    ``aten.convolution_backward``, 二者逐位一致。
    """
    return F.conv2d(X, K, bias, stride=(s_h, s_w), padding=(p_h, p_w),
                    dilation=(d_h, d_w))


def _conv_forward(X, K, bias, s_h, s_w, p_h, p_w, d_h, d_w):
    """fwd 分派: 1x1 → 切片+matmul; 大 spatial → 原生; 极小 spatial →
    unfold+matmul; 其余 → 引擎。

    极小 spatial 阈值 (_IM2COL_MAX_HW/_IM2COL_MIN_COUT) 由实测得到: 引擎
    墙钟近似随 Cout 增长而几乎不随极小 spatial 缩小, 而 unfold 拷贝量随
    spatial 缩小; 故 @12 及以下仅在大 Cout 时改走 unfold (DenseNet 的
    Cout=32 小 spatial 引擎已足够快), @6 及以下无论 Cout 均走 unfold。
    输出空间超过 _ENGINE_MAX_HW 或通道超过 _ENGINE_MAX_CH 时引擎编译
    爆炸, 改走原生 (见两常量注释)。
    """
    if K.shape[2] == 1 and K.shape[3] == 1 and (p_h, p_w) == (0, 0) \
            and (d_h, d_w) == (1, 1):
        return _fwd_1x1(X, K, bias, s_h, s_w)
    Ho = _out_size(X.shape[2], K.shape[2], s_h, p_h, d_h)
    Wo = _out_size(X.shape[3], K.shape[3], s_w, p_w, d_w)
    hw = Ho * Wo
    if hw > _ENGINE_MAX_HW or K.shape[0] > _ENGINE_MAX_CH or K.shape[1] > _ENGINE_MAX_CH:
        return _fwd_conv_native(X, K, bias, s_h, s_w, p_h, p_w, d_h, d_w)
    if hw <= _IM2COL_MAX_HW and (hw <= _IM2COL_TINY_HW or K.shape[0] > _IM2COL_MIN_COUT):
        return _fwd_conv_im2col(X, K, bias, s_h, s_w, p_h, p_w, d_h, d_w)
    return _fwd_conv(X, K, bias, s_h, s_w, p_h, p_w, d_h, d_w)


# ════════════════════════════════════════════════════════════════════
# 兼容层: 旧 v3 matmul/im2col 辅助 (供 PyPTOConvTranspose2DModule 复用)
# ════════════════════════════════════════════════════════════════════

# 统一 Cube tile shape (经 bench 验证对 LeNet 各 shape 最优)
_CUBE_TILE = [128, 128], [64, 256], [256, 256]


@pypto.frontend.jit(runtime_options={})
def _matmul_bt_kernel(
    A:     pypto.Tensor([], pypto.DT_FP32),   # (M, K)
    B:     pypto.Tensor([], pypto.DT_FP32),   # (N, K), 计算时取 B^T
    out:   pypto.Tensor([], pypto.DT_FP32),   # (M, N)
):
    """C = A @ B^T (免 B 转置拷贝; ConvTranspose fwd/dX 用)."""
    pypto.set_cube_tile_shapes(*_CUBE_TILE)
    out[:] = pypto.matmul(A, B, pypto.DT_FP32, b_trans=True)


# ── M 维分块: matmul codegen 会把整个 tile 调度静态展开成直线代码,
#    体积 ∝ tile 总数(≈M·N/tile面积), 大 shape 单 kernel 可达 MB 级/编译数十分钟.
#    限制单 kernel tile 数, 把大 matmul 切成若干同 shape 小 kernel (编译一次, 复用所有层).
_MM_MAX_TILES = 64
_MM_TILE_M, _MM_TILE_N = 128, 128


def _mm_rows_cap(M, N):
    """单次 matmul 的 M 上限, 使 tile 数 ≤ _MM_MAX_TILES."""
    n_tn = max(1, -(-int(N) // _MM_TILE_N))
    rows = (_MM_MAX_TILES // n_tn) * _MM_TILE_M
    return max(_MM_TILE_M, min(int(M), max(rows, _MM_TILE_M)))


def _matmul_bt_capped(A, B, out):
    """C = A @ B^T, M 维分块 (输出行切片, 各块独立写)."""
    M, N = A.shape[0], B.shape[0]
    step = _mm_rows_cap(M, N)
    if step >= M:
        _matmul_bt_kernel(A, B, out)
        return
    for m0 in range(0, M, step):
        m1 = min(m0 + step, M)
        _matmul_bt_kernel(A[m0:m1], B, out[m0:m1])


def _im2col_strided(X_padded, kH, kW, stride=(1, 1)):
    """X_padded (N,Cin,Hp,Wp) → cols (M, Cin*kH*kW), M=N*H_out*W_out。

    torch.as_strided 零拷贝窗口 view + 一次 permute/reshape/contiguous。
    (供 ConvTranspose 复用)
    """
    s_h, s_w = _to_pair(stride)
    N, Cin, Hp, Wp = X_padded.shape
    Ho = (Hp - kH) // s_h + 1
    Wo = (Wp - kW) // s_w + 1
    view = X_padded.as_strided(
        (N, Cin, Ho, Wo, kH, kW),
        (Cin * Hp * Wp, Hp * Wp, s_h * Wp, s_w, Wp, 1),
    )
    return view.permute(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, Cin * kH * kW).contiguous()


def _pad4d(X, p_h, p_w):
    """对称 pad (供 ConvTranspose 复用)。"""
    if p_h == 0 and p_w == 0:
        return X
    return F.pad(X, [p_w, p_w, p_h, p_h])


# ── autograd Function + nn.Module ──


class PyPTOConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, K, stride, padding, dilation, bias):
        s_h, s_w = _to_pair(stride)
        p_h, p_w = _to_pair(padding)
        d_h, d_w = _to_pair(dilation)
        X_c, K_c = X.contiguous(), K.contiguous()
        Y = _conv_forward(X_c, K_c, bias, s_h, s_w, p_h, p_w, d_h, d_w)
        ctx.save_for_backward(X_c, K_c)
        ctx.conv_args = (s_h, s_w, p_h, p_w, d_h, d_w)
        ctx.has_bias = bias is not None
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, K = ctx.saved_tensors
        s_h, s_w, p_h, p_w, d_h, d_w = ctx.conv_args
        mask = (ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[5])
        dX, dW, db = torch.ops.aten.convolution_backward(
            grad_output.contiguous(), X, K,
            [K.shape[0]] if ctx.has_bias else None,
            [s_h, s_w], [p_h, p_w], [d_h, d_w], False, [0, 0], 1, mask)
        return dX, dW, None, None, None, db


class PyPTOConv2d(nn.Module):
    """PyPTO Conv2d: fwd = pypto.conv 引擎 (k>1) / view+matmul (1x1),
    bwd = aten.convolution_backward (原生, 零编译)。

    - stride/padding/dilation 任意原生透传; transposed 不支持 (传入即抛错)
    - 路由与编译注意事项: 见模块 docstring
    """

    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 3),
                 stride=1, padding=0, dilation=1, transposed=False,
                 bias=False, device="cpu", dtype=torch.float32):
        super().__init__()
        if transposed:
            raise NotImplementedError("transposed 未实现, 请用 torch.nn.ConvTranspose2d")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size,
                                               device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.bias = (nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
                     if bias else None)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -fan_in ** -0.5, fan_in ** -0.5)

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, dilation={self.dilation}, "
                f"bias={self.bias is not None}")

    def forward(self, x):
        return PyPTOConv2dFunction.apply(
            x, self.weight, self.stride, self.padding, self.dilation, self.bias)
