"""第 11 章优化算法更新公式的 PyPTO 逐元素内核实现。

设计约定（全章统一）
--------------------
1. 每个优化器拆成若干「单输出」pypto kernel：一个 kernel 只写一个输出张量
   （多输出 kernel 在本仓库实测有兼容性问题，见 04 章 dropout 注记）。
2. 参数/梯度/状态统一展平为一维向量，按 8 对齐补零（FP32 向量 tile 末维
   必须为 8 的倍数）；kernel 内固定 ``set_vec_tile_shapes(8)`` 逐元素并行。
3. host 侧提供两类 trainer_fn（签名与 ``train_ch11`` 的 trainer_fn 完全一致）：
     pypto_<opt>(params, states, hyperparams)  # PyPTO 内核版
     torch_<opt>(params, states, hyperparams)  # 与正文原文逐字等价的 torch 对照
4. pypto_* 内部流程（每个参数）：p/g/状态展平 -> 拷入按参数缓存的补零工作区
   （NPU）-> 依次调用内核 -> 有效段拷回原设备 -> 清零 p.grad。无 NPU 时
   自动回退 torch_*。
5. 标量超参直接作为内核形参传入；Adam/Yogi 的偏差修正系数 corr1/corr2 在
   host 用 Python 浮点算出后以 (1,) 张量形参传入（标量值每步变化会触发
   内核重编译，详见 adam_update_kernel 说明）。
"""

import os

import pypto
import torch

try:
    import torch_npu  # noqa: F401  注册 npu 后端

    NPU_OK = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:  # pragma: no cover
    NPU_OK = False

_TILE = 8  # FP32 向量 tile：末维 8 对齐（32 字节）

__all__ = [
    "NPU_OK",
    "gd_step_kernel", "sgd_step_kernel",
    "momentum_state_kernel", "momentum_update_kernel",
    "adagrad_state_kernel", "adagrad_update_kernel",
    "rmsprop_state_kernel", "rmsprop_update_kernel",
    "adadelta_state_kernel", "adadelta_grad_kernel",
    "adadelta_apply_kernel", "adadelta_delta_kernel",
    "adam_v_kernel", "adam_s_kernel", "adam_update_kernel",
    "yogi_s_kernel",
    "pypto_gd", "pypto_sgd", "pypto_momentum", "pypto_adagrad",
    "pypto_rmsprop", "pypto_adadelta", "pypto_adam", "pypto_yogi",
    "torch_sgd", "torch_momentum", "torch_adagrad", "torch_rmsprop",
    "torch_adadelta", "torch_adam", "torch_yogi",
    "PyPTOSGD",
]


# ---------------------------------------------------------------------------
# 一、逐元素内核（输入/输出均为「补零后的一维向量」，单输出）
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def gd_step_kernel(p: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), lr: float):
    """梯度下降 (11.3.7)：out = p - lr * grad。"""
    pypto.set_vec_tile_shapes(_TILE)
    out.move(p - lr * grad)


# 随机梯度下降 (11.4.3)/(11.5.1) 的逐元素公式与 GD 相同（噪声来自梯度本身）
sgd_step_kernel = gd_step_kernel


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def momentum_state_kernel(v: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), out_v: pypto.Tensor([], pypto.DT_FP32), beta: float):
    """动量状态 (11.6.5 第一式)：out_v = beta * v + grad（与正文 sgd_momentum 逐字一致）。"""
    pypto.set_vec_tile_shapes(_TILE)
    out_v.move(beta * v + grad)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def momentum_update_kernel(p: pypto.Tensor([], pypto.DT_FP32), v: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), lr: float):
    """动量参数更新 (11.6.5 第二式)：out = p - lr * v。"""
    pypto.set_vec_tile_shapes(_TILE)
    out.move(p - lr * v)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def adagrad_state_kernel(s: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), out_s: pypto.Tensor([], pypto.DT_FP32)):
    """AdaGrad 状态 (11.7.1 第一式)：out_s = s + grad^2。"""
    pypto.set_vec_tile_shapes(_TILE)
    out_s.move(s + grad * grad)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def adagrad_update_kernel(p: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), s: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), lr: float, eps: float):
    """AdaGrad 参数更新 (11.7.1 第二式)：out = p - lr * grad / sqrt(s + eps)。"""
    pypto.set_vec_tile_shapes(_TILE)
    out.move(p - lr * grad / pypto.sqrt(s + eps))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rmsprop_state_kernel(s: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), out_s: pypto.Tensor([], pypto.DT_FP32), gamma: float):
    """RMSProp 状态 (11.8.3)：out_s = gamma*s + (1-gamma)*grad^2。"""
    pypto.set_vec_tile_shapes(_TILE)
    out_s.move(gamma * s + (1.0 - gamma) * grad * grad)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rmsprop_update_kernel(p: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), s: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), lr: float, eps: float):
    """RMSProp 参数更新 (11.8.4)：out = p - lr * grad / sqrt(s + eps)。"""
    pypto.set_vec_tile_shapes(_TILE)
    out.move(p - lr * grad / pypto.sqrt(s + eps))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def adadelta_state_kernel(s: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), out_s: pypto.Tensor([], pypto.DT_FP32), rho: float):
    """Adadelta 梯度二阶矩 (11.9.1)：out_s = rho*s + (1-rho)*grad^2。"""
    pypto.set_vec_tile_shapes(_TILE)
    out_s.move(rho * s + (1.0 - rho) * grad * grad)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def adadelta_grad_kernel(s: pypto.Tensor([], pypto.DT_FP32), delta: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), out_g: pypto.Tensor([], pypto.DT_FP32), eps: float):
    """Adadelta 尺度化梯度 (11.9.2)：out_g = sqrt(delta+eps)/sqrt(s+eps) * grad。"""
    pypto.set_vec_tile_shapes(_TILE)
    out_g.move(pypto.sqrt(delta + eps) / pypto.sqrt(s + eps) * grad)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def adadelta_apply_kernel(p: pypto.Tensor([], pypto.DT_FP32), g: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    """Adadelta 参数更新 (11.9.3)：out = p - g。"""
    pypto.set_vec_tile_shapes(_TILE)
    out.move(p - g)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def adadelta_delta_kernel(delta: pypto.Tensor([], pypto.DT_FP32), g: pypto.Tensor([], pypto.DT_FP32), out_d: pypto.Tensor([], pypto.DT_FP32), rho: float):
    """Adadelta 参数变化二阶矩 (11.9.4)：out_d = rho*delta + (1-rho)*g^2。"""
    pypto.set_vec_tile_shapes(_TILE)
    out_d.move(rho * delta + (1.0 - rho) * g * g)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def adam_v_kernel(v: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), out_v: pypto.Tensor([], pypto.DT_FP32), beta1: float):
    """Adam 一阶矩 (11.10.1 第一式)：out_v = beta1*v + (1-beta1)*grad。"""
    pypto.set_vec_tile_shapes(_TILE)
    out_v.move(beta1 * v + (1.0 - beta1) * grad)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def adam_s_kernel(s: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), out_s: pypto.Tensor([], pypto.DT_FP32), beta2: float):
    """Adam 二阶矩 (11.10.1 第二式)：out_s = beta2*s + (1-beta2)*grad^2。"""
    pypto.set_vec_tile_shapes(_TILE)
    out_s.move(beta2 * s + (1.0 - beta2) * grad * grad)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def adam_update_kernel(p: pypto.Tensor([], pypto.DT_FP32), v: pypto.Tensor([], pypto.DT_FP32), s: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32), corr1_t: pypto.Tensor([], pypto.DT_FP32), corr2_t: pypto.Tensor([], pypto.DT_FP32), lr: float, eps: float):
    """Adam 参数更新 (11.10.2，偏差修正后)：
    out = p - lr * (v/corr1_t) / (sqrt(s/corr2_t) + eps)。

    corr1_t/corr2_t 为 (1,) 形状张量（偏差修正系数 1-beta^t，host 计算后
    写入张量）。注意：**t 依赖的标量必须以张量形式传入**——内核编译缓存按
    标量值区分，标量每步变化会触发每步重新编译（每次编译分配约 200MB 设备
    工作区且不释放，约 300 步即耗尽 64GB HBM）；张量参数按 shape 缓存，
    值变化不触发重编译。"""
    pypto.set_vec_tile_shapes(_TILE)
    out.move(p - lr * (v / corr1_t) / (pypto.sqrt(s / corr2_t) + eps))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def yogi_s_kernel(s: pypto.Tensor([], pypto.DT_FP32), grad: pypto.Tensor([], pypto.DT_FP32), out_s: pypto.Tensor([], pypto.DT_FP32), beta2: float):
    """Yogi 二阶矩 (11.10.5)：out_s = s + (1-beta2)*sign(grad^2 - s)*grad^2。"""
    pypto.set_vec_tile_shapes(_TILE)
    out_s.move(s + (1.0 - beta2) * pypto.sign(grad * grad - s) * grad * grad)


# ---------------------------------------------------------------------------
# 二、host 侧基础设施：按参数缓存的补零工作区（避免每步重复分配）
# ---------------------------------------------------------------------------

def _pad_len(n):
    return ((n + 7) // 8) * 8


def _dev():
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    return torch.device(f"npu:{device_id}")


class _Workspace:
    """每个参数一份：p/g 为参数与梯度，a/b 为状态缓冲，o1/o2/o3 为内核输出。"""

    __slots__ = ("n", "np", "p", "g", "a", "b", "o1", "o2", "o3")

    def __init__(self, n):
        self.n = n
        self.np = _pad_len(n)
        dev = _dev()
        z = lambda: torch.zeros(self.np, device=dev, dtype=torch.float32)
        self.p, self.g = z(), z()
        self.a, self.b = z(), z()
        self.o1, self.o2, self.o3 = z(), z(), z()


_WS_CACHE = {}


def _workspace(p):
    key = (id(p), p.data_ptr())
    ws = _WS_CACHE.get(key)
    if ws is None or ws.n != p.numel():
        ws = _Workspace(p.numel())
        _WS_CACHE[key] = ws
    return ws


def _flat(t):
    return t.detach().reshape(-1)


def _copy_in(ws, src, dst):
    """src(任意设备) -> dst(NPU 补零缓冲)。"""
    n = ws.n
    dst[:n].copy_(src.to(dst.device))
    if n < ws.np:
        dst[n:].zero_()


def _copy_out(ws, src, dst):
    """内核输出 src(NPU) 的有效段 -> dst(原设备)。"""
    dst.copy_(src[: ws.n].to(dst.device))


# ---------------------------------------------------------------------------
# 三、train_ch11 兼容的 PyPTO trainer_fn
# ---------------------------------------------------------------------------

def _gd_param(p, lr):
    ws = _workspace(p)
    pf, gf = _flat(p), _flat(p.grad)
    _copy_in(ws, pf, ws.p)
    _copy_in(ws, gf, ws.g)
    gd_step_kernel(ws.p, ws.g, ws.o1, lr)
    _copy_out(ws, ws.o1, pf)
    p.grad.data.zero_()


def pypto_gd(params, states, hyperparams):
    """PyPTO 内核版梯度下降（train_ch11 的 trainer_fn 接口，states 未使用）。"""
    if not NPU_OK:
        return torch_sgd(params, states, hyperparams)
    lr = float(hyperparams["lr"])
    for p in params:
        if p.grad is None:
            continue
        _gd_param(p, lr)


def pypto_sgd(params, states, hyperparams):
    """PyPTO 内核版小批量 SGD（11.5.1）；与 gd 的逐元素公式相同。"""
    if not NPU_OK:
        return torch_sgd(params, states, hyperparams)
    lr = float(hyperparams["lr"])
    for p in params:
        if p.grad is None:
            continue
        _gd_param(p, lr)


def _momentum_param(p, v, lr, beta):
    ws = _workspace(p)
    pf, gf, vf = _flat(p), _flat(p.grad), _flat(v)
    _copy_in(ws, vf, ws.a)
    _copy_in(ws, gf, ws.g)
    momentum_state_kernel(ws.a, ws.g, ws.o1, beta)     # v' = beta*v + g
    _copy_in(ws, pf, ws.p)
    momentum_update_kernel(ws.p, ws.o1, ws.o2, lr)     # p' = p - lr*v'
    vf.copy_(ws.o1[: ws.n].to(vf.device))
    _copy_out(ws, ws.o2, pf)
    p.grad.data.zero_()


def pypto_momentum(params, states, hyperparams):
    """PyPTO 内核版动量法（11.6.5）。states 为与参数同形状的速度 v。"""
    if not NPU_OK:
        return torch_momentum(params, states, hyperparams)
    lr = float(hyperparams["lr"])
    beta = float(hyperparams["momentum"])
    for p, v in zip(params, states):
        if p.grad is None:
            continue
        _momentum_param(p, v, lr, beta)


def _adagrad_param(p, s, lr, eps):
    ws = _workspace(p)
    pf, gf, sf = _flat(p), _flat(p.grad), _flat(s)
    _copy_in(ws, sf, ws.a)
    _copy_in(ws, gf, ws.g)
    adagrad_state_kernel(ws.a, ws.g, ws.o1)            # s' = s + g^2
    _copy_in(ws, pf, ws.p)
    adagrad_update_kernel(ws.p, ws.g, ws.o1, ws.o2, lr, eps)
    sf.copy_(ws.o1[: ws.n].to(sf.device))
    _copy_out(ws, ws.o2, pf)
    p.grad.data.zero_()


def pypto_adagrad(params, states, hyperparams):
    """PyPTO 内核版 AdaGrad（11.7.1）。states 为平方梯度累加 s。"""
    if not NPU_OK:
        return torch_adagrad(params, states, hyperparams)
    lr = float(hyperparams["lr"])
    eps = float(hyperparams.get("eps", 1e-6))
    for p, s in zip(params, states):
        if p.grad is None:
            continue
        _adagrad_param(p, s, lr, eps)


def _rmsprop_param(p, s, lr, gamma, eps):
    ws = _workspace(p)
    pf, gf, sf = _flat(p), _flat(p.grad), _flat(s)
    _copy_in(ws, sf, ws.a)
    _copy_in(ws, gf, ws.g)
    rmsprop_state_kernel(ws.a, ws.g, ws.o1, gamma)     # s' = gamma*s + (1-gamma)*g^2
    _copy_in(ws, pf, ws.p)
    rmsprop_update_kernel(ws.p, ws.g, ws.o1, ws.o2, lr, eps)
    sf.copy_(ws.o1[: ws.n].to(sf.device))
    _copy_out(ws, ws.o2, pf)
    p.grad.data.zero_()


def pypto_rmsprop(params, states, hyperparams):
    """PyPTO 内核版 RMSProp（11.8.3/11.8.4）。states 为平方梯度滑动平均 s。"""
    if not NPU_OK:
        return torch_rmsprop(params, states, hyperparams)
    lr = float(hyperparams["lr"])
    gamma = float(hyperparams["gamma"])
    eps = float(hyperparams.get("eps", 1e-6))
    for p, s in zip(params, states):
        if p.grad is None:
            continue
        _rmsprop_param(p, s, lr, gamma, eps)


def _adadelta_param(p, s, delta, rho, eps):
    ws = _workspace(p)
    pf, gf, sf, df = _flat(p), _flat(p.grad), _flat(s), _flat(delta)
    _copy_in(ws, sf, ws.a)
    _copy_in(ws, df, ws.b)
    _copy_in(ws, gf, ws.g)
    adadelta_state_kernel(ws.a, ws.g, ws.o1, rho)      # s' = rho*s + (1-rho)*g^2
    adadelta_grad_kernel(ws.o1, ws.b, ws.g, ws.o2, eps)  # g' = sqrt(d+eps)/sqrt(s'+eps)*g
    _copy_in(ws, pf, ws.p)
    adadelta_apply_kernel(ws.p, ws.o2, ws.o3)          # p' = p - g'
    sf.copy_(ws.o1[: ws.n].to(sf.device))
    adadelta_delta_kernel(ws.b, ws.o2, ws.o1, rho)     # d' = rho*d + (1-rho)*g'^2
    df.copy_(ws.o1[: ws.n].to(df.device))
    _copy_out(ws, ws.o3, pf)
    p.grad.data.zero_()


def pypto_adadelta(params, states, hyperparams):
    """PyPTO 内核版 Adadelta（11.9.1~11.9.4）。
    states 为 ((s_w, delta_w), (s_b, delta_b))，无学习率。"""
    if not NPU_OK:
        return torch_adadelta(params, states, hyperparams)
    rho = float(hyperparams["rho"])
    eps = float(hyperparams.get("eps", 1e-5))
    for p, (s, delta) in zip(params, states):
        if p.grad is None:
            continue
        _adadelta_param(p, s, delta, rho, eps)


def _adam_param(p, v, s, lr, beta1, beta2, eps, corr1_t, corr2_t):
    ws = _workspace(p)
    pf, gf, vf, sf = _flat(p), _flat(p.grad), _flat(v), _flat(s)
    _copy_in(ws, vf, ws.a)
    _copy_in(ws, sf, ws.b)
    _copy_in(ws, gf, ws.g)
    adam_v_kernel(ws.a, ws.g, ws.o1, beta1)            # v' = beta1*v + (1-beta1)*g
    adam_s_kernel(ws.b, ws.g, ws.o2, beta2)            # s' = beta2*s + (1-beta2)*g^2
    _copy_in(ws, pf, ws.p)
    adam_update_kernel(ws.p, ws.o1, ws.o2, ws.o3, corr1_t, corr2_t, lr, eps)
    vf.copy_(ws.o1[: ws.n].to(vf.device))
    sf.copy_(ws.o2[: ws.n].to(sf.device))
    _copy_out(ws, ws.o3, pf)
    p.grad.data.zero_()


def pypto_adam(params, states, hyperparams):
    """PyPTO 内核版 Adam（11.10.1~11.10.2）。
    states 为 ((v_w, s_w), (v_b, s_b))；时间步 t 存于 hyperparams['t']。"""
    if not NPU_OK:
        return torch_adam(params, states, hyperparams)
    lr = float(hyperparams["lr"])
    beta1, beta2 = 0.9, 0.999
    eps = float(hyperparams.get("eps", 1e-6))
    t = int(hyperparams["t"])
    # 偏差修正系数（随 t 每步变化）以 (1,) 张量传入内核：
    # 标量值变化会触发内核重新编译（每次 ~200MB 设备内存泄漏），见
    # adam_update_kernel 文档说明。
    corr1_t = torch.tensor([1.0 - beta1 ** t], dtype=torch.float32, device=_dev())
    corr2_t = torch.tensor([1.0 - beta2 ** t], dtype=torch.float32, device=_dev())
    for p, (v, s) in zip(params, states):
        if p.grad is None:
            continue
        _adam_param(p, v, s, lr, beta1, beta2, eps, corr1_t, corr2_t)
    hyperparams["t"] += 1


def _yogi_param(p, v, s, lr, beta1, beta2, eps, corr1_t, corr2_t):
    ws = _workspace(p)
    pf, gf, vf, sf = _flat(p), _flat(p.grad), _flat(v), _flat(s)
    _copy_in(ws, vf, ws.a)
    _copy_in(ws, sf, ws.b)
    _copy_in(ws, gf, ws.g)
    adam_v_kernel(ws.a, ws.g, ws.o1, beta1)
    yogi_s_kernel(ws.b, ws.g, ws.o2, beta2)            # s' = s + (1-b2)*sign(g^2-s)*g^2
    _copy_in(ws, pf, ws.p)
    adam_update_kernel(ws.p, ws.o1, ws.o2, ws.o3, corr1_t, corr2_t, lr, eps)
    vf.copy_(ws.o1[: ws.n].to(vf.device))
    sf.copy_(ws.o2[: ws.n].to(sf.device))
    _copy_out(ws, ws.o3, pf)
    p.grad.data.zero_()


def pypto_yogi(params, states, hyperparams):
    """PyPTO 内核版 Yogi（11.10.5）：二阶矩用 sign 截断，其余同 Adam。"""
    if not NPU_OK:
        return torch_yogi(params, states, hyperparams)
    lr = float(hyperparams["lr"])
    beta1, beta2 = 0.9, 0.999
    eps = float(hyperparams.get("eps", 1e-3))
    t = int(hyperparams["t"])
    corr1_t = torch.tensor([1.0 - beta1 ** t], dtype=torch.float32, device=_dev())
    corr2_t = torch.tensor([1.0 - beta2 ** t], dtype=torch.float32, device=_dev())
    for p, (v, s) in zip(params, states):
        if p.grad is None:
            continue
        _yogi_param(p, v, s, lr, beta1, beta2, eps, corr1_t, corr2_t)
    hyperparams["t"] += 1


# ---------------------------------------------------------------------------
# 四、torch 对照实现（与正文原文逐字等价，用于数值对比）
# ---------------------------------------------------------------------------

def torch_sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()


def torch_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()


def torch_adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()


def torch_rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()


def torch_adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()


def torch_adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1


def torch_yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1


# ---------------------------------------------------------------------------
# 五、11.11 可选：兼容 torch.optim.Optimizer 接口的适配器
# ---------------------------------------------------------------------------

class PyPTOSGD(torch.optim.Optimizer):
    """用 pypto_sgd 内核驱动 step() 的 Optimizer 适配器。

    实现 11.11 训练循环用到的接口子集（zero_grad / step / param_groups），
    可直接替换 torch.optim.SGD：

        trainer = PyPTOSGD(net.parameters(), lr=lr)

    各学习率调度器用法完全不变（param_groups[0]["lr"] 仍可读写）。
    """

    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            pypto_sgd(group["params"], None, {"lr": group["lr"]})
        return loss
