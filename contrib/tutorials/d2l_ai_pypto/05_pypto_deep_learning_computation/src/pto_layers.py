import math
import os

import pypto
import torch
import torch.nn as nn


def _accelerated_device():
    """返回 pypto kernel 实际运行的 NPU 设备。

    kernel 由环境变量 TILE_FWK_DEVICE_ID 指定设备（默认 0），
    因此参数也创建在同一设备上，并同步 torch 当前设备，避免跨设备访问。
    """
    npu = getattr(torch, "npu", None)
    try:
        if npu is not None and npu.is_available():
            device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
            if 0 <= device_id < npu.device_count():
                npu.set_device(device_id)
                return torch.device("npu", device_id)
    except Exception:
        pass
    return torch.device("cpu")


# ========== 分块策略选择 ==========
def linear_tile_shapes(in_features, out_features):
    if max(in_features, out_features) <= 64:
        return [16, 16], [32, 64], [16, 16]
    return [32, 32], [64, 64], [64, 64]


# ========== kernel 定义 ==========
# pypto 0.2.1（CANN 9.1.0）支持动态形状标注 pypto.DYNAMIC：
# 将 batch 维（以及 ReLU 的特征维）声明为动态维度后，kernel 只需编译一次
# 即可服务于任意输入形状。计算时通过 pypto.loop + pypto.view 把动态维度
# 切分为固定大小的分块，再用 pypto.assemble 写回输出。

RELU_TILE_B, RELU_TILE_F = 32, 32


@pypto.frontend.jit
def pypto_relu_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    batch = x.shape[0]
    features = x.shape[1]
    n_b = pypto.ceildiv(batch, RELU_TILE_B)
    n_f = pypto.ceildiv(features, RELU_TILE_F)
    for ib in pypto.loop(0, n_b, 1, name="relu_b", idx_name="ib"):
        off_b = ib * RELU_TILE_B
        rows = (batch - off_b).min(RELU_TILE_B)
        for if_ in pypto.loop(0, n_f, 1, name="relu_f", idx_name="if_"):
            off_f = if_ * RELU_TILE_F
            cols = (features - off_f).min(RELU_TILE_F)
            xv = pypto.view(x, [RELU_TILE_B, RELU_TILE_F], [off_b, off_f],
                            valid_shape=[rows, cols])
            pypto.set_vec_tile_shapes(RELU_TILE_B, RELU_TILE_F)
            y = pypto.relu(xv)
            pypto.assemble(y, [off_b, off_f], out)


def get_pypto_linear_kernel(in_features, out_features):
    m_tile, k_tile, n_tile = linear_tile_shapes(in_features, out_features)
    basic_batch = 16

    @pypto.frontend.jit
    def linear_kernel(
            x: pypto.Tensor([pypto.DYNAMIC, in_features], pypto.DT_FP32),
            weight: pypto.Tensor([out_features, in_features], pypto.DT_FP32),
            bias: pypto.Tensor([out_features], pypto.DT_FP32),
            out: pypto.Tensor([pypto.DYNAMIC, out_features], pypto.DT_FP32),
        ):
            batch = x.shape[0]
            n_iter = pypto.ceildiv(batch, basic_batch)
            for i in pypto.loop(0, n_iter, 1, name="linear_m", idx_name="i"):
                off = i * basic_batch
                rows = (batch - off).min(basic_batch)
                xv = pypto.view(x, [basic_batch, in_features], [off, 0],
                                valid_shape=[rows, in_features])
                pypto.set_matrix_size([basic_batch, in_features, out_features])
                pypto.set_cube_tile_shapes(m_tile, k_tile, n_tile)
                h = pypto.matmul(xv, weight, pypto.DT_FP32, b_trans=True)
                pypto.set_vec_tile_shapes(m_tile[0], n_tile[0])
                y = pypto.add(h, bias)
                pypto.assemble(y, [off, 0], out)

    return linear_kernel


# ========== 自动微分包装 ==========
class PyPTOLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, kernel):
        ctx.save_for_backward(X, weight)
        out = torch.empty(X.shape[0], weight.shape[0], device=X.device, dtype=X.dtype)
        kernel(X, weight, bias, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, weight = ctx.saved_tensors
        grad_bias = grad_output.sum(dim=0) if grad_output.shape[1] > 0 else None
        grad_weight = grad_output.t() @ X
        grad_X = grad_output @ weight
        return grad_X, grad_weight, grad_bias, None


class PyPTOReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, kernel):
        ctx.save_for_backward(X)
        # 用显式 shape/dtype/device 创建输出，避免 empty_like 继承
        # torch_npu 内部存储格式而触发 UserWarning
        out = torch.empty(X.shape, device=X.device, dtype=X.dtype)
        kernel(X, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (X,) = ctx.saved_tensors
        return grad_output * (X > 0).float(), None


# ========== nn.Module 包装 ==========
class PyPTOLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = _accelerated_device()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32, device=self.device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=self.device))
        self.reset_parameters()
        self._kernel = get_pypto_linear_kernel(in_features, out_features)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[1]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_kernel', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._kernel = get_pypto_linear_kernel(self.in_features, self.out_features)

    def forward(self, X):
        if X.shape[-1] != self.in_features:
            raise ValueError(f"期望输入维度 {self.in_features}, 得到 {X.shape[-1]}")
        if X.dtype != self.weight.dtype:
            raise TypeError(f"输入类型 {X.dtype} 必须与权重类型 {self.weight.dtype} 一致")

        # 处理多维输入
        leading_shape = X.shape[:-1]  # X.shape[:-1]：取形状的除最后一个之外的所有维度
        X_2d = X.reshape(-1, self.in_features).contiguous()  # -1 表示"自动计算这个维度的大小"
        out_2d = PyPTOLinearFunction.apply(X_2d, self.weight, self.bias, self._kernel)

        return out_2d.reshape(*leading_shape, self.out_features)


class PyPTOReLU(nn.Module):
    def __init__(self):
        super().__init__()
        # 动态形状 kernel：单一 kernel 适配任意 (batch, features)，
        # 无需再按特征维度缓存多份编译结果
        self._kernel = pypto_relu_kernel

    def forward(self, X):
        leading_shape = X.shape[:-1]
        features = X.shape[-1]
        X_2d = X.reshape(-1, features).contiguous()
        out_2d = PyPTOReLUFunction.apply(X_2d, self._kernel)
        return out_2d.reshape(*leading_shape, features)


class PyPTOLazyLinear(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.device = _accelerated_device()
        self.weight = nn.UninitializedParameter()
        self.bias = nn.UninitializedParameter()
        self._kernel = None
        self._initialized = False

    def _lazy_init(self, X):
        if not self._initialized:
            self.in_features = X.shape[-1]
            self.weight.materialize((self.out_features, self.in_features),
                                    dtype=torch.float32, device=self.device)
            self.bias.materialize((self.out_features,),
                                  dtype=torch.float32, device=self.device)
            self.reset_parameters()
            self._kernel = get_pypto_linear_kernel(self.in_features, self.out_features)
            self._initialized = True

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[1]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_kernel'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._initialized:
            self._kernel = get_pypto_linear_kernel(self.in_features, self.out_features)

    def forward(self, X):
        self._lazy_init(X)

        if X.shape[-1] != self.in_features:
            raise ValueError(f"期望输入维度 {self.in_features}, 得到 {X.shape[-1]}")
        if X.dtype != self.weight.dtype:
            raise TypeError(f"输入类型 {X.dtype} 必须与权重类型 {self.weight.dtype} 一致")

        leading_shape = X.shape[:-1]
        X_2d = X.reshape(-1, self.in_features).contiguous()
        out_2d = PyPTOLinearFunction.apply(X_2d, self.weight, self.bias, self._kernel)

        return out_2d.reshape(*leading_shape, self.out_features)
