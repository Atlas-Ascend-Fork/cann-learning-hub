"""FastPyPTOLSTM：融合版高性能 PyPTO LSTM。

相比教程 src/pypto_ops.py 的 PyPTOLSTM（~10 次 kernel 调用/步，~110ms/序列），
本实现把每步压缩为 **1 次 pypto kernel 调用**，并支持 NPU 图捕获：

  * 融合 kernel：2 matmul + 4 门 + cell 更新 + H/C 输出 = 1 次调用/步
  * 重算技巧：H 链用重算的 C2b，绕过多输出 kernel 的缓冲别名 bug
  * ping-pong 双缓冲：C 状态输入==输出时避免数据竞争
  * dummy kernel 预热：规避"首编译 vec->matmul 边误编译"bug
  * NPUGraph 图捕获：整条序列烘焙成一张 ACL 图，replay 单次分发

实测（seq=35, batch=32, hidden=256, vocab=28, torch_npu 2.7.1.post4, pypto 0.2.1）：
  torch nn.LSTM           0.53 ms
  本文件 fused eager      ~10 ms
  本文件 fused + 图捕获   ~1.34 ms（cube tile N=256 + device_sched_mode=1；调优前 ~2.6-4.9 ms）

用法（接口与 PyPTOLSTM / nn.LSTM 一致）：
    lstm = FastPyPTOLSTM(vocab_size, hidden_size).to("npu:0")
    Y, (H_n, C_n) = lstm(X, state)      # X: (seq, batch, in); state: (H_stack, C_stack) 或 None

图捕获（推荐）：
    lstm(X0, None)                      # 预热 + 分配内部 buffer（必须在捕获前跑一次）
    g = lstm.make_graph(X0, None)       # 捕获一次完整 forward
    # 训练循环内喂新数据：X0.copy_(new_batch); 然后 g.replay()

注意：
  * 返回值复用内部 buffer，下次 forward 会被覆盖；需长期持有请自行 .clone()
  * 图捕获要求 batch 为 16 的倍数（ch09 用 32，无此问题），且层间不触发分配
  * 本文件只优化前向；反向融合尚未包含
  * hidden 需为 16 的倍数；权重门顺序为 [i, f, o, c]

依赖：pypto 0.2.x / torch / torch_npu（2.7.x）。本文件自包含，可独立拷贝。
"""

import functools
import os
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TILE_FWK_DEVICE_ID", "0")

import pypto
import torch
from torch import nn
from torch.nn import functional as F

try:
    import torch_npu  # noqa: F401
    import torch_npu.npu as npu
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

# 极速模式：直调 C++ 启动路径绕 Python 包装（依赖私有 API，默认关闭）
_DIRECT_LAUNCH = os.environ.get("PYTO_LSTM_DIRECT_LAUNCH", "0") == "1"


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def _dummy_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    b.move(pypto.sigmoid(a))


_dummy_done = False


def _warmup(device):
    """进程内首个 pypto kernel 必须先跑无关 kernel，否则 vec->matmul 边会被误编译。"""
    global _dummy_done
    if not _dummy_done:
        pypto.pypto_impl.DeviceInit()
        a = torch.randn(32, 256, device=device)
        b = torch.empty(32, 256, device=device)
        _dummy_kernel(a, b)
        torch.npu.synchronize()
        _dummy_done = True


def _launch(kernel, args):
    if _DIRECT_LAUNCH:
        try:
            from pypto.frontend.parser import entry as _entry
            in_tensors, ntv, defs = kernel._parse_call_args(args, {})
            kernel._validate_exact_torch_tensors(in_tensors)
            kernel._get_or_create_kmodule(ntv)
            stream = _entry._current_stream()
            pypto.pypto_impl.LaunchKernelTorch(kernel, stream, in_tensors, defs)
            return
        except Exception:
            pass
    kernel(*args)


@functools.lru_cache(maxsize=None)
def _step_kernel(hidden):
    """创建隐藏层宽度为 hidden 的融合 kernel（每步 1 次调用）。"""

    @pypto.frontend.jit(runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "device_sched_mode": 1,
    })
    def lstm_step_fused(
        x_t: pypto.Tensor([], pypto.DT_FP32),
        H: pypto.Tensor([], pypto.DT_FP32),
        C: pypto.Tensor([], pypto.DT_FP32),
        W_x: pypto.Tensor([], pypto.DT_FP32),
        W_h: pypto.Tensor([], pypto.DT_FP32),
        b_i: pypto.Tensor([], pypto.DT_FP32),
        b_f: pypto.Tensor([], pypto.DT_FP32),
        b_o: pypto.Tensor([], pypto.DT_FP32),
        b_c: pypto.Tensor([], pypto.DT_FP32),
        H_out: pypto.Tensor([], pypto.DT_FP32),
        C_out: pypto.Tensor([], pypto.DT_FP32),
    ):
        pypto.set_cube_tile_shapes([32, 32], [16, 16], [256, 256])
        pypto.set_vec_tile_shapes(128, 128)
        xw = pypto.matmul(x_t, W_x, pypto.DT_FP32)
        hw = pypto.matmul(H, W_h, pypto.DT_FP32)
        I = pypto.sigmoid(pypto.add(pypto.add(xw[:, :hidden], hw[:, :hidden]), b_i))
        F = pypto.sigmoid(pypto.add(pypto.add(xw[:, hidden:2 * hidden], hw[:, hidden:2 * hidden]), b_f))
        O = pypto.sigmoid(pypto.add(pypto.add(xw[:, 2 * hidden:3 * hidden], hw[:, 2 * hidden:3 * hidden]), b_o))
        Ct = pypto.tanh(pypto.add(pypto.add(xw[:, 3 * hidden:], hw[:, 3 * hidden:]), b_c))
        C2 = pypto.add(pypto.mul(F, C), pypto.mul(I, Ct))
        # 重算一份给 H 链：多输出 kernel 中被 move 且被其他输出读取的量会被错误复用缓冲
        C2b = pypto.add(pypto.mul(F, C), pypto.mul(I, Ct))
        H_out.move(pypto.mul(O, pypto.tanh(C2b)))
        C_out.move(C2)

    return lstm_step_fused


class FastLSTMStep(torch.autograd.Function):
    """融合 LSTM 单步的 autograd 封装。

    前向：调用融合 pypto kernel（1 次 kernel launch，与 _step_kernel 一致）。
    反向：按标准 LSTM BPTT 公式用 PyTorch 原生算子求导（NPU 上自动加速），
          重算门激活值避免额外显存占用。
    """

    @staticmethod
    def forward(ctx, x_t, H, C, W_x, W_h, b_i, b_f, b_o, b_c,
                kernel_fn, hidden_size):
        batch = x_t.shape[0]
        H_out = torch.empty(batch, hidden_size, device=x_t.device,
                            dtype=x_t.dtype)
        C_out = torch.empty(batch, hidden_size, device=x_t.device,
                            dtype=x_t.dtype)
        kernel_fn(x_t, H, C, W_x, W_h, b_i, b_f, b_o, b_c, H_out, C_out)
        ctx.save_for_backward(x_t, H, C, W_x, W_h, b_i, b_f, b_o, b_c)
        ctx.hidden_size = hidden_size
        return H_out, C_out

    @staticmethod
    def backward(ctx, grad_H, grad_C):
        x_t, H, C, W_x, W_h, b_i, b_f, b_o, b_c = ctx.saved_tensors
        hs = ctx.hidden_size

        # 重算门激活值（供反向使用）
        xw = torch.matmul(x_t, W_x)
        hw = torch.matmul(H, W_h)
        I = torch.sigmoid(xw[:, :hs] + hw[:, :hs] + b_i)
        F_g = torch.sigmoid(xw[:, hs:2 * hs] + hw[:, hs:2 * hs] + b_f)
        O = torch.sigmoid(xw[:, 2 * hs:3 * hs] + hw[:, 2 * hs:3 * hs] + b_o)
        Ct = torch.tanh(xw[:, 3 * hs:] + hw[:, 3 * hs:] + b_c)
        C_new = F_g * C + I * Ct
        tanh_C = torch.tanh(C_new)

        # 标准 LSTM 反向传播
        dC_total = grad_C + grad_H * O * (1.0 - tanh_C * tanh_C)

        dI_pre = dC_total * Ct * I * (1.0 - I)
        dF_pre = dC_total * C * F_g * (1.0 - F_g)
        dO_pre = grad_H * tanh_C * O * (1.0 - O)
        dCt_pre = dC_total * I * (1.0 - Ct * Ct)

        dC_grad = dC_total * F_g

        # 门顺序 [i, f, o, c] 与 W_x/W_h 的列顺序一致
        dgates = torch.cat([dI_pre, dF_pre, dO_pre, dCt_pre], dim=1)

        dx_t = torch.matmul(dgates, W_x.t())
        dH = torch.matmul(dgates, W_h.t())
        dW_x = torch.matmul(x_t.t(), dgates)
        dW_h = torch.matmul(H.t(), dgates)
        db_i = dI_pre.sum(dim=0)
        db_f = dF_pre.sum(dim=0)
        db_o = dO_pre.sum(dim=0)
        db_c = dCt_pre.sum(dim=0)

        return (dx_t, dH, dC_grad, dW_x, dW_h,
                db_i, db_f, db_o, db_c, None, None)


class FastPyPTOLSTM(nn.Module):
    """融合版 PyPTO LSTM（多层、单向/双向），参数名/接口与 PyPTOLSTM 一致，可替换使用。

    bidirectional=True 时，每层额外运行一个反向 LSTM（时间步倒序递推），
    前向/反向隐状态在特征维拼接为 2*hidden_size，与 nn.LSTM(bidirectional=True) 语义一致。
    state 为 (H, C)，各形状 (num_layers*num_directions, batch, hidden)，方向与层交错排列。
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0,
                 bidirectional=False):
        super().__init__()
        if not _HAS_NPU:
            raise RuntimeError("FastPyPTOLSTM 需要 torch_npu 环境（昇腾 NPU）")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # 逐方向逐层参数：方向后缀（前向 ''，反向 '_rev'）+ 层后缀（第 0 层 ''，第 l 层 '_{l}'）
        # 参数名均含 "W_" 前缀，门顺序 [i, f, o, c]
        self._flat_weights_names = []
        for direction in range(self.num_directions):
            d = '_rev' if direction == 1 else ''
            for l in range(num_layers):
                lsuf = '' if l == 0 else f'_{l}'
                suffix = f'{d}{lsuf}'
                # 第 0 层输入为 input_size；后续层输入为上一层输出 hidden_size*num_directions
                in_size = input_size if l == 0 else hidden_size * self.num_directions
                setattr(self, f'W_x{suffix}',
                        nn.Parameter(torch.randn(in_size, hidden_size * 4) * 0.01))
                setattr(self, f'W_h{suffix}',
                        nn.Parameter(torch.randn(hidden_size, hidden_size * 4) * 0.01))
                setattr(self, f'b_i{suffix}', nn.Parameter(torch.zeros(hidden_size)))
                setattr(self, f'b_f{suffix}', nn.Parameter(torch.zeros(hidden_size)))
                setattr(self, f'b_o{suffix}', nn.Parameter(torch.zeros(hidden_size)))
                setattr(self, f'b_c{suffix}', nn.Parameter(torch.zeros(hidden_size)))
                self._flat_weights_names.append(f'W_x{suffix}')
                self._flat_weights_names.append(f'W_h{suffix}')
        self._kernel = _step_kernel(hidden_size)
        self._bufs = {}

    def flatten_parameters(self):
        """与 nn.LSTM 接口兼容（权重已是独立 Parameter，无需展平）。"""
        pass

    def _bufs_for(self, seq, batch, device):
        key = (seq, batch, str(device))
        bufs = self._bufs.get(key)
        if bufs is None:
            H = self.hidden_size
            nd = self.num_directions
            bufs = {
                # y[dir][buf_idx]：每方向一套跨层 ping-pong 缓冲；层内每步写到 y[t]，天然无别名
                "y": [[torch.empty(seq, batch, H, device=device) for _ in range(2)]
                      for _ in range(nd)],
                # c[dir][layer][buf_idx]：C 状态 ping-pong 双缓冲（输入==输出必须）
                "c": [[[torch.empty(batch, H, device=device) for _ in range(2)]
                       for _ in range(self.num_layers)] for _ in range(nd)],
                "z0": torch.zeros(self.num_layers * nd, batch, H, device=device),
                "xpad": torch.empty(seq, batch, self.input_size, device=device),
            }
            self._bufs[key] = bufs
        return bufs

    def _run_pass(self, in_x, l, rev, H_init, C_init, bufs, seq):
        """跑一个方向的单层递推。

        rev=False：时间步 0→seq-1；rev=True：时间步 seq-1→0（反向递推）。
        无论方向，结果都写入 y_buf[t]（即时间步 t 的隐状态），因此输出天然为正序。

        训练时用 FastLSTMStep（autograd 封装，保证反向传播正确）；
        推理时直接调融合 kernel（零开销，无需梯度）。
        """
        d = 1 if rev else 0
        dsuf = '_rev' if rev else ''
        lsuf = '' if l == 0 else f'_{l}'
        suffix = f'{dsuf}{lsuf}'
        W_x = getattr(self, f'W_x{suffix}')
        W_h = getattr(self, f'W_h{suffix}')
        b_i, b_f, b_o, b_c = (getattr(self, f'b_{g}{suffix}') for g in "ifoc")
        H_cur, C_cur = H_init, C_init
        indices = range(seq - 1, -1, -1) if rev else range(seq)
        if self.training:
            # 训练：autograd 封装的融合 kernel，反向传播正确
            outputs = [None] * seq
            for t in indices:
                H_cur, C_cur = FastLSTMStep.apply(
                    in_x[t], H_cur, C_cur, W_x, W_h, b_i, b_f, b_o, b_c,
                    self._kernel, self.hidden_size)
                outputs[t] = H_cur
            y_buf = torch.stack(outputs, dim=0)
        else:
            # 推理：直接调融合 kernel（最快，无需梯度）
            y_buf = bufs["y"][d][l % 2]
            c_bufs = bufs["c"][d][l]
            for t in indices:
                C_new = c_bufs[t % 2]
                _launch(self._kernel, (in_x[t], H_cur, C_cur, W_x, W_h,
                                       b_i, b_f, b_o, b_c, y_buf[t], C_new))
                H_cur, C_cur = y_buf[t], C_new
        return y_buf, H_cur, C_cur

    def forward(self, X, state=None):
        """前向：X (seq, batch, in) -> Y (seq, batch, hidden*num_directions), (H_n, C_n)。

        单向时返回的 Y / H_n / C_n 复用内部 buffer，下次 forward 会被覆盖；
        需要长期持有请自行 clone()。双向时 Y 为拼接结果（新张量）。
        """
        if not _HAS_NPU:
            raise RuntimeError("FastPyPTOLSTM 需要 torch_npu 环境")
        _warmup(X.device)
        seq, orig_batch = X.shape[0], X.shape[1]
        batch, padded = orig_batch, False
        if batch % 16 != 0:
            batch += 16 - batch % 16
            padded = True
        bufs = self._bufs_for(seq, batch, X.device)
        X_use = X
        if padded:
            bufs["xpad"][:, :orig_batch].copy_(X)
            X_use = bufs["xpad"]
        nd = self.num_directions
        if state is None:
            H_stack = C_stack = bufs["z0"]
        else:
            H_stack, C_stack = state
            if padded:
                H_stack = F.pad(H_stack, (0, 0, 0, batch - orig_batch, 0, 0))
                C_stack = F.pad(C_stack, (0, 0, 0, batch - orig_batch, 0, 0))

        in_x = X_use
        h_finals, c_finals = [], []
        layer_out = None
        for l in range(self.num_layers):
            # 前向方向
            y_f, h_f, c_f = self._run_pass(
                in_x, l, False, H_stack[l * nd], C_stack[l * nd], bufs, seq)
            h_finals.append(h_f)
            c_finals.append(c_f)
            if nd == 2:
                # 反向方向
                y_r, h_r, c_r = self._run_pass(
                    in_x, l, True, H_stack[l * nd + 1], C_stack[l * nd + 1], bufs, seq)
                h_finals.append(h_r)
                c_finals.append(c_r)
                layer_out = torch.cat([y_f[:seq], y_r[:seq]], dim=-1)
            else:
                layer_out = y_f if not padded else y_f[:, :orig_batch]
            # 层间 dropout：仅训练时、仅在层之间施加（与 nn.LSTM 一致）
            if l < self.num_layers - 1:
                if self.dropout > 0 and self.training:
                    in_x = F.dropout(layer_out, p=self.dropout)
                else:
                    in_x = layer_out

        if padded:
            layer_out = layer_out[:, :orig_batch]
            h_finals = [h[:orig_batch] for h in h_finals]
            c_finals = [c[:orig_batch] for c in c_finals]
        H_n = torch.stack(h_finals)
        C_n = torch.stack(c_finals)
        return layer_out, (H_n, C_n)

    def make_graph(self, X, state=None):
        """把一次完整 forward 捕获成 NPUGraph。

        仅支持单向（双向 forward 含动态 cat 分配，无法被图捕获）。
        调用前必须已用同样 shape 跑过一次 forward（预热编译 + 预分配 buffer）。
        之后每次 g.replay() 用捕获时的同一批张量执行；喂新数据请原地写入：
        X.copy_(new_batch)，再 g.replay()。
        注意：图捕获要求 batch 为 16 的倍数，且 forward 内不触发任何内存分配。
        """
        if self.bidirectional:
            raise RuntimeError("make_graph 暂不支持 bidirectional=True")
        torch.npu.synchronize()
        g = npu.NPUGraph()
        with npu.graph(g):
            self(X, state)
        torch.npu.synchronize()
        return g


def _bench(fn, iters=200, warm=3):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    torch.npu.set_device(int(os.environ["TILE_FWK_DEVICE_ID"]))
    pypto.pypto_impl.DeviceInit()

    torch.manual_seed(0)
    seq, batch, hidden, vocab = 35, 32, 256, 28

    lstm = FastPyPTOLSTM(vocab, hidden).to("npu:0")
    X = torch.randn(seq, batch, vocab, device="npu:0")

    # 与 torch nn.LSTM 对齐验证（门顺序映射 [i,f,o,c] -> [i,f,g,o]）
    ref = nn.LSTM(vocab, hidden).to("npu:0")
    with torch.no_grad():
        w_ih = torch.cat([lstm.W_x[:, :hidden], lstm.W_x[:, hidden:2 * hidden],
                          lstm.W_x[:, 3 * hidden:], lstm.W_x[:, 2 * hidden:3 * hidden]], dim=1).t().contiguous()
        w_hh = torch.cat([lstm.W_h[:, :hidden], lstm.W_h[:, hidden:2 * hidden],
                          lstm.W_h[:, 3 * hidden:], lstm.W_h[:, 2 * hidden:3 * hidden]], dim=1).t().contiguous()
        b = torch.cat([lstm.b_i, lstm.b_f, lstm.b_c, lstm.b_o])
        ref.weight_ih_l0.copy_(w_ih)
        ref.weight_hh_l0.copy_(w_hh)
        ref.bias_ih_l0.copy_(b)
        ref.bias_hh_l0.copy_(b)

    Y, (H_n, C_n) = lstm(X, None)
    Yr, (Hr, Cr) = ref(X, None)
    torch.npu.synchronize()
    print(f"与 torch nn.LSTM 对齐: max|err| Y={(Y - Yr).abs().max().item():.2e} "
          f"H={(H_n - Hr).abs().max().item():.2e} C={(C_n - Cr).abs().max().item():.2e}")

    t_eager = _bench(lambda: lstm(X, None))
    g = lstm.make_graph(X, None)
    t_graph = _bench(lambda: g.replay())
    t_torch = _bench(lambda: ref(X, None))
    print(f"FastPyPTOLSTM eager : {t_eager * 1e3:7.2f} ms/序列")
    print(f"FastPyPTOLSTM graph  : {t_graph * 1e3:7.2f} ms/序列   (eager 的 {t_eager / t_graph:.1f}x)")
    print(f"torch nn.LSTM        : {t_torch * 1e3:7.2f} ms/序列")


if __name__ == "__main__":
    main()
