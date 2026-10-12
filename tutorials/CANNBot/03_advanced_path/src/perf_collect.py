#!/usr/bin/env python3
"""Lab C 性能采集脚本：gelu_approx 基线."""
import argparse, json, os, statistics
import torch, torch_npu
import op_gelu_approx

CASES = [
    ("L_gelu_2d",    (28, 713, 128)),           # L 档
    ("L_gelu_1d",    (71934,)),                   # L 档
    ("M_gelu_3d",    (26, 19, 50616)),            # M 档
    ("S_gelu_small", (9,)),                       # S 档
    ("S_gelu_2d",    (20, 53)),                   # S 档
    ("A_gelu_align", (54, 17, 2851),),            # 对齐对照
]

def numel(shape):
    n = 1
    for d in shape: n *= d
    return n

def run_case(name, xs, rounds=10, warmup=3):
    n = numel(xs)
    x = torch.randn(xs, dtype=torch.float32).mul_(2.0).npu()
    for _ in range(warmup):
        torch.ops.op_gelu_approx.gelu_approx(x)
    torch.npu.synchronize()
    times = []
    for _ in range(rounds):
        s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        s.record(); torch.ops.op_gelu_approx.gelu_approx(x); e.record()
        torch.npu.synchronize()
        times.append(s.elapsed_time(e))
    ms = statistics.median(times)
    bytes_moved = n * 4 * 2  # 读 x + 写 z
    gbps = bytes_moved / (ms / 1e3) / 1e9
    print(f"  {name:16s} n={n:>12,}  t={ms:10.4f} ms  BW={gbps:8.1f} GB/s")
    del x
    return {"case": name, "numel": n, "ms": round(ms, 4), "gbps": round(gbps, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/perf_baseline.json")
    args = ap.parse_args()
    print(f"NPU: {torch.npu.get_device_name(0)}")
    print("采集基线（中位 × 10 轮，warmup 3）：")
    rows = [run_case(name, xs) for name, xs in CASES]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"op": "gelu_approx", "mode": "baseline", "rows": rows}, f, indent=2)
    print(f"已写入 {args.out}（{len(rows)} 用例）")

if __name__ == "__main__":
    main()
