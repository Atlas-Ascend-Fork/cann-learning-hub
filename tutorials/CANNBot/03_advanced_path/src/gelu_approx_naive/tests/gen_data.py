#!/usr/bin/env python3
"""gelu_approx 黑盒测试数据生成——最小集."""
import numpy as np

F32 = np.float32
SUBNORMAL = np.nextafter(F32(0), F32(1))
FP32_MAX = np.finfo(np.float32).max
MIN_NORMAL = np.finfo(np.float32).tiny

# 特殊值集合
X_SPECIALS = [
    F32(0.0), F32(-0.0), np.nan, np.inf, -np.inf,
    F32(1.0), F32(-1.0), F32(3.0), F32(-3.0),
    F32(10.0), F32(-10.0), F32(15.0), F32(-15.0),
    F32(20.0), F32(-20.0), F32(50.0), F32(-50.0),
    SUBNORMAL, -SUBNORMAL, MIN_NORMAL, -MIN_NORMAL,
    F32(1e-8), F32(-1e-8), F32(1e-4), F32(-1e-4),
    F32(5.0), F32(-5.0), F32(8.0), F32(-8.0),
]

def _case(cid, level, kind, x):
    return {"case_id": cid, "level": level, "kind": kind,
            "x": x, "shape": x.shape}

def _uniform(seed, n, lo=-3.0, hi=3.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, n).astype(F32)

def build_case(cid, seed=20261007):
    builders = CASE_BUILDERS
    return builders[cid](seed)

# ── L0：基础 ──
def _l0_1(seed):
    x = _uniform(seed, 1024)
    return _case("L0-1", "L0", "normal", x)

def _l0_2(seed):
    x = _uniform(seed, 64 * 64).reshape(64, 64)
    return _case("L0-2", "L0", "normal", x)

def _l0_3(seed):
    x = np.array([F32(1.5)], dtype=F32)
    return _case("L0-3", "L0", "normal", x)

def _l0_4(seed):
    """非对齐尾块 [4093]：4093%8=5."""
    x = _uniform(seed, 4093)
    x[-16:] = np.array(X_SPECIALS[:16], dtype=F32)
    return _case("L0-4", "L0", "normal", x)

def _l0_5(seed):
    x = _uniform(seed, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    return _case("L0-5", "L0", "normal", x)

# ── L1：精度暴露 ──
def _l1_1(seed):
    """精度暴露网格 [256]：|x| ∈ [5, 20] 稠密采样."""
    x = np.linspace(5.0, 20.0, 256).astype(F32)
    return _case("L1-1", "L1", "normal", x)

def _l1_2(seed):
    """负值精度暴露 [256]."""
    x = np.linspace(-20.0, -5.0, 256).astype(F32)
    return _case("L1-2", "L1", "normal", x)

def _l1_3(seed):
    """特殊值笛卡尔积 [64]：32 个特殊值."""
    x = np.array(X_SPECIALS[:32], dtype=F32)
    return _case("L1-3", "L1", "normal", x)

def _l1_4(seed):
    """大 |x| 饱和区 [128]：|x| > 20 → z ≈ x (正) 或 z ≈ 0 (负)."""
    pos = np.linspace(20.0, 100.0, 64).astype(F32)
    neg = np.linspace(-100.0, -20.0, 64).astype(F32)
    x = np.concatenate([pos, neg])
    return _case("L1-4", "L1", "normal", x)

# ── L2：边界与异常 ──
def _l2_1(seed):
    x = np.array([0.0], dtype=F32)  # 空 shape 退化
    return _case("L2-1", "L2", "empty", x[:0])

def _l2_2(seed):
    x = np.array([np.nan, np.inf, -np.inf, F32(0), F32(-0)], dtype=F32)
    return _case("L2-2", "L2", "error_values", x)

def _l2_3(seed):
    x = np.array([F32(1e30), F32(-1e30), F32(1e35), F32(-1e35)], dtype=F32)
    return _case("L2-3", "L2", "error_values", x)

def _l2_4(seed):
    x = np.array([SUBNORMAL, -SUBNORMAL, MIN_NORMAL, -MIN_NORMAL], dtype=F32)
    return _case("L2-4", "L2", "error_values", x)

def _l2_5(seed):
    """大张量散布特殊值 [10000]."""
    x = _uniform(seed, 10000, lo=-10, hi=10)
    idx = np.linspace(0, 9999, 32).astype(int)
    n_sp = min(len(X_SPECIALS), len(idx))
    x[idx[:n_sp]] = np.array(X_SPECIALS[:n_sp], dtype=F32)
    return _case("L2-5", "L2", "normal", x)

CASE_BUILDERS = {
    "L0-1": _l0_1, "L0-2": _l0_2, "L0-3": _l0_3, "L0-4": _l0_4, "L0-5": _l0_5,
    "L1-1": _l1_1, "L1-2": _l1_2, "L1-3": _l1_3, "L1-4": _l1_4,
    "L2-1": _l2_1, "L2-2": _l2_2, "L2-3": _l2_3, "L2-4": _l2_4, "L2-5": _l2_5,
}

def assert_close_with_specials(actual, expected, rtol=1e-5, atol=1e-7):
    """三段精度判定：NaN位相等 → ±Inf同号同位 → 有限值容差."""
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"
    assert actual.dtype == np.float32 and expected.dtype == np.float32
    nan_mask = np.isnan(expected)
    assert np.array_equal(np.isnan(actual), nan_mask), \
        f"NaN position mismatch: {np.sum(np.isnan(actual))} vs {np.sum(nan_mask)}"
    inf_mask = np.isinf(expected)
    if np.any(inf_mask):
        assert np.array_equal(np.sign(actual[inf_mask]), np.sign(expected[inf_mask])), \
            "Inf sign mismatch"
    finite = ~nan_mask & ~inf_mask
    if np.any(finite):
        diff = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
        tol = atol + rtol * np.abs(expected[finite].astype(np.float64))
        bad = diff > tol
        assert not np.any(bad), \
            f"tolerance violation at {np.sum(bad)} positions, max diff={np.max(diff):.6e}, tol={np.min(tol):.6e}"
