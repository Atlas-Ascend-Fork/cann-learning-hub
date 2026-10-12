"""gelu_approx 黑盒测试入口——14 用例."""
import datetime
import json
import os

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

import op_gelu_approx  # noqa: F401  registers torch.ops.op_gelu_approx

from gen_data import CASE_BUILDERS, assert_close_with_specials
from golden.gelu_approx_golden import gelu_approx_golden

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEVEL_ORDER = ["L0", "L1", "L2"]

def _run_gm(x_np):
    x_t = torch.from_numpy(x_np).npu()
    z_t = torch.ops.op_gelu_approx.gelu_approx(x_t)
    torch.npu.synchronize()
    return z_t.cpu().numpy()

def _build_and_run(case):
    z = _run_gm(case["x"])
    g = gelu_approx_golden(case["x"])
    return z, g

def _make_test(cid):
    def test_fn():
        case = CASE_BUILDERS[cid](seed=20261007)
        if case["kind"] == "empty":
            z = _run_gm(case["x"])
            assert z.size == 0
            return
        z, g = _build_and_run(case)
        assert_close_with_specials(z, g, rtol=1e-5, atol=1e-7)
    test_fn.__name__ = f"test_gelu_approx_blackbox[{cid}]"
    return test_fn

for _cid in CASE_BUILDERS:
    globals()[f"test_gelu_approx_blackbox_{_cid.replace('-', '_')}"] = _make_test(_cid)
