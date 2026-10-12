# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You should refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""gelu_approx 直调算子 Python 包（direct launch 工程）。

构建安装：
    bash operators/gelu_approx/build.sh --install

调用：
    import op_gelu_approx
    z = op_gelu_approx.gelu_approx(x, y)              # 或
    z = torch.ops.op_gelu_approx.gelu_approx(x, y)    # x/y 为 NPU fp32 等形连续张量
"""

import torch

try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "Cannot import _C. Please make sure the `op_gelu_approx` package is properly installed "
        "(bash operators/gelu_approx/build.sh --install)."
    ) from e


def gelu_approx(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """逐元素 z = isnan(y) ? y : (x == 0 ? +0 : x * log1p(y))。

    规则顺序固定：规则1（先）x==0（含 ±0）⟹ z=+0；
    规则2（后，覆盖规则1）y 为 NaN ⟹ z=y。仅支持 fp32 等形连续输入。
    """
    return torch.ops.op_gelu_approx.gelu_approx(x, y)
