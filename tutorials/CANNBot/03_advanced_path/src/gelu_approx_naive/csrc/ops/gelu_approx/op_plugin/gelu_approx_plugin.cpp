/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A
 * PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gelu_approx_plugin.cpp
 * \brief gelu_approx API 层 - torch.library 注册（g++ 编译）
 *
 * 接口契约（需求分析.md R3/R4）：
 *   schema: gelu_approx(Tensor x) -> Tensor
 *   namespace: op_gelu_approx（torch.ops.op_gelu_approx.gelu_approx(x, y)）
 *   校验：等形、双方 fp32、row-major 连续；非等形/非 fp32 抛 RuntimeError
 *   （消息含 shape/dtype 可定位原因，L2-1~L2-4 契约）；空 tensor 直接返回
 *   （numel=0 不 launch，L2-5~L2-7 契约）。kernel 不做广播与非连续处理
 *   （R4：非等形/非连续由调用方 host 侧物化）。
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/gelu_approx_launch.h"

namespace op_gelu_approx {

// schema 注册（与需求分析.md R3 接口契约一致）
TORCH_LIBRARY_FRAGMENT(op_gelu_approx, m)
{
    m.def("gelu_approx(Tensor x) -> Tensor");
}

// 输入校验 + 输出 shape/dtype 推导（Meta 与 NPU 实现共用）
torch::Tensor gelu_approx_meta(const torch::Tensor &x)
{
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
        "gelu_approx: input must be float32 dtype, got ", x.scalar_type());
    TORCH_CHECK(x.is_contiguous(),
        "gelu_approx: input must be contiguous (row-major)");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(op_gelu_approx, Meta, m)
{
    m.impl("gelu_approx", gelu_approx_meta);
}

// NPU 实现（调 tiling + launch kernel）
torch::Tensor gelu_approx_npu(const torch::Tensor &x)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto z = gelu_approx_meta(x);
    int64_t totalLength = x.numel();
    if (totalLength == 0) {
        return z;  // 空 tensor：直接返回，不下发 kernel（R4）
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t numBlocks, blockLength, tileElems;
    std::tie(numBlocks, blockLength, tileElems) = calc_gelu_approx_tiling_params(totalLength);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto z_ptr = (GM_ADDR)z.data_ptr();

    auto acl_call = [=]() -> int {
        if (x.scalar_type() == torch::kFloat32) {
            launch_gelu_approx_kernel_float(x_ptr, z_ptr, totalLength, numBlocks, blockLength,
                                        static_cast<uint32_t>(tileElems), stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Xlog1py", acl_call);
    return z;
}

TORCH_LIBRARY_IMPL(op_gelu_approx, PrivateUse1, m)
{
    m.impl("gelu_approx", gelu_approx_npu);
}

}  // namespace op_gelu_approx
