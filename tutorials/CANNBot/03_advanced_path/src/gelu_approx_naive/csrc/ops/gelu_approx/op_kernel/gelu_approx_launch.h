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
 * \file gelu_approx_launch.h
 * \brief Launch function declarations for g++（plugin 侧可见）
 */

#ifndef GELU_APPROX_LAUNCH_H
#define GELU_APPROX_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling function declaration（返回 numBlocks, blockLength, tileElems）
std::tuple<int64_t, int64_t, int64_t> calc_gelu_approx_tiling_params(int64_t totalLength);

// Launch function declarations
extern "C" {
void launch_gelu_approx_kernel_float(GM_ADDR x, GM_ADDR z,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, void* stream);
}

#endif  // GELU_APPROX_LAUNCH_H
