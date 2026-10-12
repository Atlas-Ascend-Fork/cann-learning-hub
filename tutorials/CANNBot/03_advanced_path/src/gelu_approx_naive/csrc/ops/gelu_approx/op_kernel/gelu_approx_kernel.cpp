/**
 * \file gelu_approx_kernel.cpp
 * \brief gelu_approx kernel: z = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))
 *
 * 逐元素一元运算。近似 GELU 激活函数（LLM FFN 核心激活）。
 */

#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "gelu_approx_launch.h"

namespace {
constexpr int32_t PIPELINE_DEPTH = 2;
constexpr int64_t UB_BYTES_PER_ELEM = 56;
constexpr int64_t MAX_TILE_ELEMS = 2048;
}  // namespace

template <typename T>
__global__ __aicore__ void gelu_approx_kernel(GM_ADDR x, GM_ADDR z,
    int64_t totalLength, int64_t blockLength, uint32_t tileElems)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, zGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileElems * sizeof(T));
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileElems * sizeof(T));

    // 计算缓冲
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufCubed, bufInner, bufTanhIn, bufTanhOut, bufResult, bufHalf, bufOne, bufTemp;
    pipe.InitBuffer(bufCubed, tileElems * sizeof(T));
    pipe.InitBuffer(bufInner, tileElems * sizeof(T));
    pipe.InitBuffer(bufTanhIn, tileElems * sizeof(T));
    pipe.InitBuffer(bufTanhOut, tileElems * sizeof(T));
    pipe.InitBuffer(bufResult, tileElems * sizeof(T));
    pipe.InitBuffer(bufHalf, tileElems * sizeof(T));
    pipe.InitBuffer(bufOne, tileElems * sizeof(T));
    pipe.InitBuffer(bufTemp, tileElems * sizeof(T));

    // 常量
    {
        auto half = bufHalf.Get<T>();
        AscendC::Duplicate(half, static_cast<T>(0.5), static_cast<int32_t>(tileElems));
        auto one = bufOne.Get<T>();
        AscendC::Duplicate(one, static_cast<T>(1.0), static_cast<int32_t>(tileElems));
    }

    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());

    int64_t currentBlockLength =
        totalLength - static_cast<int64_t>(AscendC::GetBlockIdx()) * blockLength;
    if (currentBlockLength > blockLength) currentBlockLength = blockLength;
    if (currentBlockLength < 0) currentBlockLength = 0;
    int64_t tileNum = currentBlockLength / tileElems;
    int64_t tailTileElementNum = currentBlockLength - tileNum * tileElems;
    int64_t totalTiles = tileNum + (tailTileElementNum > 0 ? 1 : 0);

    for (int64_t i = 0; i < totalTiles; ++i) {
        int64_t offset = i * tileElems;
        int64_t validCount = (i < tileNum) ? tileElems : tailTileElementNum;
        int64_t computeCount = validCount;

        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(validCount * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);

        xLocal = inQueueX.DeQue<T>();
        auto zLocal = outQueueZ.AllocTensor<T>();
        auto cubed = bufCubed.Get<T>();
        auto inner = bufInner.Get<T>();
        auto tanhIn = bufTanhIn.Get<T>();
        auto tanhOut = bufTanhOut.Get<T>();
        auto result = bufResult.Get<T>();
        auto half = bufHalf.Get<T>();
        auto one = bufOne.Get<T>();
        auto temp = bufTemp.Get<T>();
        const int32_t cnt32 = static_cast<int32_t>(computeCount);
        const uint32_t cntU32 = static_cast<uint32_t>(computeCount);

        // ---- z = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) ----

        // Step 1: cubed = x * x * x  (坑3: 大 x 时溢出——naive 不做缩放)
        AscendC::Mul(cubed, xLocal, xLocal, cnt32);       // cubed = x^2
        AscendC::Mul(cubed, cubed, xLocal, cnt32);         // cubed = x^3

        // Step 2: cubed *= 0.044715
        AscendC::Muls(cubed, cubed, static_cast<T>(0.044715), cnt32);

        // Step 3: inner = x + cubed
        AscendC::Add(inner, xLocal, cubed, cnt32);

        // Step 4: tanhIn = inner * sqrt(2/pi)
        AscendC::Muls(tanhIn, inner, static_cast<T>(0.7978845608), cnt32);

        // Step 5: tanhOut = tanh(tanhIn)  (坑1: Tanh 指令精度)
        AscendC::Tanh(tanhOut, tanhIn, cnt32);

        // Step 6: tanhOut += 1
        AscendC::Add(tanhOut, tanhOut, one, cnt32);

        // Step 7: result = x * tanhOut
        AscendC::Mul(result, xLocal, tanhOut, cnt32);

        // Step 8: z = result * 0.5
        AscendC::Mul(zLocal, result, half, cnt32);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);

        zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }
}

// Tiling 计算
std::tuple<int64_t, int64_t, int64_t> calc_gelu_approx_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    numBlocks = std::max(numBlocks, static_cast<int64_t>(1));
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t budget = (ubSize > 8192) ? static_cast<int64_t>(ubSize) - 4096 : static_cast<int64_t>(ubSize);
    int64_t tileElems = budget / UB_BYTES_PER_ELEM;
    tileElems = std::min(tileElems, MAX_TILE_ELEMS);
    tileElems = tileElems / 64 * 64;
    if (tileElems < 64) tileElems = 64;
    return std::make_tuple(numBlocks, blockLength, tileElems);
}

extern "C" {
void launch_gelu_approx_kernel_float(GM_ADDR x, GM_ADDR z,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, void* stream)
{
    gelu_approx_kernel<float><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileElems);
}
}
