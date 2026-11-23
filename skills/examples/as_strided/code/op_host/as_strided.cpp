// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "../op_kernel/as_strided_tiling.h"

#include "../op_kernel/tiling_key_as_strided.h"

namespace optiling {
    static inline uint32_t AlignUp(uint32_t value, uint32_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    // 校验 ndim/size/stride 属性并填充定长数组，计算输出总元素数与 dimStride；
    // 非法参数返回 GRAPH_FAILED，避免写穿定长数组与 uint32 回绕
    static ge::graphStatus ParseShapeAttrs(const int64_t *sizeData, const int64_t *strideData,
                                           uint32_t ndim, uint32_t strideLen, uint32_t sizeArr[4],
                                           int32_t strideArr[4], uint32_t dimStrideArr[4],
                                           uint32_t &totalOutputElements) {
        // 维度数约定 1-4（见 AsStridedTilingData.ndim 注释）：ndim==0 时 ndim-1 会下溢为极大下标
        if (ndim == 0 || ndim > 4) {
            return ge::GRAPH_FAILED;
        }
        // stride 与 size 逐维对应：列表短于 ndim 时按 ndim 读 strideData 会越界
        if (strideLen != ndim) {
            return ge::GRAPH_FAILED;
        }
        // 64 位累乘并在超过 uint32 上限时拒绝，避免回绕后继续填 TilingData
        uint64_t totalOutputElements64 = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            if (sizeData[d] <= 0 || sizeData[d] > static_cast<int64_t>(UINT32_MAX)) {
                return ge::GRAPH_FAILED;
            }
            sizeArr[d] = static_cast<uint32_t>(sizeData[d]);
            strideArr[d] = static_cast<int32_t>(strideData[d]);
            totalOutputElements64 *= static_cast<uint64_t>(sizeArr[d]);
            if (totalOutputElements64 > static_cast<uint64_t>(UINT32_MAX)) {
                return ge::GRAPH_FAILED;
            }
        }
        totalOutputElements = static_cast<uint32_t>(totalOutputElements64);
        // dimStride[d] = size[d+1] * size[d+2] * ... * size[ndim-1]
        dimStrideArr[ndim - 1] = 1;
        for (int d = static_cast<int>(ndim) - 2; d >= 0; d--) {
            dimStrideArr[d] = dimStrideArr[d + 1] * sizeArr[d + 1];
        }
        return ge::GRAPH_SUCCESS;
    }

    // 路径判断和 tileSize 计算，返回 pathFlag（0=PathA, 1=PathB）
    static uint32_t CalcTileSize(uint32_t inputTotalElements, uint32_t dtypeSize, uint32_t &tileSize) {
        constexpr uint32_t UB_SIZE = 196608;  // 192KB for ascend910b (DAV_2201)
        constexpr uint32_t RESERVED = 8192;   // 预留 8KB
        uint32_t ubBudget = UB_SIZE - RESERVED;  // 188416
        uint32_t inputBytesAligned = AlignUp(inputTotalElements * dtypeSize, 32);
        uint32_t pathFlag = 1;  // 默认 Path B
        tileSize = 0;
        // Path A: 输入全量在 UB
        if (inputBytesAligned < ubBudget) {
            uint32_t remainBudget = ubBudget - inputBytesAligned;
            // offset 表: tileSize * 4, 输出: tileSize * dtypeSize * 2 (Double Buffer)
            uint32_t perTileBytes = 4 + dtypeSize * 2;
            tileSize = (remainBudget / perTileBytes) & ~0x7u;  // 向下对齐到 8 元素
            tileSize = std::max(tileSize, 8u);
            uint32_t offsetTileBytes = AlignUp(tileSize * 4, 32);
            uint32_t outputTileBytes = AlignUp(tileSize * dtypeSize, 32);
            if (inputBytesAligned + offsetTileBytes + outputTileBytes * 2 <= ubBudget) {
                pathFlag = 0;  // Path A
            }
        }
        if (pathFlag == 1) {
            // Path B: 无输入全量，tileSize 受 srcIdx 表 + 输出 tile 约束
            uint32_t remainBudget = ubBudget - 32;  // 减去单元素临时 buffer
            uint32_t perTileBytes = 4 + dtypeSize;
            tileSize = (remainBudget / perTileBytes) & ~0x7u;
            tileSize = std::max(tileSize, 8u);
        }
        return pathFlag;
    }

    // 预计算偏移表（Host侧一次性计算，消除 Kernel 内除法/取模和 SetValue）
    static void BuildOffsetTable(const AsStridedTilingData &tiling, uint32_t dtypeSize,
                                 std::vector<uint32_t> &offsetTable) {
        for (uint32_t f = 0; f < tiling.totalOutputElements; f++) {
            // 从平坦索引反推多维索引
            int32_t srcIdx = tiling.storageOffset;
            uint32_t remaining = f;
            for (uint32_t d = 0; d < tiling.ndim; d++) {
                uint32_t i_d = remaining / tiling.dimStride[d];
                remaining = remaining % tiling.dimStride[d];
                srcIdx += static_cast<int32_t>(i_d) * tiling.stride[d];
            }
            // 防御性 clamp：防止负 stride 或越界参数导致 Gather 偏移越界
            if (srcIdx < 0 || static_cast<uint32_t>(srcIdx) >= tiling.inputTotalElements) {
                srcIdx = 0;
            }
            if (tiling.pathFlag == 0) {
                // Path A: byte offset for Gather
                offsetTable[f] = static_cast<uint32_t>(srcIdx) * dtypeSize;
            } else {
                // Path B: element index for DataCopyPad
                offsetTable[f] = static_cast<uint32_t>(srcIdx);
            }
        }
    }

    // 容量校验通过后才预计算偏移表并追加到 tiling data（避免 Kernel 内 SetValue 循环），成功返回 1；
    // 容量不足时不建表（Kernel 回退自算偏移），避免大 shape 下无意义的巨额 host 分配
    static uint32_t AppendOffsetTable(gert::TilingContext *context, const AsStridedTilingData &tiling,
                                      uint32_t dtypeSize) {
        auto rawTiling = context->GetRawTilingData();
        if (rawTiling == nullptr) {
            return 0;
        }
        size_t tableBytes = static_cast<size_t>(tiling.totalOutputElements) * sizeof(uint32_t);
        if (rawTiling->GetDataSize() + tableBytes > rawTiling->GetCapacity()) {
            return 0;
        }
        std::vector<uint32_t> offsetTable(tiling.totalOutputElements);
        BuildOffsetTable(tiling, dtypeSize, offsetTable);
        rawTiling->Append(offsetTable.data(), static_cast<uint32_t>(offsetTable.size()));
        return 1;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 1. 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        // 2. 获取算子输入信息
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        int dtype_size = ge::GetSizeByDataType(dtype_input_x);  // 字长: 2/4/4
        uint32_t inputTotalElements = tensor_input_x->GetShapeSize();

        // 3. 获取属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const gert::TypedContinuousVector<int64_t> *attr_size = attrs->GetListInt(0);
        const gert::TypedContinuousVector<int64_t> *attr_stride = attrs->GetListInt(1);
        const int64_t *attr_storage_offset = attrs->GetInt(2);
        uint32_t ndim = static_cast<uint32_t>(attr_size->GetSize());
        int32_t storageOffset = static_cast<int32_t>(*attr_storage_offset);

        // 4. 校验属性并计算输出总元素数和 dimStride
        uint32_t sizeArr[4] = {1, 1, 1, 1};
        int32_t strideArr[4] = {0, 0, 0, 0};
        uint32_t dimStrideArr[4] = {1, 1, 1, 1};
        uint32_t totalOutputElements = 0;
        if (ParseShapeAttrs(attr_size->GetData(), attr_stride->GetData(), ndim,
                            static_cast<uint32_t>(attr_stride->GetSize()),
                            sizeArr, strideArr, dimStrideArr, totalOutputElements) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        // 5. 路径判断和 tileSize 计算
        uint32_t tileSize = 0;
        uint32_t pathFlag = CalcTileSize(inputTotalElements, static_cast<uint32_t>(dtype_size), tileSize);

        // 6. 多核切分
        uint32_t blockDim = std::min(static_cast<uint32_t>(num_cores_aiv), totalOutputElements);
        if (blockDim == 0) blockDim = 1;

        // 7. 填充 TilingData
        AsStridedTilingData *tiling = context->GetTilingData<AsStridedTilingData>();
        tiling->totalOutputElements = totalOutputElements;
        tiling->tileSize = tileSize;
        tiling->inputTotalElements = inputTotalElements;
        tiling->ndim = ndim;
        for (uint32_t d = 0; d < 4; d++) {
            tiling->size[d] = sizeArr[d];
            tiling->stride[d] = strideArr[d];
            tiling->dimStride[d] = dimStrideArr[d];
        }
        tiling->storageOffset = storageOffset;
        tiling->pathFlag = pathFlag;
        tiling->blockDim = blockDim;

        // 8. 容量校验通过后预计算偏移表并追加到 tiling data
        tiling->offsetTableInTiling = AppendOffsetTable(context, *tiling, static_cast<uint32_t>(dtype_size));

        // 9. 配置 tiling key
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // 10. 配置启动核数
        context->SetBlockDim(blockDim);

        // 11. 配置 workspace 大小（偏移表已存入 tiling data，无需 workspace）
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const gert::TypedContinuousVector<int64_t> *attr_size = attrs->GetListInt(0);

        gert::Shape *output_shape = context->GetOutputShape(0);
        const int64_t *sizeData = attr_size->GetData();
        for (size_t i = 0; i < attr_size->GetSize(); i++) {
            output_shape->AppendDim(sizeData[i]);
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType input_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_dtype);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class AsStrided : public OpDef {
    public:
        explicit AsStrided(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("input_size").AttrType(REQUIRED).ListInt();
            this->Attr("input_stride").AttrType(REQUIRED).ListInt();
            this->Attr("input_storage_offset").AttrType(REQUIRED).Int();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(AsStrided);
}  // namespace ops
