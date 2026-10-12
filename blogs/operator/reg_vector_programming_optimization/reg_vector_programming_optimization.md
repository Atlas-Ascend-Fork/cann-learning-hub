# Reg矢量计算编程优化实践

## 背景介绍

Ascend C基础API提供了简易的矢量算子实现接口，开发者通过直接输入输出LocalMemory的数据便能完成算子的调用，轻松地实现算子开发。但对于性能需求更高的场景，基础API以LocalTensor为输入和输出，数据搬运与矢量计算过程的控制粒度有限，难以充分发挥RegBase架构的硬件能力。

为此，Ascend 950支持了Reg矢量计算编程，由Ascend C提供了相应的接口，允许开发者直接操作芯片中的Vector计算寄存器，并自主组织Unified Buffer（UB）与寄存器之间的数据加载、计算和存储过程。计算数据可在寄存器中连续完成多步处理，减少不必要的中间结果落盘及重复搬运，从而提升数据复用效率和算子执行性能。

本文将介绍Reg矢量编程如何通过对寄存器分组利用以提高寄存器利用率的优化思路。

## 问题引入

我们以矢量融合算子为例，假设存在一个矢量融合算子计算公式如下：

```
y = (x1 + x2) * x3 + x4 * (1 - x3)
```

数据类型特点如下：

- 输入 x1、x2、x3、x4 为BF16

- 输出 y 为BF16

- 中间计算使用 FP32，以保证计算精度

那由于中间计算和输入输出数据类型不一致，因此在计算中需要频繁执行BF16 -> FP32 和 FP32->BF16的转换。

优化前实现的主要特点：

- 每轮处理 64 个 BF16元素

- BF16->FP32 使用 CAST_BF16_TO_FP32_ZERO，只形成一组FP32寄存器计算链路

- 主路径使用 DIST_UNPACK_B16 加载BF16，使用 DIST_PACK_B32 写回BF16

- 计算链路为单组

该实现逻辑简单清晰，但没有充分利用BF16低位宽数据在Regbase中可映射到 ZERO/ONE 两个布局位置的特性。对Ascend950PR 的Regbase API 而言，128个 BF16 元素可以拆成 64 个 FP32 lane 并行处理，因此原实现存在向量粒度和寄存器利用不足的问题。

## 优化思路

本次优化核心思路是利用 RegLayout::ZERO 和 RegLayout::ONE 将 BF16 数据拆成两组 FP32 寄存器链路，并在同一轮循环中完成两组数据的计算。

优化后主路径处理流程：

- 每轮加载 128 个 BF16 元素

- 对同一组 BF16 寄存器执行两次 cast

  - CAST_BF16_TO_FP32_ZERO 生成 zero 侧 FP32 寄存器

  - CAST_BF16_TO_FP32_ONE 生成 one 侧 FP32 寄存器

- 对 zero/one 两组 FP32 寄存器分别执行同构算术计算

- 写回 BF16 时先cast one侧结果，再cast zero侧结果

- 最后通过 DIST_NORM 将合并后的 BF16 寄存器写回 UB

该策略本质是将原来的单链路：

```
64 BF16 -> 64 FP32 -> compute -> 64 BF16
```

扩展为双链路：

```
128 BF16 -> 64 FP32(ZERO) + 64 FP32(ONE) -> compute twice -> 128 BF16
```

## 关键设计点

### 新增 CastTrait

新增四组 AscendC::MicroAPI::CastTrait，显式描述 BF16/FP32 转换时的寄存器布局，饱和模式、mask merge 模式和舍入模式。

```cpp
constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_F32_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN
};

constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_F32_ONE = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN
};

constexpr AscendC::MicroAPI::CastTrait CAST_F32_TO_BF16_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_RINT
};

constexpr AscendC::MicroAPI::CastTrait CAST_F32_TO_BF16_ONE = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};
```

### 主循环向量长度从64扩展为128

优化前：

```cpp
constexpr uint32_t VL = 64;
uint32_t loopNum = count / VL;
uint32_t tailCount = count % VL;
MaskReg maskFull = UpdateMask<float>(VL);
```

优化后：

```cpp
constexpr uint32_t VL = 128;
uint32_t loopNum = count / VL;
uint32_t tailCount = count % VL;
uint32_t vl = VL / 2;
MaskReg maskBF16 = CreateMask<bfloat16_t>();
MaskReg maskFP32 = UpdateMask<float>(vl);
```

这里 VL=128 表示每轮处理 128 个BF16元素；vl=64 表示拆分后，每组FP32寄存器处理64个元素。

### BF16到FP32 拆分成 zero/one 两路

以上述算子为例，优化后每个输入都生成两组 FP32 寄存器。并且LoadDist的配置，需要从DIST_UNPACK_B16，修改为DIST_NORM。

```cpp
LoadAlign<bfloat16_t, LoadDist::DIST_NORM>(vb16, x1Ub + base);
Cast<float, bfloat16_t, CAST_BF16_TO_F32_ZERO>(vx1, vb16, maskBF16);
Cast<float, bfloat16_t, CAST_BF16_TO_F32_ONE>(vx1One, vb16, maskBF16);

LoadAlign<bfloat16_t, LoadDist::DIST_NORM>(vb16, x2Ub + base);
Cast<float, bfloat16_t, CAST_BF16_TO_F32_ZERO>(vx2, vb16, maskBF16);
Cast<float, bfloat16_t, CAST_BF16_TO_F32_ONE>(vx2One, vb16, maskBF16);
```

x3/x4 同样按 zero/one 拆分。这样后续计算可以在两套 FP32 寄存器中展开。

### FP32计算链路复制为两组

优化前只计算一组：

```cpp
Add(vresult, vx1, vx2, maskFull);
Mul(vresult, vresult, vx3, maskFull);
... ...
```

优化后 zero/one 两组并行展开：

```cpp
Add(vresult, vx1, vx2, maskFP32);
Add(vresultOne, vx1One, vx2One, maskFP32);
Mul(vresult, vresult, vx3, maskFP32);
Mul(vresultOne, vresultOne, vx3One, maskFP32);
... ...
```

### FP32到BF16 按one到zero的顺序合并

写回时，需要先处理ONE，再处理ZERO

```cpp
Cast<bfloat16_t, float, CAST_F32_TO_BF16_ONE>(vb16, vresultOne, maskFP32);
Cast<bfloat16_t, float, CAST_F32_TO_BF16_ZERO>(vb16, vresult, maskFP32);
StoreAlign<bfloat16_t, StoreDist::DIST_NORM>(yUb + base, vb16, maskBF16);
```

这里顺序不能随意调整，原因是 CAST_F32_TO_BF16_ZERO 使用 MaskMergeMode::MERGING, 用于最终BF16寄存器中合并zero侧结果；而 CAST_F32_TO_BF16_ONE 被限制，只能使用 MaskMergeMode::ZEROING，用于先写入one侧结果，其他未写入的位置默认置零，留给zero侧结果覆盖写入。这里的StoreAlign也需要将StoreDist，由DIST_PACK_B32，修改为DIST_NORM。

## 优化前后对比

| 维度 | 优化前 | 优化后 |
| --- | --- | --- |
| 主循环VL | 64 | 128 |
| 每轮BF16元素数 | 64 | 128 |
| FP32计算链路 | 单链路 | zero/one 双链路 |
| BF16 -> FP32 cast | layoutZMrgZ | CAST_BF16_TO_F32_ZERO + CAST_BF16_TO_F32_ONE |
| FP32 -> BF16 cast | layoutZMrgZRndRSatNS | CAST_F32_TO_BF16_ONE + CAST_F32_TO_BF16_ZERO |
| 主路径 load/store | DIST_UNPACK_B16 / DIST_PACK_B32 | DIST_NORM / DIST_NORM |
| 主要收益点 | 实现简单 | 更高向量粒度和寄存器利用率 |

## 总结

该优化模式适用于以下类型的Regbase算子：

- 输入/输出为 BF16

- 中间计算为 FP32

- 主路径存在频繁的 bf16 -> fp32 和 fp32 -> bf16 转换

- 单轮向量粒度较小，未充分利用低位宽数据布局

- 大shape主循环耗时占比高

复用步骤如下：

1. 确认原始主循环的 BF16 处理粒度和 FP32 寄存器使用方式

2. 定义 zero/one 两组 CastTrait

3. 将 BF16 主路径 load/store 切换至 DIST_NORM

4. 将每个 BF16 输入cast成 zero/one 两组 FP32 寄存器

5. 将 FP32 计算链路复制为 zero/one 两套

6. 写回 BF16 时，先cast one，再cast zero

7. 对齐路径和尾块路径，分别验证精度

8. 使用 msprof 对大shape做before/after 对比
