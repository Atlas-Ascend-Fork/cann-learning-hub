# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# gelu_approx 算子自注册

# Kernel源文件(bisheng编译)
set(XLOG1PY_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gelu_approx_kernel.cpp
)

# Plugin源文件(g++编译)
set(XLOG1PY_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/gelu_approx_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${XLOG1PY_KERNEL_SRCS}"
    op_kernel
    "${XLOG1PY_PLUGIN_SRCS}"
    op_kernel
)
