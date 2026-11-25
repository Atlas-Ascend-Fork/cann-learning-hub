#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""YOLOv13 pyACL 推理脚本：官方案例 yolov13_offline_inference 第 3 章 ACL 离线推理
C++ 程序（yolov13_main）的免编译备选。

在缺少 C++ 编译环境（如未安装 g++/gcc-c++）或希望快速验证的场景下，可用本脚本
完成与 yolov13_main 相同的推理流程：加载 OM 模型 → 执行推理 → 解析输出并生成
与 C++ 程序格式一致的 `<输入名>_result.txt`，可直接用官方案例的
scripts/draw_boxes.py 可视化。

依赖：CANN Toolkit（含 pyACL）、numpy。OM 模型与预处理 BIN 输入可按官方案例
（仓库根 reference_practice/yolov13_offline_inference/）第 2 章生成后复用。执行：

    python3 scripts/infer_pyacl.py --model=model/yolov13.om --input=data/bus.bin

参数与官方案例 scripts/run.sh 保持一致（--model/--input），置信度阈值与 NMS IoU
阈值与 sample_yolov13.cpp 的 kConfThresh(0.25)/kIouThresh(0.45) 默认值相同。
"""

import argparse
import logging
import sys

import numpy as np
import acl

ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

NUM_ANCHORS = 8400
NUM_CLASSES = 80
INPUT_W = INPUT_H = 640

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

VEHICLE_CLASSES = {2, 3, 5, 7}  # 与 sample_yolov13.cpp OutputResult 一致（car/motorcycle/bus/truck）

LOG = logging.getLogger("infer_pyacl")
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s", stream=sys.stdout)


def norm(ret):
    """兼容不同版本 pyACL：部分接口返回 (ptr, ret) 元组。"""
    return ret[-1] if isinstance(ret, tuple) else ret


def check(ret, message):
    """pyACL 返回值校验：非 0 时抛出带返回码的异常（禁用 assert，G.TES.01）。"""
    if ret != 0:
        raise RuntimeError(f"{message}: ret={ret}")


def nms(dets, iou_thresh):
    """与 sample_yolov13.cpp 的 Nms 一致：仅同类框间做 IoU 抑制，IoU 分母含 1e-6 平滑项。"""
    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            if k[0] != d[0]:
                continue
            ix1, iy1 = max(d[2], k[2]), max(d[3], k[3])
            ix2, iy2 = min(d[4], k[4]), min(d[5], k[5])
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            union = (d[4] - d[2]) * (d[5] - d[3]) + (k[4] - k[2]) * (k[5] - k[3]) - inter
            if inter / (union + 1e-6) > iou_thresh:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def parse_yolo_output(out, conf_thresh):
    """解析 (84, 8400) 输出：每列为一个 anchor，前 4 行框参数，后 80 行类别分数。"""
    box = out.reshape(4 + NUM_CLASSES, NUM_ANCHORS)
    cx, cy, bw, bh = box[0], box[1], box[2], box[3]
    scores = box[4:]
    cls_ids = np.argmax(scores, axis=0)
    confs = scores[cls_ids, np.arange(NUM_ANCHORS)]
    dets = []
    for i in np.where(confs >= conf_thresh)[0]:  # 与 C++ 一致：max_conf < conf_thresh 才跳过
        x1 = max(0.0, float(cx[i] - bw[i] * 0.5))
        y1 = max(0.0, float(cy[i] - bh[i] * 0.5))
        x2 = min(float(cx[i] + bw[i] * 0.5), float(INPUT_W))
        y2 = min(float(cy[i] + bh[i] * 0.5), float(INPUT_H))
        dets.append((int(cls_ids[i]), float(confs[i]), x1, y1, x2, y2))
    return dets


def parse_args():
    """解析命令行参数，默认值与官方案例 run.sh / sample_yolov13.cpp 对齐。"""
    parser = argparse.ArgumentParser(
        description="YOLOv13 pyACL inference (no-build alternative to yolov13_main)")
    parser.add_argument("--model", default="model/yolov13.om", help="OM 模型路径（默认 model/yolov13.om）")
    parser.add_argument("--input", default="data/bus.bin", help="预处理 BIN 输入路径（默认 data/bus.bin）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25，与 C++ 样例一致）")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值（默认 0.45，与 C++ 样例一致）")
    return parser.parse_args()


def init_acl():
    """初始化 ACL 运行时资源，返回 (context, stream)。"""
    check(acl.init(), "acl.init failed")
    check(acl.rt.set_device(0), "set_device 0 failed")
    context, ret = acl.rt.create_context(0)
    check(ret, "create_context failed")
    stream, ret = acl.rt.create_stream()
    check(ret, "create_stream failed")
    return context, stream


def load_model(model_path):
    """加载 OM 模型并创建 desc，返回 (model_id, desc)。"""
    model_id, ret = acl.mdl.load_from_file(model_path)
    check(ret, f"load model {model_path} failed")
    desc = acl.mdl.create_desc()
    check(acl.mdl.get_desc(desc, model_id), "get model desc failed")
    LOG.info("load model %s success.", model_path)
    return model_id, desc


def build_dataset_from_buffer(host_in, size, is_input):
    """malloc Device 内存并包装为 dataset；输入侧完成 H2D 拷贝，返回 (dataset, buf)。"""
    ds = acl.mdl.create_dataset()
    buf, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
    if ret != 0:
        raise RuntimeError(f"malloc {'input' if is_input else 'output'} failed: ret={ret}")
    if is_input:
        ptr = acl.util.bytes_to_ptr(host_in)
        check(norm(acl.rt.memcpy(buf, size, ptr, size, ACL_MEMCPY_HOST_TO_DEVICE)), "memcpy H2D failed")
    check(norm(acl.mdl.add_dataset_buffer(ds, acl.create_data_buffer(buf, size))), "add_dataset_buffer failed")
    return ds, buf


def fetch_output(obuf, out_size):
    """D2H 回读输出并校验张量尺寸（与 C++ PostProcessOutput 的 expected_len 校验一致）。"""
    host_out, ret = acl.rt.malloc_host(out_size)
    check(ret, "malloc_host failed")
    check(norm(acl.rt.memcpy(host_out, out_size, obuf, out_size, ACL_MEMCPY_DEVICE_TO_HOST)), "memcpy D2H failed")
    out = np.frombuffer(acl.util.ptr_to_bytes(host_out, out_size), dtype=np.float32)
    expected = NUM_ANCHORS * (NUM_CLASSES + 4)
    if out.size != expected:
        raise RuntimeError(
            f"输出张量大小异常: {out.size} floats（期望 {expected}），"
            "请确认模型为官方案例第 2 章编译的 yolov13.om")
    return out


def report_and_save(keep, input_path):
    """打印检测明细与统计，并写 _result.txt（与 C++ SaveResultFile 格式一致）。"""
    LOG.info("detected %d objects:", len(keep))
    for idx, (cls_id, conf, x1, y1, x2, y2) in enumerate(keep):
        label = COCO_LABELS[cls_id] if 0 <= cls_id < NUM_CLASSES else "unknown"
        LOG.info("  [%d] %s (%.2f)  bbox: [%d,%d,%d,%d]",
                 idx, label, conf, int(x1), int(y1), int(x2), int(y2))
    persons = sum(1 for d in keep if d[0] == 0)
    vehicles = sum(1 for d in keep if d[0] in VEHICLE_CLASSES)
    LOG.info("recognized persons: %d, vehicles: %d", persons, vehicles)
    result_path = input_path.rsplit(".bin", 1)[0] + "_result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        for cls_id, conf, x1, y1, x2, y2 in keep:
            label = COCO_LABELS[cls_id] if 0 <= cls_id < NUM_CLASSES else "unknown"
            f.write(f"{label} {conf:.6f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")
    LOG.info("result saved to %s", result_path)


def infer_once(desc, model_id, input_path, conf, iou):
    """单个输入的全流程：读输入 → 构造 dataset → 推理 → 解析 → 报告与落盘。"""
    in_ds = out_ds = dbuf = obuf = None
    try:
        with open(input_path, "rb") as f:
            host_in = f.read()
        in_size = acl.mdl.get_input_size_by_index(desc, 0)
        if len(host_in) != in_size:
            raise RuntimeError(
                f"输入大小与模型不符: 文件 {len(host_in)}B vs 模型期望 {in_size}B，"
                "请先用官方案例 scripts/preprocess.py 生成输入 BIN")
        in_ds, dbuf = build_dataset_from_buffer(host_in, in_size, is_input=True)
        out_size = acl.mdl.get_output_size_by_index(desc, 0)
        out_ds, obuf = build_dataset_from_buffer(None, out_size, is_input=False)
        check(norm(acl.mdl.execute(model_id, in_ds, out_ds)), "aclmdlExecute failed")
        out = fetch_output(obuf, out_size)
        keep = nms(parse_yolo_output(out, conf), iou)
        report_and_save(keep, input_path)
    finally:
        for ds in (in_ds, out_ds):
            if ds is not None:
                acl.mdl.destroy_dataset(ds)
        for buf in (dbuf, obuf):
            if buf is not None:
                acl.rt.free(buf)


def main():
    """入口：解析参数 → 初始化 ACL → 加载模型并推理 → 统一释放资源。"""
    args = parse_args()
    context, stream = init_acl()
    model_id, desc = None, None
    try:
        model_id, desc = load_model(args.model)
        infer_once(desc, model_id, args.input, args.conf, args.iou)
    finally:
        if model_id is not None:
            acl.mdl.unload(model_id)
        if desc is not None:
            acl.mdl.destroy_desc(desc)
        acl.rt.destroy_context(context)
        acl.rt.destroy_stream(stream)
        acl.rt.reset_device(0)
        acl.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
