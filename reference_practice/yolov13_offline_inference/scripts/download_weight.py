# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import argparse
import json
import logging
import shutil
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s : %(message)s"
)

LFS_REPO = "hf_mirrors/atalaydenknalbant/Yolov13"
LFS_OID = "6653035017b0f111f80ec11ed914874ea85699b104aeac1e46e517d16889d6b7"
LFS_SIZE = 10570688


def download(output):
    if output.exists() and output.stat().st_size == LFS_SIZE:
        logging.info("%s already exists, skip download.", output)
        return

    request = urllib.request.Request(
        f"https://gitcode.com/{LFS_REPO}.git/info/lfs/objects/batch",
        data=json.dumps(
            {
                "operation": "download",
                "transfers": ["basic"],
                "objects": [{"oid": LFS_OID, "size": LFS_SIZE}],
            }
        ).encode(),
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        href = json.load(response)["objects"][0]["actions"]["download"]["href"]

    with urllib.request.urlopen(href) as response, output.open("wb") as f:
        shutil.copyfileobj(response, f)

    if output.stat().st_size != LFS_SIZE:
        raise RuntimeError(
            f"unexpected file size {output.stat().st_size}, expected {LFS_SIZE}"
        )
    logging.info("saved %s (%s bytes).", output, LFS_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download yolov13n.pt from the GitCode model repository (Git LFS)"
    )
    parser.add_argument(
        "--output", default="yolov13n.pt", help="output path of the weight file"
    )
    args = parser.parse_args()
    download(Path(args.output))
