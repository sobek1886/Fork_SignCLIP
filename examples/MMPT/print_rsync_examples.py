#!/usr/bin/env python3
"""Print rsync commands to copy 5 matched sign examples to local."""

import json
import re
import sys
from pathlib import Path

MANIFEST   = "/home/psobecki/ngt_pair_manifest.json"
BUSHUIS_DIR = "/scratch-shared/psobecki/BushuisMP4"
PIOTR_MKV_DIR = "/scratch-shared/psobecki/piotrAnims/mkv"
LOCAL_DIR  = "~/Desktop/ngt_examples"
N = 5

with open(MANIFEST) as f:
    manifest = json.load(f)

_BUSHUIS_RE = re.compile(r'^bushuis_(M\d{8}_\d+)\.npy$')

# Find sign IDs that have a Bushuis MP4
bushuis_ids = set()
for p in Path(BUSHUIS_DIR).glob("*.MP4"):
    # key is the stem e.g. M20241106_6037
    bushuis_ids.add(p.stem)

eval_ids = sorted(set(manifest) & bushuis_ids)[:N]

if not eval_ids:
    sys.exit("No matched IDs found.")

print(f"mkdir -p {LOCAL_DIR}")
print()

for sid in eval_ids:
    paths = manifest[sid]

    # piotrAnims MIDDLE: .npy path → derive MKV filename
    # e.g. /path/piotr_anims_M_..._MIDDLE_....npy → MKV_DIR/M_..._MIDDLE_....mkv
    middle_npy = next((p for p in paths["real"] if "_MIDDLE_" in p), paths["real"][1])
    mkv_name = Path(middle_npy).name.replace("piotr_anims_", "").replace(".npy", ".mkv")
    mkv_path = f"{PIOTR_MKV_DIR}/{mkv_name}"

    # Bushuis MP4
    bushuis_path = f"{BUSHUIS_DIR}/{sid}.MP4"

    # palmer: first unreal path → .npy → derive PNG dir
    # unreal paths look like: .../ngt_aug_palmer_M..._cam3.npy
    palmer_npy = paths["unreal"][1]  # cam3
    palmer_name = Path(palmer_npy).stem  # ngt_aug_palmer_M..._cam3
    # PNG sequences are typically in a dir named after the sign+cam
    # Print the .npy path so user knows where to look
    print(f"# Sign: {sid}")
    print(f"rsync -avz snellius:{mkv_path} {LOCAL_DIR}/{sid}_piotr_middle.mkv")
    print(f"rsync -avz snellius:{bushuis_path} {LOCAL_DIR}/{sid}_bushuis.MP4")
    print(f"# palmer npy: {palmer_npy}")
    print()
