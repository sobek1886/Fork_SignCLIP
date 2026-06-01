#!/usr/bin/env python3
"""Convert palmer + digits PNG sequences to MP4 and print rsync commands for 5 matched signs.

Palmer unreal indices in manifest: [palmer_cam2, palmer_cam3, palmer_cam4, digits_cam2, digits_cam3, digits_cam4]
We pick cam3 for both characters (indices 1 and 4).

PNG dirs live at PALMER_RENDER/{character}_{key}_{session}_{recnum}_{cam}/
Converted MP4s are written to TMP_DIR (original PNG dirs untouched).
"""

import json
import sys
from pathlib import Path

import cv2

MANIFEST      = "/home/psobecki/ngt_pair_manifest.json"
BUSHUIS_DIR   = "/scratch-shared/psobecki/BushuisMP4"
PIOTR_MKV_DIR = "/scratch-shared/psobecki/piotrAnims/mkv"
PALMER_RENDER = "/scratch-shared/psobecki/Custom_from_FBX/Saved/Renders"
TMP_DIR       = "/scratch-shared/psobecki/ngt_examples"
LOCAL_DIR     = "~/Desktop/ngt_examples"
N             = 5

Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

with open(MANIFEST) as f:
    manifest = json.load(f)

bushuis_ids = {p.stem for p in Path(BUSHUIS_DIR).glob("*.MP4")}
eval_ids = sorted(set(manifest) & bushuis_ids)[:N]

if not eval_ids:
    sys.exit("No matched IDs found.")


def png_dir_to_mp4(render_dir: Path, out_path: str, fps: float = 30.0):
    """Convert a PNG sequence directory to an MP4. Returns True on success."""
    png_files = sorted(render_dir.glob("*.png"))
    if not png_files:
        print(f"  WARNING: no PNGs in {render_dir}")
        return False
    first = cv2.imread(str(png_files[0]))
    if first is None:
        print(f"  WARNING: could not read {png_files[0]}")
        return False
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in png_files:
        frame = cv2.imread(str(p))
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f"  wrote {out_path}")
    return True


print(f"mkdir -p {LOCAL_DIR}")
print()

for sid in eval_ids:
    paths = manifest[sid]

    # ── piotrAnims: LEFT, MIDDLE, RIGHT ────────────────────────────────────
    piotr_mkv_paths = {}
    for view in ("LEFT", "MIDDLE", "RIGHT"):
        npy = next((p for p in paths["real"] if f"_{view}_" in p), None)
        if npy:
            mkv_name = Path(npy).name.replace("piotr_anims_", "").replace(".npy", ".mkv")
            piotr_mkv_paths[view] = f"{PIOTR_MKV_DIR}/{mkv_name}"

    # ── Bushuis MP4 ─────────────────────────────────────────────────────────
    bushuis_path = f"{BUSHUIS_DIR}/{sid}.MP4"

    # ── Palmer cam2/3/4 (unreal indices 0,1,2) ─────────────────────────────
    palmer_mp4s = {}
    for i, cam in enumerate(("cam2", "cam3", "cam4")):
        npy = paths["unreal"][i]
        render_dir = Path(PALMER_RENDER) / Path(npy).stem.replace("ngt_aug_", "")
        out = f"{TMP_DIR}/{sid}_palmer_{cam}.mp4"
        if not Path(out).exists():
            print(f"# Converting palmer {cam} {sid}")
            png_dir_to_mp4(render_dir, out)
        palmer_mp4s[cam] = out

    # ── Digits cam2/3/4 (unreal indices 3,4,5) ─────────────────────────────
    digits_mp4s = {}
    for i, cam in enumerate(("cam2", "cam3", "cam4")):
        npy = paths["unreal"][3 + i]
        render_dir = Path(PALMER_RENDER) / Path(npy).stem.replace("ngt_aug_", "")
        out = f"{TMP_DIR}/{sid}_digits_{cam}.mp4"
        if not Path(out).exists():
            print(f"# Converting digits {cam} {sid}")
            png_dir_to_mp4(render_dir, out)
        digits_mp4s[cam] = out

    # ── rsync commands ───────────────────────────────────────────────────────
    print(f"# Sign: {sid}")
    for view, p in piotr_mkv_paths.items():
        print(f"rsync -avz snellius:{p} {LOCAL_DIR}/{sid}_piotr_{view.lower()}.mkv")
    print(f"rsync -avz snellius:{bushuis_path} {LOCAL_DIR}/{sid}_bushuis.MP4")
    for cam, p in palmer_mp4s.items():
        print(f"rsync -avz snellius:{p} {LOCAL_DIR}/{sid}_palmer_{cam}.mp4")
    for cam, p in digits_mp4s.items():
        print(f"rsync -avz snellius:{p} {LOCAL_DIR}/{sid}_digits_{cam}.mp4")
    print()
