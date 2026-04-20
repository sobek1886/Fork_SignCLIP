#!/usr/bin/env python3
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
extract_asl_citizen_poses.py – Parallel MediaPipe pose extraction for ASL-Citizen.

Reads video IDs from the ASL-Citizen split CSVs (train/val/test), extracts
MediaPipe Holistic poses for each video, and saves them as .pose files using
the pose_format library.  Output files are named:
    {dataset_name}_{video_stem}.pose   (e.g. asl_citizen_abc123.pose)

This matches the naming convention used by SignCLIPVideoCSVMetaProcessor so
the resulting .pose files can be used directly with RWTHFSPoseProcessor.

Usage:
    python extract_asl_citizen_poses.py \
        --video_dir  /scratch-shared/psobecki/ASL_Citizen/videos \
        --splits_dir /home/psobecki/ASL_Citizen/splits \
        --output_dir /home/psobecki/ASL_Citizen/mediapipe_poses \
        [--workers 8]  [--dataset_name asl_citizen]  [--overwrite]
"""

import argparse
import contextlib
import csv
import multiprocessing as mp
import os
from pathlib import Path

import cv2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_c_stderr():
    """Silence MediaPipe ABSL/C++ warnings by redirecting fd 2 at the OS level."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def _extract_one(task: tuple) -> tuple:
    """
    Worker function executed in a child process.
    Returns (feat_id, status) where status is one of:
      'ok'      – pose extracted and saved
      'skip'    – .pose file already exists (and --overwrite not set)
      'missing' – video file not found on disk
      'empty'   – video decoded to zero frames
      'error'   – any other exception
    """
    feat_id, video_path, output_dir, overwrite = task

    cache_path = Path(output_dir) / f"{feat_id}.pose"
    if cache_path.exists() and not overwrite:
        return feat_id, "skip"

    video_path = Path(video_path)
    if not video_path.exists():
        return feat_id, "missing"

    try:
        from pose_format.utils.holistic import load_holistic

        cap    = cv2.VideoCapture(str(video_path))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            return feat_id, "empty"

        with _suppress_c_stderr():
            pose = load_holistic(
                frames, fps=fps, width=width, height=height, progress=False
            )

        with open(cache_path, "wb") as f:
            pose.write(f)

        return feat_id, "ok"

    except Exception as exc:
        return feat_id, f"error: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _collect_tasks(splits_dir, video_dir, output_dir, dataset_name, overwrite):
    """Read all split CSVs and build task list of (feat_id, video_path, ...)."""
    tasks = []
    seen = set()
    for split_file in ("train.csv", "val.csv", "test.csv"):
        path = os.path.join(splits_dir, split_file)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                video_file = row["Video file"]
                video_stem = os.path.splitext(video_file)[0]
                feat_id    = f"{dataset_name}_{video_stem}"
                if feat_id in seen:
                    continue
                seen.add(feat_id)
                # Try common video extensions
                video_path = None
                for ext in (video_file, video_stem + ".mp4", video_stem + ".webm"):
                    candidate = os.path.join(video_dir, ext)
                    if os.path.exists(candidate):
                        video_path = candidate
                        break
                if video_path is None:
                    video_path = os.path.join(video_dir, video_file)  # will show as missing
                tasks.append((feat_id, video_path, output_dir, overwrite))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video_dir",    required=True,
                        help="Directory containing ASL-Citizen video files")
    parser.add_argument("--splits_dir",   required=True,
                        help="Directory containing train.csv / val.csv / test.csv")
    parser.add_argument("--output_dir",   required=True,
                        help="Directory to write .pose files into")
    parser.add_argument("--workers",      type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--dataset_name", default="asl_citizen",
                        help="Dataset name prefix for output files (default: asl_citizen)")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-extract even if a .pose file already exists")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(
        args.splits_dir, args.video_dir, args.output_dir,
        args.dataset_name, args.overwrite,
    )

    already_done = sum(
        1 for feat_id, _, out_dir, _ in tasks
        if (Path(out_dir) / f"{feat_id}.pose").exists()
    )
    print(f"Total videos     : {len(tasks)}")
    print(f"Already extracted: {already_done}"
          + ("  (will overwrite)" if args.overwrite else "  (skipping)"))
    print(f"Workers          : {args.workers}")
    print(f"Output dir       : {args.output_dir}\n")

    counts = {"ok": 0, "skip": 0, "missing": 0, "empty": 0, "error": 0}

    with mp.Pool(processes=args.workers, maxtasksperchild=100) as pool:
        for _, status in tqdm(
            pool.imap_unordered(_extract_one, tasks, chunksize=1),
            total=len(tasks),
            desc="Extracting",
            unit="vid",
        ):
            key = status if status in counts else "error"
            counts[key] += 1

    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Extracted (ok)   : {counts['ok']}")
    print(f"  Skipped (cached) : {counts['skip']}")
    print(f"  Missing video    : {counts['missing']}")
    print(f"  Empty video      : {counts['empty']}")
    print(f"  Errors           : {counts['error']}")
    print(f"  Total            : {sum(counts.values())}")
    print("=" * 50)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
