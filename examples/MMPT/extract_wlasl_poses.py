#!/usr/bin/env python3
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
extract_wlasl_poses.py  –  Parallel MediaPipe pose extraction for WLASL videos.

Reads all video IDs from WLASL_v0.3.json, skips videos that are already
cached, and extracts MediaPipe Holistic poses using a multiprocessing pool.
Each worker process loads its own MediaPipe model, so workers are independent.

After this script completes, run wlasl_mediapipe_signer_classifier.py and
all poses will be served from the cache with no extraction overhead.

Usage
  python extract_wlasl_poses.py
  python extract_wlasl_poses.py --workers 8
  python extract_wlasl_poses.py --workers 4 --overwrite
"""

import argparse
import contextlib
import json
import multiprocessing as mp
from pathlib import Path

import cv2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WLASL_JSON = Path("/Users/piotr/Projects/Thesis/Data/WLASL/WLASL_v0.3.json")
VIDEO_DIR  = Path("/Users/piotr/Projects/Thesis/Data/WLASL/videos")
POSE_CACHE = Path("/Users/piotr/Projects/Thesis/Data/WLASL/pose_cache")


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


def _extract_one(task: tuple) -> tuple[int, str]:
    """
    Worker function executed in a child process.

    Returns (video_id, status) where status is one of:
      'ok'      – pose extracted and saved
      'skip'    – .pose file already exists (and --overwrite not set)
      'missing' – MP4 not found on disk
      'empty'   – video decoded to zero frames
      'error'   – any other exception
    """
    video_id, cache_dir, overwrite = task

    cache_path = Path(cache_dir) / f"{video_id:05d}.pose"
    if cache_path.exists() and not overwrite:
        return video_id, "skip"

    video_path = VIDEO_DIR / f"{video_id:05d}.mp4"
    if not video_path.exists():
        return video_id, "missing"

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
            return video_id, "empty"

        with _suppress_c_stderr():
            pose = load_holistic(
                frames, fps=fps, width=width, height=height, progress=False
            )

        with open(cache_path, "wb") as f:
            pose.write(f)

        return video_id, "ok"

    except Exception as exc:
        return video_id, f"error: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=POSE_CACHE,
        help="Directory to write .pose files into (default: %(default)s)",
    )
    parser.add_argument(
        "--min_signers", type=int, default=3,
        help="Only extract videos from glosses with at least this many signers "
             "(default: 3, must match the value used in the classifier)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract even if a .pose file already exists",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect video IDs from qualifying glosses only
    # ------------------------------------------------------------------
    print(f"Loading WLASL metadata from {WLASL_JSON} ...")
    with open(WLASL_JSON) as f:
        data = json.load(f)

    # Group video IDs by gloss, then filter by signer count
    from collections import defaultdict
    by_gloss: dict[str, list[int]] = defaultdict(list)
    for entry in data:
        for inst in entry["instances"]:
            if "video_id" in inst:
                by_gloss[entry["gloss"]].append(int(inst["video_id"]))

    video_ids = sorted({
        vid
        for instances in by_gloss.values()
        if len(instances) >= args.min_signers
        for vid in instances
    })

    already_cached = sum(
        1 for vid in video_ids
        if (args.cache_dir / f"{vid:05d}.pose").exists()
    )

    qualifying_glosses = sum(1 for v in by_gloss.values() if len(v) >= args.min_signers)
    print(f"Glosses with ≥{args.min_signers} signers : {qualifying_glosses} / {len(by_gloss)}")
    print(f"Videos to extract  : {len(video_ids)}")
    print(f"Already cached     : {already_cached}"
          + ("  (will overwrite)" if args.overwrite else "  (skipping)"))
    print(f"Workers            : {args.workers}")
    print(f"Output dir         : {args.cache_dir}\n")

    tasks = [(vid, args.cache_dir, args.overwrite) for vid in video_ids]

    # ------------------------------------------------------------------
    # Parallel extraction
    # maxtasksperchild restarts workers periodically to prevent MediaPipe
    # memory from accumulating over thousands of videos.
    # ------------------------------------------------------------------
    counts: dict[str, int] = {"ok": 0, "skip": 0, "missing": 0,
                               "empty": 0, "error": 0}

    with mp.Pool(processes=args.workers, maxtasksperchild=100) as pool:
        for _, status in tqdm(
            pool.imap_unordered(_extract_one, tasks, chunksize=1),
            total=len(tasks),
            desc="Extracting",
            unit="vid",
        ):
            key = status if status in counts else "error"
            counts[key] += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Extracted (ok)   : {counts['ok']}")
    print(f"  Skipped (cached) : {counts['skip']}")
    print(f"  Missing MP4      : {counts['missing']}")
    print(f"  Empty video      : {counts['empty']}")
    print(f"  Errors           : {counts['error']}")
    print(f"  Total            : {sum(counts.values())}")
    print("=" * 50)


if __name__ == "__main__":
    # 'spawn' is the default on macOS (Python 3.8+) and is required to avoid
    # forking MediaPipe's internal thread pool into child processes.
    mp.set_start_method("spawn", force=True)
    main()
