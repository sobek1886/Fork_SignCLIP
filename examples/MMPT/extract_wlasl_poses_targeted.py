#!/usr/bin/env python3
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
extract_wlasl_poses_targeted.py  –  Targeted pose extraction for near-complete glosses.

Scans the pose cache against WLASL metadata and finds glosses that are nearly
complete (i.e. only a few signers' videos are missing).  Extracts only the
missing videos, making those glosses eligible for --complete_only evaluation
in wlasl_mediapipe_signer_classifier.py.

Workflow
  1. Report a coverage table: for each gloss, how many signers are cached
     vs. total in the metadata.
  2. Extract only the missing videos for glosses satisfying:
       total signers >= --min_signers  AND  missing videos <= --max_missing

Usage
  # Dry run – just show coverage, extract nothing
  python extract_wlasl_poses_targeted.py --dry_run

  # Extract missing videos for glosses missing at most 2
  python extract_wlasl_poses_targeted.py --max_missing 2

  # Stricter: only glosses with ≥5 signers, missing at most 1
  python extract_wlasl_poses_targeted.py --min_signers 5 --max_missing 1
"""

import argparse
import contextlib
import json
import multiprocessing as mp
from collections import defaultdict
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
# Helpers  (shared with extract_wlasl_poses.py)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_c_stderr():
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
    """Worker: extract and cache one video. Returns (video_id, status)."""
    video_id, cache_dir = task

    cache_path = Path(cache_dir) / f"{video_id:05d}.pose"
    if cache_path.exists():
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
        "--min_signers", type=int, default=3,
        help="Minimum signers a gloss must have in metadata (default: 3)",
    )
    parser.add_argument(
        "--max_missing", type=int, default=2,
        help="Maximum number of uncached videos allowed per gloss (default: 2)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel worker processes for extraction (default: 4)",
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=POSE_CACHE,
        help="Pose cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print coverage report only; do not extract anything",
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of near-complete glosses to show in the report (default: 30)",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load metadata and compute per-gloss cache coverage
    # ------------------------------------------------------------------
    print(f"Loading WLASL metadata from {WLASL_JSON} ...")
    with open(WLASL_JSON) as f:
        data = json.load(f)

    by_gloss: dict[str, list[int]] = defaultdict(list)
    for entry in data:
        for inst in entry["instances"]:
            if "video_id" in inst:
                by_gloss[entry["gloss"]].append(int(inst["video_id"]))

    # For each gloss compute: total, cached, missing, on-disk-but-not-cached
    coverage = []
    for gloss, vids in by_gloss.items():
        if len(vids) < args.min_signers:
            continue
        cached  = sum(1 for v in vids if (args.cache_dir / f"{v:05d}.pose").exists())
        on_disk = sum(1 for v in vids if (VIDEO_DIR / f"{v:05d}.mp4").exists())
        missing_cache  = len(vids) - cached
        missing_video  = len(vids) - on_disk
        coverage.append({
            "gloss":         gloss,
            "total":         len(vids),
            "cached":        cached,
            "on_disk":       on_disk,
            "missing_cache": missing_cache,   # in metadata, not yet in cache
            "missing_video": missing_video,   # in metadata, no MP4 on disk
        })

    # Sort by fewest missing cache files first, then by total signers descending
    coverage.sort(key=lambda r: (r["missing_cache"], -r["total"]))

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------
    complete     = [r for r in coverage if r["missing_cache"] == 0]
    near_complete = [r for r in coverage
                     if 0 < r["missing_cache"] <= args.max_missing
                     and r["missing_cache"] <= r["on_disk"] - r["cached"]]

    print(f"\nGlosses with ≥{args.min_signers} signers in metadata : {len(coverage)}")
    print(f"Already complete (all cached)                    : {len(complete)}")
    print(f"Near-complete (missing ≤{args.max_missing}, MP4s exist)    : {len(near_complete)}")
    print(f"Videos to extract for near-complete glosses      : "
          f"{sum(r['missing_cache'] for r in near_complete)}")

    print(f"\n{'Gloss':<25} {'Total':>6} {'Cached':>7} {'Missing':>8} {'On disk':>8}")
    print("-" * 58)
    shown = [r for r in coverage if r["missing_cache"] <= args.max_missing][:args.top]
    for r in shown:
        status = "COMPLETE" if r["missing_cache"] == 0 else f"-{r['missing_cache']}"
        print(f"{r['gloss']:<25} {r['total']:>6} {r['cached']:>7} "
              f"{r['missing_cache']:>8} {r['on_disk']:>8}  {status}")

    if args.dry_run:
        print("\n(dry run — no extraction performed)")
        return

    # ------------------------------------------------------------------
    # Extract missing videos for near-complete glosses
    # ------------------------------------------------------------------
    # Collect video IDs that are: missing from cache AND the MP4 exists
    to_extract: list[int] = []
    for r in near_complete:
        gloss_vids = by_gloss[r["gloss"]]
        for vid in gloss_vids:
            if (not (args.cache_dir / f"{vid:05d}.pose").exists()
                    and (VIDEO_DIR / f"{vid:05d}.mp4").exists()):
                to_extract.append(vid)

    to_extract = sorted(set(to_extract))

    if not to_extract:
        print("\nNothing to extract — all near-complete glosses are already fully cached.")
        return

    print(f"\nExtracting {len(to_extract)} missing videos across "
          f"{len(near_complete)} near-complete glosses ...")

    tasks  = [(vid, args.cache_dir) for vid in to_extract]
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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    newly_complete = sum(
        1 for r in near_complete
        if all((args.cache_dir / f"{v:05d}.pose").exists()
               for v in by_gloss[r["gloss"]])
    )

    print("\n" + "=" * 50)
    print("TARGETED EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Extracted (ok)      : {counts['ok']}")
    print(f"  Skipped (cached)    : {counts['skip']}")
    print(f"  Missing MP4         : {counts['missing']}")
    print(f"  Empty video         : {counts['empty']}")
    print(f"  Errors              : {counts['error']}")
    print(f"  Glosses now complete: {len(complete) + newly_complete}")
    print("=" * 50)
    print(f"\nRun the classifier with:")
    print(f"  python wlasl_mediapipe_signer_classifier.py "
          f"--min_signers {args.min_signers} --complete_only")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
