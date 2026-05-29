#!/usr/bin/env python3
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
extract_wlasl100_poses.py – MediaPipe holistic pose extraction for WLASL100.

Reads the three split CSVs produced by generate_wlasl100_splits.py and extracts
(N_frames, 609) .npy pose features using the same pipeline as
extract_asl_citizen_poses.py.

Split/directory layout on Snellius:
  train  →  augmented_videos/{video_id}/original.mp4    feat_id: wlasl100_{video_id}_original
  val    →  val/{video_id}.mp4                         feat_id: wlasl100_val_{video_id}
  test   →  test/{video_id}.mp4                        feat_id: wlasl100_test_{video_id}

The val CSV stores 'val_{video_id}.mp4' as the Video file (so the feat_id
includes the 'val_' prefix), but the actual file on disk has no prefix.
This script handles that mapping transparently.

Usage:
    python extract_wlasl100_poses.py \\
        --splits_dir      /home/psobecki/wlasl100/splits \\
        --train_video_dir /scratch-shared/psobecki/wlasl100/augmented_videos \\
        --val_video_dir   /scratch-shared/psobecki/wlasl100/val \\
        --test_video_dir  /scratch-shared/psobecki/wlasl100/test \\
        --output_dir      /scratch-shared/psobecki/wlasl100/pose_features \\
        --model_path      /home/psobecki/Fork_SignCLIP/examples/MMPT/holistic_landmarker.task \\
        [--workers 60]    [--overwrite]
"""

import argparse
import contextlib
import csv
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reduced-face landmark indices (identical to extract_asl_citizen_poses.py)
# ---------------------------------------------------------------------------
FACEMESH_CONTOURS_POINTS = [
    0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63,
    65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109,
    127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158,
    159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251,
    263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300,
    308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361,
    362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388,
    389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466,
]  # 128 contour points

_landmarker = None


def _init_worker(model_path):
    global _landmarker
    import mediapipe as mp_lib
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions

    options = HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp_tasks.vision.RunningMode.IMAGE,
    )
    _landmarker = HolisticLandmarker.create_from_options(options)


def _lm_to_array(lms, n):
    if not lms:
        return np.zeros((n, 3), dtype=np.float32)
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms[:n]], dtype=np.float32)
    if len(arr) < n:
        arr = np.vstack([arr, np.zeros((n - len(arr), 3), dtype=np.float32)])
    return arr


def _result_to_feat(result):
    pose_arr = _lm_to_array(result.pose_landmarks, 33)
    face_arr = _lm_to_array(result.face_landmarks, 478)
    lh_arr   = _lm_to_array(result.left_hand_landmarks, 21)
    rh_arr   = _lm_to_array(result.right_hand_landmarks, 21)

    face_reduced = face_arr[FACEMESH_CONTOURS_POINTS]
    combined = np.concatenate([pose_arr, face_reduced, lh_arr, rh_arr], axis=0)  # (203, 3)

    ls = pose_arr[11, :2]
    rs = pose_arr[12, :2]
    shoulder_dist = float(np.linalg.norm(ls - rs))
    if shoulder_dist > 1e-6:
        midpoint = (ls + rs) / 2.0
        combined[:, :2] = (combined[:, :2] - midpoint) / shoulder_dist
        combined[:, 2]  =  combined[:, 2] / shoulder_dist

    return np.nan_to_num(combined.reshape(-1)).astype(np.float32)  # (609,)


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


def _extract_one(task):
    feat_id, video_path, output_dir, overwrite = task

    cache_path = Path(output_dir) / f"{feat_id}.npy"
    if cache_path.exists() and not overwrite:
        return feat_id, "skip"

    video_path = Path(video_path)
    if not video_path.exists():
        return feat_id, "missing"

    try:
        import mediapipe as mp_lib

        cap    = cv2.VideoCapture(str(video_path))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            return feat_id, "empty"

        features = []
        with _suppress_c_stderr():
            for frame in frames:
                mp_image = mp_lib.Image(
                    image_format=mp_lib.ImageFormat.SRGB, data=frame
                )
                result = _landmarker.detect(mp_image)
                features.append(_result_to_feat(result))

        np.save(str(cache_path), np.stack(features, axis=0))  # (N_frames, 609)
        return feat_id, "ok"

    except Exception as exc:
        return feat_id, f"error: {exc}"


# ---------------------------------------------------------------------------
# Task collection — handles per-split video directory and val prefix stripping
# ---------------------------------------------------------------------------

def _collect_tasks(splits_dir, train_video_dir, val_video_dir, test_video_dir,
                   output_dir, overwrite):
    """
    Split  | CSV Video file        | feat_id                 | actual video path
    -------|-----------------------|-------------------------|-------------------------------------------
    train  | {vid}_original.mp4    | wlasl100_{vid}_original | train_video_dir/{vid}/original.mp4
    val    | val_{vid}.mp4         | wlasl100_val_{vid}      | val_video_dir/{vid}.mp4
    test   | {vid}.mp4             | wlasl100_test_{vid}     | test_video_dir/{vid}.mp4
    """
    tasks = []
    seen  = set()

    # --- train ---
    csv_path = os.path.join(splits_dir, "train.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                video_file = row["Video file"]               # e.g. "05237_original.mp4"
                video_stem = os.path.splitext(video_file)[0] # "05237_original"
                feat_id    = f"wlasl100_{video_stem}"
                if feat_id in seen:
                    continue
                seen.add(feat_id)
                # "05237_original" → subdir "05237", filename "original.mp4"
                video_id = video_stem.replace("_original", "")
                video_path = os.path.join(train_video_dir, video_id, "original.mp4")
                tasks.append((feat_id, video_path, output_dir, overwrite))

    # --- val ---
    csv_path = os.path.join(splits_dir, "val.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                video_file = row["Video file"]               # e.g. "val_69422.mp4"
                video_stem = os.path.splitext(video_file)[0] # "val_69422"
                feat_id    = f"wlasl100_{video_stem}"
                if feat_id in seen:
                    continue
                seen.add(feat_id)
                # strip "val_" prefix to get the actual filename on disk
                actual_filename = video_file[len("val_"):]   # "69422.mp4"
                video_path = os.path.join(val_video_dir, actual_filename)
                tasks.append((feat_id, video_path, output_dir, overwrite))

    # --- test ---
    csv_path = os.path.join(splits_dir, "test.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                video_file = row["Video file"]               # e.g. "51068.mp4"
                video_stem = os.path.splitext(video_file)[0] # "51068"
                feat_id    = f"wlasl100_test_{video_stem}"
                if feat_id in seen:
                    continue
                seen.add(feat_id)
                video_path = os.path.join(test_video_dir, video_file)
                tasks.append((feat_id, video_path, output_dir, overwrite))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--splits_dir",      required=True,
                        help="Directory containing train.csv / val.csv / test.csv")
    parser.add_argument("--train_video_dir", required=True,
                        help="Directory containing augmented train videos")
    parser.add_argument("--val_video_dir",   required=True,
                        help="Directory containing val videos (plain {video_id}.mp4)")
    parser.add_argument("--test_video_dir",  required=True,
                        help="Directory containing test videos")
    parser.add_argument("--output_dir",      required=True,
                        help="Output directory for .npy pose feature files")
    parser.add_argument("--model_path",      required=True,
                        help="Path to holistic_landmarker.task model file")
    parser.add_argument("--workers",         type=int, default=8)
    parser.add_argument("--overwrite",       action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(
        args.splits_dir,
        args.train_video_dir, args.val_video_dir, args.test_video_dir,
        args.output_dir, args.overwrite,
    )
    already_done = sum(
        1 for feat_id, _, out_dir, _ in tasks
        if (Path(out_dir) / f"{feat_id}.npy").exists()
    )
    print(f"Total videos     : {len(tasks)}")
    print(f"Already extracted: {already_done}"
          + ("  (will overwrite)" if args.overwrite else "  (skipping)"))
    print(f"Workers          : {args.workers}")
    print(f"Model            : {args.model_path}")
    print(f"Output dir       : {args.output_dir}\n")

    counts = {"ok": 0, "skip": 0, "missing": 0, "empty": 0, "error": 0}

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.model_path,),
        maxtasksperchild=200,
    ) as pool:
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
