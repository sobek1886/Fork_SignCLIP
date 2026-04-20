#!/usr/bin/env python3
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
extract_asl_citizen_poses.py – MediaPipe holistic pose extraction for ASL-Citizen.
Uses the mediapipe 0.10+ Tasks API (HolisticLandmarker).

Outputs (N_frames, 609) .npy files with the same reduced_face feature layout
as SignCLIP's PoseProcessor:
    33 POSE_LANDMARKS  +  128 FACE contour pts  +  21 LEFT_HAND  +  21 RIGHT_HAND
    = 203 landmarks × 3 coords = 609 dims

Landmarks are normalised so that the shoulder width = 1 (matching pose_format
convention), centred at the midpoint between the two shoulders.

Requires the holistic landmarker model file – download once:
    wget -q -O holistic_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task

Usage:
    python extract_asl_citizen_poses.py \
        --video_dir  /scratch-shared/psobecki/ASL_Citizen/videos \
        --splits_dir $HOME/ASL_Citizen/splits \
        --output_dir $HOME/ASL_Citizen/mediapipe_features \
        --model_path ./holistic_landmarker.task \
        [--workers 8]  [--dataset_name asl_citizen]  [--overwrite]
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
# Reduced-face landmark indices (matches SignCLIP's FACEMESH_CONTOURS_POINTS)
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

# ---------------------------------------------------------------------------
# Per-process landmarker (created once per worker in the pool initializer)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _lm_to_array(lms, n):
    """Convert a list of NormalizedLandmark to (n, 3) float32 array."""
    if not lms:
        return np.zeros((n, 3), dtype=np.float32)
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms[:n]], dtype=np.float32)
    if len(arr) < n:
        pad = np.zeros((n - len(arr), 3), dtype=np.float32)
        arr = np.vstack([arr, pad])
    return arr


def _result_to_feat(result):
    """Convert HolisticLandmarkerResult to a (609,) float32 feature vector."""
    import mediapipe as mp_lib

    pose_arr = _lm_to_array(result.pose_landmarks, 33)         # (33, 3)
    face_arr = _lm_to_array(result.face_landmarks, 478)         # (478, 3)
    lh_arr   = _lm_to_array(result.left_hand_landmarks, 21)    # (21, 3)
    rh_arr   = _lm_to_array(result.right_hand_landmarks, 21)   # (21, 3)

    face_reduced = face_arr[FACEMESH_CONTOURS_POINTS]           # (128, 3)
    combined = np.concatenate([pose_arr, face_reduced, lh_arr, rh_arr], axis=0)  # (203, 3)

    # Normalise: shoulder width = 1, centred at shoulder midpoint
    ls = pose_arr[11, :2]   # left shoulder  (x, y)
    rs = pose_arr[12, :2]   # right shoulder (x, y)
    shoulder_dist = float(np.linalg.norm(ls - rs))
    if shoulder_dist > 1e-6:
        midpoint = (ls + rs) / 2.0
        combined[:, :2] = (combined[:, :2] - midpoint) / shoulder_dist
        combined[:, 2]  =  combined[:, 2] / shoulder_dist

    return np.nan_to_num(combined.reshape(-1)).astype(np.float32)  # (609,)


# ---------------------------------------------------------------------------
# Worker
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


def _extract_one(task: tuple) -> tuple:
    """
    Worker function. Returns (feat_id, status) where status is one of:
      'ok' | 'skip' | 'missing' | 'empty' | 'error:<msg>'
    """
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
# Main
# ---------------------------------------------------------------------------

def _collect_tasks(splits_dir, video_dir, output_dir, dataset_name, overwrite):
    tasks = []
    seen  = set()
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
                for candidate_name in (video_file, video_stem + ".mp4", video_stem + ".webm"):
                    candidate = os.path.join(video_dir, candidate_name)
                    if os.path.exists(candidate):
                        video_path = candidate
                        break
                if video_path is None:
                    video_path = os.path.join(video_dir, video_file)
                tasks.append((feat_id, video_path, output_dir, overwrite))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video_dir",    required=True)
    parser.add_argument("--splits_dir",   required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--model_path",   required=True,
                        help="Path to holistic_landmarker.task model file")
    parser.add_argument("--workers",      type=int, default=8)
    parser.add_argument("--dataset_name", default="asl_citizen")
    parser.add_argument("--overwrite",    action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(
        args.splits_dir, args.video_dir, args.output_dir,
        args.dataset_name, args.overwrite,
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
