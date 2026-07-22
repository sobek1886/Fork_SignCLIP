#!/usr/bin/env python3
import os
import contextlib
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect OS-level stderr (fd 2) to /dev/null for the duration of the block.
    Necessary to silence C++/ABSL warnings from MediaPipe that bypass Python logging."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)

"""
wlasl_mediapipe_signer_classifier.py  –  Cross-gloss signer retrieval probe on WLASL
using MediaPipe Holistic pose features.

Uses the same cross-gloss LOO retrieval design as wlasl_signclip_signer_classifier.py
for a direct comparison between raw pose features and SignCLIP embeddings.

  For each video v (signer s, gloss g) as a query:
    - Build a prototype for every signer t as the mean of their 1218-dim feature
      vectors for all videos EXCEPT those of gloss g (content contamination control).
    - 1-NN: predict the signer whose prototype is most similar to v.
    - Correct if the nearest prototype belongs to signer s.

  Chance level = 1 / (number of signers with a valid prototype).

Features
  Standard SignCLIP preprocessing (mirrors signer_classifier.py / demo_sign.py):
    select components → normalise (shoulder width = 1) → zero legs
    → (T, 609) per frame → mean + std pooling → 1218 dims

Usage
  python wlasl_mediapipe_signer_classifier.py
  python wlasl_mediapipe_signer_classifier.py --min_signer_videos 10
  python wlasl_mediapipe_signer_classifier.py --save_features mp_features.pkl
  python wlasl_mediapipe_signer_classifier.py --load_features mp_features.pkl
"""

import argparse
import json
import pickle
import concurrent.futures
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WLASL_JSON = Path("/Users/piotr/Projects/Thesis/Data/WLASL/WLASL_v0.3.json")
VIDEO_DIR  = Path("/Users/piotr/Projects/Thesis/Data/WLASL/videos")
POSE_CACHE = Path("/Users/piotr/Projects/Thesis/Data/WLASL/pose_cache")

# ---------------------------------------------------------------------------
# MediaPipe / pose_format constants  (identical to signer_classifier.py)
# ---------------------------------------------------------------------------
FACEMESH_CONTOURS_POINTS = [
    '0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46',
    '52', '53', '54', '55', '58', '61', '63', '65', '66', '67', '70', '78',
    '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107',
    '109', '127', '132', '133', '136', '144', '145', '146', '148', '149', '150',
    '152', '153', '154', '155', '157', '158', '159', '160', '161', '162', '163',
    '172', '173', '176', '178', '181', '185', '191', '234', '246', '249', '251',
    '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291',
    '293', '295', '296', '297', '300', '308', '310', '311', '312', '314', '317',
    '318', '321', '323', '324', '332', '334', '336', '338', '356', '361', '362',
    '365', '373', '374', '375', '377', '378', '379', '380', '381', '382', '384',
    '385', '386', '387', '388', '389', '390', '397', '398', '400', '402', '405',
    '409', '415', '454', '466',
]


# ---------------------------------------------------------------------------
# Pose extraction helpers
# ---------------------------------------------------------------------------

def _extract_from_video(video_path: Path):
    """Run MediaPipe Holistic on every frame of video_path via pose_format."""
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
        return None

    with _suppress_c_stderr():
        return load_holistic(frames, fps=fps, width=width, height=height, progress=False)


def _get_pose(video_id: int, cache_dir: Path):
    """Return a Pose object from cache; returns None if not cached."""
    from pose_format import Pose

    cache_path = cache_dir / f"{video_id:05d}.pose"
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        return Pose.read(f.read())


# ---------------------------------------------------------------------------
# Feature preprocessing  (mirrors signer_classifier.py / demo_sign.py)
# ---------------------------------------------------------------------------

def _normalization_info(pose_header):
    name = pose_header.components[0].name
    if name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"),
        )
    raise ValueError(f"Unknown pose schema: {name}")


def _hide_legs(pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        leg_pts = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        indices = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + pt)
            for pt in leg_pts
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, indices] = 0
        pose.body.data[:, :, indices, :] = 0
        return pose
    raise ValueError("Cannot hide legs for this pose schema")


def _pose_to_matrix(pose) -> np.ndarray | None:
    """Apply SignCLIP preprocessing → (T, 609) float32."""
    try:
        pose = pose.get_components(
            ["POSE_LANDMARKS", "FACE_LANDMARKS",
             "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
            {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
        )
        pose = pose.normalize(_normalization_info(pose.header))
        pose = _hide_legs(pose)
        feat = np.nan_to_num(np.asarray(pose.body.data))  # (T, 1, n_pts, 3)
        feat = feat[:, 0, :, :]                            # (T, n_pts, 3)
        feat = feat.reshape(feat.shape[0], -1)             # (T, 609)
        return feat.astype(np.float32)
    except Exception:
        return None


def _load_one(args: tuple) -> tuple[int, np.ndarray | None]:
    """Worker: load pose from cache and extract 1218-dim feature vector."""
    video_id, cache_dir = args
    pose = _get_pose(video_id, Path(cache_dir))
    if pose is None:
        return video_id, None
    feat = _pose_to_matrix(pose)
    if feat is None or len(feat) == 0:
        return video_id, None
    return video_id, np.concatenate([feat.mean(0), feat.std(0)])  # (1218,)


def load_all_features(video_ids: list[int], cache_dir: Path,
                      io_workers: int = 8) -> dict[int, np.ndarray]:
    """Load 1218-dim features for all cached videos in parallel."""
    tasks = [(vid_id, str(cache_dir)) for vid_id in video_ids
             if (cache_dir / f"{vid_id:05d}.pose").exists()]

    features = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as ex:
        for vid_id, feat in tqdm(
            ex.map(_load_one, tasks), total=len(tasks), desc="Loading features"
        ):
            if feat is not None:
                features[vid_id] = feat
    return features


# ---------------------------------------------------------------------------
# Cross-gloss LOO retrieval  (identical logic to wlasl_signclip_signer_classifier.py)
# ---------------------------------------------------------------------------

def cosine_sim_matrix(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """(Q, D) × (K, D) → (Q, K) cosine similarity matrix."""
    q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    k = keys    / (np.linalg.norm(keys,    axis=1, keepdims=True) + 1e-8)
    return q @ k.T


def run_retrieval(videos: list[dict], min_signer_videos: int) -> list[dict]:
    """
    For each query video v (signer s, gloss g):
      - prototype_t = mean feature of all videos by signer t where gloss != g
      - predict = argmax_t cosine_sim(v, prototype_t)
      - record hit / miss / rank
    """
    by_signer: dict[int, list[dict]] = defaultdict(list)
    for v in videos:
        by_signer[v["signer_id"]].append(v)

    query_signers = {sid for sid, vlist in by_signer.items()
                     if len(vlist) >= min_signer_videos}

    results = []
    for v in tqdm(videos, desc="Retrieval"):
        sid   = v["signer_id"]
        gloss = v["gloss"]

        if sid not in query_signers:
            continue

        prototypes = {}
        for t, vlist in by_signer.items():
            other = [x["feat"] for x in vlist if x["gloss"] != gloss]
            if other:
                prototypes[t] = np.mean(other, axis=0)

        if sid not in prototypes:
            continue

        proto_ids = sorted(prototypes.keys())
        proto_mat = np.stack([prototypes[t] for t in proto_ids])
        query_vec = v["feat"][np.newaxis, :]

        sims      = cosine_sim_matrix(query_vec, proto_mat)[0]
        pred_idx  = int(np.argmax(sims))
        correct   = int(proto_ids[pred_idx] == sid)

        true_idx  = proto_ids.index(sid)
        rank      = int(np.sum(sims >= sims[true_idx]))

        results.append({
            "video_id":  v["video_id"],
            "signer_id": sid,
            "gloss":     gloss,
            "correct":   correct,
            "rank":      rank,
            "n_pool":    len(proto_ids),
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: list[dict], min_signer_videos: int):
    if not results:
        print("No results.")
        return

    n_pool  = results[0]["n_pool"]
    correct = [r["correct"] for r in results]
    ranks   = [r["rank"]    for r in results]

    overall_acc = np.mean(correct)
    mean_rank   = np.mean(ranks)
    chance      = 1.0 / n_pool
    ratio       = overall_acc / chance
    mrr         = np.mean([1.0 / r for r in ranks])

    by_signer: dict[int, list] = defaultdict(list)
    for r in results:
        by_signer[r["signer_id"]].append(r["correct"])

    W = 62
    print("\n" + "=" * W)
    print("WLASL MEDIAPIPE CROSS-GLOSS SIGNER RETRIEVAL PROBE")
    print("=" * W)
    print(f"\n  Design     : LOO cross-gloss 1-NN prototype retrieval")
    print(f"  Features   : MediaPipe Holistic mean+std  (1218 dims)")
    print(f"  Pool size  : {n_pool} signers")
    print(f"  Chance     : 1/{n_pool} = {chance:.4f}")
    print(f"  Queries    : {len(results)} videos  "
          f"(signers with ≥{min_signer_videos} videos)")
    print()
    print(f"  Top-1 accuracy : {overall_acc:.4f}  ({ratio:.2f}x chance)")
    print(f"  Mean rank      : {mean_rank:.2f}  (chance = {n_pool / 2:.1f})")
    print(f"  MRR            : {mrr:.4f}  (chance = {1 / n_pool:.4f})")

    print(f"\n{'Signer':>8}  {'Queries':>8}  {'Accuracy':>10}  {'Ratio':>7}")
    print("-" * 42)
    for sid, hits in sorted(by_signer.items(), key=lambda x: -len(x[1])):
        acc = np.mean(hits)
        print(f"  {sid:>6}  {len(hits):>8}  {acc:>10.3f}  {acc/chance:>6.2f}x")

    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--min_signer_videos", type=int, default=5,
                        help="Minimum cached videos for a signer to serve as a query "
                             "(default: 5)")
    parser.add_argument("--pose_cache", type=Path, default=POSE_CACHE)
    parser.add_argument("--io_workers", type=int, default=8,
                        help="Threads for parallel feature loading (default: 8)")
    parser.add_argument("--save_features", type=Path, default=None,
                        help="Save computed features to this pickle file for reuse")
    parser.add_argument("--load_features", type=Path, default=None,
                        help="Load pre-computed features instead of re-extracting")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    print(f"Loading WLASL metadata ...")
    with open(WLASL_JSON) as f:
        data = json.load(f)

    vid_to_meta: dict[int, dict] = {}
    for entry in data:
        for inst in entry["instances"]:
            if "video_id" in inst:
                vid_to_meta[int(inst["video_id"])] = {
                    "gloss":     entry["gloss"],
                    "signer_id": inst["signer_id"],
                }

    cached_ids = sorted(
        vid for vid in vid_to_meta
        if (args.pose_cache / f"{vid:05d}.pose").exists()
    )
    print(f"Cached videos: {len(cached_ids)}  across "
          f"{len(set(vid_to_meta[v]['signer_id'] for v in cached_ids))} signers, "
          f"{len(set(vid_to_meta[v]['gloss'] for v in cached_ids))} glosses")

    # ------------------------------------------------------------------
    # Load / compute features
    # ------------------------------------------------------------------
    if args.load_features and args.load_features.exists():
        print(f"Loading features from {args.load_features} ...")
        with open(args.load_features, "rb") as f:
            feat_dict = pickle.load(f)
        print(f"Loaded {len(feat_dict)} feature vectors")
    else:
        print(f"\nExtracting features from {len(cached_ids)} cached poses ...")
        feat_dict = load_all_features(cached_ids, cache_dir=args.pose_cache,
                                      io_workers=args.io_workers)
        print(f"Successfully extracted {len(feat_dict)} / {len(cached_ids)} features")

        if args.save_features:
            with open(args.save_features, "wb") as f:
                pickle.dump(feat_dict, f)
            print(f"Features saved to {args.save_features}")

    # ------------------------------------------------------------------
    # Build video list
    # ------------------------------------------------------------------
    videos = []
    for vid_id, feat in feat_dict.items():
        meta = vid_to_meta.get(vid_id)
        if meta is None:
            continue
        videos.append({
            "video_id":  vid_id,
            "signer_id": meta["signer_id"],
            "gloss":     meta["gloss"],
            "feat":      feat,
        })

    # ------------------------------------------------------------------
    # Retrieval probe
    # ------------------------------------------------------------------
    print(f"\nRunning cross-gloss signer retrieval "
          f"(min_signer_videos={args.min_signer_videos}) ...")
    results = run_retrieval(videos, min_signer_videos=args.min_signer_videos)
    print(f"Evaluated {len(results)} queries")

    print_results(results, min_signer_videos=args.min_signer_videos)


if __name__ == "__main__":
    main()
