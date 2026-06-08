#!/usr/bin/env python3
"""
eval_asl_citizen_retrieval.py  –  Frozen-feature dictionary retrieval on
                                   ASL-Citizen (no downstream training).

Reproduces the ASL-Citizen benchmark retrieval protocol used by the SignRep
paper (Wong et al., ICCV 2025, Table 4) for evaluating a feature extractor
*without* fine-tuning.  This is a faithful, vectorised port of the official
ASL-Citizen reference implementation:
    microsoft/ASL-citizen-code → "I3D Features/test_features.py"

Protocol
  - Gallery / dictionary = TRAIN split videos (each carries a gloss label).
  - Queries              = TEST  split videos.
  - Each video feature (T x D per-clip .npy) is mean-pooled over time → (D,).
  - For a query, compute COSINE distance to every gallery video; for each
    gloss keep the MINIMUM distance (per-gloss nearest neighbour), then rank
    all glosses by ascending distance.
  - Let r be the 0-indexed rank of the query's ground-truth gloss:
        DCG  = 1 / log2(r + 2)        (single-relevant-item gain)
        MRR  = 1 / (r + 1)
        Rec@k = 1 if r < k            (k = 1, 5, 10, 20)
  - Report the mean over all queries, scaled x100 (as in SignRep Table 4,
    where "Rec@1"/"Rec@5" are the ASL-Citizen "Top-1"/"Top-5" accuracies).

Weighted pooling (--weighted, requires --pose_dir)
  Approximates SignRep's activity-weighted temporal pooling (eq. 11-12).
  For each video clip t, a hand-activity score γ_t is derived from the
  MediaPipe pose features (same files as asl_citizen_feature_l2_probe.py):
    - Pose .npy files are (N_frames, 609): flattened (203 landmarks × 3 coords).
      Layout: 33 POSE | 128 FACE_contour | 21 LEFT_HAND | 21 RIGHT_HAND.
    - Left wrist y  = pose frame index 46 (landmark 15, coord 1).
    - Right wrist y = pose frame index 49 (landmark 16, coord 1).
    - Left/right hip y = indices 70, 73 (landmarks 23/24).
    - In normalized coords (y ↓, centered at shoulder midpoint, shoulder-width=1):
        stomach_y  = mean(left_hip_y, right_hip_y) / 2
        above_stomach(h) = wrist_h_y < stomach_y
        moving(h)        = std(wrist_h_xy over clip) > MOTION_THRESH (0.05)
        active(h)        = above_stomach(h) OR moving(h)
        γ_t = max(active_LH_t, active_RH_t)
  - Pose frames are uniformly resampled to match the T Logos clips.
  - If pose is missing, falls back to mean pooling for that video.
  - Final weighted vector: sum(γ_t * feat_t) / sum(γ_t), or mean if all γ=0.

This is generic over the feature extractor: point --feature_dir at the Logos
features now, and at the SignRep features later, to compare feature quality
on identical footing.

Feature files
  Named  {dataset_name}_{video_basename_no_ext}.npy  (shape T x D, or D).

Usage
  # Frozen Logos — average pooling (SignRep "avg" row)
  python eval_asl_citizen_retrieval.py \\
      --feature_dir /home/psobecki/ASL_Citizen/logos_features \\
      --splits_dir  /home/psobecki/ASL_Citizen/splits \\
      --feature_name logos

  # Frozen Logos — activity-weighted pooling (SignRep "weighted" row)
  python eval_asl_citizen_retrieval.py \\
      --feature_dir /home/psobecki/ASL_Citizen/logos_features \\
      --splits_dir  /home/psobecki/ASL_Citizen/splits \\
      --feature_name logos_weighted --weighted \\
      --pose_dir /home/psobecki/ASL_Citizen/mediapipe_features

  # Later: frozen SignRep features
  python eval_asl_citizen_retrieval.py \\
      --feature_dir /home/psobecki/ASL_Citizen/signrep_features \\
      --splits_dir  /home/psobecki/ASL_Citizen/splits \\
      --feature_name signrep

  # Quick sanity check on a subset
  python eval_asl_citizen_retrieval.py --feature_name logos \\
      --max_gallery 2000 --max_queries 500
"""

import argparse
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults (Snellius paths) — mirror asl_citizen_feature_l2_probe.py
# ---------------------------------------------------------------------------
DEFAULT_LOGOS_DIR = "/home/psobecki/ASL_Citizen/logos_features"
DEFAULT_I3D_DIR   = "/home/psobecki/ASL_Citizen/i3d_wlasl_features"
DEFAULT_SPLITS    = "/home/psobecki/ASL_Citizen/splits"
DEFAULT_POSE_DIR  = "/home/psobecki/ASL_Citizen/mediapipe_features"
DATASET_NAME      = "asl_citizen"

_SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}

# Pose landmark flat indices in the (609,) vector
# Layout: [33 POSE | 128 FACE | 21 LH | 21 RH] × 3 coords, row-major
_LW_X, _LW_Y = 15*3,   15*3+1   # left wrist
_RW_X, _RW_Y = 16*3,   16*3+1   # right wrist
_LH_Y        = 23*3+1            # left hip y
_RH_Y        = 24*3+1            # right hip y

# Motion threshold in shoulder-width units (hand std over a clip)
_MOTION_THRESH = 0.05


# ---------------------------------------------------------------------------
# Activity-weighted pooling helpers
# ---------------------------------------------------------------------------

def _activity_weights(pose: np.ndarray, T: int) -> np.ndarray:
    """Compute per-clip activity weights γ_t ∈ {0, 1}  (shape T,).

    Approximates SignRep eq. 11-12: γ_t = max(LH_active_t, RH_active_t).
    A hand is active in clip t if it is above the stomach midpoint OR moving.
    Falls back to uniform weights (all ones) if pose is unusable.
    """
    N = pose.shape[0]
    if N == 0 or T == 0:
        return np.ones(T, dtype=np.float32)

    # Per-frame: wrist positions and hip y
    lw_x = pose[:, _LW_X];  lw_y = pose[:, _LW_Y]
    rw_x = pose[:, _RW_X];  rw_y = pose[:, _RW_Y]
    lh_y = pose[:, _LH_Y];  rh_y = pose[:, _RH_Y]

    # Stomach midpoint y: halfway between shoulder midpoint (y=0) and hips.
    # Falls back to 0.8 shoulder-widths if hip detection failed (zeros).
    hip_y = (lh_y + rh_y) / 2.0
    stomach_y = np.where(np.abs(hip_y) > 0.1, hip_y / 2.0, 0.8)

    # Assign each pose frame to one of T clips (uniform split).
    clip_idx = (np.arange(N) * T / N).astype(int).clip(0, T - 1)

    weights = np.zeros(T, dtype=np.float32)
    for t in range(T):
        mask = clip_idx == t
        if not mask.any():
            weights[t] = 1.0   # no pose frames for this clip → treat as active
            continue

        s_y = stomach_y[mask]

        lw_xt, lw_yt = lw_x[mask], lw_y[mask]
        rw_xt, rw_yt = rw_x[mask], rw_y[mask]

        # Position: hand above stomach (wrist_y < stomach_y in ↓-positive coords)
        lh_above = (lw_yt < s_y).any()
        rh_above = (rw_yt < s_y).any()

        # Motion: std of wrist (x, y) over the clip frames
        lh_moving = float(np.std(lw_xt)) + float(np.std(lw_yt)) > _MOTION_THRESH
        rh_moving = float(np.std(rw_xt)) + float(np.std(rw_yt)) > _MOTION_THRESH

        lh_active = lh_above or lh_moving
        rh_active = rh_above or rh_moving
        weights[t] = float(lh_active or rh_active)

    # If all clips are inactive (pose entirely failed), fall back to uniform.
    if weights.sum() == 0:
        weights[:] = 1.0

    return weights


def _weighted_pool(feat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average of (T, D) feature clips by (T,) activity weights."""
    w = weights / weights.sum()
    return (feat * w[:, None]).sum(axis=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metadata(splits_dir: Path, split: str) -> list[dict]:
    """Read one CSV split → list of {feat_id, gloss}."""
    csv_path = splits_dir / _SPLIT_FILES[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"split file not found: {csv_path}")
    records = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            video_basename = os.path.splitext(row["Video file"])[0]
            records.append({
                "feat_id": f"{DATASET_NAME}_{video_basename}",
                "gloss":   row["Gloss"].strip(),
            })
    return records


def load_features(
    records: list[dict],
    feature_dir: Path,
    max_videos: int | None = None,
    desc: str = "Loading features",
    pose_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load .npy features, pool over time → (N, D) matrix + gloss list.

    If pose_dir is given, uses activity-weighted pooling (SignRep weighted
    variant); otherwise falls back to simple mean pooling (avg variant).
    Videos whose pose file is missing fall back to mean pooling individually.
    """
    weighted = pose_dir is not None
    feats, glosses = [], []
    missing = 0
    pose_fallback = 0

    for rec in tqdm(records, desc=desc):
        npy_path = feature_dir / (rec["feat_id"] + ".npy")
        if not npy_path.exists():
            missing += 1
            continue
        try:
            feat = np.load(npy_path).astype(np.float32)   # (T, D) or (D,)
        except Exception:
            missing += 1
            continue
        if feat.ndim == 2:
            if feat.shape[0] == 0:
                missing += 1
                continue
            if weighted:
                pose_path = pose_dir / (rec["feat_id"] + ".npy")
                if pose_path.exists():
                    try:
                        pose = np.load(pose_path).astype(np.float32)  # (N_frames, 609)
                        w = _activity_weights(pose, feat.shape[0])
                        feat = _weighted_pool(feat, w)
                    except Exception:
                        feat = feat.mean(axis=0)
                        pose_fallback += 1
                else:
                    feat = feat.mean(axis=0)
                    pose_fallback += 1
            else:
                feat = feat.mean(axis=0)                   # → (D,)
        elif feat.ndim != 1:
            missing += 1
            continue
        feats.append(feat)
        glosses.append(rec["gloss"])
        if max_videos and len(feats) >= max_videos:
            break

    if not feats:
        raise RuntimeError(f"No features loaded from {feature_dir}")
    msg = f"  {desc}: loaded {len(feats)} / {len(records)} ({missing} missing .npy)"
    if weighted:
        msg += f", {pose_fallback} mean-pool fallbacks (no pose)"
    print(msg)
    return np.stack(feats), glosses


# ---------------------------------------------------------------------------
# Retrieval (faithful vectorised port of ASL-Citizen test_features.py)
# ---------------------------------------------------------------------------

def evaluate(
    gallery_feats: np.ndarray,
    gallery_glosses: list[str],
    query_feats: np.ndarray,
    query_glosses: list[str],
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Per-gloss nearest-neighbour retrieval → DCG / MRR / Top-{1,5,10,20}.

    Ranking by ascending cosine distance == descending cosine similarity, so
    we L2-normalise and rank glosses by the MAX similarity to any gallery
    video of that gloss (the min-distance rule in the reference code).
    """
    # Gloss vocabulary defined by the gallery (dictionary), as in the reference.
    g_dict = OrderedDict()
    for g in gallery_glosses:
        if g not in g_dict:
            g_dict[g] = len(g_dict)
    n_gloss = len(g_dict)

    gallery_gloss_idx = torch.tensor(
        [g_dict[g] for g in gallery_glosses], dtype=torch.long, device=device
    )

    # L2-normalise so that dot product == cosine similarity.
    G = torch.from_numpy(gallery_feats).to(device)
    G = torch.nn.functional.normalize(G, dim=-1)                  # (Ng, D)

    # Queries whose gloss is absent from the gallery cannot be retrieved;
    # the reference code assumes a closed vocabulary, so we skip + warn.
    skipped = [g for g in query_glosses if g not in g_dict]
    if skipped:
        print(f"  WARNING: {len(skipped)} query videos have a gloss not in the "
              f"gallery vocabulary — skipped (closed-vocab protocol).")

    sums = np.zeros(6, dtype=np.float64)   # [DCG, Top1, Top5, Top10, Top20, MRR]
    n_eval = 0

    idx_expand_full = gallery_gloss_idx.unsqueeze(0)             # (1, Ng)

    for start in tqdm(range(0, len(query_glosses), batch_size), desc="Retrieval"):
        batch_glosses = query_glosses[start:start + batch_size]
        keep = [i for i, g in enumerate(batch_glosses) if g in g_dict]
        if not keep:
            continue
        rows = [start + i for i in keep]
        Qb = torch.from_numpy(query_feats[rows]).to(device)
        Qb = torch.nn.functional.normalize(Qb, dim=-1)           # (b, D)
        labels = torch.tensor(
            [g_dict[batch_glosses[i]] for i in keep],
            dtype=torch.long, device=device,
        )

        sim = Qb @ G.T                                           # (b, Ng)

        # Per-gloss MAX similarity (== per-gloss MIN cosine distance).
        gloss_sim = torch.full((sim.size(0), n_gloss), float("-inf"),
                               device=device)
        gloss_sim.scatter_reduce_(
            1, idx_expand_full.expand(sim.size(0), -1), sim,
            reduce="amax", include_self=True,
        )

        # 0-indexed rank of the ground-truth gloss = #glosses scoring strictly
        # higher than it (ties broken toward a better rank, matching argsort's
        # stable order in the float-distinct reference case).
        gt_sim = gloss_sim.gather(1, labels.unsqueeze(1))        # (b, 1)
        ranks = (gloss_sim > gt_sim).sum(dim=1).double()         # (b,)

        dcg = 1.0 / torch.log2(ranks + 2.0)
        mrr = 1.0 / (ranks + 1.0)
        top1  = (ranks < 1).double()
        top5  = (ranks < 5).double()
        top10 = (ranks < 10).double()
        top20 = (ranks < 20).double()

        sums += np.array([
            dcg.sum().item(), top1.sum().item(), top5.sum().item(),
            top10.sum().item(), top20.sum().item(), mrr.sum().item(),
        ])
        n_eval += sim.size(0)

    if n_eval == 0:
        raise RuntimeError("No evaluable queries (gloss vocabulary mismatch?).")

    means = sums / n_eval
    return {
        "n_queries":   n_eval,
        "n_gallery":   len(gallery_glosses),
        "n_glosses":   n_gloss,
        "DCG":   100.0 * means[0],
        "Rec@1": 100.0 * means[1],
        "Rec@5": 100.0 * means[2],
        "Rec@10": 100.0 * means[3],
        "Rec@20": 100.0 * means[4],
        "MRR":   100.0 * means[5],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(res: dict, feature_name: str, feat_dim: int, weighted: bool = False):
    W = 60
    pool = "activity-weighted" if weighted else "mean"
    print("\n" + "=" * W)
    print(f"ASL-CITIZEN  FROZEN-FEATURE  RETRIEVAL  ({feature_name})")
    print(f"(no downstream training — SignRep Table 4 protocol, {pool} pooling)")
    print("=" * W)
    print(f"  Feature dim     : {feat_dim}")
    print(f"  Gallery videos  : {res['n_gallery']}  ({res['n_glosses']} glosses)")
    print(f"  Query videos    : {res['n_queries']}")
    print("  " + "-" * (W - 2))
    print(f"  DCG    : {res['DCG']:6.2f}")
    print(f"  MRR    : {res['MRR']:6.2f}")
    print(f"  Rec@1  : {res['Rec@1']:6.2f}")
    print(f"  Rec@5  : {res['Rec@5']:6.2f}")
    print(f"  Rec@10 : {res['Rec@10']:6.2f}")
    print(f"  Rec@20 : {res['Rec@20']:6.2f}")
    print("=" * W)
    print("  Compare against SignRep Table 4 (ASL-Citizen, frozen):")
    print("    HieraMAE-Kinetic  DCG 11.64  Rec@1  0.25  Rec@5  0.84")
    print("    Hand Keypoints    DCG 25.91  Rec@1  7.96  Rec@5 19.58")
    print("    SignRep (avg)     DCG 61.40  Rec@1 37.47  Rec@5 68.77")
    print("    SignRep (weighted)DCG 71.21  Rec@1 49.95  Rec@5 80.09")
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global DATASET_NAME
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--feature_dir", type=Path, default=None,
                        help="Directory with .npy feature files "
                             "(default: logos path on Snellius)")
    parser.add_argument("--splits_dir", type=Path, default=Path(DEFAULT_SPLITS),
                        help=f"Directory with train/val/test CSVs "
                             f"(default: {DEFAULT_SPLITS})")
    parser.add_argument("--feature_name", type=str, default="logos",
                        help="Label for display + default --feature_dir "
                             "(e.g. logos, i3d_wlasl, signrep)")
    parser.add_argument("--gallery_split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Dictionary/gallery split (default: train)")
    parser.add_argument("--query_split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Query split (default: test)")
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME,
                        help=f"Prefix in .npy filenames (default: {DATASET_NAME})")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Query batch size for the similarity matmul")
    parser.add_argument("--max_gallery", type=int, default=None,
                        help="Cap gallery videos (quick sanity check)")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Cap query videos (quick sanity check)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weighted", action="store_true",
                        help="Use activity-weighted temporal pooling instead of "
                             "mean pooling (approximates SignRep 'weighted' row). "
                             "Requires --pose_dir.")
    parser.add_argument("--pose_dir", type=Path,
                        default=Path(DEFAULT_POSE_DIR),
                        help=f"Directory with MediaPipe pose .npy files — "
                             f"only used when --weighted is set "
                             f"(default: {DEFAULT_POSE_DIR})")
    parser.add_argument("--output_json", type=Path, default=None,
                        help="If set, write the metrics dict to this JSON file")
    args = parser.parse_args()

    if args.feature_dir is None:
        name = args.feature_name.lower()
        if "logos" in name:
            args.feature_dir = Path(DEFAULT_LOGOS_DIR)
        else:
            args.feature_dir = Path(DEFAULT_I3D_DIR)

    if not args.feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {args.feature_dir}")
    if not args.splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {args.splits_dir}")

    DATASET_NAME = args.dataset_name
    device = torch.device(args.device)

    pose_dir = args.pose_dir if args.weighted else None
    if args.weighted and not args.pose_dir.exists():
        raise FileNotFoundError(f"Pose directory not found: {args.pose_dir}")

    print(f"Feature dir   : {args.feature_dir}")
    print(f"Splits dir    : {args.splits_dir}")
    print(f"Gallery/query : {args.gallery_split} → {args.query_split}")
    print(f"Pooling       : {'activity-weighted (pose_dir=' + str(pose_dir) + ')' if args.weighted else 'mean'}")
    print(f"Device        : {device}")

    gallery_recs = load_metadata(args.splits_dir, args.gallery_split)
    query_recs   = load_metadata(args.splits_dir, args.query_split)

    gallery_feats, gallery_glosses = load_features(
        gallery_recs, args.feature_dir, args.max_gallery,
        desc=f"Loading gallery ({args.gallery_split})", pose_dir=pose_dir)
    query_feats, query_glosses = load_features(
        query_recs, args.feature_dir, args.max_queries,
        desc=f"Loading queries ({args.query_split})", pose_dir=pose_dir)

    feat_dim = gallery_feats.shape[1]

    res = evaluate(gallery_feats, gallery_glosses, query_feats, query_glosses,
                   device=device, batch_size=args.batch_size)

    print_results(res, args.feature_name, feat_dim, weighted=args.weighted)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"feature_name": args.feature_name, "weighted": args.weighted,
                       "feature_dim": feat_dim, **res}, f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
