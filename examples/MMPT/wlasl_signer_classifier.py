#!/usr/bin/env python3
"""
wlasl_signer_classifier.py  –  Per-gloss signer identity probe on WLASL SMPL-X data.

For each gloss (isolated sign) with multiple signers, trains a separate signer
classifier and evaluates with leave-one-out CV (LOO).  Reports per-gloss
accuracy, chance baseline, and the overall average across glosses.

This is the cleanest possible content-controlled experiment:
  - Each video = one isolated sign (not a sentence)
  - Gloss label is held constant within each classification problem
  - Each signer appears exactly once per gloss
  - No aggregation across content is needed

Features
  All SMPL-X parameters except betas/shape (dims 0:175):
    root_pose 0:3, body_pose 3:66, lhand_pose 66:111, rhand_pose 111:156,
    jaw+eyes 156:165, expr 165:175  → 175 dims/frame.
  Mean + std pooled over all valid frames → 350 dims.

Classifier
  1-Nearest-Neighbour with LOO-CV.
  With one instance per signer per gloss, 1-NN is the natural choice:
  it asks "which training signer's average pose configuration is most
  similar to this test instance?"  No parameters to fit, no overfitting risk.

Usage
  python wlasl_signer_classifier.py
  python wlasl_signer_classifier.py --min_signers 3
  python wlasl_signer_classifier.py --glosses accident approve apple
"""

import io
import pickle
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WLASL_JSON = Path("/Users/piotr/Projects/Thesis/Data/WLASL_v0.3.json")
PKL_DIR    = Path("/Users/piotr/Projects/Thesis/Data/SignAvatars/wlasl_pkls_cropFalse_defult_shape")

# SMPL-X parameter layout inside the 182-dim 'smplx' array:
#   root_pose   0:3    (3)    ← used
#   body_pose   3:66   (63)   ← used
#   lhand_pose  66:111 (45)   ← used
#   rhand_pose  111:156 (45)  ← used
#   jaw + eyes  156:165 (9)   ← used
#   expr        165:175 (10)  ← used
#   shape/betas 175:182 (7)   ✗ excluded (body shape, not pose/style)
POSE_SLICE = slice(0, 175)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class _CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_pose_features(video_id: int) -> np.ndarray | None:
    """
    Load one WLASL PKL and return a 350-dim mean+std feature vector.

    Extracts all SMPL-X parameters except betas/shape (dims 0:175) from
    each frame of the 'smplx' array, then applies mean + std pooling over time.

    Returns None if the file is missing or unreadable.
    """
    path = PKL_DIR / f"{video_id:05d}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            d = _CPU_Unpickler(f).load()
    except Exception:
        return None

    smplx = np.asarray(d["smplx"], dtype=np.float32)   # (T, 182)
    pose  = smplx[:, POSE_SLICE]                         # (T, 175)

    return np.concatenate([pose.mean(0), pose.std(0)])   # (350,)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_gloss(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Run LOO-CV with a 1-NN classifier on (N, 180) features.
    Returns accuracy (fraction correct).
    """
    n = len(labels)
    if n < 2:
        return float("nan")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(n_neighbors=1, metric="euclidean")),
    ])

    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(features):
        pipe.fit(features[train_idx], labels[train_idx])
        pred = pipe.predict(features[test_idx])
        if pred[0] == labels[test_idx[0]]:
            correct += 1

    return correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--min_signers", type=int, default=2,
                        help="Only include glosses with at least this many signers (default: 2)")
    parser.add_argument("--glosses", nargs="+", default=None,
                        help="Restrict to specific glosses (default: all multi-signer glosses)")
    args = parser.parse_args()

    # Load metadata
    print(f"Loading WLASL metadata from {WLASL_JSON} ...")
    with open(WLASL_JSON) as f:
        data = json.load(f)

    vid_to_meta = {
        int(inst["video_id"]): {"gloss": entry["gloss"], "signer_id": inst["signer_id"]}
        for entry in data
        for inst in entry["instances"]
        if "video_id" in inst
    }

    # Discover local PKLs
    local_ids = sorted(
        int(p.stem) for p in PKL_DIR.glob("*.pkl") if "_" not in p.stem
    )

    # Group by gloss
    by_gloss: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for vid_id in local_ids:
        if vid_id not in vid_to_meta:
            continue
        m = vid_to_meta[vid_id]
        by_gloss[m["gloss"]].append((vid_id, m["signer_id"]))

    # Filter
    if args.glosses:
        by_gloss = {g: v for g, v in by_gloss.items() if g in args.glosses}
    by_gloss = {g: v for g, v in by_gloss.items() if len(v) >= args.min_signers}

    glosses_sorted = sorted(by_gloss, key=lambda g: len(by_gloss[g]), reverse=True)
    print(f"Glosses to classify: {len(glosses_sorted)}  "
          f"(≥{args.min_signers} signers each)\n")

    # Per-gloss classification
    results = []
    failed_loads = 0

    print(f"{'Gloss':<25} {'N':>4}  {'Chance':>7}  {'LOO acc':>8}  {'Ratio':>6}")
    print("-" * 60)

    for gloss in glosses_sorted:
        instances = by_gloss[gloss]   # list of (video_id, signer_id)

        feats, signer_ids = [], []
        for vid_id, sid in instances:
            vec = load_pose_features(vid_id)
            if vec is None:
                failed_loads += 1
                continue
            feats.append(vec)
            signer_ids.append(sid)

        n = len(feats)
        if n < 2:
            continue

        X = np.stack(feats)
        y = np.array(signer_ids)

        acc   = classify_gloss(X, y)
        chance = 1.0 / n
        ratio = acc / chance if chance > 0 else float("nan")

        results.append({
            "gloss":  gloss,
            "n":      n,
            "chance": chance,
            "acc":    acc,
            "ratio":  ratio,
        })

        marker = " ✓" if acc > chance + 0.1 else ""
        print(f"{gloss:<25} {n:>4}  {chance:>7.3f}  {acc:>8.3f}  {ratio:>6.2f}x{marker}")

    if failed_loads:
        print(f"\n(Could not load {failed_loads} PKL files)")

    # Summary
    if not results:
        print("No results.")
        return

    accs   = [r["acc"]   for r in results]
    ratios = [r["ratio"] for r in results]
    ns     = [r["n"]     for r in results]
    chances = [r["chance"] for r in results]

    mean_acc    = np.mean(accs)
    mean_chance = np.mean(chances)
    mean_ratio  = np.mean(ratios)

    # Weighted average (by number of signers)
    weights    = np.array(ns, dtype=float)
    w_acc      = np.average(accs,   weights=weights)
    w_chance   = np.average(chances, weights=weights)

    above_chance = sum(1 for r in results if r["acc"] > r["chance"])

    print("\n" + "=" * 60)
    print("WLASL SIGNER IDENTITY PROBE – SUMMARY (all pose, no betas)")
    print("=" * 60)
    print(f"  Glosses evaluated         : {len(results)}")
    print(f"  Instances total           : {sum(ns)}")
    print(f"  Glosses above chance      : {above_chance} / {len(results)}")
    print()
    print(f"  Mean accuracy (unweighted): {mean_acc:.4f}")
    print(f"  Mean chance   (unweighted): {mean_chance:.4f}")
    print(f"  Mean ratio                : {mean_ratio:.2f}x")
    print()
    print(f"  Mean accuracy (weighted)  : {w_acc:.4f}")
    print(f"  Mean chance   (weighted)  : {w_chance:.4f}")
    print(f"  Weighted ratio            : {w_acc / w_chance:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
