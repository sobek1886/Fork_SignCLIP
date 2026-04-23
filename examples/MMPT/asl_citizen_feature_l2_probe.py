#!/usr/bin/env python3
"""
asl_citizen_feature_l2_probe.py  –  L2-distance signer-bias probe on ASL-Citizen
                                     appearance features (Logos / I3D).

For each gloss that has videos from ≥ 2 signers:
  - Compute a mean feature vector per (gloss, signer) pair.
  - Intra-signer L2:  mean pairwise distance between individual videos and
                      their own signer-gloss prototype (within-signer spread).
  - Inter-signer L2:  mean pairwise distance between signer-gloss prototypes
                      of *different* signers (between-signer spread).

If features were already signer-invariant, inter-signer L2 ≈ 0.
We expect inter >> intra, confirming the features carry signer identity and
that a signer-invariance training intervention is warranted.

Feature files
  Named  {dataset_name}_{video_basename_no_ext}.npy  (shape T × D).
  Each file is mean-pooled to a single D-dim vector for this analysis.

Usage
  # Logos (768-dim)
  python asl_citizen_feature_l2_probe.py \\
      --feature_dir /home/psobecki/ASL_Citizen/logos_features \\
      --splits_dir  /home/psobecki/ASL_Citizen/splits \\
      --feature_name logos

  # I3D WLASL (1024-dim)
  python asl_citizen_feature_l2_probe.py \\
      --feature_dir /home/psobecki/ASL_Citizen/i3d_wlasl_features \\
      --splits_dir  /home/psobecki/ASL_Citizen/splits \\
      --feature_name i3d_wlasl

  # Quick sanity check (first 200 videos, test split only)
  python asl_citizen_feature_l2_probe.py \\
      --feature_dir /home/psobecki/ASL_Citizen/logos_features \\
      --splits_dir  /home/psobecki/ASL_Citizen/splits \\
      --feature_name logos --splits test --max_videos 200
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults (Snellius paths)
# ---------------------------------------------------------------------------
DEFAULT_LOGOS_DIR = "/home/psobecki/ASL_Citizen/logos_features"
DEFAULT_I3D_DIR   = "/home/psobecki/ASL_Citizen/i3d_wlasl_features"
DEFAULT_SPLITS    = "/home/psobecki/ASL_Citizen/splits"
DATASET_NAME      = "asl_citizen"

_SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metadata(splits_dir: Path, splits: list[str]) -> list[dict]:
    """Read CSV splits → list of {video_id, gloss, signer_id}."""
    records = []
    for split in splits:
        csv_path = splits_dir / _SPLIT_FILES[split]
        if not csv_path.exists():
            print(f"WARNING: split file not found: {csv_path}")
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                video_basename = os.path.splitext(row["Video file"])[0]
                feat_id = f"{DATASET_NAME}_{video_basename}"
                records.append({
                    "feat_id":    feat_id,
                    "gloss":      row["Gloss"].strip(),
                    "signer_id":  row["Participant ID"].strip(),
                })
    return records


def load_features(
    records: list[dict],
    feature_dir: Path,
    max_videos: int | None = None,
) -> list[dict]:
    """Load .npy files, mean-pool temporal axis → D-dim vector per video."""
    loaded = []
    missing = 0
    for rec in tqdm(records, desc="Loading .npy features"):
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
            feat = feat.mean(axis=0)                   # → (D,)
        elif feat.ndim != 1:
            missing += 1
            continue
        loaded.append({**rec, "feat": feat})
        if max_videos and len(loaded) >= max_videos:
            break

    print(f"Loaded {len(loaded)} / {len(records)} features  "
          f"({missing} missing .npy files)")
    return loaded


# ---------------------------------------------------------------------------
# L2 analysis
# ---------------------------------------------------------------------------

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def analyse(
    videos: list[dict],
    min_signers: int = 2,
    top_glosses: int = 20,
) -> dict:
    """
    Compute intra-signer and inter-signer L2 distances for each gloss.

    Returns dict with per-gloss results and aggregate statistics.
    """
    # Group by gloss → signer → list of feature vectors
    by_gloss: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for v in videos:
        by_gloss[v["gloss"]][v["signer_id"]].append(v["feat"])

    gloss_results = []

    for gloss, signer_dict in by_gloss.items():
        if len(signer_dict) < min_signers:
            continue

        # Signer-gloss prototypes (mean over multiple recordings)
        prototypes: dict[str, np.ndarray] = {}
        intra_dists: list[float] = []

        for sid, feats in signer_dict.items():
            proto = np.mean(feats, axis=0)
            prototypes[sid] = proto
            # Intra-signer spread: individual videos vs their own prototype
            for f in feats:
                intra_dists.append(l2(f, proto))

        # Inter-signer: pairwise L2 between prototypes of different signers
        signer_ids = sorted(prototypes.keys())
        inter_dists: list[float] = []
        for i in range(len(signer_ids)):
            for j in range(i + 1, len(signer_ids)):
                inter_dists.append(l2(prototypes[signer_ids[i]],
                                      prototypes[signer_ids[j]]))

        gloss_results.append({
            "gloss":        gloss,
            "n_signers":    len(signer_dict),
            "n_videos":     sum(len(fs) for fs in signer_dict.values()),
            "intra_mean":   float(np.mean(intra_dists)) if intra_dists else 0.0,
            "intra_std":    float(np.std(intra_dists))  if intra_dists else 0.0,
            "inter_mean":   float(np.mean(inter_dists)) if inter_dists else 0.0,
            "inter_std":    float(np.std(inter_dists))  if inter_dists else 0.0,
            "n_inter_pairs": len(inter_dists),
        })

    if not gloss_results:
        return {"gloss_results": [], "aggregate": None}

    # Aggregate over all glosses (weighted by number of inter-signer pairs)
    all_intra  = [r["intra_mean"] for r in gloss_results]
    all_inter  = [r["inter_mean"] for r in gloss_results]
    ratios     = [r["inter_mean"] / (r["intra_mean"] + 1e-8) for r in gloss_results]

    aggregate = {
        "n_glosses_evaluated":   len(gloss_results),
        "mean_intra_L2":         float(np.mean(all_intra)),
        "median_intra_L2":       float(np.median(all_intra)),
        "mean_inter_L2":         float(np.mean(all_inter)),
        "median_inter_L2":       float(np.median(all_inter)),
        "mean_ratio_inter_intra": float(np.mean(ratios)),
        "median_ratio":          float(np.median(ratios)),
        "pct_inter_gt_intra":    float(np.mean([r > 1.0 for r in ratios])) * 100,
    }

    return {
        "gloss_results": gloss_results,
        "aggregate":     aggregate,
        "top_glosses":   top_glosses,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(result: dict, feature_name: str, feat_dim: int):
    gloss_results = result["gloss_results"]
    agg           = result["aggregate"]
    top_n         = result.get("top_glosses", 20)

    W = 68
    print("\n" + "=" * W)
    print(f"ASL-CITIZEN  {feature_name.upper()}  FEATURE  L2-DISTANCE PROBE")
    print("=" * W)

    if agg is None:
        print("  No glosses met the minimum-signers criterion.")
        return

    print(f"\n  Feature dim  : {feat_dim}")
    print(f"  Glosses      : {agg['n_glosses_evaluated']} with ≥ 2 signers")
    print()
    print(f"  ── Intra-signer L2  (video vs own signer-gloss prototype) ──")
    print(f"     mean   {agg['mean_intra_L2']:.4f}")
    print(f"     median {agg['median_intra_L2']:.4f}")
    print()
    print(f"  ── Inter-signer L2  (prototype vs prototype, different signer) ──")
    print(f"     mean   {agg['mean_inter_L2']:.4f}")
    print(f"     median {agg['median_inter_L2']:.4f}")
    print()
    print(f"  ── Ratio  inter / intra ──")
    print(f"     mean   {agg['mean_ratio_inter_intra']:.2f}x")
    print(f"     median {agg['median_ratio']:.2f}x")
    print(f"     % glosses where inter > intra: {agg['pct_inter_gt_intra']:.1f}%")

    # Verdict
    ratio = agg["mean_ratio_inter_intra"]
    if agg["mean_inter_L2"] < 1e-3:
        verdict = "INVARIANT: inter-signer L2 ≈ 0 — features collapse across signers."
    elif ratio > 2.0:
        verdict = "STRONG signer signal: features clearly differ across signers."
    elif ratio > 1.3:
        verdict = "MODERATE signer signal: some signer-specific variation."
    else:
        verdict = "WEAK signer signal: inter- ≈ intra-signer L2."
    print(f"\n  Verdict: {verdict}")

    # Per-gloss table (sorted by inter_mean desc, capped at top_n)
    sorted_glosses = sorted(gloss_results, key=lambda r: -r["inter_mean"])[:top_n]
    print(f"\n  Top-{top_n} glosses by inter-signer L2:")
    print(f"  {'Gloss':<22}  {'#sig':>4}  {'#vid':>4}  "
          f"{'intra':>8}  {'inter':>8}  {'ratio':>6}")
    print("  " + "-" * 62)
    for r in sorted_glosses:
        ratio_g = r["inter_mean"] / (r["intra_mean"] + 1e-8)
        print(f"  {r['gloss']:<22}  {r['n_signers']:>4}  {r['n_videos']:>4}  "
              f"{r['intra_mean']:>8.4f}  {r['inter_mean']:>8.4f}  {ratio_g:>6.2f}x")
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
                        help="Directory containing .npy feature files "
                             "(default: logos path on Snellius)")
    parser.add_argument("--splits_dir", type=Path, default=Path(DEFAULT_SPLITS),
                        help=f"Directory with train/val/test CSV splits "
                             f"(default: {DEFAULT_SPLITS})")
    parser.add_argument("--feature_name", type=str, default="logos",
                        help="Label for the feature type, e.g. logos or i3d_wlasl "
                             "(used for display and default --feature_dir)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"],
                        help="Which CSV splits to include (default: all three)")
    parser.add_argument("--min_signers", type=int, default=2,
                        help="Minimum distinct signers per gloss to include it "
                             "in the analysis (default: 2)")
    parser.add_argument("--top_glosses", type=int, default=20,
                        help="Number of per-gloss rows to print (default: 20)")
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME,
                        help=f"Dataset prefix used in .npy filenames "
                             f"(default: {DATASET_NAME})")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Stop after loading this many videos (quick sanity check)")
    args = parser.parse_args()

    # Resolve default feature_dir based on feature_name
    if args.feature_dir is None:
        if "logos" in args.feature_name.lower():
            args.feature_dir = Path(DEFAULT_LOGOS_DIR)
        else:
            args.feature_dir = Path(DEFAULT_I3D_DIR)

    if not args.feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {args.feature_dir}")
    if not args.splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {args.splits_dir}")

    DATASET_NAME = args.dataset_name

    print(f"Feature dir : {args.feature_dir}")
    print(f"Splits dir  : {args.splits_dir}")
    print(f"Splits used : {args.splits}")

    # Load metadata
    records = load_metadata(args.splits_dir, args.splits)
    print(f"Metadata records: {len(records)}")

    # Load features
    videos = load_features(records, args.feature_dir, max_videos=args.max_videos)
    if not videos:
        raise RuntimeError("No features loaded – check paths and .npy filenames.")

    feat_dim = videos[0]["feat"].shape[0]
    print(f"Feature dimension: {feat_dim}")

    # Analyse
    result = analyse(videos, min_signers=args.min_signers,
                     top_glosses=args.top_glosses)

    print_results(result, args.feature_name, feat_dim)


if __name__ == "__main__":
    main()
