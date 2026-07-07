#!/usr/bin/env python3
"""
king_queen_analogy.py - Reproduce SignCLIP's Figure 2 (King - Man + Woman = Queen)
                         analogy check, but on raw, frozen Logos features that never
                         saw any contrastive text-alignment training. Tests whether
                         this kind of distributional-semantic structure requires
                         language-alignment training, or falls out of a purely
                         supervised visual backbone for free.

Two checks, run on BOTH ASL-Citizen WOMAN variants (WOMAN1 = plain "WOMAN.mp4",
WOMAN2 = "WOMAN 2.mp4"):

  1. Quantitative (the real test): is QUEEN's per-gloss mean feature vector the
     nearest neighbor, by cosine similarity, to (KING - MAN + WOMAN) among ALL
     gloss centers in the split? A 2D t-SNE projection can visually suggest
     structure that a full-dimensional nearest-neighbor check shows isn't there.
  2. Visual: a t-SNE plot in the style of the paper's Figure 2, for direct
     side-by-side comparison.

Reuses eval_asl_citizen_retrieval.load_metadata/load_features so features are
loaded identically (same mean-pooling over clip frames) to every other reported
number in this project.

Usage
  python king_queen_analogy.py \
      --feature_dir /scratch-shared/psobecki/ASL_Citizen/logos_features_native \
      --splits_dir  /home/psobecki/ASL_Citizen/splits \
      --split train \
      --out_png king_queen_analogy.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from eval_asl_citizen_retrieval import load_metadata, load_features, DEFAULT_SPLITS

TARGET_GLOSSES = ["KING", "MAN", "WOMAN1", "WOMAN2", "QUEEN"]
ANALOGY_PAIRS = [("WOMAN1", "King - Man + Woman1"), ("WOMAN2", "King - Man + Woman2")]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature_dir", type=Path,
                     default=Path("/scratch-shared/psobecki/ASL_Citizen/logos_features_native"))
    ap.add_argument("--splits_dir", type=Path, default=Path(DEFAULT_SPLITS))
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--n_examples_per_gloss", type=int, default=14,
                     help="matches the paper's Figure 2 (14 videos/sign)")
    ap.add_argument("--out_png", type=Path, default=Path("king_queen_analogy.png"))
    ap.add_argument("--load_workers", type=int, default=32)
    args = ap.parse_args()

    print(f"Loading {args.split} metadata/features from {args.feature_dir} ...")
    recs = load_metadata(args.splits_dir, args.split)
    feats, glosses = load_features(
        recs, args.feature_dir, None,
        desc=f"Loading {args.split} features", workers=args.load_workers,
        min_coverage=0, allow_partial=True)
    glosses = np.array(glosses)
    print(f"Loaded {feats.shape[0]} videos, {feats.shape[1]}-dim features, "
          f"{len(set(glosses.tolist()))} unique glosses.")

    # L2-normalize once so cosine similarity == plain dot product from here on.
    feats = F.normalize(torch.from_numpy(feats).float(), dim=-1).numpy()

    # ---- 1. Per-gloss cluster centers for the WHOLE vocabulary (for the ranked NN test) ----
    unique_glosses = sorted(set(glosses.tolist()))
    centers = np.stack([feats[glosses == g].mean(axis=0) for g in unique_glosses])
    centers = F.normalize(torch.from_numpy(centers).float(), dim=-1).numpy()
    gloss_to_idx = {g: i for i, g in enumerate(unique_glosses)}

    missing = [g for g in TARGET_GLOSSES if g not in gloss_to_idx]
    if missing:
        raise SystemExit(f"Missing glosses in {args.split} split: {missing}")

    king = centers[gloss_to_idx["KING"]]
    man = centers[gloss_to_idx["MAN"]]

    print(f"\n=== Quantitative check: rank of QUEEN among all {len(unique_glosses)} "
          "gloss centers, by cosine similarity to (King - Man + Woman) ===")
    for woman_gloss, label in ANALOGY_PAIRS:
        woman = centers[gloss_to_idx[woman_gloss]]
        pred = king - man + woman
        pred = pred / np.linalg.norm(pred)
        sims = centers @ pred  # cosine similarity to every gloss center
        order = np.argsort(-sims)
        rank = int(np.where(order == gloss_to_idx["QUEEN"])[0][0])  # 0-indexed, 0 = nearest
        sim_to_queen = float(sims[gloss_to_idx["QUEEN"]])
        top5 = [(unique_glosses[i], round(float(sims[i]), 4)) for i in order[:5]]
        print(f"\n  {label}:")
        print(f"    cosine(pred, QUEEN)               = {sim_to_queen:.4f}")
        print(f"    QUEEN rank among {len(unique_glosses)} glosses = {rank} (0 = nearest)")
        print(f"    top-5 nearest to predicted vector  = {top5}")

    # ---- 2. t-SNE plot, paper-Figure-2 style ----
    print(f"\nBuilding t-SNE plot ({args.n_examples_per_gloss} examples/gloss) -> {args.out_png}")
    rng = np.random.default_rng(0)
    sampled_feats, sampled_labels = [], []
    for g in TARGET_GLOSSES:
        idx = np.where(glosses == g)[0]
        take = rng.choice(idx, size=min(args.n_examples_per_gloss, len(idx)), replace=False)
        sampled_feats.append(feats[take])
        sampled_labels += [g] * len(take)
    sampled_feats = np.concatenate(sampled_feats, axis=0)
    sampled_labels = np.array(sampled_labels)

    proj = TSNE(n_components=2, perplexity=15, init="pca",
                random_state=0).fit_transform(sampled_feats)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"KING": "tab:red", "MAN": "tab:blue", "WOMAN1": "tab:green",
              "WOMAN2": "tab:orange", "QUEEN": "tab:purple"}
    markers = {"KING": "s", "MAN": "^", "WOMAN1": "o", "WOMAN2": "D", "QUEEN": "v"}
    for g in TARGET_GLOSSES:
        m = sampled_labels == g
        ax.scatter(proj[m, 0], proj[m, 1], c=colors[g], marker=markers[g],
                   s=30, alpha=0.6, label=g)
        cx, cy = proj[m, 0].mean(), proj[m, 1].mean()
        ax.scatter([cx], [cy], c=colors[g], marker=markers[g], s=250,
                   edgecolors="black", linewidths=1.5, zorder=5)
    ax.legend()
    ax.set_title("Logos features: King / Man / Woman / Queen (t-SNE, perplexity=15)")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved plot to {args.out_png}")


if __name__ == "__main__":
    main()
