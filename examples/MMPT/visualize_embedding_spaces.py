#!/usr/bin/env python3
"""
visualize_embedding_spaces.py - t-SNE (and optionally UMAP) comparison of
                                ASL-Citizen embedding spaces, colored by gloss
                                and by signer, plus scalar cluster-quality
                                summaries (silhouette scores).

Companion to the L2 probes (asl_citizen_feature_l2_probe.py /
asl_citizen_signclip_l2_probe.py): the probes give one scalar per space; this
gives the qualitative picture of HOW each training condition reshapes the
space — does contrastive training on synthetic counterfactuals (SupCon-on-
pairs) collapse signer structure differently than plain augmentation-as-data?

Every space is a directory of {dataset_name}_{video_id}.npy files (T x D or D),
the common currency of this project: raw Logos features, backbone-FT
re-extractions (logos_features_run2 etc.), and SignCLIP pooled_video exports
(export_signclip_features.py). Features are mean-pooled per video and
L2-normalized, exactly as in eval_asl_citizen_retrieval.py.

Outputs (into --out_dir):
  embedding_spaces_by_gloss.png   one t-SNE panel per space, colored by gloss
  embedding_spaces_by_signer.png  same projections, colored by signer
  embedding_spaces_summary.json   silhouette(gloss) and silhouette(signer) per
                                  space in the full-dimensional space — the
                                  quantitative counterpart of the plots

Usage (on Snellius)
  python visualize_embedding_spaces.py \
      --space raw=/scratch-shared/psobecki/ASL_Citizen/logos_features_native \
      --space ft_ce=/scratch-shared/psobecki/ASL_Citizen/logos_features_run2 \
      --space signclip_scratch=/scratch-shared/psobecki/ASL_Citizen/signclip_features_scratch \
      --space pair_k5=/scratch-shared/psobecki/ASL_Citizen/signclip_features_pair_k5 \
      --splits_dir /home/psobecki/ASL_Citizen/splits \
      --split train --n_glosses 12 --n_per_gloss 20 \
      --out_dir runs/embedding_viz
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from eval_asl_citizen_retrieval import load_metadata, DEFAULT_SPLITS, _SPLIT_FILES


def parse_space(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--space must be NAME=DIR, got {s!r}")
    name, d = s.split("=", 1)
    return name, Path(d)


def load_signer_map(splits_dir: Path, split: str) -> dict:
    """feat_id → Participant ID, from the same CSV load_metadata reads."""
    signer = {}
    with open(splits_dir / _SPLIT_FILES[split], newline="") as f:
        for row in csv.DictReader(f):
            basename = os.path.splitext(row["Video file"])[0]
            signer[f"asl_citizen_{basename}"] = str(row["Participant ID"]).strip()
    return signer


def load_space_aligned(sample_recs, feat_dir):
    """Load + mean-pool features for sample_recs, keeping ALIGNED indices.

    Returns (feats (N,D) L2-normalized, kept_indices) — records whose .npy is
    missing/invalid are dropped from BOTH, so label arrays can be subset with
    kept_indices and stay aligned (load_features drops rows silently, which
    would misalign the signer labels — hence this local loader)."""
    feats, kept = [], []
    for i, rec in enumerate(sample_recs):
        try:
            f = np.load(Path(feat_dir) / (rec["feat_id"] + ".npy")).astype(np.float32)
        except Exception:
            continue
        if f.ndim == 2:
            if f.shape[0] == 0:
                continue
            f = f.mean(axis=0)
        feats.append(f)
        kept.append(i)
    feats = np.stack(feats)
    feats = F.normalize(torch.from_numpy(feats).float(), dim=-1).numpy()
    return feats, np.array(kept)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", action="append", type=parse_space, required=True,
                    help="NAME=FEATURE_DIR; repeat for each space to compare")
    ap.add_argument("--splits_dir", type=Path, default=Path(DEFAULT_SPLITS))
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--n_glosses", type=int, default=12,
                    help="glosses sampled for the plots (most-frequent first, "
                         "deterministic across spaces)")
    ap.add_argument("--n_per_gloss", type=int, default=20)
    ap.add_argument("--umap", action="store_true",
                    help="also produce UMAP panels if umap-learn is installed")
    ap.add_argument("--load_workers", type=int, default=32)
    ap.add_argument("--out_dir", type=Path, default=Path("runs/embedding_viz"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = load_metadata(args.splits_dir, args.split)
    signer_map = load_signer_map(args.splits_dir, args.split)

    # ---- choose a deterministic gloss/video sample shared by ALL spaces ----
    from collections import Counter, defaultdict
    gloss_counts = Counter(r["gloss"] for r in recs)
    chosen_glosses = [g for g, _ in gloss_counts.most_common(args.n_glosses)]
    rng = np.random.default_rng(0)
    by_gloss = defaultdict(list)
    for r in recs:
        if r["gloss"] in chosen_glosses:
            by_gloss[r["gloss"]].append(r)
    sample_recs = []
    for g in chosen_glosses:
        idx = rng.permutation(len(by_gloss[g]))[:args.n_per_gloss]
        sample_recs += [by_gloss[g][i] for i in idx]
    all_glosses = np.array([r["gloss"] for r in sample_recs])
    all_signers = np.array([signer_map.get(r["feat_id"], "unknown")
                            for r in sample_recs])
    print(f"Sample: {len(sample_recs)} videos, {len(chosen_glosses)} glosses, "
          f"{len(set(all_signers.tolist()))} signers")

    # ---- load each space on the identical sample, project, score ----
    # Per-space kept-index bookkeeping: a space may be missing a few .npy files
    # (e.g. SignCLIP exports on a subset); labels are subset per space so they
    # stay aligned with the loaded features.
    spaces = dict(args.space)
    projections, labels_per_space, summary = {}, {}, {}
    for name, feat_dir in spaces.items():
        print(f"\n=== Space: {name} ({feat_dir}) ===")
        feats, kept = load_space_aligned(sample_recs, feat_dir)
        glosses = all_glosses[kept]
        signers = all_signers[kept]
        labels_per_space[name] = (glosses, signers)
        sil_gloss = float(silhouette_score(feats, glosses))
        sil_signer = float(silhouette_score(feats, signers))
        summary[name] = {"silhouette_gloss": sil_gloss,
                         "silhouette_signer": sil_signer,
                         "n_videos": int(feats.shape[0]),
                         "n_missing": int(len(sample_recs) - feats.shape[0]),
                         "dim": int(feats.shape[1])}
        print(f"  {feats.shape[0]}/{len(sample_recs)} videos loaded, "
              f"silhouette(gloss)={sil_gloss:.4f}  "
              f"silhouette(signer)={sil_signer:.4f}")
        projections[name] = {"tsne": TSNE(
            n_components=2, perplexity=15, init="pca",
            random_state=0).fit_transform(feats)}
        if args.umap:
            try:
                import umap
                projections[name]["umap"] = umap.UMAP(
                    n_components=2, random_state=0).fit_transform(feats)
            except ImportError:
                print("  umap-learn not installed — skipping UMAP")

    with open(args.out_dir / "embedding_spaces_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_dir / 'embedding_spaces_summary.json'}")

    # ---- grid figures: one panel per space, colored by gloss / by signer ----
    def grid_figure(label_kind, title, out_png, method="tsne"):
        names = [n for n in projections if method in projections[n]]
        if not names:
            return
        ncols = min(3, len(names))
        nrows = (len(names) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4.4 * nrows), squeeze=False)
        label_idx = 0 if label_kind == "gloss" else 1
        uniq = sorted({lab for n in names
                       for lab in labels_per_space[n][label_idx].tolist()})
        cmap = plt.get_cmap("tab20")
        for ax_i, name in enumerate(names):
            ax = axes[ax_i // ncols][ax_i % ncols]
            proj = projections[name][method]
            color_labels = labels_per_space[name][label_idx]
            for ci, lab in enumerate(uniq):
                m = color_labels == lab
                if not m.any():
                    continue
                ax.scatter(proj[m, 0], proj[m, 1], s=14, alpha=0.65,
                           color=cmap(ci % 20), label=str(lab))
            ax.set_title(f"{name}  "
                         f"(sil_g={summary[name]['silhouette_gloss']:.2f}, "
                         f"sil_s={summary[name]['silhouette_signer']:.2f})",
                         fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(len(names), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(uniq), 10), fontsize=7, frameon=False)
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_png}")

    grid_figure("gloss", f"{args.split} split — colored by GLOSS (t-SNE)",
                args.out_dir / "embedding_spaces_by_gloss.png")
    grid_figure("signer", f"{args.split} split — colored by SIGNER (t-SNE)",
                args.out_dir / "embedding_spaces_by_signer.png")
    if args.umap:
        grid_figure("gloss", f"{args.split} split — colored by GLOSS (UMAP)",
                    args.out_dir / "embedding_spaces_by_gloss_umap.png",
                    method="umap")
        grid_figure("signer", f"{args.split} split — colored by SIGNER (UMAP)",
                    args.out_dir / "embedding_spaces_by_signer_umap.png",
                    method="umap")


if __name__ == "__main__":
    main()
