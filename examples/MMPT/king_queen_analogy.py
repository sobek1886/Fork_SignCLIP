#!/usr/bin/env python3
"""
king_queen_analogy.py - Analogy structure (KING - MAN + WOMAN = QUEEN) in sign
                         embedding spaces.

Originally: reproduce SignCLIP's Figure 2 analogy check on raw, frozen Logos
features (does distributional-semantic structure require language-alignment
training, or fall out of a supervised visual backbone for free?).

EXTENDED (2026-07-11) to separate two hypotheses the original check confounds:
  H-visual:   QUEEN lands near (KING - MAN + WOMAN) merely because the QUEEN
              and KING signs are visually/motion-similar (same handshape at
              the head, mirrored path), so anything near KING is near QUEEN.
  H-semantic: the space encodes the gendered-pair relation itself, so applying
              the (WOMAN - MAN) offset moves you toward QUEEN *beyond* the
              similarity KING already has to QUEEN.

Design:
  1. A battery of gendered pairs that VARY in visual similarity — KING/QUEEN
     (and PRINCE/PRINCESS when available) as visually-similar pairs,
     MAN/WOMAN1, MAN/WOMAN2 and BOY/GIRL as visually-dissimilar pairs.
     Availability of each gloss is checked against the split at runtime;
     missing glosses are skipped with a warning.
     FIX 2026-07-19: the vocabulary spells the WOMAN variants WOMAN1/WOMAN2;
     earlier revisions queried "WOMAN"/"WOMAN 2" and silently dropped both
     MAN pairs (the 2026-07-19 multi_space.json therefore covers only
     KING/QUEEN and BOY/GIRL). PRINCESS is confirmed absent from the
     2,731-gloss vocabulary. Rerun needed for the enlarged battery.

ICONICITY EXTENSION (2026-07-19): see iconicity_analysis.py for the
ASL-LEX-based high- vs low-iconicity contrast and the gender-offset vs
location-direction probe; it reuses this script's --space convention.
  2. Cross-pair offset transfer: for source pair (mA, fA) and target pair
     (mB, fB), pred = c(mB) + (c(fA) - c(mA)). Report, per space:
       - rank(fB | pred) among all gloss centers      (the analogy test)
       - delta_cos = cos(pred, fB) - cos(c(mB), fB)   (the H-visual control:
         positive = the gender offset moves toward the target BEYOND the
         male anchor's own similarity; H-visual alone predicts ~0)
       - rank_gain = rank(fB | c(mB)) - rank(fB | pred)  (same control in
         rank terms; positive = offset helps)
       - cos(c(mB), c(fB))                            (visual-similarity
         covariate: in the RAW space this measures how visually alike the
         pair is; read delta_cos against it)
  3. Offset consistency: mean pairwise cosine among normalized (f - m)
     offsets across pairs — does a shared "gender direction" exist at all?
  4. Multiple spaces via --space NAME=FEATURE_DIR (raw Logos native,
     backbone-FT re-extractions, and SignCLIP variants via
     export_signclip_features.py exports). If no --space is given, falls back
     to the original single-space behavior on --feature_dir.

If SignCLIP's text alignment adds semantic structure beyond visual
similarity, delta_cos/rank_gain should be larger in text-aligned spaces than
in raw Logos, and the gap should be largest for visually-DISSIMILAR pairs.

Outputs: printed per-space tables, a JSON dump (--out_json), and the original
Figure-2-style t-SNE per space ({out_png stem}_{space}.png).

Usage (original, single space)
  python king_queen_analogy.py \
      --feature_dir /scratch-shared/psobecki/ASL_Citizen/logos_features_native \
      --splits_dir  /home/psobecki/ASL_Citizen/splits \
      --split train --out_png king_queen_analogy.png

Usage (multi-space comparison; export SignCLIP spaces first with
export_signclip_features.py)
  python king_queen_analogy.py \
      --space raw=/scratch-shared/psobecki/ASL_Citizen/logos_features_native \
      --space signclip_scratch=/scratch-shared/psobecki/ASL_Citizen/signclip_features_scratch \
      --space signclip_aug_ft=/scratch-shared/psobecki/ASL_Citizen/signclip_features_aug_ft \
      --space pair_k5=/scratch-shared/psobecki/ASL_Citizen/signclip_features_pair_k5 \
      --splits_dir /home/psobecki/ASL_Citizen/splits --split train \
      --out_json runs/king_queen_analogy/multi_space.json \
      --out_png  runs/king_queen_analogy/analogy.png
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from eval_asl_citizen_retrieval import load_metadata, load_features, DEFAULT_SPLITS

# Gendered-pair battery: (male gloss, female gloss, visual-similarity prior).
# The prior is a human label used only for reading the results, not computed.
#
# Gloss names verified against the ASL Citizen train split (2026-07-19):
# the vocabulary uses WOMAN1 / WOMAN2 (two genuine lexical variants of
# WOMAN with different movement, per ASL Citizen's variant-numbering
# convention, cf. NIGHT1/NIGHT2, DOG1/DOG2), not "WOMAN" / "WOMAN 2" as
# earlier revisions assumed - which is why the MAN/WOMAN pairs silently
# dropped out of the 2026-07-19 run. PRINCE exists but PRINCESS is
# genuinely absent from the 2,731-gloss vocabulary, so that pair stays
# listed only to make the skip explicit in the logs.
DEFAULT_PAIRS = [
    ("KING", "QUEEN", "visually-similar"),
    ("PRINCE", "PRINCESS", "visually-similar"),   # PRINCESS absent: skipped
    ("MAN", "WOMAN1", "visually-dissimilar"),
    ("MAN", "WOMAN2", "visually-dissimilar"),     # second WOMAN variant
    ("BOY", "GIRL", "visually-dissimilar"),
]

# Aliases so earlier spellings keep working in --pairs / --tsne_glosses.
ALIASES = {"WOMAN": "WOMAN1", "WOMAN 2": "WOMAN2"}


def parse_space(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--space must be NAME=DIR, got {s!r}")
    name, d = s.split("=", 1)
    return name, Path(d)


def build_centers(feats, glosses):
    """Per-gloss L2-normalized mean vectors + index map."""
    unique_glosses = sorted(set(glosses.tolist()))
    centers = np.stack([feats[glosses == g].mean(axis=0) for g in unique_glosses])
    centers = F.normalize(torch.from_numpy(centers).float(), dim=-1).numpy()
    return centers, unique_glosses, {g: i for i, g in enumerate(unique_glosses)}


def rank_of(centers, query_vec, target_idx):
    """0-indexed rank of target among all centers by cosine to query."""
    q = query_vec / np.linalg.norm(query_vec)
    sims = centers @ q
    order = np.argsort(-sims)
    return int(np.where(order == target_idx)[0][0]), sims


def analyze_space(space_name, feats, glosses, pairs, out):
    feats = F.normalize(torch.from_numpy(feats).float(), dim=-1).numpy()
    glosses = np.array(glosses)
    centers, unique_glosses, g2i = build_centers(feats, glosses)
    n_gloss = len(unique_glosses)

    avail = [(m, f, tag) for (m, f, tag) in pairs
             if m in g2i and f in g2i]
    skipped = [(m, f) for (m, f, _) in pairs if m not in g2i or f not in g2i]
    for m, f in skipped:
        print(f"  SKIP pair ({m}, {f}) — gloss not in this split")
    if len(avail) < 2:
        print(f"  Not enough available pairs in space {space_name}; need >= 2")
        out[space_name] = {"error": "fewer than 2 available pairs"}
        return

    space_result = {"n_glosses": n_gloss, "pairs": {}, "transfers": []}

    # 1. within-pair similarity (visual-similarity covariate in the raw space)
    print(f"\n  Within-pair centroid cosine ({space_name}):")
    for m, f, tag in avail:
        c = float(centers[g2i[m]] @ centers[g2i[f]])
        space_result["pairs"][f"{m}|{f}"] = {"cos": c, "prior": tag}
        print(f"    cos({m:>8}, {f:<10}) = {c:.4f}   [{tag}]")

    # 2. cross-pair offset transfer with the H-visual control
    print(f"\n  Offset transfer (source offset applied to target male anchor):")
    print(f"    {'source':>18} -> {'target':<18} "
          f"{'rank(pred)':>10} {'rank(anchor)':>12} {'rank_gain':>9} "
          f"{'d_cos':>8}")
    for (mA, fA, _tA) in avail:
        offset = centers[g2i[fA]] - centers[g2i[mA]]
        for (mB, fB, _tB) in avail:
            if (mA, fA) == (mB, fB):
                continue
            pred = centers[g2i[mB]] + offset
            r_pred, sims_pred = rank_of(centers, pred, g2i[fB])
            r_anchor, _ = rank_of(centers, centers[g2i[mB]], g2i[fB])
            d_cos = float(sims_pred[g2i[fB]]
                          - centers[g2i[mB]] @ centers[g2i[fB]])
            rec = {
                "source": [mA, fA], "target": [mB, fB],
                "rank_pred": r_pred, "rank_anchor": r_anchor,
                "rank_gain": r_anchor - r_pred, "delta_cos": d_cos,
                "cos_pred_target": float(sims_pred[g2i[fB]]),
            }
            space_result["transfers"].append(rec)
            print(f"    {mA + '-' + fA:>18} -> {mB + '-' + fB:<18} "
                  f"{r_pred:>10} {r_anchor:>12} {r_anchor - r_pred:>9} "
                  f"{d_cos:>8.4f}")

    # 3. offset consistency across pairs
    offsets = []
    for m, f, _ in avail:
        o = centers[g2i[f]] - centers[g2i[m]]
        offsets.append(o / (np.linalg.norm(o) + 1e-12))
    cons = [float(a @ b) for a, b in combinations(offsets, 2)]
    space_result["offset_consistency_mean"] = float(np.mean(cons))
    space_result["offset_consistency_all"] = cons
    print(f"\n  Offset consistency (mean pairwise cosine of gender offsets): "
          f"{np.mean(cons):.4f}")

    # 4. the original KING - MAN + WOMAN check, kept verbatim for continuity
    for woman_gloss in ("WOMAN1", "WOMAN2"):
        if all(g in g2i for g in ("KING", "MAN", woman_gloss, "QUEEN")):
            pred = (centers[g2i["KING"]] - centers[g2i["MAN"]]
                    + centers[g2i[woman_gloss]])
            r, sims = rank_of(centers, pred, g2i["QUEEN"])
            order = np.argsort(-sims)
            top5 = [(unique_glosses[i], round(float(sims[i]), 4))
                    for i in order[:5]]
            key = f"king_man_{woman_gloss.replace(' ', '')}"
            space_result[key] = {"queen_rank": r, "top5": top5,
                                 "cos_to_queen": float(sims[g2i['QUEEN']])}
            print(f"\n  King - Man + {woman_gloss}: QUEEN rank {r} of "
                  f"{n_gloss} (0=nearest), cos={sims[g2i['QUEEN']]:.4f}")
            print(f"    top-5: {top5}")

    out[space_name] = space_result


def tsne_plot(space_name, feats, glosses, target_glosses, n_per_gloss, out_png):
    feats = F.normalize(torch.from_numpy(feats).float(), dim=-1).numpy()
    glosses = np.array(glosses)
    present = [g for g in target_glosses if (glosses == g).any()]
    rng = np.random.default_rng(0)
    sampled_feats, sampled_labels = [], []
    for g in present:
        idx = np.where(glosses == g)[0]
        take = rng.choice(idx, size=min(n_per_gloss, len(idx)), replace=False)
        sampled_feats.append(feats[take])
        sampled_labels += [g] * len(take)
    sampled_feats = np.concatenate(sampled_feats, axis=0)
    sampled_labels = np.array(sampled_labels)

    proj = TSNE(n_components=2, perplexity=15, init="pca",
                random_state=0).fit_transform(sampled_feats)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = plt.get_cmap("tab10")
    markers = ["s", "^", "o", "D", "v", "P", "X", "*", "<", ">"]
    for i, g in enumerate(present):
        m = sampled_labels == g
        ax.scatter(proj[m, 0], proj[m, 1], color=cmap(i % 10),
                   marker=markers[i % len(markers)], s=30, alpha=0.6, label=g)
        cx, cy = proj[m, 0].mean(), proj[m, 1].mean()
        ax.scatter([cx], [cy], color=cmap(i % 10),
                   marker=markers[i % len(markers)], s=250,
                   edgecolors="black", linewidths=1.5, zorder=5)
    ax.legend(fontsize=8)
    ax.set_title(f"{space_name}: gendered-pair glosses (t-SNE, perplexity=15)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved t-SNE plot to {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature_dir", type=Path,
                    default=Path("/scratch-shared/psobecki/ASL_Citizen/logos_features_native"),
                    help="single-space fallback when no --space is given")
    ap.add_argument("--space", action="append", type=parse_space, default=None,
                    help="NAME=FEATURE_DIR; repeat to compare spaces "
                         "(SignCLIP spaces via export_signclip_features.py)")
    ap.add_argument("--splits_dir", type=Path, default=Path(DEFAULT_SPLITS))
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--pairs", type=str, default=None,
                    help="override battery: 'KING:QUEEN,MAN:WOMAN,...' "
                         "(default: built-in battery incl. PRINCE/PRINCESS "
                         "when present)")
    ap.add_argument("--n_examples_per_gloss", type=int, default=14,
                    help="matches the paper's Figure 2 (14 videos/sign)")
    ap.add_argument("--out_png", type=Path, default=Path("king_queen_analogy.png"))
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--load_workers", type=int, default=32)
    args = ap.parse_args()

    if args.pairs:
        pairs = []
        for p in args.pairs.split(","):
            m, f = p.split(":")
            m = ALIASES.get(m.strip(), m.strip())
            f = ALIASES.get(f.strip(), f.strip())
            pairs.append((m, f, "user-specified"))
    else:
        pairs = DEFAULT_PAIRS

    spaces = dict(args.space) if args.space else {"logos_native": args.feature_dir}
    recs = load_metadata(args.splits_dir, args.split)
    battery_glosses = sorted({g for m, f, _ in pairs for g in (m, f)})

    results = {}
    for name, feat_dir in spaces.items():
        print(f"\n================ Space: {name} ({feat_dir}) ================")
        feats, glosses = load_features(
            recs, feat_dir, None, desc=f"Loading {name}",
            workers=args.load_workers, min_coverage=0, allow_partial=True)
        print(f"Loaded {feats.shape[0]} videos, {feats.shape[1]}-dim, "
              f"{len(set(glosses))} unique glosses.")
        analyze_space(name, feats, glosses, pairs, results)
        out_png = args.out_png.with_name(
            f"{args.out_png.stem}_{name}{args.out_png.suffix}")
        tsne_plot(name, feats, glosses, battery_glosses,
                  args.n_examples_per_gloss, out_png)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
