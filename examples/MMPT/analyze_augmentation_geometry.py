#!/usr/bin/env python3
"""
analyze_augmentation_geometry.py - What do the Flux appearance edits DO to the
                                    feature space? (thesis analyses S1 + S3)

S1 — displacement-direction analysis. For every video with augmented copies,
compute the displacement Delta_v = f(aug_v) - f(orig) per variant type, in one
or more embedding spaces. If the Delta vectors of a variant share a direction
across videos (high mean pairwise cosine, dominant first PCA component), that
variant is a COHERENT APPEARANCE AXIS the space encodes — and a working
invariance objective should selectively collapse it (smaller ||Delta||, lower
direction consistency) relative to the frozen space. If Delta is isotropic
noise, "appearance" is not a learnable direction and augmentation can only act
as generic data noise. This upgrades diagnose_appearance_invariance.py's
scalar cosine to a mechanism-level readout.

S3 — counterfactual confusion test. For each augmented feature: (a) does 1-NN
retrieval over the ORIGINAL gallery (own source video excluded) still find the
source's gloss (motion retained)?  (b) does its nearest per-signer prototype
flip away from the source video's nearest prototype (appearance moved)?
Expected ordering if the edits work as designed: gloss retention high for all
variants; signer-flip rate low for shirt/glasses, higher for signer_swap/skin.
A space trained for invariance should keep (a) and suppress (b).

Feature pairing is discovered by filename: {dataset}_{id}.npy (original) and
{dataset}_{id}_{variant}.npy (augment) in the SAME directory — the layout the
extraction jobs produce. Spaces without variant files are skipped with a note
(e.g. backbone-FT re-extractions that only contain originals).

Usage (on Snellius)
  python analyze_augmentation_geometry.py \
      --space frozen_logos=/home/psobecki/ASL_Citizen/logos_features \
      --space i3d_wlasl=/home/psobecki/ASL_Citizen/i3d_wlasl_features \
      --splits_dir /home/psobecki/ASL_Citizen/splits \
      --split train \
      --out_json runs/aug_geometry/aug_geometry.json
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_asl_citizen_per_signer import read_records, load_with_records

DEFAULT_VARIANTS = ["glasses", "shirt_1", "signer_swap", "skin_mst_diffusion"]


def parse_space(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--space must be NAME=DIR, got {s!r}")
    name, d = s.split("=", 1)
    return name, Path(d)


def pooled(path):
    f = np.load(path).astype(np.float32)
    if f.ndim == 2:
        if f.shape[0] == 0:
            raise ValueError("empty feature")
        f = f.mean(axis=0)
    return f


def l2n(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def discover_pairs(feat_dir: Path, variants):
    """variant → list of (orig_path, aug_path, feat_id). Pure filename logic."""
    pairs = defaultdict(list)
    suffixes = {v: f"_{v}" for v in variants}
    for p in feat_dir.glob("*.npy"):
        stem = p.stem
        for v, suf in suffixes.items():
            if stem.endswith(suf):
                orig = feat_dir / (stem[: -len(suf)] + ".npy")
                if orig.exists():
                    pairs[v].append((orig, p, stem[: -len(suf)]))
                break
    return pairs


def direction_stats(deltas):
    """deltas (N, D) → consistency, PCA EVR, magnitudes."""
    mags = np.linalg.norm(deltas, axis=1)
    dhat = l2n(deltas)
    # mean pairwise cosine, computed exactly from the Gram trick:
    # sum_{i!=j} d_i·d_j = ||sum_i d_i||^2 - N
    n = dhat.shape[0]
    s = np.linalg.norm(dhat.sum(axis=0)) ** 2 - n
    mean_pairwise_cos = float(s / (n * (n - 1))) if n > 1 else float("nan")
    evr1 = evr5 = float("nan")
    if n > 5:
        try:
            from sklearn.decomposition import PCA
            k = min(5, n - 1, deltas.shape[1])
            pca = PCA(n_components=k).fit(deltas)
            evr1 = float(pca.explained_variance_ratio_[0])
            evr5 = float(pca.explained_variance_ratio_[:k].sum())
        except ImportError:
            pass
    return {
        "n": int(n),
        "mean_delta_norm": float(mags.mean()),
        "mean_pairwise_cos": mean_pairwise_cos,
        "pca_evr_top1": evr1,
        "pca_evr_top5": evr5,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", action="append", type=parse_space, required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--dataset_name", default="asl_citizen")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="cap pairs per variant (debug)")
    ap.add_argument("--load_workers", type=int, default=32)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    # gloss + signer of each source video, from the split CSV
    recs = read_records(args.splits_dir, args.split, args.dataset_name,
                        with_signer=True)
    meta = {r["feat_id"]: (r["gloss"], r["signer"]) for r in recs}

    results = {}
    for space_name, feat_dir in dict(args.space).items():
        print(f"\n================ Space: {space_name} ({feat_dir}) ================")
        pairs = discover_pairs(feat_dir, args.variants)
        if not any(pairs.values()):
            print("  no variant feature files found — skipping "
                  "(this space has originals only)")
            results[space_name] = {"skipped": "no variant files"}
            continue

        # ---- gallery of split originals (for S3), aligned with gloss/signer ----
        gallery_feats, gallery_recs = load_with_records(
            recs, feat_dir, workers=args.load_workers,
            desc=f"{space_name} gallery")
        gallery = l2n(gallery_feats)
        g_gloss = np.array([r["gloss"] for r in gallery_recs])
        g_signer = np.array([r["signer"] for r in gallery_recs])
        g_id = {r["feat_id"]: i for i, r in enumerate(gallery_recs)}
        signers = sorted(set(g_signer.tolist()))
        proto = l2n(np.stack([gallery[g_signer == s].mean(axis=0)
                              for s in signers]))
        print(f"  gallery: {gallery.shape[0]} originals, "
              f"{len(signers)} signers")

        space_out = {"variants": {}, "cross_variant_direction_cos": None}
        mean_dirs = {}
        for v in args.variants:
            vp = pairs.get(v, [])
            if args.max_pairs:
                vp = vp[: args.max_pairs]
            if len(vp) < 5:
                print(f"  {v}: only {len(vp)} pairs — skipped")
                continue
            origs, augs, ids = [], [], []
            for orig_path, aug_path, feat_id in vp:
                try:
                    o, a = pooled(orig_path), pooled(aug_path)
                except Exception:
                    continue
                origs.append(o); augs.append(a); ids.append(feat_id)
            origs = l2n(np.stack(origs))
            augs = l2n(np.stack(augs))
            deltas = augs - origs

            stats = direction_stats(deltas)
            stats["mean_cos_orig_aug"] = float((origs * augs).sum(axis=1).mean())
            mean_dirs[v] = l2n(deltas.mean(axis=0))

            # ---- S3: gloss retention + signer-prototype flip ----
            known = [i for i, fid in enumerate(ids) if fid in g_id]
            if known:
                A = augs[known]
                O = origs[known]
                own_idx = np.array([g_id[ids[i]] for i in known])
                true_gloss = g_gloss[own_idx]

                sims = A @ gallery.T                        # (n, N)
                sims[np.arange(len(known)), own_idx] = -np.inf  # exclude self
                nn1 = sims.argmax(axis=1)
                stats["gloss_retention_at1"] = float(
                    (g_gloss[nn1] == true_gloss).mean())
                top5 = np.argpartition(-sims, 5, axis=1)[:, :5]
                stats["gloss_retention_at5"] = float(np.mean([
                    true_gloss[i] in g_gloss[top5[i]].tolist()
                    for i in range(len(known))]))

                near_sig_aug = np.array(signers)[(A @ proto.T).argmax(axis=1)]
                near_sig_orig = np.array(signers)[(O @ proto.T).argmax(axis=1)]
                stats["signer_flip_rate"] = float(
                    (near_sig_aug != near_sig_orig).mean())
                stats["orig_signer_proto_acc"] = float(
                    (near_sig_orig == g_signer[own_idx]).mean())
                stats["n_scored"] = int(len(known))

            space_out["variants"][v] = stats
            print(f"  {v}: n={stats['n']}  cos(orig,aug)="
                  f"{stats['mean_cos_orig_aug']:.4f}  "
                  f"dir-consistency={stats['mean_pairwise_cos']:.4f}  "
                  f"EVR1={stats['pca_evr_top1']:.3f}  "
                  f"gloss@1={stats.get('gloss_retention_at1', float('nan')):.3f}  "
                  f"signer-flip={stats.get('signer_flip_rate', float('nan')):.3f}")

        if len(mean_dirs) > 1:
            names = sorted(mean_dirs)
            M = np.stack([mean_dirs[n] for n in names])
            space_out["cross_variant_direction_cos"] = {
                "order": names, "matrix": (M @ M.T).round(4).tolist()}
        results[space_name] = space_out

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
