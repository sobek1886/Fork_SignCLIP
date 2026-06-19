#!/usr/bin/env python3
"""
diagnose_appearance_invariance.py — does a backbone pull a sign's appearance variants
together?  (Operates ONLY on already-extracted .npy features — no model, no extraction.)

Mechanistic test of the consistency hypothesis. For every augmented video that has BOTH an
original feature and an augmentation feature in the given dirs, computes the cosine similarity
between the original CLS feature z_i and its appearance-augmented copy z_i^a
(signer_swap / skin_mst_diffusion). Run once per feature set and compare:

  * baseline : features from the Logos init (no fine-tuning)        — e.g. logos_features
  * run1     : features from the CE+consistency backbone (top-2)    — e.g. logos_features_run1

Reading (compare matched_mean across the two runs):
  - run1 matched_mean  >>  baseline matched_mean
        → consistency DID pull appearance variants together (top-2 unfreeze sufficed). Since
          this did not improve in-domain retrieval, appearance is not the in-domain bottleneck;
          the bottom-unfreeze two-stage would only matter cross-domain.
  - run1 matched_mean  ≈  baseline matched_mean
        → top-only consistency could NOT remove appearance (the appearance-encoding lower
          blocks were frozen) → unfreeze the bottom blocks for the consistency stage next.

The 'random' floor = cos between unrelated originals; 'gap' = matched − random shows how much
appearance-invariant structure exists above that floor.

Feature files: {dataset}_{video_id}.npy (original) and {dataset}_{video_id}_{aug}.npy (aug),
each (T, D) per-clip (mean-pooled here) or (D,). The aug features must have been extracted with
the SAME checkpoint as the originals in --orig_dir (i.e. point both dirs at one model's run).

Usage (run twice, then diff the JSONs):
  python diagnose_appearance_invariance.py --name baseline \
      --orig_dir /home/psobecki/ASL_Citizen/logos_features \
      --output_json runs/appearance_diag/baseline.json
  python diagnose_appearance_invariance.py --name run1 \
      --orig_dir /scratch-shared/psobecki/ASL_Citizen/logos_features_run1 \
      --aug_dir  /scratch-shared/psobecki/ASL_Citizen/logos_features_run1 \
      --output_json runs/appearance_diag/run1.json
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

DATASET_NAME = "asl_citizen"
DEFAULT_AUGS = ("signer_swap", "skin_mst_diffusion")


def _load_pooled(path):
    """Load a .npy feature, mean-pool clips if 2-D, L2-normalise → (D,) or None."""
    try:
        feat = np.load(path).astype(np.float32)
    except Exception:
        return None
    if feat.ndim == 2:
        if feat.shape[0] == 0:
            return None
        feat = feat.mean(axis=0)
    elif feat.ndim != 1:
        return None
    n = np.linalg.norm(feat)
    return feat / n if n > 0 else None


def _stats(name, vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {f"{name}_n": 0, f"{name}_mean": float("nan"), f"{name}_median": float("nan"),
                f"{name}_std": float("nan"), f"{name}_p10": float("nan"), f"{name}_p90": float("nan")}
    return {f"{name}_n": int(a.size), f"{name}_mean": float(a.mean()),
            f"{name}_median": float(np.median(a)), f"{name}_std": float(a.std()),
            f"{name}_p10": float(np.percentile(a, 10)), f"{name}_p90": float(np.percentile(a, 90))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig_dir", required=True, help="dir with {dataset}_{id}.npy originals")
    ap.add_argument("--aug_dir", default=None,
                    help="dir with {dataset}_{id}_{aug}.npy (default: same as --orig_dir)")
    ap.add_argument("--augs", nargs="+", default=list(DEFAULT_AUGS))
    ap.add_argument("--dataset_name", default=DATASET_NAME)
    ap.add_argument("--name", default="model")
    ap.add_argument("--max_videos", type=int, default=None, help="cap matched videos (speed)")
    ap.add_argument("--n_random", type=int, default=2000, help="random unrelated-pair control")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    orig_dir = Path(args.orig_dir)
    aug_dir = Path(args.aug_dir) if args.aug_dir else orig_dir
    pfx = f"{args.dataset_name}_"

    # Find every aug feature file and pair it with its original.
    pairs = []   # (video_id, aug, aug_path, orig_path)
    for aug in args.augs:
        suffix = f"_{aug}.npy"
        for ap_path in sorted(aug_dir.glob(f"{pfx}*{suffix}")):
            vid = ap_path.name[len(pfx):-len(suffix)]
            opath = orig_dir / f"{pfx}{vid}.npy"
            if opath.exists():
                pairs.append((vid, aug, ap_path, opath))
    if args.max_videos:
        pairs = pairs[:args.max_videos]
    if not pairs:
        raise SystemExit(
            f"No (orig, aug) pairs found.\n  aug files {args.augs} in {aug_dir}\n  "
            f"originals in {orig_dir}\nIf this is a fine-tuned run, its augmentation features "
            f"may not have been extracted with that checkpoint yet (the backbone re-extraction "
            f"only does originals). Extract the augs for this run first, then point --aug_dir "
            f"at them.")
    print(f"[{args.name}] {len(pairs)} (orig,aug) pairs across {args.augs}")

    # Load all needed features once (unique paths), in parallel.
    need = {p for _, _, ap_path, op in pairs for p in (ap_path, op)}
    cache = {}

    def _do(p):
        return p, _load_pooled(p)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for p, v in tqdm(ex.map(_do, need), total=len(need), desc=f"load {args.name}"):
            cache[p] = v

    per_aug = {a: [] for a in args.augs}
    orig_vecs = {}
    for vid, aug, ap_path, opath in pairs:
        za, zo = cache.get(ap_path), cache.get(opath)
        if za is None or zo is None:
            continue
        per_aug[aug].append(float(np.dot(zo, za)))   # both unit-norm → cosine
        orig_vecs[vid] = zo

    # Random unrelated-pair floor among the originals we loaded.
    rnd = random.Random(0)
    ids = list(orig_vecs)
    rand_cos = []
    if len(ids) >= 2:
        for _ in range(args.n_random):
            i, j = rnd.sample(ids, 2)
            rand_cos.append(float(np.dot(orig_vecs[i], orig_vecs[j])))

    matched_all = [c for v in per_aug.values() for c in v]
    out = {"name": args.name, "orig_dir": str(orig_dir), "aug_dir": str(aug_dir),
           "n_videos": len(orig_vecs)}
    out.update(_stats("matched", matched_all))
    out.update(_stats("random", rand_cos))
    for a in args.augs:
        out.update(_stats(f"matched_{a}", per_aug[a]))
    out["gap_matched_minus_random"] = out["matched_mean"] - out["random_mean"]

    W = 64
    print("\n" + "=" * W)
    print(f"APPEARANCE-INVARIANCE DIAGNOSTIC  ({args.name})")
    print("=" * W)
    print(f"  videos                : {out['n_videos']}")
    print(f"  matched cos (orig↔aug): {out['matched_mean']:.4f}  "
          f"(median {out['matched_median']:.4f}, n={out['matched_n']})")
    for a in args.augs:
        if out[f"matched_{a}_n"]:
            print(f"     {a:<20}: {out[f'matched_{a}_mean']:.4f}  (n={out[f'matched_{a}_n']})")
    print(f"  random cos (unrelated): {out['random_mean']:.4f}  (n={out['random_n']})")
    print(f"  gap (matched−random)  : {out['gap_matched_minus_random']:.4f}")
    print("=" * W)
    print("  Compare matched_mean across feature sets: a large baseline→run1 jump means")
    print("  consistency reduced appearance distance (top-2 sufficed); ≈no change means the")
    print("  appearance-encoding lower blocks were frozen → unfreeze the bottom next.")
    print("=" * W)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"→ {args.output_json}")


if __name__ == "__main__":
    main()
