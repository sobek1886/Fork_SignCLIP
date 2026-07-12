#!/usr/bin/env python3
"""
analyze_space_information.py - How much SIGNER information is extractable from
                               each embedding space, and how far apart are the
                               spaces themselves? (thesis analyses S2 + S6)

S2 — signer-identity linear probe. Train a logistic-regression classifier to
predict the signer from a video's embedding (per-signer stratified holdout
within one split, so every signer appears in both fit and test). Accuracy far
above chance = signer identity is LINEARLY EXTRACTABLE — a stronger,
information-flavoured statement than the unsupervised inter/intra L2 ratio
(asl_citizen_feature_l2_probe.py), and the direct quantitative form of the
thesis's "signer identity is linearly recoverable" contribution. Comparing
across training conditions shows whether any objective actually REMOVED
extractable signer information, or merely rearranged it.

S6 — linear CKA between spaces. One number per space pair (0..1) on the same
video sample: how similar are the representational geometries? Answers "did
SupCon-on-pairs actually reshape the space, or barely move it?" without any
labels.

All spaces are feature dirs ({dataset}_{id}.npy), the project's common
currency (raw Logos, backbone-FT re-extractions, SignCLIP pooled_video
exports from export_signclip_features.py).

Usage (on Snellius)
  python analyze_space_information.py \
      --space raw=/scratch-shared/psobecki/ASL_Citizen/logos_features_native \
      --space ft_ce=/scratch-shared/psobecki/ASL_Citizen/logos_features_run2 \
      --space pair_k5=/scratch-shared/psobecki/ASL_Citizen/signclip_features_pair_k5 \
      --splits_dir /home/psobecki/ASL_Citizen/splits \
      --split train --max_videos 8000 \
      --out_json runs/space_info/space_info.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_asl_citizen_per_signer import read_records, load_with_records


def parse_space(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--space must be NAME=DIR, got {s!r}")
    name, d = s.split("=", 1)
    return name, Path(d)


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def signer_probe(feats, signers, seed=0):
    """Per-signer stratified 80/20 holdout + logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for s in sorted(set(signers.tolist())):
        idx = np.where(signers == s)[0]
        idx = idx[rng.permutation(len(idx))]
        k = max(1, int(0.8 * len(idx)))
        tr_idx += idx[:k].tolist()
        te_idx += idx[k:].tolist()
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(feats[tr_idx], signers[tr_idx])
    pred = clf.predict(feats[te_idx])
    acc = float((pred == signers[te_idx]).mean())
    f1 = float(f1_score(signers[te_idx], pred, average="macro"))
    return {"probe_acc": acc, "probe_macro_f1": f1,
            "chance": 1.0 / len(set(signers.tolist())),
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))}


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019) between (N,Dx) and (N,Dy)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xty = np.linalg.norm(X.T @ Y) ** 2
    xtx = np.linalg.norm(X.T @ X)
    yty = np.linalg.norm(Y.T @ Y)
    return float(xty / (xtx * yty + 1e-12))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", action="append", type=parse_space, required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--dataset_name", default="asl_citizen")
    ap.add_argument("--max_videos", type=int, default=8000,
                    help="deterministic subsample shared by all spaces "
                         "(keeps the probe + CKA tractable)")
    ap.add_argument("--load_workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    recs = read_records(args.splits_dir, args.split, args.dataset_name,
                        with_signer=True)
    rng = np.random.default_rng(args.seed)
    if args.max_videos and len(recs) > args.max_videos:
        keep = rng.permutation(len(recs))[: args.max_videos]
        recs = [recs[i] for i in sorted(keep)]
    print(f"Sample: {len(recs)} videos from the {args.split} split")

    spaces = dict(args.space)
    loaded = {}     # name → (feats normalized, feat_id list, signer array)
    results = {"probe": {}, "cka": {}}
    for name, feat_dir in spaces.items():
        print(f"\n=== Space: {name} ({feat_dir}) ===")
        feats, kept = load_with_records(recs, feat_dir,
                                        workers=args.load_workers,
                                        desc=f"load {name}")
        feats = l2n(feats)
        signers = np.array([r["signer"] for r in kept])
        ids = [r["feat_id"] for r in kept]
        loaded[name] = (feats, ids, signers)
        res = signer_probe(feats, signers, seed=args.seed)
        res["n_videos"] = int(feats.shape[0])
        res["dim"] = int(feats.shape[1])
        results["probe"][name] = res
        print(f"  signer probe: acc={res['probe_acc']:.4f} "
              f"(chance {res['chance']:.4f}), macro-F1={res['probe_macro_f1']:.4f}")

    # ---- pairwise linear CKA on the intersection of ids ----
    names = list(loaded)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ids_a = {fid: k for k, fid in enumerate(loaded[a][1])}
            common = [(ids_a[fid], k) for k, fid in enumerate(loaded[b][1])
                      if fid in ids_a]
            if len(common) < 100:
                print(f"CKA({a},{b}): <100 common videos — skipped")
                continue
            ia, ib = map(np.array, zip(*common))
            cka = linear_cka(loaded[a][0][ia], loaded[b][0][ib])
            results["cka"][f"{a}|{b}"] = {"cka": round(cka, 4),
                                          "n_common": int(len(common))}
            print(f"CKA({a}, {b}) = {cka:.4f}  (n={len(common)})")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
