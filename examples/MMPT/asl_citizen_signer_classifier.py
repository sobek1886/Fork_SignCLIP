#!/usr/bin/env python3
"""
asl_citizen_signer_classifier.py — E3 signer-DECODING probe on ASL-Citizen
feature spaces (companion to the E2 distance probe in
asl_citizen_feature_l2_probe.py; thesis sec:results-ft-probes).

A linear head p(s|z) = softmax(Wz + b) is trained on frozen, mean-pooled
features z (768-d) to predict the Participant ID s, with stratified K-fold
cross-validation over videos. Reported per space and per split:
  - top-1 accuracy (mean ± sd over folds) vs chance 1/S and the
    majority-class baseline (signer video counts are unbalanced),
  - balanced (macro) accuracy,
  - S, N.
The two probes can dissociate: identity information that is linearly
decodable but small relative to gloss variation is invisible to the distance
probe — an intervention that genuinely removes appearance identity must
lower BOTH readouts.

Splits are evaluated separately and are signer-disjoint (train: 35 signers,
test: 11). The TEST split is the primary readout for the fine-tuned spaces:
those videos were never seen by the fine-tuning, so decodability there
measures what the representation carries, not what it memorised.

Feature files: {prefix}{video_basename}.npy, shape (T, D), mean-pooled here.

Usage (Snellius):
  python asl_citizen_signer_classifier.py \
      --space baseline_native=/scratch-shared/psobecki/ASL_Citizen/logos_features_native \
      --space run2=/scratch-shared/psobecki/ASL_Citizen/logos_features_run2 \
      --splits_dir /home/psobecki/ASL_Citizen/splits \
      --splits test train --out_json runs/signer_probe/signer_decoding.json
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def load_split(splits_dir, split):
    rows = []
    with open(Path(splits_dir) / _SPLIT_FILES[split], newline="") as f:
        for row in csv.DictReader(f):
            rows.append((os.path.splitext(row["Video file"])[0],
                         row["Participant ID"].strip()))
    return rows


def load_space(feature_dir, rows, prefix, l2n):
    X, y, missing = [], [], 0
    for base, signer in rows:
        p = Path(feature_dir) / f"{prefix}{base}.npy"
        if not p.exists():
            missing += 1
            continue
        v = np.load(p).astype(np.float32).mean(axis=0)
        if l2n:
            v /= (np.linalg.norm(v) + 1e-8)
        X.append(v)
        y.append(signer)
    if missing:
        print(f"    ({missing} missing feature files skipped)")
    return np.stack(X), np.array(y)


def probe(X, y, folds, seed, max_iter):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, baccs = [], []
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=max_iter, tol=1e-3)),
        ])
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        accs.append(float((pred == y[te]).mean()))
        baccs.append(float(balanced_accuracy_score(y[te], pred)))
    return accs, baccs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", action="append", required=True,
                    metavar="NAME=DIR", help="feature space, repeatable")
    ap.add_argument("--splits_dir", default="/home/psobecki/ASL_Citizen/splits")
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--prefix", default="asl_citizen_")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--l2n", action="store_true",
                    help="also run on l2-normalised features")
    ap.add_argument("--out_json", default="runs/signer_probe/signer_decoding.json")
    args = ap.parse_args()

    spaces = [s.split("=", 1) for s in args.space]
    results = {"folds": args.folds, "seed": args.seed, "spaces": {}}

    for name, d in spaces:
        if not Path(d).is_dir():
            print(f"== skip {name} (no dir {d}) ==")
            continue
        results["spaces"][name] = {}
        for split in args.splits:
            rows = load_split(args.splits_dir, split)
            print(f"== {name} / {split}: loading {len(rows)} videos from {d}")
            variants = [("raw", False)] + ([("l2n", True)] if args.l2n else [])
            for vname, l2n in variants:
                X, y = load_space(d, rows, args.prefix, l2n)
                S = len(set(y))
                counts = np.unique(y, return_counts=True)[1]
                majority = float(counts.max() / counts.sum())
                accs, baccs = probe(X, y, args.folds, args.seed, args.max_iter)
                r = {"n": int(len(y)), "n_signers": S, "chance": 1.0 / S,
                     "majority_baseline": majority,
                     "top1_mean": float(np.mean(accs)),
                     "top1_sd": float(np.std(accs)),
                     "balanced_acc_mean": float(np.mean(baccs)),
                     "fold_top1": accs}
                results["spaces"][name][f"{split}_{vname}"] = r
                print(f"   {split}/{vname}: top1={r['top1_mean']:.4f}"
                      f"±{r['top1_sd']:.4f}  balanced={r['balanced_acc_mean']:.4f}"
                      f"  chance={r['chance']:.4f}  majority={majority:.4f}"
                      f"  (S={S}, N={r['n']})")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
