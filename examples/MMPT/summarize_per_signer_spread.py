#!/usr/bin/env python3
"""
summarize_per_signer_spread.py - Cross-condition fairness comparison from
                                 eval_asl_citizen_per_signer.py outputs
                                 (thesis analysis S4).

eval_asl_citizen_per_signer.py writes one CSV (+ _summary.json) per feature
set. This aggregates them: per condition, the per-signer Rec@1 spread (std,
max-min gap, worst-k signers) and, against a chosen baseline condition, the
per-signer paired deltas — did the augmentation/invariance objective lift the
WEAK signers (fairness gain) or only the average?

Usage
  python summarize_per_signer_spread.py \
      --csv baseline=runs/per_signer/baseline_native.csv \
      --csv ft_ce=runs/per_signer/run2.csv \
      --csv ft_aug=runs/per_signer/run3.csv \
      --baseline baseline \
      --out_json runs/per_signer/spread_summary.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_csv_arg(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--csv must be NAME=PATH, got {s!r}")
    name, p = s.split("=", 1)
    return name, Path(p)


def read_per_signer(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["signer"]] = {"n": int(row["n"]),
                                   "rec1": float(row["rec1"]),
                                   "rec5": float(row["rec5"]),
                                   "mrr": float(row["mrr"])}
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", type=parse_csv_arg, required=True)
    ap.add_argument("--baseline", default=None,
                    help="condition name to compute paired deltas against")
    ap.add_argument("--worst", type=int, default=3)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    conds = {name: read_per_signer(p) for name, p in dict(args.csv).items()
             if p.exists() or print(f"SKIP {name}: {p} not found") or False}
    if not conds:
        raise SystemExit("no per-signer CSVs found")

    out = {"conditions": {}, "paired_vs_baseline": {}}
    for name, rows in conds.items():
        r1 = np.array([v["rec1"] for v in rows.values()])
        signers = list(rows)
        worst = sorted(signers, key=lambda s: rows[s]["rec1"])[: args.worst]
        out["conditions"][name] = {
            "n_signers": len(rows),
            "macro_rec1": float(r1.mean()),
            "std": float(r1.std()),
            "gap_max_minus_min": float(r1.max() - r1.min()),
            "worst_signers": {s: rows[s]["rec1"] for s in worst},
        }
        print(f"{name:>16}: macro R@1 {r1.mean():.4f}  std {r1.std():.4f}  "
              f"gap {r1.max()-r1.min():.4f}  worst {out['conditions'][name]['worst_signers']}")

    base = args.baseline
    if base and base in conds:
        b = conds[base]
        for name, rows in conds.items():
            if name == base:
                continue
            common = sorted(set(rows) & set(b))
            d = np.array([rows[s]["rec1"] - b[s]["rec1"] for s in common])
            base_r1 = np.array([b[s]["rec1"] for s in common])
            # correlation between a signer's baseline strength and its gain:
            # negative correlation = the intervention lifts WEAK signers most.
            corr = float(np.corrcoef(base_r1, d)[0, 1]) if len(common) > 2 else None
            out["paired_vs_baseline"][name] = {
                "baseline": base, "n_common_signers": len(common),
                "mean_delta_rec1": float(d.mean()),
                "delta_std_change": float(np.array([rows[s]["rec1"] for s in common]).std()
                                          - base_r1.std()),
                "corr_baseline_strength_vs_gain": corr,
                "per_signer_delta": {s: round(float(rows[s]["rec1"] - b[s]["rec1"]), 4)
                                     for s in common},
            }
            print(f"{name} vs {base}: mean Δ {d.mean():+.4f}, "
                  f"std change {out['paired_vs_baseline'][name]['delta_std_change']:+.4f}, "
                  f"strength-vs-gain corr {corr}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
