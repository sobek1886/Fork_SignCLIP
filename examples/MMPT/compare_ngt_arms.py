#!/usr/bin/env python3
"""Paired per-sentence significance between two NGT experiment arms.

Both arms score the SAME test sentences, so the right test is paired:
exact McNemar on per-sentence top-1 correctness plus a percentile-bootstrap
CI on the R@1 difference, per seed-matched run pair, for the front-view
readout in both retrieval directions. Headline pairs of the Flux-vs-Unreal
ladder: E3|E4 (matched composition) and E5|E6 (full strength).

Requires eval_results.json files with "per_sentence" ranks, i.e. produced by
eval_ngt_retrieval.py on a take-list eval manifest ("style": "front").

Usage (on Snellius; CPU, instant):
    python compare_ngt_arms.py \
        --a ngt_sv_unreal_full \
        --b ngt_sv_flux_full \
        --base_dir /scratch-shared/psobecki/runs/retri_ngt \
        --k 5 \
        --out_json runs/ngt_compare/e5_vs_e6.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_runs(base_dir, experiment, k):
    runs = {}
    for i in range(1, k + 1):
        p = Path(base_dir) / f"{experiment}_run{i}" / "eval_results.json"
        if not p.exists():
            print(f"WARNING: missing {p}")
            continue
        with open(p) as f:
            r = json.load(f)
        if "per_sentence" not in r:
            raise SystemExit(f"{p} has no per-sentence ranks — re-run eval "
                             "with the take-list eval manifest.")
        runs[i] = r["per_sentence"]
    if not runs:
        raise SystemExit(f"No runs found for {experiment} under {base_dir}")
    return runs


def compare_direction(pa, pb, rank_key, n_bootstrap, rng):
    """Paired comparison of one seed pair in one direction."""
    common = sorted(set(pa) & set(pb))
    ca = np.array([pa[s][rank_key] == 0 for s in common])
    cb = np.array([pb[s][rank_key] == 0 for s in common])
    disc_b = int((ca & ~cb).sum())   # a right, b wrong
    disc_c = int((~ca & cb).sum())   # a wrong, b right
    n = len(common)
    diffs = np.empty(n_bootstrap)
    for t in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        diffs[t] = cb[idx].mean() - ca[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_sentences": n,
        "r1_a": float(ca.mean()), "r1_b": float(cb.mean()),
        "delta_r1_b_minus_a": float(cb.mean() - ca.mean()),
        "a_right_b_wrong": disc_b, "a_wrong_b_right": disc_c,
        "mcnemar_p_two_sided": mcnemar_exact(disc_b, disc_c),
        "bootstrap_ci95": [float(lo), float(hi)],
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="experiment name of arm A")
    ap.add_argument("--b", required=True, help="experiment name of arm B")
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--k", type=int, default=5, help="runs per arm (seeds)")
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    runs_a = load_runs(args.base_dir, args.a, args.k)
    runs_b = load_runs(args.base_dir, args.b, args.k)
    seeds = sorted(set(runs_a) & set(runs_b))
    if not seeds:
        raise SystemExit("No seed-matched run pairs.")
    rng = np.random.default_rng(args.seed)

    results = {"arm_a": args.a, "arm_b": args.b, "per_seed": {}, "summary": {}}
    for direction, rank_key in [("b2s_front", "b2s_front_rank"),
                                ("s2b_front", "s2b_front_rank")]:
        per_seed = []
        for i in seeds:
            per_seed.append(compare_direction(
                runs_a[i], runs_b[i], rank_key, args.n_bootstrap, rng))
            results["per_seed"].setdefault(str(i), {})[direction] = per_seed[-1]
        deltas = [r["delta_r1_b_minus_a"] for r in per_seed]
        ps = [r["mcnemar_p_two_sided"] for r in per_seed]
        results["summary"][direction] = {
            "n_seed_pairs": len(seeds),
            "mean_delta_r1": float(np.mean(deltas)),
            "std_delta_r1": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "seed_deltas": deltas,
            "seed_mcnemar_ps": ps,
            "n_seeds_significant_005": int(sum(p < 0.05 for p in ps)),
        }
        print(f"\n=== {direction}: {args.b} - {args.a} ===")
        for i, r in zip(seeds, per_seed):
            print(f"  seed pair {i}: ΔR@1={r['delta_r1_b_minus_a']:+.4f} "
                  f"(b={r['a_right_b_wrong']}, c={r['a_wrong_b_right']}, "
                  f"p={r['mcnemar_p_two_sided']:.4g}, "
                  f"CI [{r['bootstrap_ci95'][0]:+.4f}, {r['bootstrap_ci95'][1]:+.4f}])")
        s = results["summary"][direction]
        print(f"  over seeds: mean ΔR@1={s['mean_delta_r1']:+.4f} "
              f"± {s['std_delta_r1']:.4f}, "
              f"{s['n_seeds_significant_005']}/{len(seeds)} seeds p<0.05")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
