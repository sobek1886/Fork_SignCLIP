#!/usr/bin/env python3
"""
eval_asl_citizen_per_signer.py  –  Per-signer retrieval breakdown on ASL-Citizen.

Diagnoses whether appearance / signer identity is the bottleneck for a feature set:
computes the video-to-video NN dictionary-retrieval rank (same protocol as
eval_asl_citizen_retrieval.py / Eval A) for every TEST query, groups by the signer
(`Participant ID`), and reports the DISTRIBUTION of per-signer Rec@1.

Interpretation:
  - per-signer Rec@1 ~uniform & high  -> the model already generalises across signers;
    little headroom for an appearance-invariance intervention (explains a flat aug result).
  - per-signer Rec@1 spread / a low tail -> appearance/signer IS a lever; a working
    invariance method should lift the weak signers.

Works on ANY feature dir (Logos 768-d, I3D 1024-d, fine-tuned, ...) — dimension-agnostic.

Usage:
  python eval_asl_citizen_per_signer.py \\
      --feature_dir /home/psobecki/ASL_Citizen/logos_features_native \\
      --splits_dir  /home/psobecki/ASL_Citizen/splits \\
      --feature_name baseline_native --output_csv runs/per_signer/baseline_native.csv
"""

import argparse
import csv
import json
import os
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from eval_asl_citizen_retrieval import DEFAULT_LOGOS_DIR, DEFAULT_SPLITS, DATASET_NAME

_SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def read_records(splits_dir, split, dataset_name, with_signer=False):
    """Read a split CSV → list of {feat_id, gloss[, signer]}."""
    recs = []
    with open(Path(splits_dir) / _SPLIT_FILES[split], newline="") as f:
        for row in csv.DictReader(f):
            base = os.path.splitext(row["Video file"])[0]
            rec = {"feat_id": f"{dataset_name}_{base}", "gloss": row["Gloss"].strip()}
            if with_signer:
                rec["signer"] = row.get("Participant ID", "?").strip()
            recs.append(rec)
    return recs


def load_with_records(records, feature_dir, workers=32, desc="load"):
    """Parallel .npy load (mean-pool clips) → (feats (N,D), kept_records) preserving alignment."""
    feature_dir = Path(feature_dir)

    def _load(rec):
        try:
            feat = np.load(feature_dir / (rec["feat_id"] + ".npy")).astype(np.float32)
        except Exception:
            return None
        if feat.ndim == 2:
            if feat.shape[0] == 0:
                return None
            feat = feat.mean(axis=0)
        elif feat.ndim != 1:
            return None
        return feat, rec

    feats, kept = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in tqdm(ex.map(_load, records), total=len(records), desc=desc):
            if res is not None:
                feats.append(res[0])
                kept.append(res[1])
    if not feats:
        raise RuntimeError(f"No features loaded from {feature_dir}")
    return np.stack(feats), kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--splits_dir", default=DEFAULT_SPLITS)
    ap.add_argument("--feature_name", default="features")
    ap.add_argument("--dataset_name", default=DATASET_NAME)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--load_workers", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_csv", default=None, help="per-signer rows → CSV")
    ap.add_argument("--worst", type=int, default=15, help="how many worst signers to print")
    args = ap.parse_args()
    device = torch.device(args.device)

    gallery_recs = read_records(args.splits_dir, "train", args.dataset_name)
    query_recs = read_records(args.splits_dir, "test", args.dataset_name, with_signer=True)
    G, gal_kept = load_with_records(gallery_recs, args.feature_dir, args.load_workers,
                                    desc="gallery(train)")
    Q, q_kept = load_with_records(query_recs, args.feature_dir, args.load_workers,
                                  desc="query(test)")

    # gloss vocab from the gallery (same OrderedDict rule as Eval A)
    g_dict = OrderedDict()
    for r in gal_kept:
        if r["gloss"] not in g_dict:
            g_dict[r["gloss"]] = len(g_dict)
    n_gloss = len(g_dict)
    gal_idx = torch.tensor([g_dict[r["gloss"]] for r in gal_kept],
                           dtype=torch.long, device=device)
    Gn = torch.nn.functional.normalize(torch.from_numpy(G).to(device), dim=-1)

    # rank each query (per-gloss max similarity → rank of the GT gloss)
    ranks = np.full(len(q_kept), -1, dtype=np.int64)
    keep = [i for i, r in enumerate(q_kept) if r["gloss"] in g_dict]
    for s in tqdm(range(0, len(keep), args.batch_size), desc="rank"):
        idx = keep[s:s + args.batch_size]
        Qb = torch.nn.functional.normalize(
            torch.from_numpy(Q[idx]).to(device), dim=-1)
        sim = Qb @ Gn.T
        gloss_sim = torch.full((sim.size(0), n_gloss), float("-inf"), device=device)
        gloss_sim.scatter_reduce_(1, gal_idx.unsqueeze(0).expand(sim.size(0), -1), sim,
                                  reduce="amax", include_self=True)
        labels = torch.tensor([g_dict[q_kept[i]["gloss"]] for i in idx],
                              dtype=torch.long, device=device)
        gt = gloss_sim.gather(1, labels.unsqueeze(1))
        r = (gloss_sim > gt).sum(dim=1).cpu().numpy()
        for j, i in enumerate(idx):
            ranks[i] = r[j]

    # group by signer
    by_signer = defaultdict(list)
    for i, rec in enumerate(q_kept):
        if ranks[i] >= 0:
            by_signer[rec["signer"]].append(ranks[i])

    rows = []
    for sgn, rks in by_signer.items():
        rks = np.array(rks)
        rows.append({
            "signer": sgn, "n": len(rks),
            "rec1": 100.0 * np.mean(rks < 1),
            "rec5": 100.0 * np.mean(rks < 5),
            "mrr": 100.0 * np.mean(1.0 / (rks + 1.0)),
        })
    rows.sort(key=lambda d: d["rec1"])

    rec1s = np.array([r["rec1"] for r in rows])
    all_ranks = np.concatenate([np.array(v) for v in by_signer.values()])
    micro_rec1 = 100.0 * np.mean(all_ranks < 1)

    W = 64
    print("\n" + "=" * W)
    print(f"PER-SIGNER RETRIEVAL  ({args.feature_name})")
    print("=" * W)
    print(f"  test signers     : {len(rows)}   queries: {len(all_ranks)}")
    print(f"  micro Rec@1      : {micro_rec1:6.2f}  (over all queries)")
    print(f"  macro Rec@1      : {rec1s.mean():6.2f}  (mean over signers)")
    print(f"  per-signer std   : {rec1s.std():6.2f}")
    print(f"  min / max signer : {rec1s.min():6.2f} / {rec1s.max():6.2f}")
    print(f"  spread (max-min) : {rec1s.max() - rec1s.min():6.2f}")
    print(f"\n  Worst {args.worst} signers by Rec@1:")
    print(f"  {'signer':<24} {'n':>4} {'Rec@1':>7} {'Rec@5':>7}")
    for r in rows[:args.worst]:
        print(f"  {r['signer']:<24} {r['n']:>4} {r['rec1']:>7.1f} {r['rec5']:>7.1f}")
    print("=" * W)
    print("  High std / low tail → appearance is a lever (invariance has room to help).")
    print("  Uniform & high      → backbone already generalises across signers.")
    print("=" * W)

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["signer", "n", "rec1", "rec5", "mrr"])
            w.writeheader()
            w.writerows(rows)
        print(f"per-signer rows → {args.output_csv}")
        summ = args.output_csv.replace(".csv", "_summary.json")
        with open(summ, "w") as f:
            json.dump({"feature_name": args.feature_name, "n_signers": len(rows),
                       "micro_rec1": micro_rec1, "macro_rec1": float(rec1s.mean()),
                       "std": float(rec1s.std()), "min": float(rec1s.min()),
                       "max": float(rec1s.max())}, f, indent=2)


if __name__ == "__main__":
    main()
