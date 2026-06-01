#!/usr/bin/env python3
"""
Diagnostic: raw Logos feature retrieval (no model).

Bypasses MMBert entirely — just mean-pools each .npy file's clips and
measures retrieval.  Answers the question:

  "Are piotrAnims / palmer Logos features already similar to Bushuis before
   any fine-tuning?"

Usage:
    python diagnose_raw_retrieval.py \
        --manifest    /home/psobecki/ngt_pair_manifest.json \
        --bushuis_dir /scratch-shared/psobecki/Bushuis/logos_features
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def mean_pool(path: str) -> np.ndarray:
    feat = np.load(path).astype(np.float32)   # (N_clips, 768)
    v = feat.mean(axis=0)                      # (768,)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def retrieval_metrics(Q: np.ndarray, G: np.ndarray) -> dict:
    """R@1/5/10 and MRR for aligned Q/G (query i matches gallery i)."""
    sims = Q @ G.T   # (N, N)
    N = sims.shape[0]
    r1 = r5 = r10 = 0
    rr = 0.0
    for i in range(N):
        order = np.argsort(-sims[i])
        rank = np.where(order == i)[0][0]
        if rank < 1:  r1 += 1
        if rank < 5:  r5 += 1
        if rank < 10: r10 += 1
        rr += 1.0 / (rank + 1)
    return {
        "R@1":  100.0 * r1 / N,
        "R@5":  100.0 * r5 / N,
        "R@10": 100.0 * r10 / N,
        "MRR":  100.0 * rr / N,
    }


def fmt(m: dict) -> str:
    return (f"R@1={m['R@1']:.1f}%  R@5={m['R@5']:.1f}%  "
            f"R@10={m['R@10']:.1f}%  MRR={m['MRR']:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",    required=True)
    ap.add_argument("--bushuis_dir", required=True)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    _BUSHUIS_RE = re.compile(r'^bushuis_(M\d{8}_\d+)\.npy$')
    bushuis_map = {}
    for p in Path(args.bushuis_dir).glob("*.npy"):
        m = _BUSHUIS_RE.match(p.name)
        if m:
            bushuis_map[m.group(1)] = str(p)

    eval_ids = sorted(set(manifest) & set(bushuis_map))
    if not eval_ids:
        sys.exit("ERROR: no matched sign IDs.")

    print(f"Evaluating {len(eval_ids)} signs\n")

    # Clip count statistics
    piotr_clips, palmer_clips, bushuis_clips = [], [], []

    piotr_mid, palmer_mid, bushuis_vecs = [], [], []

    for sid in eval_ids:
        paths = manifest[sid]

        # piotrAnims MIDDLE (index 1)
        real_paths = paths["real"]
        piotr_feat = np.load(real_paths[1]).astype(np.float32)
        piotr_clips.append(piotr_feat.shape[0])
        v = piotr_feat.mean(axis=0)
        norm = np.linalg.norm(v)
        piotr_mid.append(v / norm if norm > 1e-8 else v)

        # palmer cam3 (unreal index 1 = palmer_cam3)
        unreal_paths = paths.get("unreal", [])
        if len(unreal_paths) >= 2:
            palmer_feat = np.load(unreal_paths[1]).astype(np.float32)
            palmer_clips.append(palmer_feat.shape[0])
            v = palmer_feat.mean(axis=0)
            norm = np.linalg.norm(v)
            palmer_mid.append(v / norm if norm > 1e-8 else v)
        else:
            palmer_clips.append(0)
            palmer_mid.append(np.zeros(768, dtype=np.float32))

        # Bushuis
        bushuis_feat = np.load(bushuis_map[sid]).astype(np.float32)
        bushuis_clips.append(bushuis_feat.shape[0])
        v = bushuis_feat.mean(axis=0)
        norm = np.linalg.norm(v)
        bushuis_vecs.append(v / norm if norm > 1e-8 else v)

    print("=== Clip count statistics ===")
    for name, counts in [("piotrAnims", piotr_clips),
                          ("palmer",    palmer_clips),
                          ("Bushuis",   bushuis_clips)]:
        arr = np.array(counts, dtype=float)
        print(f"  {name:12s}  mean={arr.mean():.1f}  "
              f"median={np.median(arr):.1f}  "
              f"min={arr.min():.0f}  max={arr.max():.0f}")

    Q_piotr   = np.stack(piotr_mid)
    Q_palmer  = np.stack(palmer_mid)
    Q_bushuis = np.stack(bushuis_vecs)

    print("\n=== RAW Logos feature retrieval (no model, mean-pooled) ===")
    m_p2b = retrieval_metrics(Q_piotr, Q_bushuis)
    m_b2p = retrieval_metrics(Q_bushuis, Q_piotr)
    m_a2b = retrieval_metrics(Q_palmer, Q_bushuis)
    m_b2a = retrieval_metrics(Q_bushuis, Q_palmer)

    print(f"  piotrAnims → Bushuis:  {fmt(m_p2b)}")
    print(f"  Bushuis → piotrAnims:  {fmt(m_b2p)}")
    print(f"  palmer    → Bushuis:   {fmt(m_a2b)}")
    print(f"  Bushuis → palmer:      {fmt(m_b2a)}")


if __name__ == "__main__":
    main()
