#!/usr/bin/env python3
"""
eval_asl_citizen_linear_probe.py  –  Linear-probe evaluation of frozen features
                                      on ASL-Citizen (SignRep Table 3 protocol).

Companion to eval_asl_citizen_retrieval.py (Eval A, video-to-video NN retrieval,
= SignRep Table 4). This is Eval B: train a single Linear(768 -> n_gloss) classifier
on the FROZEN train features, evaluate on the frozen test features, and report the
SAME ASL-Citizen metrics (DCG / MRR / Rec@1/5/10/20) using identical definitions.

Protocol
  - Train features  = TRAIN split (gallery) mean-pooled (T,768) -> (768,) features.
  - Test features   = TEST  split (query)  features, same pooling.
  - Gloss vocabulary is the TRAIN gloss set (same OrderedDict rule as Eval A's
    `evaluate`), so gloss indices are consistent.
  - Fit Linear(768 -> n_gloss) with cross-entropy (AdamW, GPU). At test, rank the
    n_gloss classes by logit; let r be the 0-indexed rank of the GT gloss:
        DCG  = 1 / log2(r + 2)
        MRR  = 1 / (r + 1)
        Rec@k = 1 if r < k        (k = 1, 5, 10, 20)
    Report the mean over test queries, x100. (Metric formulas copied verbatim from
    eval_asl_citizen_retrieval.evaluate to guarantee identical numbers.)

Reuses eval_asl_citizen_retrieval.load_metadata / load_features so feature loading
and clip mean-pooling are byte-for-byte identical to Eval A.

Usage
  python eval_asl_citizen_linear_probe.py \\
      --feature_dir /home/psobecki/ASL_Citizen/logos_features_run2 \\
      --splits_dir  /home/psobecki/ASL_Citizen/splits \\
      --feature_name logos_run2 \\
      --output_json runs/run2_evalB.json
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from eval_asl_citizen_retrieval import (
    load_metadata, load_features, DEFAULT_LOGOS_DIR, DEFAULT_SPLITS, DATASET_NAME,
)


def train_probe(train_feats, train_labels, n_gloss, device,
                epochs=50, batch_size=1024, lr=1e-3, weight_decay=1e-4,
                l2_normalize=False):
    """Fit a Linear(D -> n_gloss) classifier on frozen features with AdamW."""
    X = torch.from_numpy(train_feats).float()
    if l2_normalize:
        X = torch.nn.functional.normalize(X, dim=-1)
    y = torch.from_numpy(train_labels).long()
    N, D = X.shape

    probe = nn.Linear(D, n_gloss).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    X, y = X.to(device), y.to(device)
    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = ce(probe(X[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  probe epoch {epoch+1}/{epochs}  ce={total/N:.4f}")
    return probe


@torch.no_grad()
def evaluate_probe(probe, test_feats, test_labels, device,
                   batch_size=4096, l2_normalize=False):
    """Rank glosses by probe logits; compute DCG/MRR/Rec@k (identical to Eval A)."""
    X = torch.from_numpy(test_feats).float()
    if l2_normalize:
        X = torch.nn.functional.normalize(X, dim=-1)
    y = torch.from_numpy(test_labels).long().to(device)

    sums = np.zeros(6, dtype=np.float64)   # [DCG, Top1, Top5, Top10, Top20, MRR]
    n_eval = 0
    for i in range(0, X.size(0), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size]
        logits = probe(xb)                                  # (b, n_gloss)
        gt = logits.gather(1, yb.unsqueeze(1))              # (b,1)
        ranks = (logits > gt).sum(dim=1).double()           # 0-indexed rank of GT

        dcg = 1.0 / torch.log2(ranks + 2.0)
        mrr = 1.0 / (ranks + 1.0)
        sums += np.array([
            dcg.sum().item(),
            (ranks < 1).double().sum().item(),
            (ranks < 5).double().sum().item(),
            (ranks < 10).double().sum().item(),
            (ranks < 20).double().sum().item(),
            mrr.sum().item(),
        ])
        n_eval += xb.size(0)

    means = sums / n_eval
    return {
        "n_queries": n_eval,
        "DCG":   100.0 * means[0],
        "Rec@1": 100.0 * means[1],
        "Rec@5": 100.0 * means[2],
        "Rec@10": 100.0 * means[3],
        "Rec@20": 100.0 * means[4],
        "MRR":   100.0 * means[5],
    }


def print_results(res, feature_name, feat_dim, n_gloss):
    W = 60
    print("\n" + "=" * W)
    print(f"ASL-CITIZEN  LINEAR-PROBE  ({feature_name})")
    print("(frozen features + trained linear classifier — SignRep Table 3 protocol)")
    print("=" * W)
    print(f"  Feature dim   : {feat_dim}")
    print(f"  Gloss classes : {n_gloss}")
    print(f"  Query videos  : {res['n_queries']}")
    print("  " + "-" * (W - 2))
    print(f"  DCG    : {res['DCG']:6.2f}")
    print(f"  MRR    : {res['MRR']:6.2f}")
    print(f"  Rec@1  : {res['Rec@1']:6.2f}")
    print(f"  Rec@5  : {res['Rec@5']:6.2f}")
    print(f"  Rec@10 : {res['Rec@10']:6.2f}")
    print(f"  Rec@20 : {res['Rec@20']:6.2f}")
    print("=" * W)
    print("  SignRep Table 3 (ASL-Citizen, frozen encoder + linear classifier):")
    print("    ST-GCN*  DCG 76.37  MRR 69.97  Rec@1 59.52  Rec@5 82.68")
    print("    I3D*     DCG 79.13  MRR 73.32  Rec@1 63.10  Rec@5 86.09")
    print("    SignRep  DCG 90.84  MRR 88.05  Rec@1 81.37  Rec@5 96.11")
    print("=" * W)


def main():
    global DATASET_NAME
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature_dir", type=Path, default=Path(DEFAULT_LOGOS_DIR))
    ap.add_argument("--splits_dir", type=Path, default=Path(DEFAULT_SPLITS))
    ap.add_argument("--feature_name", type=str, default="logos")
    ap.add_argument("--gallery_split", type=str, default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--query_split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    ap.add_argument("--l2_normalize", action="store_true",
                    help="L2-normalize features before the probe (match retrieval geometry)")
    ap.add_argument("--probe_epochs", type=int, default=50)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--probe_batch_size", type=int, default=1024)
    ap.add_argument("--max_gallery", type=int, default=None)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_json", type=Path, default=None)
    args = ap.parse_args()

    import eval_asl_citizen_retrieval as evalA
    evalA.DATASET_NAME = args.dataset_name
    DATASET_NAME = args.dataset_name
    device = torch.device(args.device)

    if not args.feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {args.feature_dir}")

    print(f"Feature dir   : {args.feature_dir}")
    print(f"Gallery/query : {args.gallery_split} → {args.query_split}")
    print(f"Device        : {device}")

    train_recs = load_metadata(args.splits_dir, args.gallery_split)
    test_recs = load_metadata(args.splits_dir, args.query_split)

    train_feats, train_glosses = load_features(
        train_recs, args.feature_dir, args.max_gallery,
        desc=f"Loading train ({args.gallery_split})")
    test_feats, test_glosses = load_features(
        test_recs, args.feature_dir, args.max_queries,
        desc=f"Loading test ({args.query_split})")

    # Gloss vocab from the gallery (train) — same OrderedDict rule as Eval A's evaluate().
    g_dict = OrderedDict()
    for g in train_glosses:
        if g not in g_dict:
            g_dict[g] = len(g_dict)
    n_gloss = len(g_dict)

    train_labels = np.array([g_dict[g] for g in train_glosses], dtype=np.int64)

    # Drop test queries whose gloss is absent from the gallery vocab (closed-vocab).
    keep = [i for i, g in enumerate(test_glosses) if g in g_dict]
    if len(keep) < len(test_glosses):
        print(f"  WARNING: {len(test_glosses)-len(keep)} test queries have a gloss not in "
              f"the gallery vocab — skipped (closed-vocab).")
    test_feats = test_feats[keep]
    test_labels = np.array([g_dict[test_glosses[i]] for i in keep], dtype=np.int64)

    feat_dim = train_feats.shape[1]
    print(f"Train: {train_feats.shape[0]}  Test: {test_feats.shape[0]}  "
          f"Glosses: {n_gloss}  Dim: {feat_dim}")

    probe = train_probe(train_feats, train_labels, n_gloss, device,
                        epochs=args.probe_epochs, batch_size=args.probe_batch_size,
                        lr=args.probe_lr, weight_decay=args.probe_wd,
                        l2_normalize=args.l2_normalize)
    res = evaluate_probe(probe, test_feats, test_labels, device,
                         l2_normalize=args.l2_normalize)

    print_results(res, args.feature_name, feat_dim, n_gloss)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"feature_name": args.feature_name, "eval": "linear_probe",
                       "feature_dim": feat_dim, "n_gloss": n_gloss, **res}, f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
