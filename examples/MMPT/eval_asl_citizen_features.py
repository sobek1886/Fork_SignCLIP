#!/usr/bin/env python3
"""
eval_asl_citizen_features.py  –  Run BOTH Eval A (raw NN retrieval) and Eval B
                                  (linear probe) on one ASL-Citizen feature dir,
                                  loading the features ONCE and logging both into a
                                  SINGLE MLflow run for easy A-vs-B comparison.

This is the combined driver for the per-feature-set evaluation. It reuses:
  - eval_asl_citizen_retrieval.{load_metadata,load_features,evaluate}  (Eval A)
  - eval_asl_citizen_linear_probe.{train_probe,evaluate_probe}          (Eval B)
so the metrics are byte-identical to the standalone scripts.

One fresh MLflow run per invocation (no reuse/overwrite), named --feature_name,
containing evalA_* and evalB_* metrics side by side.

Usage
  python eval_asl_citizen_features.py \\
      --feature_name baseline_endanchor \\
      --feature_dir  /home/psobecki/ASL_Citizen/logos_features_endanchor \\
      --splits_dir   /home/psobecki/ASL_Citizen/splits \\
      --mlflow --output_json runs/asl_ft_eval/baseline_endanchor.json
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from eval_asl_citizen_retrieval import (
    load_metadata, load_features, evaluate, DEFAULT_LOGOS_DIR, DEFAULT_SPLITS, DATASET_NAME,
)
from eval_asl_citizen_linear_probe import train_probe, evaluate_probe


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
    ap.add_argument("--batch_size", type=int, default=256, help="Eval A query batch")
    # Eval B (linear probe) hyperparams
    ap.add_argument("--l2_normalize", action="store_true")
    ap.add_argument("--probe_epochs", type=int, default=50)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--probe_batch_size", type=int, default=1024)
    ap.add_argument("--max_gallery", type=int, default=None)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--load_workers", type=int, default=32,
                    help="threads for parallel .npy loading (scratch-shared is I/O-latency bound)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_json", type=Path, default=None)
    ap.add_argument("--mlflow", action="store_true",
                    help="Log evalA_* and evalB_* into ONE fresh MLflow run")
    ap.add_argument("--mlflow_experiment", type=str, default="asl-citizen-feature-eval")
    ap.add_argument("--mlflow_run_name", type=str, default=None,
                    help="MLflow run name (default: --feature_name)")
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

    # ── Load features ONCE ────────────────────────────────────────────────────
    gallery_recs = load_metadata(args.splits_dir, args.gallery_split)
    query_recs = load_metadata(args.splits_dir, args.query_split)
    gallery_feats, gallery_glosses = load_features(
        gallery_recs, args.feature_dir, args.max_gallery,
        desc=f"Loading gallery ({args.gallery_split})", workers=args.load_workers)
    query_feats, query_glosses = load_features(
        query_recs, args.feature_dir, args.max_queries,
        desc=f"Loading queries ({args.query_split})", workers=args.load_workers)
    feat_dim = gallery_feats.shape[1]

    # ── Eval A: raw NN retrieval ──────────────────────────────────────────────
    print("\n=== Eval A: raw video-to-video NN retrieval ===")
    resA = evaluate(gallery_feats, gallery_glosses, query_feats, query_glosses,
                    device=device, batch_size=args.batch_size)
    for k in ("DCG", "MRR", "Rec@1", "Rec@5", "Rec@10", "Rec@20"):
        print(f"  A {k:6s}: {resA[k]:6.2f}")

    # ── Eval B: linear probe ──────────────────────────────────────────────────
    print("\n=== Eval B: linear probe ===")
    g_dict = OrderedDict()
    for g in gallery_glosses:
        if g not in g_dict:
            g_dict[g] = len(g_dict)
    n_gloss = len(g_dict)
    train_labels = np.array([g_dict[g] for g in gallery_glosses], dtype=np.int64)
    keep = [i for i, g in enumerate(query_glosses) if g in g_dict]
    qfeats_b = query_feats[keep]
    qlabels_b = np.array([g_dict[query_glosses[i]] for i in keep], dtype=np.int64)

    probe = train_probe(gallery_feats, train_labels, n_gloss, device,
                        epochs=args.probe_epochs, batch_size=args.probe_batch_size,
                        lr=args.probe_lr, weight_decay=args.probe_wd,
                        l2_normalize=args.l2_normalize)
    resB = evaluate_probe(probe, qfeats_b, qlabels_b, device,
                          l2_normalize=args.l2_normalize)
    for k in ("DCG", "MRR", "Rec@1", "Rec@5", "Rec@10", "Rec@20"):
        print(f"  B {k:6s}: {resB[k]:6.2f}")

    # ── Log both into ONE MLflow run ──────────────────────────────────────────
    metric_keys = ["DCG", "MRR", "Rec@1", "Rec@5", "Rec@10", "Rec@20"]
    if args.mlflow:
        from mlflow_eval_logging import log_evals_to_mlflow
        log_evals_to_mlflow(
            experiment=args.mlflow_experiment,
            run_name=args.mlflow_run_name or args.feature_name,
            groups={"evalA_": {k: resA[k] for k in metric_keys},
                    "evalB_": {k: resB[k] for k in metric_keys}},
            params={"feature_dim": feat_dim, "n_gallery": resA["n_gallery"],
                    "n_gloss": n_gloss, "n_queries": resA["n_queries"],
                    "feature_dir": str(args.feature_dir),
                    "probe_epochs": args.probe_epochs, "l2_normalize": args.l2_normalize},
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"feature_name": args.feature_name, "feature_dim": feat_dim,
                       "evalA": {k: resA[k] for k in metric_keys},
                       "evalB": {k: resB[k] for k in metric_keys}}, f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
