#!/usr/bin/env python3
"""
NGT zero-shot retrieval evaluation.

Evaluates whether the intervention model generalises better to unseen Bushuis
signers by computing cross-domain retrieval metrics.

For each matched sign ID: embed piotrAnims views and the Bushuis view through
the trained video encoder, then run cross-domain retrieval.

Usage:
    python eval_ngt_retrieval.py \\
        --config      projects/retri/signclip_ngt/ngt_baseline.yaml \\
        --checkpoint  runs/retri_ngt/ngt_baseline/checkpoint_last.pt \\
        --manifest    /home/psobecki/ngt_pair_manifest.json \\
        --bushuis_dir /home/psobecki/Bushuis/logos_features \\
        [--device     cuda]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmpt.tasks import Task


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    config = OmegaConf.load(config_path)
    task = Task.config_task(config)
    task.build_model()
    task.load_checkpoint(checkpoint_path)
    model = task.model.eval().to(device)
    return model, config


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_npy(
    path: str,
    model: torch.nn.Module,
    max_video_len: int,
    cls_id: int,
    sep_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Load a .npy feature file and return the pooled video embedding (D,)."""
    feat = np.load(path).astype(np.float32)   # (N_clips, 768)
    N = feat.shape[0]
    T = max_video_len

    vfeats = np.zeros((1, T, 768), dtype=np.float32)
    vmasks = np.zeros((1, T), dtype=bool)
    n = min(N, T)
    vfeats[0, :n] = feat[:n]
    vmasks[0, :n] = True

    vfeats_t = torch.from_numpy(vfeats).to(device)          # (1, T, 768)
    vmasks_t = torch.from_numpy(vmasks).to(device)          # (1, T)
    caps     = torch.tensor([[cls_id, sep_id]], dtype=torch.long, device=device)
    cmasks   = torch.ones(1, 2, dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model.forward_video(vfeats_t, vmasks_t, caps, cmasks)  # (1, D)

    return out.squeeze(0).float()


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def retrieval_metrics(query_vecs: torch.Tensor, gallery_vecs: torch.Tensor) -> dict:
    """R@1/5/10 and MRR for aligned query/gallery (query i matches gallery i)."""
    N = query_vecs.size(0)
    q = torch.nn.functional.normalize(query_vecs, dim=-1)
    g = torch.nn.functional.normalize(gallery_vecs, dim=-1)
    sims = q @ g.T   # (N, N)

    r1 = r5 = r10 = 0
    rr = 0.0
    for i in range(N):
        order = sims[i].argsort(descending=True).tolist()
        rank  = order.index(i)
        if rank < 1:
            r1 += 1
        if rank < 5:
            r5 += 1
        if rank < 10:
            r10 += 1
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",      required=True,
                        help="Task config YAML (e.g. ngt_baseline.yaml)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Trained checkpoint (.pt)")
    parser.add_argument("--manifest",    required=True,
                        help="Path to ngt_pair_manifest.json")
    parser.add_argument("--bushuis_dir", required=True,
                        help="Directory containing bushuis_M*.npy features")
    parser.add_argument("--use_unreal", action="store_true",
                        help="Use unreal (palmer) features instead of real (piotrAnims)")
    parser.add_argument("--unreal_character", type=int, default=0,
                        help="Which unreal character to use: 0=palmer, 1=digits (default: 0)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print(f"Loading model from {args.checkpoint} ...")
    model, config = load_model(args.config, args.checkpoint, device)
    max_video_len = config.dataset.max_video_len

    tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # ------------------------------------------------------------------ #
    # Sign matching
    # ------------------------------------------------------------------ #
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
        sys.exit("ERROR: no matched sign IDs between manifest and bushuis_dir.")

    # ------------------------------------------------------------------ #
    # Embedding extraction
    # ------------------------------------------------------------------ #
    emb_left    = {}
    emb_middle  = {}
    emb_right   = {}
    emb_bushuis = {}

    _VIEWS_PER_CHARACTER = 3
    signer_label = "palmer" if args.use_unreal else "piotrAnims"

    for sign_id in tqdm(eval_ids, desc="Embedding"):
        if args.use_unreal:
            start = args.unreal_character * _VIEWS_PER_CHARACTER
            paths = manifest[sign_id]["unreal"][start:start + _VIEWS_PER_CHARACTER]
        else:
            paths = manifest[sign_id]["real"]   # [LEFT, MIDDLE, RIGHT]
        emb_left[sign_id]    = embed_npy(paths[0], model, max_video_len, cls_id, sep_id, device)
        emb_middle[sign_id]  = embed_npy(paths[1], model, max_video_len, cls_id, sep_id, device)
        emb_right[sign_id]   = embed_npy(paths[2], model, max_video_len, cls_id, sep_id, device)
        emb_bushuis[sign_id] = embed_npy(
            bushuis_map[sign_id], model, max_video_len, cls_id, sep_id, device
        )

    # ------------------------------------------------------------------ #
    # Stack tensors for both gallery modes
    # ------------------------------------------------------------------ #
    middle_vecs  = torch.stack([emb_middle[s]  for s in eval_ids])
    bushuis_vecs = torch.stack([emb_bushuis[s] for s in eval_ids])
    avg_vecs     = torch.stack([
        (emb_left[s] + emb_middle[s] + emb_right[s]) / 3.0
        for s in eval_ids
    ])

    # ------------------------------------------------------------------ #
    # Retrieval results
    # ------------------------------------------------------------------ #
    print(f"\nSigns evaluated: {len(eval_ids)}\n")

    print("Gallery: MIDDLE only")
    print(f"  Bushuis → {signer_label}:  {fmt(retrieval_metrics(bushuis_vecs, middle_vecs))}")
    print(f"  {signer_label} → Bushuis:  {fmt(retrieval_metrics(middle_vecs, bushuis_vecs))}")

    print("\nGallery: avg(LEFT + MIDDLE + RIGHT)")
    print(f"  Bushuis → {signer_label}:  {fmt(retrieval_metrics(bushuis_vecs, avg_vecs))}")
    print(f"  {signer_label} → Bushuis:  {fmt(retrieval_metrics(avg_vecs, bushuis_vecs))}")


if __name__ == "__main__":
    main()
