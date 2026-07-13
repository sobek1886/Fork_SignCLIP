#!/usr/bin/env python3
"""
analyze_v2t_significance.py - Paired significance and error structure for the
                              SignCLIP video-to-text results (analyses S5 + S7)

S5 — McNemar test + bootstrap CI. The in-domain headline "+1.1 R@1 from
augmentation (0.789 -> 0.800)" is currently reported without any significance
statement. Because both systems score the SAME test videos, the right test is
paired: McNemar's exact test on the per-video top-1 correctness of two
configurations, plus a percentile-bootstrap CI on the R@1 difference.

S7 — variant-gloss error concentration. Validates mechanism B of
context/v2t-performance-analysis.md: if the text tower struggles to separate
numbered variant glosses ("CONFUSED" vs "CONFUSED 2", identical strings except
one token), V2T errors should concentrate on within-family confusions far
above the chance rate implied by family sizes.

Per config, this recomputes per-video V2T top-1 predictions from the trained
checkpoint: pooled_video per test video, pooled_text per UNIQUE caption
(identical strings give identical embeddings in eval mode, so one embedding
per gloss — matching RWTHFSV2TMetric's dedup semantics), rank the unique
texts per video. R@1 printed per config should match the reported table
values; if it doesn't, stop and investigate before quoting anything.

Usage (on Snellius, GPU)
  python analyze_v2t_significance.py \
      --config ft=projects/retri/signclip_asl/test_asl_citizen_cnn_logos_ft10.yaml \
      --config ft_aug=projects/retri/signclip_asl/test_asl_citizen_cnn_aug_logos_ft.yaml \
      --split test \
      --out_json runs/v2t_significance/ft_vs_aug.json
"""
import argparse
import json
import math
import re
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asl_citizen_signclip_l2_probe import (  # noqa: E402
    build_dataloader, load_model, disable_struct_recursive,
)
from mmpt.utils import load_config  # noqa: E402


def parse_config(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--config must be NAME=PATH, got {s!r}")
    name, p = s.split("=", 1)
    return name, p


def gloss_base(text):
    """'<en> <ase> CONFUSED 2' → ('CONFUSED 2', 'CONFUSED')."""
    gloss = re.sub(r"^<[^>]*>\s*(<[^>]*>\s*)*", "", text).strip()
    base = re.sub(r"\s+\d+$", "", gloss)
    return gloss, base


@torch.no_grad()
def predict_config(config_path, split, batch_size, device):
    """→ dict video_id → (true_text, pred_text, correct, rank)."""
    config = load_config(Namespace(taskconfig=str(config_path)))
    disable_struct_recursive(config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name)
    model = load_model(config)
    model.eval().to(device)
    dataloader = build_dataloader(config, split, batch_size)

    vids, vecs, texts = [], [], []
    text_emb = {}       # unique caption string → embedding
    for batch in tqdm(dataloader, desc=f"forward {Path(str(config_path)).stem}"):
        video_ids = None
        if isinstance(batch, (list, tuple)) and len(batch) >= 5:
            video_ids = batch[4]
            batch = {"caps": batch[0], "cmasks": batch[1],
                     "vfeats": batch[2], "vmasks": batch[3]}
        elif isinstance(batch, dict) and "video_id" in batch:
            video_ids = batch.pop("video_id")
        caps_cpu = batch["caps"]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(**batch)
        pv = out["pooled_video"].float().cpu().numpy()
        pt = out["pooled_text"].float().cpu().numpy()
        for j in range(pv.shape[0]):
            cap = tokenizer.decode(caps_cpu[j], skip_special_tokens=True)
            vid = video_ids[j] if video_ids is not None else len(vids)
            if torch.is_tensor(vid):
                vid = vid.item()
            vids.append(str(vid)); vecs.append(pv[j]); texts.append(cap)
            if cap not in text_emb:
                text_emb[cap] = pt[j]

    V = np.stack(vecs)
    uniq = sorted(text_emb)
    T = np.stack([text_emb[u] for u in uniq])
    t_idx = {u: i for i, u in enumerate(uniq)}
    sims = V @ T.T                                   # (N videos, U texts)
    order = np.argsort(-sims, axis=1)
    preds = {}
    for i, vid in enumerate(vids):
        own = t_idx[texts[i]]
        rank = int(np.where(order[i] == own)[0][0])
        preds[vid] = {"true": texts[i], "pred": uniq[order[i][0]],
                      "correct": bool(rank == 0), "rank": rank}
    r1 = float(np.mean([p["correct"] for p in preds.values()]))
    print(f"  {len(preds)} videos, {len(uniq)} unique texts, R@1={r1:.4f} "
          f"(sanity-check against the reported table value)")
    anchor_stats = text_anchor_proximity(uniq, T)
    return preds, anchor_stats


def text_anchor_proximity(uniq, T, seed=0):
    """Are variant-family text anchors ('X' vs 'X 2') really close? (todo #18)

    Returns mean cosine between text anchors WITHIN a variant family vs the
    mean cosine of random unrelated anchor pairs — the direct measurement of
    the anchor-proximity hypothesis in the Discussion's mechanism B."""
    Tn = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    fam = defaultdict(list)
    for i, u in enumerate(uniq):
        _, base = gloss_base(u)
        fam[base].append(i)
    within = []
    for idxs in fam.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                within.append(float(Tn[idxs[a]] @ Tn[idxs[b]]))
    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(max(1000, len(within) * 10)):
        i, j = rng.integers(0, len(uniq), 2)
        if i != j:
            rand.append(float(Tn[i] @ Tn[j]))
    stats = {
        "n_family_pairs": len(within),
        "mean_cos_within_family": float(np.mean(within)) if within else None,
        "mean_cos_random_pairs": float(np.mean(rand)),
        "p90_cos_random_pairs": float(np.percentile(rand, 90)),
    }
    if within:
        print(f"  text-anchor proximity: within-family cos "
              f"{stats['mean_cos_within_family']:.4f} vs random "
              f"{stats['mean_cos_random_pairs']:.4f} "
              f"(random p90 {stats['p90_cos_random_pairs']:.4f}, "
              f"{len(within)} family pairs)")
    return stats


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", type=parse_config, required=True,
                    help="NAME=test_yaml; repeat (>=2 for McNemar pairs)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    systems = {}
    anchor_proximity = {}
    for name, path in dict(args.config).items():
        print(f"\n=== Config: {name} ({path}) ===")
        systems[name], anchor_proximity[name] = predict_config(
            path, args.split, args.batch_size, device)

    results = {"r1": {}, "mcnemar": {}, "variant_errors": {},
               "text_anchor_proximity": anchor_proximity}
    for name, preds in systems.items():
        results["r1"][name] = float(np.mean([p["correct"]
                                             for p in preds.values()]))

    # ---- S5: McNemar + bootstrap for every config pair ----
    rng = np.random.default_rng(args.seed)
    names = list(systems)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = sorted(set(systems[a]) & set(systems[b]))
            ca = np.array([systems[a][v]["correct"] for v in common])
            cb = np.array([systems[b][v]["correct"] for v in common])
            disc_b = int((ca & ~cb).sum())   # a right, b wrong
            disc_c = int((~ca & cb).sum())   # a wrong, b right
            p = mcnemar_exact(disc_b, disc_c)
            diffs = []
            n = len(common)
            for _ in range(args.n_bootstrap):
                idx = rng.integers(0, n, n)
                diffs.append(cb[idx].mean() - ca[idx].mean())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            results["mcnemar"][f"{a}|{b}"] = {
                "n_common": n, "r1_a": float(ca.mean()), "r1_b": float(cb.mean()),
                "delta_r1_b_minus_a": float(cb.mean() - ca.mean()),
                "a_right_b_wrong": disc_b, "a_wrong_b_right": disc_c,
                "mcnemar_p_two_sided": p,
                "bootstrap_ci95": [float(lo), float(hi)],
            }
            print(f"\nMcNemar {a} vs {b}: ΔR@1={cb.mean()-ca.mean():+.4f} "
                  f"(b={disc_b}, c={disc_c}, p={p:.4g}, "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}])")

    # ---- S7: variant-family error concentration, per config ----
    for name, preds in systems.items():
        fam = defaultdict(set)
        for p in preds.values():
            g, base = gloss_base(p["true"])
            fam[base].add(g)
        n_gloss = len({gloss_base(p["true"])[0] for p in preds.values()})
        errors = [p for p in preds.values() if not p["correct"]]
        sib = 0
        for p in errors:
            tg, tb = gloss_base(p["true"])
            pg, pb = gloss_base(p["pred"])
            if tb == pb and tg != pg:
                sib += 1
        # chance: probability a uniformly-random wrong gloss is a sibling
        chance = float(np.mean([
            (len(fam[gloss_base(p["true"])[1]]) - 1) / max(1, n_gloss - 1)
            for p in errors])) if errors else 0.0
        frac = sib / len(errors) if errors else 0.0
        results["variant_errors"][name] = {
            "n_errors": len(errors), "sibling_errors": sib,
            "sibling_fraction": frac, "chance_fraction": chance,
            "enrichment": (frac / chance) if chance > 0 else None,
            "n_variant_families_gt1": int(sum(len(v) > 1 for v in fam.values())),
        }
        print(f"\n{name}: {sib}/{len(errors)} errors are within a variant "
              f"family ({frac:.3f} vs chance {chance:.4f}"
              + (f", enrichment {frac/chance:.0f}x" if chance else "") + ")")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
