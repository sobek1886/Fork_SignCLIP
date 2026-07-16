#!/usr/bin/env python3
"""
NGT zero-shot cross-domain sentence retrieval evaluation.

Evaluates whether a trained model generalises to the unseen Bushuis test
signers (different studio, disjoint signers) by cross-domain retrieval:
embed the train-signer views and the Bushuis view of each matched sentence,
then rank.

Two manifest styles are auto-detected:

  NEW (take-list, from make_ngt_singleview_manifest.py):
      sentence -> [ {"real": [FRONT, ...other views], "unreal": [...]}, ... ]
    Take 0 is used deterministically. Primary readout = the FRONT view alone
    (train/eval consistent for the front-anchored arms); secondary = average
    of all available real views. Also reports:
      - per-sentence ranks (for paired significance between arms), and
      - a length-only shortcut baseline: gallery ranked by |Δ clip count|.
        Sentence duration could partially identify the match when each query
        has exactly one relevant item; every arm must clear this floor.

  LEGACY (dict, from make_ngt_pair_manifest.py):
      sentence -> {"real": [LEFT, MIDDLE, RIGHT], "unreal": [...]}
    Original behaviour is preserved (MIDDLE and avg galleries, --use_unreal).

Usage:
    python eval_ngt_retrieval.py \\
        --config      projects/retri/signclip_ngt/ngt_sv_flux_full.yaml \\
        --checkpoint  runs/retri_ngt/ngt_sv_flux_full_run1/checkpoint_last.pt \\
        --manifest    /home/psobecki/ngt_sv_eval_manifest.json \\
        --bushuis_dir /scratch-shared/psobecki/Bushuis/logos_features \\
        --output_json runs/.../eval_results.json
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


def clip_count(path: str, max_video_len: int) -> int:
    """Number of Logos clips the model actually sees (truncated at T)."""
    feat = np.load(path, mmap_mode='r')
    return int(min(feat.shape[0], max_video_len))


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def retrieval_ranks(query_vecs: torch.Tensor, gallery_vecs: torch.Tensor):
    """Rank (0-based) of the matching gallery item for every query (i ↔ i)."""
    q = torch.nn.functional.normalize(query_vecs, dim=-1)
    g = torch.nn.functional.normalize(gallery_vecs, dim=-1)
    sims = q @ g.T                                   # (N, N)
    order = sims.argsort(dim=1, descending=True)
    targets = torch.arange(len(q), device=order.device).unsqueeze(1)
    ranks = (order == targets).nonzero()[:, 1]
    return ranks.tolist()


def metrics_from_ranks(ranks) -> dict:
    ranks = np.asarray(ranks)
    N = len(ranks)
    return {
        "R@1":  100.0 * float((ranks < 1).mean()),
        "R@5":  100.0 * float((ranks < 5).mean()),
        "R@10": 100.0 * float((ranks < 10).mean()),
        "MRR":  100.0 * float((1.0 / (ranks + 1)).mean()),
        "N":    N,
    }


def retrieval_metrics(query_vecs: torch.Tensor, gallery_vecs: torch.Tensor) -> dict:
    return metrics_from_ranks(retrieval_ranks(query_vecs, gallery_vecs))


def length_baseline_ranks(query_lens, gallery_lens, seed=0):
    """Rank gallery by |Δ clip count| (seeded random tiebreak)."""
    rng = np.random.default_rng(seed)
    q = np.asarray(query_lens, dtype=np.float64)
    g = np.asarray(gallery_lens, dtype=np.float64)
    ranks = []
    for i in range(len(q)):
        score = -np.abs(g - q[i]) + rng.uniform(0, 1e-6, size=len(g))
        order = np.argsort(-score)
        ranks.append(int(np.where(order == i)[0][0]))
    return ranks


def fmt(m: dict) -> str:
    return (f"R@1={m['R@1']:.1f}%  R@5={m['R@5']:.1f}%  "
            f"R@10={m['R@10']:.1f}%  MRR={m['MRR']:.1f}%")


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def _log_eval_metrics_to_mlflow(config, config_path: str, metrics: dict):
    """Log NGT retrieval metrics into the corresponding MLflow training run."""
    try:
        import mlflow as _mlflow
        experiment_name = getattr(config, "mlflow_experiment", None)
        if not experiment_name:
            print("[MLflow] 'mlflow_experiment' not set in config; skipping.")
            return
        run_name = os.path.splitext(os.path.basename(config_path))[0]
        _mlflow.set_tracking_uri("https://mlflow.ai.mytkhgroup.com/")
        _mlflow.set_experiment(experiment_name)
        client = _mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            print(f"[MLflow] Experiment '{experiment_name}' not found; skipping.")
            return
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.run_name = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            print(f"[MLflow] No run named '{run_name}' found; skipping.")
            return
        run_id = runs[0].info.run_id
        flat = {f"eval_{k}": float(v) for k, v in metrics.items()}
        with _mlflow.start_run(run_id=run_id):
            _mlflow.log_metrics(flat)
        print(f"[MLflow] Logged to run '{run_name}' ({run_id}): "
              + ", ".join(f"{k}={v:.2f}" for k, v in flat.items()))
    except Exception as exc:
        print(f"[MLflow] Warning: could not log eval metrics: {exc}")


def flatten_metric_groups(groups: dict) -> dict:
    out = {}
    for key, m in groups.items():
        for metric, val in m.items():
            if metric == "N":
                continue
            out[f"{key}_{metric.replace('@', '')}"] = val
    return out


# ---------------------------------------------------------------------------
# NEW style: take-list manifests (front-anchored arms)
# ---------------------------------------------------------------------------

def eval_front_style(args, model, config, eval_ids, manifest, bushuis_map,
                     max_video_len, cls_id, sep_id, device):
    emb_front, emb_avg, emb_bushuis = {}, {}, {}
    len_front, len_bushuis = {}, {}

    n_views = None
    for sid in tqdm(eval_ids, desc="Embedding"):
        take0 = manifest[sid][0]
        real_paths = take0["real"]           # FRONT first, by builder contract
        if n_views is None:
            n_views = len(real_paths)
            print(f"Real views per sentence: {n_views} (front first)")
        views = [embed_npy(p, model, max_video_len, cls_id, sep_id, device)
                 for p in real_paths]
        emb_front[sid] = views[0]
        emb_avg[sid] = torch.stack(views).mean(0)
        emb_bushuis[sid] = embed_npy(
            bushuis_map[sid], model, max_video_len, cls_id, sep_id, device)
        len_front[sid] = clip_count(real_paths[0], max_video_len)
        len_bushuis[sid] = clip_count(bushuis_map[sid], max_video_len)

    front_vecs   = torch.stack([emb_front[s]   for s in eval_ids])
    avg_vecs     = torch.stack([emb_avg[s]     for s in eval_ids])
    bushuis_vecs = torch.stack([emb_bushuis[s] for s in eval_ids])

    b2s_front_ranks = retrieval_ranks(bushuis_vecs, front_vecs)
    s2b_front_ranks = retrieval_ranks(front_vecs, bushuis_vecs)

    groups = {
        "b2s_front": metrics_from_ranks(b2s_front_ranks),
        "s2b_front": metrics_from_ranks(s2b_front_ranks),
        "b2s_avg":   retrieval_metrics(bushuis_vecs, avg_vecs),
        "s2b_avg":   retrieval_metrics(avg_vecs, bushuis_vecs),
    }

    lens_q = [len_bushuis[s] for s in eval_ids]
    lens_g = [len_front[s] for s in eval_ids]
    length_groups = {
        "len_b2s": metrics_from_ranks(length_baseline_ranks(lens_q, lens_g)),
        "len_s2b": metrics_from_ranks(length_baseline_ranks(lens_g, lens_q)),
    }

    print(f"\nSentences evaluated: {len(eval_ids)}\n")
    print("Query/gallery: FRONT view (primary)")
    print(f"  Bushuis → Signer:  {fmt(groups['b2s_front'])}")
    print(f"  Signer → Bushuis:  {fmt(groups['s2b_front'])}")
    print("\nQuery/gallery: avg of real views (secondary)")
    print(f"  Bushuis → Signer:  {fmt(groups['b2s_avg'])}")
    print(f"  Signer → Bushuis:  {fmt(groups['s2b_avg'])}")
    print("\nLength-only shortcut baseline (|Δ clip count|):")
    print(f"  Bushuis → Signer:  {fmt(length_groups['len_b2s'])}")
    print(f"  Signer → Bushuis:  {fmt(length_groups['len_s2b'])}")

    results = {
        "style": "front",
        "n_sentences": len(eval_ids),
        **groups,
        "length_baseline": length_groups,
        "per_sentence": {
            sid: {"b2s_front_rank": int(b2s_front_ranks[i]),
                  "s2b_front_rank": int(s2b_front_ranks[i])}
            for i, sid in enumerate(eval_ids)
        },
    }
    mlflow_metrics = flatten_metric_groups({**groups, **length_groups})
    return results, mlflow_metrics


# ---------------------------------------------------------------------------
# LEGACY style: dict manifests from make_ngt_pair_manifest.py
# ---------------------------------------------------------------------------

def eval_legacy_style(args, model, config, eval_ids, manifest, bushuis_map,
                      max_video_len, cls_id, sep_id, device):
    emb_left, emb_middle, emb_right, emb_bushuis = {}, {}, {}, {}

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

    middle_vecs  = torch.stack([emb_middle[s]  for s in eval_ids])
    bushuis_vecs = torch.stack([emb_bushuis[s] for s in eval_ids])
    avg_vecs     = torch.stack([
        (emb_left[s] + emb_middle[s] + emb_right[s]) / 3.0
        for s in eval_ids
    ])

    print(f"\nSigns evaluated: {len(eval_ids)}\n")

    groups = {
        "b2s_mid": retrieval_metrics(bushuis_vecs, middle_vecs),
        "s2b_mid": retrieval_metrics(middle_vecs, bushuis_vecs),
        "b2s_avg": retrieval_metrics(bushuis_vecs, avg_vecs),
        "s2b_avg": retrieval_metrics(avg_vecs, bushuis_vecs),
    }

    print("Gallery: MIDDLE only")
    print(f"  Bushuis → {signer_label}:  {fmt(groups['b2s_mid'])}")
    print(f"  {signer_label} → Bushuis:  {fmt(groups['s2b_mid'])}")
    print("\nGallery: avg(LEFT + MIDDLE + RIGHT)")
    print(f"  Bushuis → {signer_label}:  {fmt(groups['b2s_avg'])}")
    print(f"  {signer_label} → Bushuis:  {fmt(groups['s2b_avg'])}")

    return dict(groups), flatten_metric_groups(groups)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",      required=True,
                        help="Task config YAML (e.g. ngt_sv_flux_full.yaml)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Trained checkpoint (.pt)")
    parser.add_argument("--manifest",    required=True,
                        help="Eval manifest (take-list style: "
                             "ngt_sv_eval_manifest.json; legacy: "
                             "ngt_pair_manifest.json)")
    parser.add_argument("--bushuis_dir", required=True,
                        help="Directory containing bushuis_M*.npy features")
    parser.add_argument("--use_unreal", action="store_true",
                        help="[legacy manifests only] query with unreal features")
    parser.add_argument("--unreal_character", type=int, default=0,
                        help="[legacy] 0=palmer, 1=digits (default: 0)")
    parser.add_argument("--exclude_ids", default=None,
                        help="Optional text file with sentence IDs to exclude "
                             "(one per line, e.g. pruned test duplicates)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", default=None,
                        help="If given, save eval results dict to this JSON file")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint} ...")
    model, config = load_model(args.config, args.checkpoint, device)
    max_video_len = config.dataset.max_video_len

    tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    with open(args.manifest) as f:
        manifest = json.load(f)

    _BUSHUIS_RE = re.compile(r'^bushuis_(M\d{8}_\d+)\.npy$')
    bushuis_map = {}
    for p in Path(args.bushuis_dir).glob("*.npy"):
        m = _BUSHUIS_RE.match(p.name)
        if m:
            bushuis_map[m.group(1)] = str(p)

    eval_ids = sorted(set(manifest) & set(bushuis_map))
    if args.exclude_ids:
        with open(args.exclude_ids) as f:
            excl = {line.strip() for line in f if line.strip()}
        before = len(eval_ids)
        eval_ids = [s for s in eval_ids if s not in excl]
        print(f"Excluded {before - len(eval_ids)} of {len(excl)} listed IDs.")
    if not eval_ids:
        sys.exit("ERROR: no matched sentence IDs between manifest and bushuis_dir.")

    new_style = isinstance(manifest[eval_ids[0]], list)
    print(f"Manifest style: {'take-list (front)' if new_style else 'legacy dict'}")

    if new_style:
        results, mlflow_metrics = eval_front_style(
            args, model, config, eval_ids, manifest, bushuis_map,
            max_video_len, cls_id, sep_id, device)
    else:
        results, mlflow_metrics = eval_legacy_style(
            args, model, config, eval_ids, manifest, bushuis_map,
            max_video_len, cls_id, sep_id, device)

    _log_eval_metrics_to_mlflow(config, args.config, mlflow_metrics)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
