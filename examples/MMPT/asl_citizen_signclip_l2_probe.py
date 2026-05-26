#!/usr/bin/env python3
"""
asl_citizen_signclip_l2_probe.py  –  L2-distance signer-bias probe on SignCLIP
                                      *output embeddings* (pooled_video).

Mirrors asl_citizen_feature_l2_probe.py but instead of probing raw input
features (Logos / I3D / MediaPipe), it probes the final pooled_video embeddings
produced by a trained SignCLIP model.  This answers:

  "Has SignCLIP training already compressed the signer signal, or is it
   still present in the output space where contrastive loss acts?"

Requires a GPU.

Usage
  # Logos model
  python asl_citizen_signclip_l2_probe.py \\
      --config projects/retri/signclip_asl/test_asl_citizen_cnn_scratch_logos.yaml \\
      --model_name logos

  # I3D WLASL model
  python asl_citizen_signclip_l2_probe.py \\
      --config projects/retri/signclip_asl/test_asl_citizen_cnn_scratch_wlasl.yaml \\
      --model_name i3d_wlasl

  # Use train split instead of test (more coverage)
  python asl_citizen_signclip_l2_probe.py \\
      --config projects/retri/signclip_asl/test_asl_citizen_cnn_scratch_logos.yaml \\
      --model_name logos --split train

  # Cache embeddings and skip re-running the model on subsequent runs
  python asl_citizen_signclip_l2_probe.py \\
      --config projects/retri/signclip_asl/test_asl_citizen_cnn_scratch_logos.yaml \\
      --model_name logos --cache_dir /scratch-shared/psobecki/signclip_embed_cache
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# mmpt imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # fairseq root

from mmpt.utils import load_config
from mmpt.tasks import Task
from mmpt import processors
from mmpt.datasets import MMDataset
from torch.utils.data import DataLoader

DATASET_NAME  = "asl_citizen"
DEFAULT_SPLITS_DIR = "/home/psobecki/ASL_Citizen/splits"
_SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_signer_map(splits_dir: Path) -> dict[str, tuple[str, str]]:
    """Build feat_id → (gloss, signer_id) from all three CSVs."""
    mapping = {}
    for csv_path in splits_dir.glob("*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                video_basename = os.path.splitext(row["Video file"])[0]
                feat_id = f"{DATASET_NAME}_{video_basename}"
                mapping[feat_id] = (row["Gloss"].strip(), row["Participant ID"].strip())
    return mapping


# ---------------------------------------------------------------------------
# Model + dataloader setup
# ---------------------------------------------------------------------------

def disable_struct_recursive(cfg):
    """Recursively disable OmegaConf struct mode on all nested configs."""
    from omegaconf import OmegaConf, DictConfig, ListConfig
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
        for key in cfg:
            try:
                disable_struct_recursive(cfg[key])
            except Exception:
                pass
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            try:
                disable_struct_recursive(item)
            except Exception:
                pass


def build_dataloader(config, split: str, batch_size: int) -> DataLoader:
    """Build a dataloader for a given split, overriding config.dataset.split."""
    config.dataset.split = split

    meta_processor_cls  = getattr(processors, config.dataset.meta_processor)
    video_processor_cls = getattr(processors, config.dataset.video_processor)
    text_processor_cls  = getattr(processors, config.dataset.text_processor)
    aligner_cls         = getattr(processors, config.dataset.aligner)

    meta_processor  = meta_processor_cls(config.dataset)
    video_processor = video_processor_cls(config.dataset)
    text_processor  = text_processor_cls(config.dataset)
    aligner         = aligner_cls(config.dataset)

    dataset = MMDataset(meta_processor, video_processor, text_processor, aligner)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collater,
    )
    return dataloader, meta_processor.data


def load_model(config) -> torch.nn.Module:
    checkpoint_path = config.fairseq.common_eval.path
    print(f"Loading checkpoint: {checkpoint_path}")
    mmtask = Task.config_task(config)
    mmtask.build_model()
    mmtask.load_checkpoint(checkpoint_path)
    return mmtask.model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    meta_data: list,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run model forward pass and collect feat_id → pooled_video embedding.

    Video IDs are read from the batch when available (works for all meta
    processors including SignCLIPMetaProcessor), falling back to index-based
    lookup from meta_data otherwise.
    """
    model.eval()
    model.to(device)

    # Fallback index list — works for SignCLIPVideoCSVMetaProcessor ("id" key)
    # and SignCLIPMetaProcessor ("pose" key, strip extension to get video_id)
    def _meta_id(d):
        if "id" in d:
            return d["id"]
        if "pose" in d:
            return d["pose"].replace(".pose", "")
        return None

    fallback_ids = [_meta_id(d) for d in meta_data]
    idx = 0
    embeddings = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Extract video_ids before converting batch to tensors
            video_ids = None
            if isinstance(batch, (list, tuple)) and len(batch) >= 5:
                video_ids = batch[4]
                batch = {"caps": batch[0], "cmasks": batch[1],
                         "vfeats": batch[2], "vmasks": batch[3]}
            elif isinstance(batch, dict) and "video_id" in batch:
                video_ids = batch.pop("video_id")

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            pooled = outputs["pooled_video"].cpu().numpy()  # (B, D)

            for j in range(pooled.shape[0]):
                if video_ids is not None:
                    vid = video_ids[j]
                    vid = vid.item() if hasattr(vid, "item") else str(vid)
                elif idx < len(fallback_ids):
                    vid = fallback_ids[idx]
                else:
                    idx += 1
                    continue
                embeddings[vid] = pooled[j]
                idx += 1

    print(f"Extracted {len(embeddings)} embeddings")
    return embeddings


# ---------------------------------------------------------------------------
# L2 analysis  (identical logic to asl_citizen_feature_l2_probe.py)
# ---------------------------------------------------------------------------

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def analyse(
    embeddings: dict[str, np.ndarray],
    signer_map: dict[str, tuple[str, str]],
    min_signers: int = 2,
    top_glosses: int = 20,
) -> dict:
    by_gloss: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for feat_id, emb in embeddings.items():
        if feat_id not in signer_map:
            continue
        gloss, signer = signer_map[feat_id]
        by_gloss[gloss][signer].append(emb)

    gloss_results = []
    for gloss, signer_dict in by_gloss.items():
        if len(signer_dict) < min_signers:
            continue

        prototypes = {}
        intra_dists = []
        for sid, feats in signer_dict.items():
            proto = np.mean(feats, axis=0)
            prototypes[sid] = proto
            for f in feats:
                intra_dists.append(l2(f, proto))

        signer_ids = sorted(prototypes.keys())
        inter_dists = []
        for i in range(len(signer_ids)):
            for j in range(i + 1, len(signer_ids)):
                inter_dists.append(l2(prototypes[signer_ids[i]],
                                      prototypes[signer_ids[j]]))

        gloss_results.append({
            "gloss":         gloss,
            "n_signers":     len(signer_dict),
            "n_videos":      sum(len(fs) for fs in signer_dict.values()),
            "intra_mean":    float(np.mean(intra_dists)) if intra_dists else 0.0,
            "inter_mean":    float(np.mean(inter_dists)) if inter_dists else 0.0,
        })

    if not gloss_results:
        return {"gloss_results": [], "aggregate": None}

    all_intra = [r["intra_mean"] for r in gloss_results]
    all_inter = [r["inter_mean"] for r in gloss_results]
    ratios    = [r["inter_mean"] / (r["intra_mean"] + 1e-8) for r in gloss_results]

    return {
        "gloss_results": gloss_results,
        "aggregate": {
            "n_glosses":              len(gloss_results),
            "mean_intra_L2":          float(np.mean(all_intra)),
            "median_intra_L2":        float(np.median(all_intra)),
            "mean_inter_L2":          float(np.mean(all_inter)),
            "median_inter_L2":        float(np.median(all_inter)),
            "mean_ratio":             float(np.mean(ratios)),
            "median_ratio":           float(np.median(ratios)),
            "pct_inter_gt_intra":     float(np.mean([r > 1.0 for r in ratios])) * 100,
        },
        "top_glosses": top_glosses,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(result: dict, model_name: str, embed_dim: int):
    gloss_results = result["gloss_results"]
    agg           = result["aggregate"]
    top_n         = result.get("top_glosses", 20)
    W = 68

    print("\n" + "=" * W)
    print(f"ASL-CITIZEN  {model_name.upper()}  SIGNCLIP EMBEDDING  L2-DISTANCE PROBE")
    print("=" * W)

    if agg is None:
        print("  No glosses met the minimum-signers criterion.")
        return

    print(f"\n  Embedding dim : {embed_dim}")
    print(f"  Glosses       : {agg['n_glosses']} with ≥ 2 signers")
    print()
    print(f"  ── Intra-signer L2 ──")
    print(f"     mean   {agg['mean_intra_L2']:.4f}")
    print(f"     median {agg['median_intra_L2']:.4f}")
    print()
    print(f"  ── Inter-signer L2 ──")
    print(f"     mean   {agg['mean_inter_L2']:.4f}")
    print(f"     median {agg['median_inter_L2']:.4f}")
    print()
    print(f"  ── Ratio  inter / intra ──")
    print(f"     mean   {agg['mean_ratio']:.2f}x")
    print(f"     median {agg['median_ratio']:.2f}x")
    print(f"     % glosses where inter > intra: {agg['pct_inter_gt_intra']:.1f}%")

    ratio = agg["mean_ratio"]
    if agg["mean_inter_L2"] < 1e-3:
        verdict = "INVARIANT: embeddings collapse across signers."
    elif ratio > 2.0:
        verdict = "STRONG signer signal in SignCLIP embeddings."
    elif ratio > 1.3:
        verdict = "MODERATE signer signal in SignCLIP embeddings."
    else:
        verdict = "WEAK signer signal — embeddings approaching invariance."
    print(f"\n  Verdict: {verdict}")

    sorted_glosses = sorted(gloss_results, key=lambda r: -r["inter_mean"])[:top_n]
    print(f"\n  Top-{top_n} glosses by inter-signer L2:")
    print(f"  {'Gloss':<22}  {'#sig':>4}  {'#vid':>4}  {'intra':>8}  {'inter':>8}  {'ratio':>6}")
    print("  " + "-" * 62)
    for r in sorted_glosses:
        ratio_g = r["inter_mean"] / (r["intra_mean"] + 1e-8)
        print(f"  {r['gloss']:<22}  {r['n_signers']:>4}  {r['n_videos']:>4}  "
              f"{r['intra_mean']:>8.4f}  {r['inter_mean']:>8.4f}  {ratio_g:>6.2f}x")
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Test YAML config for the SignCLIP model")
    parser.add_argument("--model_name", type=str, default="signclip",
                        help="Label for display (e.g. logos, i3d_wlasl, mediapipe)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to run inference on (default: test)")
    parser.add_argument("--splits_dir", type=Path, default=Path(DEFAULT_SPLITS_DIR),
                        help="Directory with CSV splits for signer metadata")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_signers", type=int, default=2)
    parser.add_argument("--top_glosses", type=int, default=20)
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="If set, save/load embeddings as a .npy cache to skip re-running the model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config and disable struct mode so missing keys return None
    from argparse import Namespace
    from omegaconf import OmegaConf
    config = load_config(Namespace(taskconfig=str(args.config)))
    disable_struct_recursive(config)

    # Cache path
    cache_path = None
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = args.cache_dir / f"{args.model_name}_{args.split}_embeddings.npz"

    if cache_path and cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        feat_ids     = [str(x) for x in data["feat_ids"]]
        embed_matrix = data["embeddings"]
        embeddings   = {fid: embed_matrix[i] for i, fid in enumerate(feat_ids)}
    else:
        # Build dataloader
        print(f"Building dataloader for split={args.split} ...")
        dataloader, meta_data = build_dataloader(config, args.split, args.batch_size)

        # Load model
        model = load_model(config)

        # Extract embeddings
        embeddings = extract_embeddings(model, dataloader, meta_data, device)

        if cache_path:
            feat_ids_arr = list(embeddings.keys())
            embed_matrix = np.stack([embeddings[k] for k in feat_ids_arr])
            np.savez(cache_path, feat_ids=feat_ids_arr, embeddings=embed_matrix)
            print(f"Cached embeddings to {cache_path}")

    if not embeddings:
        raise RuntimeError("No embeddings extracted.")

    embed_dim = next(iter(embeddings.values())).shape[0]
    print(f"Embedding dimension: {embed_dim}")

    # Load signer metadata
    signer_map = load_signer_map(args.splits_dir)

    # Analyse
    result = analyse(embeddings, signer_map,
                     min_signers=args.min_signers,
                     top_glosses=args.top_glosses)

    print_results(result, args.model_name, embed_dim)


if __name__ == "__main__":
    main()
