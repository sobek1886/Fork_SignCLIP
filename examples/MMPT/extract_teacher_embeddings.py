#!/usr/bin/env python3
"""
Extract video embeddings from a trained Logos SignCLIP checkpoint for use as
teacher targets in knowledge distillation training.

For each video in the ASL-Citizen dataset, runs the trained video encoder
on pre-extracted Logos features and saves the resulting 768-dim embedding
as a .npy file. The naming convention matches that of the other .npy features:
    {dataset_name}_{video_id}.npy   (e.g. asl_citizen_abc123.npy)

Run this once before distillation training.

Usage:
    python extract_teacher_embeddings.py \
        --config  projects/retri/signclip_asl/asl_citizen_cnn_scratch_logos.yaml \
        --checkpoint  /path/to/checkpoint_best.pt \
        --logos_dir   /path/to/logos_features \
        --splits_dir  /path/to/ASL_Citizen/splits \
        --output_dir  /path/to/logos_teacher_embeddings \
        [--batch_size 256]  [--max_video_len 32]  [--dataset_name asl_citizen]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mmpt import utils
from mmpt.models import MMFusionSeparate


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_model(config_path, checkpoint_path):
    config = utils.load_config(config_file=config_path)
    model = MMFusionSeparate(config)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]

    from collections import OrderedDict
    clean = OrderedDict()
    for k, v in state.items():
        clean[k[len("mmmodel."):] if k.startswith("mmmodel.") else k] = v

    model.load_state_dict(clean, strict=False)
    return model.cuda().eval()


def _read_video_ids(splits_dir, dataset_name):
    """Return all feat_ids ({dataset_name}_{video_stem}) from all split CSVs."""
    feat_ids = []
    for fname in ("train.csv", "val.csv", "test.csv"):
        path = os.path.join(splits_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                video_stem = os.path.splitext(row["Video file"])[0]
                feat_ids.append(f"{dataset_name}_{video_stem}")
    return list(dict.fromkeys(feat_ids))  # deduplicate, preserve order


def _extract(model, logos_dir, feat_ids, output_dir, batch_size, max_video_len):
    os.makedirs(output_dir, exist_ok=True)

    todo = [fid for fid in feat_ids
            if not os.path.exists(os.path.join(output_dir, fid + ".npy"))]
    print(f"Extracting {len(todo)} / {len(feat_ids)} videos "
          f"({len(feat_ids) - len(todo)} already done)")

    for start in tqdm(range(0, len(todo), batch_size)):
        batch_ids = todo[start: start + batch_size]
        B = len(batch_ids)

        batch_feats = np.zeros((B, max_video_len, 768), dtype=np.float32)
        batch_masks = np.zeros((B, max_video_len), dtype=bool)

        for i, fid in enumerate(batch_ids):
            feat = np.load(os.path.join(logos_dir, fid + ".npy"))  # (N, 768)
            n = min(len(feat), max_video_len)
            batch_feats[i, :n] = feat[:n]
            batch_masks[i, :n] = True

        vfeats = torch.from_numpy(batch_feats).cuda()          # (B, T, 768)
        vmasks = torch.from_numpy(batch_masks).cuda()          # (B, T)
        # forward_video only uses caps[:, :2] for [CLS]/[SEP] token ids
        caps   = torch.full((B, 2), fill_value=101, dtype=torch.long).cuda()
        caps[:, 1] = 102
        cmasks = torch.ones(B, 2, dtype=torch.bool).cuda()

        with torch.no_grad():
            pooled = model.forward_video(vfeats, vmasks, caps, cmasks)  # (B, 768)

        embeddings = pooled.float().cpu().numpy()
        for fid, emb in zip(batch_ids, embeddings):
            np.save(os.path.join(output_dir, fid + ".npy"), emb)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",       required=True,
                        help="Logos training config YAML (e.g. asl_citizen_cnn_scratch_logos.yaml)")
    parser.add_argument("--checkpoint",   required=True,
                        help="Trained Logos checkpoint (.pt)")
    parser.add_argument("--logos_dir",    required=True,
                        help="Directory containing pre-extracted Logos .npy features")
    parser.add_argument("--splits_dir",   required=True,
                        help="Directory containing train.csv / val.csv / test.csv")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to save teacher embedding .npy files")
    parser.add_argument("--batch_size",   type=int, default=256)
    parser.add_argument("--max_video_len", type=int, default=32)
    parser.add_argument("--dataset_name", default="asl_citizen")
    args = parser.parse_args()

    feat_ids = _read_video_ids(args.splits_dir, args.dataset_name)
    print(f"Total videos across all splits: {len(feat_ids)}")

    model = _load_model(args.config, args.checkpoint)
    _extract(model, args.logos_dir, feat_ids, args.output_dir,
             args.batch_size, args.max_video_len)
    print("Done.")


if __name__ == "__main__":
    main()
