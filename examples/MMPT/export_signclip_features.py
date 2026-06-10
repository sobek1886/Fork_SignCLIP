#!/usr/bin/env python3
"""
export_signclip_features.py  –  Dump a trained SignCLIP model's pooled_video
                                 embeddings as per-video .npy files.

Purpose (control experiment): evaluate the VIDEO side of a SignCLIP model with the
SAME protocol used for raw Logos features, to answer "did the contrastive video→text
training degrade the Logos visual features, or just re-task them toward text?".

The SignCLIP video encoder consumes the EXISTING frozen Logos features (vfeat_dir in
the config) and produces one pooled_video vector (768,) per video. We save each as
    {output_dir}/{id}.npy        where id = {dataset_name}_{video_basename}
which is exactly the filename contract of eval_asl_citizen_retrieval.py (Eval A) and
eval_asl_citizen_linear_probe.py (Eval B). So after exporting you simply point both
eval scripts at --feature_dir <output_dir> and compare against raw Logos features.

Reuses the embedding extraction machinery from asl_citizen_signclip_l2_probe.py
(load_model / build_dataloader / extract_embeddings) so the embeddings are identical
to those used by the L2 probe. Requires a GPU and a test_*.yaml config whose
fairseq.common_eval.path points at the trained checkpoint.

Usage
  python export_signclip_features.py \\
      --config     projects/retri/signclip_asl/test_asl_citizen_cnn_aug_logos_ft.yaml \\
      --output_dir /home/psobecki/ASL_Citizen/signclip_features_aug_logos_ft \\
      --splits train test
"""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asl_citizen_signclip_l2_probe import (  # noqa: E402
    build_dataloader, load_model, extract_embeddings, disable_struct_recursive,
)
from mmpt.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="test_*.yaml whose common_eval.path is the trained checkpoint")
    ap.add_argument("--output_dir", required=True,
                    help="dir to write {id}.npy pooled_video embeddings")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "valid", "test"],
                    help="splits to export (need train as gallery + test as query)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-export even if the output dir already has .npy files")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(Namespace(taskconfig=str(args.config)))
    disable_struct_recursive(config)

    print(f"Loading SignCLIP model from {config.fairseq.common_eval.path} ...")
    model = load_model(config)

    total = 0
    for split in args.splits:
        dataloader, meta_data = build_dataloader(config, split, args.batch_size)
        emb = extract_embeddings(model, dataloader, meta_data, device)
        saved = 0
        for vid, vec in emb.items():
            out_path = os.path.join(args.output_dir, f"{vid}.npy")
            if os.path.exists(out_path) and not args.overwrite:
                continue
            np.save(out_path, np.asarray(vec, dtype=np.float32))   # (768,)
            saved += 1
        total += saved
        print(f"  split={split}: {len(emb)} embeddings, {saved} saved")

    print(f"\nDone. {total} .npy files in {args.output_dir}")
    print(f"  shape per file: (768,)  |  filename: {{dataset_name}}_{{basename}}.npy")
    print("Now evaluate with (MMPT venv):")
    print(f"  python eval_asl_citizen_retrieval.py   --feature_dir {args.output_dir} ...")
    print(f"  python eval_asl_citizen_linear_probe.py --feature_dir {args.output_dir} ...")


if __name__ == "__main__":
    main()
