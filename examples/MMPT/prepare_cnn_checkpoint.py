"""Prepare an E7.2/E8 pose checkpoint for CNN-based fine-tuning.

The input projection in the video encoder (videomlp.linear1.weight) has shape
[768, 609] for pose features.  When training with 1024-dim I3D features, this
layer must be re-initialised.  This script strips that weight from the checkpoint
so fairseq initialises it fresh while loading all other weights (12 video BERT
layers, text encoder, multimodal projection) from the E7.2 checkpoint.

Usage:
    python prepare_cnn_checkpoint.py \\
        --input  /path/to/baseline_temporal_checkpoint_best.pt \\
        --output /path/to/baseline_temporal_cnn_ready.pt
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="E7.2 pose checkpoint")
    parser.add_argument("--output", required=True, help="Path to save modified checkpoint")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.input}")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    model_state = ckpt["model"]

    # Only linear1.weight depends on vfeat_dim (shape [768, vfeat_dim]).
    # linear1.bias is always [768], so it transfers without change.
    key_to_remove = "mmmodel.video_encoder.videomlp.linear1.weight"

    if key_to_remove in model_state:
        old_shape = model_state[key_to_remove].shape
        del model_state[key_to_remove]
        print(f"Removed '{key_to_remove}' (was {list(old_shape)}) — will be re-initialised")
    else:
        print(f"WARNING: '{key_to_remove}' not found in checkpoint. Keys present:")
        for k in model_state:
            if "videomlp" in k:
                print(f"  {k}: {list(model_state[k].shape)}")

    torch.save(ckpt, args.output)
    print(f"Saved CNN-ready checkpoint to {args.output}")
    print("videomlp layers remaining:")
    for k, v in model_state.items():
        if "videomlp" in k:
            print(f"  {k}: {list(v.shape)}")


if __name__ == "__main__":
    main()
