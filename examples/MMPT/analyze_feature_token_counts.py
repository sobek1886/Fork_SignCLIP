"""Measure the temporal token counts (.npy shape[0]) of the Logos features.

This is the sequence length SignCLIP actually sees BEFORE truncation to
max_video_len (=32). It tells us whether the 32-token cap in
_build_video_seq discards a large fraction of each sign.

  shape[0] = number of feature tokens (clips) per video
  shape[1] = feature dim (768 for MViTv2-S Logos)

Usage:
    python analyze_feature_token_counts.py \\
        --feat_dir /home/psobecki/ASL_Citizen/logos_features \\
        --sample 4000 \\
        --max_video_len 32
"""

import argparse
import glob
import os

import numpy as np


def summarize(name, counts, max_video_len):
    counts = np.asarray(counts)
    if counts.size == 0:
        print(f"\n=== {name} ===\n  (no files)")
        return
    print(f"\n=== {name} ===")
    print(f"  files analysed : {counts.size}")
    print(f"  tokens min     : {counts.min()}")
    print(f"  tokens median  : {int(np.median(counts))}")
    print(f"  tokens mean    : {counts.mean():.1f}")
    print(f"  tokens max     : {counts.max()}")
    print(f"  tokens std     : {counts.std():.1f}")
    for thr in (max_video_len, 2 * max_video_len):
        frac = 100.0 * (counts > thr).mean()
        print(f"  pct > {thr:<3d} tokens : {frac:.1f}%  ({int((counts > thr).sum())} files)")
    kept = np.minimum(counts, max_video_len)
    coverage = 100.0 * (kept / counts).mean()
    print(f"  mean temporal coverage at max_video_len={max_video_len}: {coverage:.1f}%"
          f"  (fraction of each sequence the model actually sees)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="/home/psobecki/ASL_Citizen/logos_features")
    ap.add_argument("--sample", type=int, default=4000,
                    help="max files to load per group (0 = all)")
    ap.add_argument("--max_video_len", type=int, default=32)
    args = ap.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.feat_dir, "*.npy")))
    print(f"Found {len(all_files)} .npy files in {args.feat_dir}")

    # Split into original vs augmented by the _glasses / _shirt suffix convention.
    aug_files = [f for f in all_files
                 if "_glasses" in os.path.basename(f) or "_shirt" in os.path.basename(f)]
    orig_files = [f for f in all_files if f not in set(aug_files)]

    def load_counts(files):
        if args.sample and len(files) > args.sample:
            # deterministic stride sample, no RNG
            step = len(files) // args.sample
            files = files[::step][:args.sample]
        counts, shapes = [], []
        for f in files:
            try:
                arr = np.load(f, mmap_mode="r")
                counts.append(arr.shape[0])
                if len(shapes) < 5:
                    shapes.append(arr.shape)
            except Exception as e:
                print(f"  skip {os.path.basename(f)}: {e}")
        return counts, shapes

    orig_counts, orig_shapes = load_counts(orig_files)
    aug_counts, aug_shapes = load_counts(aug_files)

    print(f"\nExample original shapes : {orig_shapes}")
    print(f"Example augmented shapes: {aug_shapes}")

    summarize("Original features", orig_counts, args.max_video_len)
    summarize("Augmented features", aug_counts, args.max_video_len)
    summarize("All features", orig_counts + aug_counts, args.max_video_len)


if __name__ == "__main__":
    main()
