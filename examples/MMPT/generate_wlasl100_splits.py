"""Generate train/val/test CSVs for WLASL100 from nslt_100.json.

Output columns: Video file, Gloss

The "Video file" value encodes the feature-file suffix used by
SignCLIPVideoCSVMetaProcessor:
    feat_id = {dataset_name}_{os.path.splitext(Video_file)[0]}

Split | Video file format        | training dataset_name | resolved .npy
------|--------------------------|-----------------------|------------------------------
train | {video_id}_{variant}.mp4 | wlasl100              | wlasl100_{video_id}_{variant}.npy
val   | val_{video_id}.mp4       | wlasl100              | wlasl100_val_{video_id}.npy
test  | {video_id}.mp4           | wlasl100_test         | wlasl100_test_{video_id}.npy

--train_variants controls which augmentation variants appear in train.csv.
Multiple variants are all written to the same train.csv (one row each).

Examples:
    # Original-only (default, backward-compatible):
    python generate_wlasl100_splits.py ... --output_dir splits

    # All augmentations combined:
    python generate_wlasl100_splits.py ... --train_variants original glasses shirt_1 \\
        --output_dir splits_aug

    # Synthetic-only (shirt_1):
    python generate_wlasl100_splits.py ... --train_variants shirt_1 \\
        --output_dir splits_shirt1

    # Synthetic-only (glasses):
    python generate_wlasl100_splits.py ... --train_variants glasses \\
        --output_dir splits_glasses

Usage:
    python generate_wlasl100_splits.py \\
      --nslt_json /path/to/nslt_100.json \\
      --class_list /path/to/wlasl_class_list.txt \\
      --output_dir /tmp/wlasl100_splits
"""

import argparse
import csv
import json
import os

VALID_VARIANTS = ['original', 'glasses', 'shirt_1']


def load_class_list(path):
    """Return dict {class_id (int): gloss (str)} from tab-separated file."""
    classes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, gloss = line.split('\t', 1)
            classes[int(idx)] = gloss
    return classes


def main():
    parser = argparse.ArgumentParser(description='Generate WLASL100 split CSVs.')
    parser.add_argument('--nslt_json', required=True,
                        help='Path to nslt_100.json')
    parser.add_argument('--class_list', required=True,
                        help='Path to wlasl_class_list.txt')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to write train.csv / val.csv / test.csv')
    parser.add_argument('--train_variants', nargs='+', default=['original'],
                        choices=VALID_VARIANTS,
                        help='Augmentation variants to include in train.csv '
                             '(default: original). Multiple values are all written '
                             'to the same train.csv.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    classes = load_class_list(args.class_list)

    with open(args.nslt_json) as f:
        data = json.load(f)

    splits = {'train': [], 'val': [], 'test': []}

    for video_id, info in data.items():
        subset = info['subset']
        class_id = info['action'][0]
        gloss = classes[class_id]

        if subset == 'train':
            for variant in args.train_variants:
                splits['train'].append({
                    'Video file': f'{video_id}_{variant}.mp4',
                    'Gloss': gloss,
                })
        elif subset == 'val':
            splits['val'].append({'Video file': f'val_{video_id}.mp4', 'Gloss': gloss})
        elif subset == 'test':
            splits['test'].append({'Video file': f'{video_id}.mp4', 'Gloss': gloss})

    for split_name, rows in splits.items():
        out_path = os.path.join(args.output_dir, f'{split_name}.csv')
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Video file', 'Gloss'])
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {len(rows):4d} rows → {out_path}')


if __name__ == '__main__':
    main()
