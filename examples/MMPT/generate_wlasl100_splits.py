"""Generate train/val/test CSVs for WLASL100 from nslt_100.json.

Output columns: Video file, Gloss

The "Video file" value encodes the feature-file suffix used by
SignCLIPVideoCSVMetaProcessor:
    feat_id = {dataset_name}_{os.path.splitext(Video_file)[0]}

Split | Video file format        | training dataset_name | resolved .npy
------|--------------------------|-----------------------|------------------------------
train | {video_id}_original.mp4  | wlasl100              | wlasl100_{video_id}_original.npy
val   | val_{video_id}.mp4       | wlasl100              | wlasl100_val_{video_id}.npy
test  | {video_id}.mp4           | wlasl100_test         | wlasl100_test_{video_id}.npy

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
            video_file = f'{video_id}_original.mp4'
        elif subset == 'val':
            video_file = f'val_{video_id}.mp4'
        elif subset == 'test':
            video_file = f'{video_id}.mp4'
        else:
            continue

        splits[subset].append({'Video file': video_file, 'Gloss': gloss})

    for split_name, rows in splits.items():
        out_path = os.path.join(args.output_dir, f'{split_name}.csv')
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Video file', 'Gloss'])
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {len(rows):4d} rows → {out_path}')


if __name__ == '__main__':
    main()
