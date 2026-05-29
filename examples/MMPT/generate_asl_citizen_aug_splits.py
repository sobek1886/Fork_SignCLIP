"""Generate augmented train/val/test CSVs for ASL-Citizen.

Reads the original splits and aug_video_list.txt (from analyze_asl_citizen_for_aug.py),
then writes a new splits directory where train.csv contains:
  - all original training rows (unchanged)
  - one additional row per augmented variant for each selected video

Output Video file format for augmented rows:
  {video_id}_{variant}.mp4   e.g.  7410564482917914-CRACK_glasses.mp4

The feature loader resolves this to:
  feat_id = asl_citizen_7410564482917914-CRACK_glasses
  file    = logos_features/asl_citizen_7410564482917914-CRACK_glasses.npy

val.csv and test.csv are copied unchanged.

Usage:
    python generate_asl_citizen_aug_splits.py \\
        --train_csv      /home/psobecki/ASL_Citizen/splits/train.csv \\
        --val_csv        /home/psobecki/ASL_Citizen/splits/val.csv \\
        --test_csv       /home/psobecki/ASL_Citizen/splits/test.csv \\
        --aug_video_list /home/psobecki/ASL_Citizen/aug_selection/aug_video_list.txt \\
        --variants       glasses shirt_1 \\
        --output_dir     /home/psobecki/ASL_Citizen/splits_aug
"""

import argparse
import csv
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--aug_video_list', required=True,
                        help='One video_id per line (no .mp4 extension)')
    parser.add_argument('--variants', nargs='+', default=['glasses', 'shirt_1'],
                        help='Augmentation variants to add to train split')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read selected video IDs
    with open(args.aug_video_list) as f:
        aug_ids = set(line.strip() for line in f if line.strip())
    print(f'Augmented video list: {len(aug_ids)} videos')
    print(f'Variants to add: {args.variants}')

    # Read original train CSV and build video_id → gloss mapping
    original_rows = []
    id_to_gloss = {}
    with open(args.train_csv, newline='') as f:
        for row in csv.DictReader(f):
            video_id = os.path.splitext(row['Video file'])[0]
            gloss = row['Gloss']
            original_rows.append({'Video file': row['Video file'], 'Gloss': gloss})
            id_to_gloss[video_id] = gloss

    # Build augmented rows for selected videos
    aug_rows = []
    missing_glosses = 0
    for video_id in sorted(aug_ids):
        gloss = id_to_gloss.get(video_id)
        if gloss is None:
            missing_glosses += 1
            continue
        for variant in args.variants:
            aug_rows.append({
                'Video file': f'{video_id}_{variant}.mp4',
                'Gloss': gloss,
            })

    if missing_glosses:
        print(f'WARNING: {missing_glosses} aug video IDs not found in train.csv (no row added)')

    all_train_rows = original_rows + aug_rows

    # Write train.csv
    train_out = os.path.join(args.output_dir, 'train.csv')
    with open(train_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Video file', 'Gloss'])
        writer.writeheader()
        writer.writerows(all_train_rows)
    print(f'Wrote {len(original_rows)} original + {len(aug_rows)} augmented = '
          f'{len(all_train_rows)} rows → {train_out}')

    # Copy val and test unchanged
    for src, name in [(args.val_csv, 'val.csv'), (args.test_csv, 'test.csv')]:
        dst = os.path.join(args.output_dir, name)
        shutil.copy2(src, dst)
        print(f'Copied {src} → {dst}')


if __name__ == '__main__':
    main()
