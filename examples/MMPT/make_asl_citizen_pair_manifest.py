"""Build the pair manifest for ASL-Citizen appearance-invariant fine-tuning.

For each video in aug_video_list.txt, checks whether all three .npy files
exist in the Logos features directory:
  - original:  asl_citizen_{video_id}.npy
  - glasses:   asl_citizen_{video_id}_glasses.npy
  - shirt_1:   asl_citizen_{video_id}_shirt_1.npy

Only videos with ALL THREE files are included (fixed K=3 required by NGTPairAligner).

Output format matches what NGTPairVideoProcessor expects:
  {
    "7410564482917914-CRACK": {
      "real":   ["/abs/path/asl_citizen_7410564482917914-CRACK.npy"],
      "unreal": ["/abs/path/asl_citizen_7410564482917914-CRACK_glasses.npy",
                 "/abs/path/asl_citizen_7410564482917914-CRACK_shirt_1.npy"]
    }
  }

Usage (on Snellius):
    python make_asl_citizen_pair_manifest.py \\
        --aug_video_list /home/psobecki/ASL_Citizen/aug_selection/aug_video_list.txt \\
        --logos_dir      /home/psobecki/ASL_Citizen/logos_features \\
        --output         /home/psobecki/ASL_Citizen/asl_citizen_pair_manifest.json
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_video_list', required=True,
                        help='aug_video_list.txt from analyze_asl_citizen_for_aug.py')
    parser.add_argument('--logos_dir', required=True,
                        help='Directory containing asl_citizen_*.npy feature files')
    parser.add_argument('--output', required=True,
                        help='Output JSON manifest path')
    args = parser.parse_args()

    with open(args.aug_video_list) as f:
        video_ids = [line.strip() for line in f if line.strip()]
    print(f'Video list: {len(video_ids)} entries')

    manifest = {}
    n_all3 = n_glasses_only = n_shirt_only = n_neither = 0

    for video_id in video_ids:
        original = os.path.join(args.logos_dir, f'asl_citizen_{video_id}.npy')
        glasses  = os.path.join(args.logos_dir, f'asl_citizen_{video_id}_glasses.npy')
        shirt_1  = os.path.join(args.logos_dir, f'asl_citizen_{video_id}_shirt_1.npy')

        has_orig    = os.path.exists(original)
        has_glasses = os.path.exists(glasses)
        has_shirt   = os.path.exists(shirt_1)

        if has_orig and has_glasses and has_shirt:
            manifest[video_id] = {
                'real':   [os.path.abspath(original)],
                'unreal': [os.path.abspath(glasses), os.path.abspath(shirt_1)],
            }
            n_all3 += 1
        elif has_glasses and not has_shirt:
            n_glasses_only += 1
        elif has_shirt and not has_glasses:
            n_shirt_only += 1
        else:
            n_neither += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nResults:')
    print(f'  All 3 (original + glasses + shirt_1) : {n_all3}  → included in manifest')
    print(f'  Glasses only                         : {n_glasses_only}')
    print(f'  Shirt_1 only                         : {n_shirt_only}')
    print(f'  Neither augmentation found           : {n_neither}')
    print(f'\nWrote {n_all3} entries → {args.output}')


if __name__ == '__main__':
    main()
