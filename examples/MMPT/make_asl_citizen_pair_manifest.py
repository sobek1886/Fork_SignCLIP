"""Build the pair manifest for ASL-Citizen appearance-invariant fine-tuning.

For each video in aug_video_list.txt, checks whether the original .npy file
and every requested augmentation variant's .npy file exist in the Logos
features directory:
  - original:  asl_citizen_{video_id}.npy
  - variants:  asl_citizen_{video_id}_{variant}.npy   (--variants, default
               glasses shirt_1; pass all four of glasses shirt_1 signer_swap
               skin_mst_diffusion for the K=5 counterfactual-contrastive run)

Only videos with the original AND every requested variant are included, so K
is uniform across the manifest (required by NGTPairAligner collation).

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
    parser.add_argument('--variants', nargs='+',
                        default=['glasses', 'shirt_1'],
                        help='Augmentation variants to pair with the original '
                             '(default: glasses shirt_1 → the original K=3 setup). '
                             'Pass all four for K=5: --variants glasses shirt_1 '
                             'signer_swap skin_mst_diffusion. Only videos with the '
                             'original AND every requested variant are included, so '
                             'K is uniform across the manifest (required by '
                             'NGTPairAligner collation).')
    parser.add_argument('--output', required=True,
                        help='Output JSON manifest path')
    args = parser.parse_args()

    with open(args.aug_video_list) as f:
        video_ids = [line.strip() for line in f if line.strip()]
    print(f'Video list: {len(video_ids)} entries')
    print(f'Variants required: {args.variants} (K = {1 + len(args.variants)})')

    manifest = {}
    n_complete = 0
    n_missing_orig = 0
    n_dropped_for = {v: 0 for v in args.variants}

    for video_id in video_ids:
        original = os.path.join(args.logos_dir, f'asl_citizen_{video_id}.npy')
        variant_paths = {
            v: os.path.join(args.logos_dir, f'asl_citizen_{video_id}_{v}.npy')
            for v in args.variants
        }

        if not os.path.exists(original):
            n_missing_orig += 1
            continue

        missing = [v for v, p in variant_paths.items() if not os.path.exists(p)]
        if missing:
            for v in missing:
                n_dropped_for[v] += 1
            continue

        manifest[video_id] = {
            'real':   [os.path.abspath(original)],
            'unreal': [os.path.abspath(variant_paths[v]) for v in args.variants],
        }
        n_complete += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nResults:')
    print(f'  Complete (original + {len(args.variants)} variants) : {n_complete}  → included in manifest')
    print(f'  Missing original feature : {n_missing_orig}')
    for v in args.variants:
        print(f'  Dropped for missing {v} : {n_dropped_for[v]}')
    print(f'\nWrote {n_complete} entries → {args.output}')


if __name__ == '__main__':
    main()
