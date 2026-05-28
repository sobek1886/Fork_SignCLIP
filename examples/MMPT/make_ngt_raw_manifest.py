"""Generate a raw-path manifest for end-to-end training (Approach C).

Reads the existing .npy pair manifest and derives raw video / PNG-sequence
paths for each view.  The output manifest maps:

  sign_id → {
      "real":   [<mkv_path>, <mkv_path>, <mkv_path>],   # 3 piotrAnims views
      "unreal": [<png_dir>,  …]                          # 6 Unreal Engine dirs
  }

Path derivation rules (matching the extraction scripts):
  real .npy  : .../piotranims_{video_name}.npy
    → {real_mkv_dir}/{video_name}.mkv

  unreal .npy: .../ngt_aug_{seq_name}.npy
    → {unreal_render_dir}/{seq_name}/

Usage (on cluster):
  python make_ngt_raw_manifest.py \\
      --npy_manifest  /home/psobecki/ngt_pair_manifest.json \\
      --output        /home/psobecki/ngt_raw_manifest.json \\
      --real_mkv_dir  /scratch-shared/psobecki/piotrAnims/mkv \\
      --unreal_dir    /scratch-shared/psobecki/Custom_from_FBX/Saved/Renders

Optional flags:
  --real_prefix   piotranims_   (default)
  --unreal_prefix ngt_aug_      (default)
  --check                       verify each derived path actually exists
"""

import argparse
import json
import os
from pathlib import Path


def derive_raw_path(npy_path, prefix, base_dir, is_dir):
    """Strip dataset prefix and .npy suffix, then build raw path."""
    basename = os.path.basename(npy_path)
    if not basename.startswith(prefix):
        raise ValueError(
            f"Expected basename to start with '{prefix}', got '{basename}'"
        )
    name = basename[len(prefix):-len('.npy')]
    if is_dir:
        return os.path.join(base_dir, name)
    else:
        return os.path.join(base_dir, name + '.mkv')


def main():
    parser = argparse.ArgumentParser(
        description='Convert .npy manifest to raw-path manifest for E2E training.'
    )
    parser.add_argument('--npy_manifest', required=True,
                        help='Path to existing ngt_pair_manifest.json')
    parser.add_argument('--output', required=True,
                        help='Output path for ngt_raw_manifest.json')
    parser.add_argument('--real_mkv_dir',
                        default='/scratch-shared/psobecki/piotrAnims/mkv',
                        help='Directory containing piotrAnims .mkv files')
    parser.add_argument('--unreal_dir',
                        default='/scratch-shared/psobecki/Custom_from_FBX/Saved/Renders',
                        help='Parent directory of Unreal PNG-sequence sub-dirs')
    parser.add_argument('--real_prefix', default='piotranims_',
                        help='Filename prefix for real .npy files')
    parser.add_argument('--unreal_prefix', default='ngt_aug_',
                        help='Filename prefix for unreal .npy files')
    parser.add_argument('--check', action='store_true',
                        help='Verify each derived raw path actually exists')
    args = parser.parse_args()

    with open(args.npy_manifest) as f:
        npy_manifest = json.load(f)

    raw_manifest = {}
    missing = []

    for sign_id, paths in npy_manifest.items():
        real_raw = [
            derive_raw_path(p, args.real_prefix, args.real_mkv_dir, is_dir=False)
            for p in paths['real']
        ]
        unreal_raw = [
            derive_raw_path(p, args.unreal_prefix, args.unreal_dir, is_dir=True)
            for p in paths.get('unreal', [])
        ]

        if args.check:
            for p in real_raw + unreal_raw:
                if not os.path.exists(p):
                    missing.append(p)

        raw_manifest[sign_id] = {
            'real': real_raw,
            'unreal': unreal_raw,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(raw_manifest, f, indent=2)

    print(f'Wrote raw manifest for {len(raw_manifest)} signs → {args.output}')
    if args.check:
        if missing:
            print(f'WARNING: {len(missing)} paths not found:')
            for p in missing[:10]:
                print(f'  {p}')
            if len(missing) > 10:
                print(f'  ... and {len(missing) - 10} more')
        else:
            print('All raw paths verified OK.')


if __name__ == '__main__':
    main()
