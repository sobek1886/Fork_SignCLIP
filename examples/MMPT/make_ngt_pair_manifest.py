"""Build the NGT pair manifest used by NGTPairMetaProcessor.

Matches Bushuis real-camera features against NGT_Aug (Unreal Engine palmer)
features using the shared sign key  M{YYYYMMDD}_{4digits}.

File naming (produced by extract_logos_features.py):
    Bushuis:  bushuis_{key}.npy
    NGT_Aug:  ngt_aug_palmer_{key}_{date}_{idx}_cam{N}.npy
    piotrAnims (TEST SET — excluded):
              piotr_anims_{key}_{date}_{idx}_{view}_{timestamp}.npy

Only keys present in BOTH directories are emitted.
Light-variation renders (palmer_test_*) are automatically excluded.

Usage:
    python make_ngt_pair_manifest.py \\
        --bushuis_dir  /home/psobecki/Bushuis/logos_features \\
        --ngt_aug_dir  /home/psobecki/NGT_Aug/logos_features \\
        --frontal_cam  cam4 \\
        --output       /home/psobecki/ngt_pair_manifest.json

Output JSON:
    {
      "M20260227_6007": {
        "real":   "/home/.../bushuis_M20260227_6007.npy",
        "unreal": "/home/.../ngt_aug_palmer_M20260227_6007_260316_0_cam4.npy"
      },
      ...
    }
"""

import argparse
import json
import re
import sys
from pathlib import Path


# Pattern for the shared sign key inside Bushuis filenames.
# bushuis_M20260227_6007.npy  →  M20260227_6007
_BUSHUIS_KEY_RE = re.compile(r'^bushuis_(M\d{8}_\d+)\.npy$')

# Pattern for the key inside NGT_Aug filenames (frontal-cam only).
# ngt_aug_palmer_M20260227_6007_260316_0_cam4.npy  →  (key=M20260227_6007, cam=cam4)
# We capture everything between "ngt_aug_palmer_" and the date-suffix "_YYYYMMDD_*".
_NGTAUG_KEY_RE = re.compile(
    r'^ngt_aug_palmer_(M\d{8}_\d+)_\d+_\d+_(cam\d+)\.npy$'
)


def _extract_bushuis(bushuis_dir: Path) -> dict:
    """Return {sign_key: absolute_path} for all Bushuis feature files."""
    result = {}
    for p in sorted(bushuis_dir.glob('bushuis_*.npy')):
        m = _BUSHUIS_KEY_RE.match(p.name)
        if m:
            result[m.group(1)] = str(p)
        else:
            print(f'  SKIP (unexpected name): {p.name}', file=sys.stderr)
    return result


def _extract_ngtaug(ngt_aug_dir: Path, frontal_cam: str) -> dict:
    """Return {sign_key: absolute_path} for NGT_Aug files with the frontal cam.

    Files matching palmer_test_* (light variations) are excluded.
    If multiple files match the same key + cam, the first one (sorted) is kept.
    """
    result = {}
    for p in sorted(ngt_aug_dir.glob('ngt_aug_palmer_*.npy')):
        # Exclude light-variation renders
        if 'palmer_test_' in p.name:
            continue
        m = _NGTAUG_KEY_RE.match(p.name)
        if not m:
            print(f'  SKIP (unexpected name): {p.name}', file=sys.stderr)
            continue
        key, cam = m.group(1), m.group(2)
        if cam != frontal_cam:
            continue
        if key not in result:
            result[key] = str(p)
        else:
            print(f'  WARN duplicate key {key} ({cam}), keeping first', file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description='Build NGT pair manifest JSON.')
    parser.add_argument('--bushuis_dir', required=True,
                        help='Directory of Bushuis logos_features (.npy files)')
    parser.add_argument('--ngt_aug_dir', required=True,
                        help='Directory of NGT_Aug logos_features (.npy files)')
    parser.add_argument('--frontal_cam', default='cam4',
                        help='Which camera to treat as frontal (default: cam4)')
    parser.add_argument('--output', required=True,
                        help='Path to write the JSON manifest')
    args = parser.parse_args()

    bushuis_dir = Path(args.bushuis_dir)
    ngt_aug_dir = Path(args.ngt_aug_dir)

    if not bushuis_dir.is_dir():
        sys.exit(f'ERROR: bushuis_dir not found: {bushuis_dir}')
    if not ngt_aug_dir.is_dir():
        sys.exit(f'ERROR: ngt_aug_dir not found: {ngt_aug_dir}')

    print(f'Scanning Bushuis:  {bushuis_dir}')
    bushuis = _extract_bushuis(bushuis_dir)
    print(f'  Found {len(bushuis)} Bushuis entries')

    print(f'Scanning NGT_Aug:  {ngt_aug_dir}  (frontal={args.frontal_cam})')
    ngtaug = _extract_ngtaug(ngt_aug_dir, args.frontal_cam)
    print(f'  Found {len(ngtaug)} NGT_Aug entries for {args.frontal_cam}')

    common_keys = sorted(set(bushuis) & set(ngtaug))
    print(f'Matched pairs:     {len(common_keys)}  '
          f'(Bushuis-only: {len(bushuis) - len(common_keys)}, '
          f'NGT_Aug-only: {len(ngtaug) - len(common_keys)})')

    if not common_keys:
        sys.exit('ERROR: no matched pairs — check directory contents and --frontal_cam')

    manifest = {
        key: {'real': bushuis[key], 'unreal': ngtaug[key]}
        for key in common_keys
    }

    # Print a few examples for sanity-checking
    for key in common_keys[:3]:
        print(f'  {key}')
        print(f'    real:   {manifest[key]["real"]}')
        print(f'    unreal: {manifest[key]["unreal"]}')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote {len(manifest)} pairs → {out_path}')


if __name__ == '__main__':
    main()
