"""Build the NGT pair manifest used by NGTPairMetaProcessor / NGTPairVideoProcessor.

Matches piotrAnims (real mocap) features against NGT_Aug (Unreal Engine) features
from both 'palmer' and 'digits' characters, using the shared sign key M{YYYYMMDD}_{N}.

File naming (produced by extract_logos_features*.py):
    piotrAnims:  piotr_anims_M20260227_5998_260316_1_LEFT_2026-03-16_12-55-25.npy
                 piotr_anims_M20260227_5998_260316_1_MIDDLE_2026-03-16_12-55-25.npy
                 piotr_anims_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25.npy
                 piotr_anims_M20260227_5998_260316_1_2026-03-16_12-55-25.npy  ← excluded
    NGT_Aug palmer:  ngt_aug_palmer_M20260227_6007_260316_0_cam4.npy
    NGT_Aug digits:  ngt_aug_digits_M20260127_2228_260316_0_cam3.npy

Manifest format:
    {
      "M20260227_5998": {
        "real":   [".../piotr_anims_..._LEFT_....npy",
                   ".../piotr_anims_..._MIDDLE_....npy",
                   ".../piotr_anims_..._RIGHT_....npy"],
        "unreal": [".../ngt_aug_palmer_..._cam2.npy",
                   ".../ngt_aug_palmer_..._cam3.npy",
                   ".../ngt_aug_palmer_..._cam4.npy",
                   ".../ngt_aug_digits_..._cam2.npy",
                   ".../ngt_aug_digits_..._cam3.npy",
                   ".../ngt_aug_digits_..._cam4.npy"]
      }
    }

Only signs with ALL piotr_views AND ALL ngt_cams for BOTH characters are emitted.
Signs where any view/cam is missing are dropped with a warning.

Usage:
    python make_ngt_pair_manifest.py \\
        --piotr_dir   /home/psobecki/piotrAnims/logos_features \\
        --ngt_aug_dir /home/psobecki/NGT_Aug/logos_features \\
        --piotr_views LEFT,MIDDLE,RIGHT \\
        --ngt_cams    cam2,cam3,cam4 \\
        --output      /home/psobecki/ngt_pair_manifest.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# piotr_anims_M20260227_5998_260316_1_LEFT_2026-03-16_12-55-25.npy  →  (key, view)
# piotr_anims_M20260227_5998_260316_1_2026-03-16_12-55-25.npy        →  (key, None) — excluded
_PIOTR_RE = re.compile(
    r'^piotr_anims_(M\d{8}_\d+)_\d+_\d+_(?:(LEFT|MIDDLE|RIGHT)_)?\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.npy$'
)

# ngt_aug_palmer_M20260227_6007_260316_0_cam4.npy  →  (key, cam)
_PALMER_RE = re.compile(r'^ngt_aug_palmer_(M\d{8}_\d+)_\d+_\d+_(cam\d+)\.npy$')

# ngt_aug_digits_M20260127_2228_260316_0_cam3.npy  →  (key, cam)
_DIGITS_RE = re.compile(r'^ngt_aug_digits_(M\d{8}_\d+)_\d+_\d+_(cam\d+)\.npy$')


def _scan_piotr(piotr_dir: Path, views: list) -> dict:
    """Return {sign_key: {view: path}} for piotrAnims features.

    Files without an explicit LEFT/MIDDLE/RIGHT view token are excluded
    (they are composite side-by-side renders, not individual views).
    """
    result = defaultdict(dict)
    for p in sorted(piotr_dir.glob('piotr_anims_*.npy')):
        m = _PIOTR_RE.match(p.name)
        if not m:
            print(f'  SKIP piotr (unexpected name): {p.name}', file=sys.stderr)
            continue
        key, view = m.group(1), m.group(2)
        if view is None:
            # Composite side-by-side file — silently skip
            continue
        if view not in views:
            continue
        if view in result[key]:
            print(f'  WARN piotr duplicate key={key} view={view}, keeping first',
                  file=sys.stderr)
            continue
        result[key][view] = str(p)
    return dict(result)


def _scan_ngtaug(ngt_aug_dir: Path, pattern: re.Pattern, glob_prefix: str,
                 cams: list) -> dict:
    """Return {sign_key: {cam: path}} for NGT_Aug features matching pattern."""
    result = defaultdict(dict)
    for p in sorted(ngt_aug_dir.glob(f'{glob_prefix}*.npy')):
        m = pattern.match(p.name)
        if not m:
            print(f'  SKIP {glob_prefix} (unexpected name): {p.name}', file=sys.stderr)
            continue
        key, cam = m.group(1), m.group(2)
        if cam not in cams:
            continue
        if cam in result[key]:
            print(f'  WARN {glob_prefix} duplicate key={key} cam={cam}, keeping first',
                  file=sys.stderr)
            continue
        result[key][cam] = str(p)
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description='Build NGT pair manifest JSON.')
    parser.add_argument('--piotr_dir', required=True,
                        help='Directory of piotrAnims logos_features (.npy files)')
    parser.add_argument('--ngt_aug_dir', required=True,
                        help='Directory of NGT_Aug logos_features (.npy files, '
                             'both palmer and digits characters)')
    parser.add_argument('--piotr_views', default='LEFT,MIDDLE,RIGHT',
                        help='Comma-separated piotrAnims views to include '
                             '(default: LEFT,MIDDLE,RIGHT)')
    parser.add_argument('--ngt_cams', default='cam2,cam3,cam4',
                        help='Comma-separated NGT_Aug cameras to include for each '
                             'character (default: cam2,cam3,cam4)')
    parser.add_argument('--output', required=True,
                        help='Path to write the JSON manifest')
    args = parser.parse_args()

    views = [v.strip() for v in args.piotr_views.split(',')]
    cams = [c.strip() for c in args.ngt_cams.split(',')]

    piotr_dir = Path(args.piotr_dir)
    ngt_aug_dir = Path(args.ngt_aug_dir)

    if not piotr_dir.is_dir():
        sys.exit(f'ERROR: piotr_dir not found: {piotr_dir}')
    if not ngt_aug_dir.is_dir():
        sys.exit(f'ERROR: ngt_aug_dir not found: {ngt_aug_dir}')

    print(f'Scanning piotrAnims:     {piotr_dir}  (views={views})')
    piotr = _scan_piotr(piotr_dir, views)
    full_piotr = sum(1 for d in piotr.values() if all(v in d for v in views))
    print(f'  Found {len(piotr)} keys, {full_piotr} with all {len(views)} views')

    print(f'Scanning NGT_Aug palmer: {ngt_aug_dir}  (cams={cams})')
    palmer = _scan_ngtaug(ngt_aug_dir, _PALMER_RE, 'ngt_aug_palmer_', cams)
    full_palmer = sum(1 for d in palmer.values() if all(c in d for c in cams))
    print(f'  Found {len(palmer)} keys, {full_palmer} with all {len(cams)} cams')

    print(f'Scanning NGT_Aug digits: {ngt_aug_dir}  (cams={cams})')
    digits = _scan_ngtaug(ngt_aug_dir, _DIGITS_RE, 'ngt_aug_digits_', cams)
    full_digits = sum(1 for d in digits.values() if all(c in d for c in cams))
    print(f'  Found {len(digits)} keys, {full_digits} with all {len(cams)} cams')

    # Only emit signs present in all three sources
    candidate_keys = sorted(set(piotr) & set(palmer) & set(digits))
    matched = []
    dropped = 0
    for key in candidate_keys:
        missing_piotr = [v for v in views if v not in piotr[key]]
        missing_palmer = [c for c in cams if c not in palmer[key]]
        missing_digits = [c for c in cams if c not in digits[key]]
        if missing_piotr or missing_palmer or missing_digits:
            parts = []
            if missing_piotr:
                parts.append(f'piotr views={missing_piotr}')
            if missing_palmer:
                parts.append(f'palmer cams={missing_palmer}')
            if missing_digits:
                parts.append(f'digits cams={missing_digits}')
            print(f'  DROP {key}: missing {", ".join(parts)}', file=sys.stderr)
            dropped += 1
            continue
        matched.append(key)

    print(f'Matched: {len(matched)}  (dropped {dropped} with incomplete views/cams)')

    if not matched:
        sys.exit('ERROR: no matched signs — check directory contents and '
                 '--piotr_views / --ngt_cams arguments')

    manifest = {}
    for key in matched:
        manifest[key] = {
            'real': [piotr[key][v] for v in views],
            'unreal': [palmer[key][c] for c in cams] + [digits[key][c] for c in cams],
        }

    # Print a few examples for sanity-checking
    for key in matched[:3]:
        print(f'  {key}')
        print(f'    real:   {manifest[key]["real"]}')
        print(f'    unreal: {manifest[key]["unreal"]}')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote {len(manifest)} signs → {out_path}')


if __name__ == '__main__':
    main()
