"""Build the three single-view NGT manifests for the Flux-vs-Unreal comparison.

Motivation (thesis story point 12): Unreal can vary camera viewpoint AND
appearance from one recording, while Flux (a 2D image edit) can only vary
appearance from a single fixed view. To isolate the appearance axis, all three
arms are restricted to the SAME single real view (the OBS RIGHT camera — the
view the Flux augmentations were generated from):

  arm "real"   : {real: [RIGHT.npy],  unreal: []}                       K=1
  arm "unreal" : {real: [RIGHT.npy],  unreal: [palmer_CAM, digits_CAM]} K=3
  arm "flux"   : {real: [RIGHT.npy],  unreal: [flux variants of RIGHT]} K=1+V

Only signs present and complete in ALL THREE arms are emitted, so every arm
trains on the identical sign set. Output feeds NGTPairMetaProcessor /
NGTPairVideoProcessor unchanged (per-arm manifest = per-arm view selection;
no processor changes needed).

File naming expected (produced by extract_logos_features*.py):
  piotrAnims:  piotr_anims_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25.npy
  NGT_Aug:     ngt_aug_palmer_M20260227_5998_260316_0_cam4.npy
               ngt_aug_digits_M20260227_5998_260316_0_cam3.npy
  NGT Flux:    ngt_flux_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25_signer_swap.npy
               (from extract_logos_features_ngt_flux.py in Fork_Logos, run on
                nano_banana/output/flux-aug/ngt/2026-03-16/augmented_frames)

Sign key = M{########}_{N} (first two underscore-separated fields), matching
make_ngt_pair_manifest.py.

NOTE (confirm with Piotr before sbatch): --unreal_cam defaults to cam4 on the
assumption that cam2/cam3/cam4 = LEFT/MIDDLE/RIGHT, i.e. cam4 is the render
camera matching the real RIGHT view. If the mapping differs, pass the right
one explicitly.

Usage (on Snellius):
    python make_ngt_singleview_manifest.py \\
        --piotr_dir   /scratch-shared/psobecki/piotrAnims/logos_features \\
        --ngt_aug_dir /scratch-shared/psobecki/NGT_Aug/logos_features \\
        --flux_dir    /scratch-shared/psobecki/NGT_Flux/logos_features \\
        --real_view   RIGHT \\
        --unreal_cam  cam4 \\
        --flux_variants signer_swap skin_mst_diffusion \\
        --output_dir  /home/psobecki
    → writes ngt_sv_real_manifest.json, ngt_sv_unreal_manifest.json,
             ngt_sv_flux_manifest.json into --output_dir
"""

import argparse
import json
import os
import re
from pathlib import Path

SIGN_KEY_RE = re.compile(r'(M\d{8}_\d+)')


def sign_key(filename):
    m = SIGN_KEY_RE.search(filename)
    return m.group(1) if m else None


def index_dir(directory, prefix, must_contain=None, suffix=None):
    """Map sign_key → absolute path for .npy files matching the filters."""
    out = {}
    for p in sorted(Path(directory).glob(f'{prefix}*.npy')):
        name = p.stem
        if must_contain and must_contain not in name:
            continue
        if suffix and not name.endswith(suffix):
            continue
        key = sign_key(name)
        if key is None:
            continue
        if key in out:
            print(f'  WARN duplicate {prefix} entry for {key}: keeping {out[key]}, '
                  f'ignoring {p}')
            continue
        out[key] = str(p.resolve())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--piotr_dir', required=True,
                        help='piotrAnims Logos features (piotr_anims_*.npy)')
    parser.add_argument('--ngt_aug_dir', required=True,
                        help='Unreal render Logos features (ngt_aug_{palmer,digits}_*.npy)')
    parser.add_argument('--flux_dir', required=True,
                        help='Flux-augmented Logos features (ngt_flux_*_{variant}.npy)')
    parser.add_argument('--real_view', default='RIGHT',
                        help='Which real camera view all arms share (default RIGHT — '
                             'the view Flux augmented)')
    parser.add_argument('--unreal_cam', default='cam4',
                        help='Which render camera matches --real_view (default cam4; '
                             'CONFIRM the cam↔view mapping before sbatch)')
    parser.add_argument('--unreal_characters', nargs='+', default=['palmer', 'digits'],
                        help='Avatars for the unreal arm (default both → K=3, matching '
                             'the default 2-variant flux arm)')
    parser.add_argument('--flux_variants', nargs='+',
                        default=['signer_swap', 'skin_mst_diffusion'],
                        help='Flux variants for the flux arm. Default signer_swap + '
                             'skin_mst_diffusion (the two identity-changing edits) so '
                             'K=3 matches the unreal arm; pass all four for a K=5 arm '
                             '(then also adjust batch size in the yaml).')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    view_tag = f'_{args.real_view}_'
    print(f'Indexing real view ({args.real_view}) from {args.piotr_dir} ...')
    real = index_dir(args.piotr_dir, 'piotr_anims_', must_contain=view_tag)
    print(f'  {len(real)} signs with a {args.real_view} view')

    unreal = {}
    for char in args.unreal_characters:
        print(f'Indexing unreal {char}/{args.unreal_cam} from {args.ngt_aug_dir} ...')
        unreal[char] = index_dir(args.ngt_aug_dir, f'ngt_aug_{char}_',
                                 suffix=f'_{args.unreal_cam}')
        print(f'  {len(unreal[char])} signs')

    flux = {}
    for variant in args.flux_variants:
        print(f'Indexing flux variant {variant} from {args.flux_dir} ...')
        flux[variant] = index_dir(args.flux_dir, 'ngt_flux_',
                                  must_contain=view_tag, suffix=f'_{variant}')
        print(f'  {len(flux[variant])} signs')

    # Intersect: a sign must be complete in every arm.
    keys = set(real)
    for char in args.unreal_characters:
        keys &= set(unreal[char])
    for variant in args.flux_variants:
        keys &= set(flux[variant])
    keys = sorted(keys)
    print(f'\nSigns complete in all three arms: {len(keys)}')
    if not keys:
        raise SystemExit('No overlapping signs — check dirs, --real_view, '
                         '--unreal_cam, and that the flux extraction ran on the '
                         'piotrAnims-session date batch (2026-03-16).')

    manifests = {
        'ngt_sv_real_manifest.json': {
            k: {'real': [real[k]], 'unreal': []} for k in keys},
        'ngt_sv_unreal_manifest.json': {
            k: {'real': [real[k]],
                'unreal': [unreal[c][k] for c in args.unreal_characters]}
            for k in keys},
        'ngt_sv_flux_manifest.json': {
            k: {'real': [real[k]],
                'unreal': [flux[v][k] for v in args.flux_variants]}
            for k in keys},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    for name, manifest in manifests.items():
        out_path = os.path.join(args.output_dir, name)
        with open(out_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        k_views = 1 + len(next(iter(manifest.values()))['unreal'])
        print(f'  Wrote {len(manifest)} signs (K={k_views}) → {out_path}')


if __name__ == '__main__':
    main()
