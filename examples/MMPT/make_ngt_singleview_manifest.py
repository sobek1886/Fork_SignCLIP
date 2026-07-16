"""Build the per-arm NGT manifests for the Flux-vs-Unreal augmentation comparison.

Experiment ladder (each arm = one change over the E1 baseline; SupCon unless
noted; K = views per training item; batch = round(192/K) keeps the encoder-pass
budget constant):

  E1  ngt_sv_real            FRONT only (VidNCELoss, K=1)
  E2  ngt_sv_real_multiview  FRONT + other real camera views        (K=3)
  E3  ngt_sv_unreal          FRONT + palmer/digits at the front cam (K=3)
  E4  ngt_sv_flux            FRONT + signer_swap + skin_mst         (K=3)
  E5  ngt_sv_unreal_full     FRONT + 2 avatars x 3 cams             (K=7)
  E6  ngt_sv_flux_full       FRONT + signer_swap + skin + shirt_1   (K=4)
  E7  ngt_sv_flux_glasses    FRONT + all 4 Flux variants            (K=5)
  E8  ngt_sv_combined        FRONT + 6 Unreal + 3 Flux              (K=10)
  E9  ngt_sv_unreal_only     palmer x 3 cams, NO real               (K=3)
  E10 ngt_sv_flux_only       3 Flux variants, NO real               (K=3)

Subset tier (low-data robustness; sentences restricted to those with a Bushuis
test counterpart, ONE fixed take each):
  E1s ngt_subset_real / E5s ngt_subset_unreal_full / E6s ngt_subset_flux_full

Plus ngt_sv_eval_manifest.json: the shared evaluation manifest (real views of
take 0 per sentence, FRONT first) used by eval_ngt_retrieval.py for every arm.

View naming: the real files labeled "RIGHT" are the FRONT (centre) camera;
the matching Unreal render camera is cam4 (--front_cam). This mapping is a
recording artefact documented here only — thesis text just says "front view".

Manifest entry format: sentence_id -> LIST of take dicts
    [{"real": [...], "unreal": [...]}, ...]
one dict per repeat take of the sentence (sorted by session, then take index).
NGTPairVideoProcessor samples one take per training step, so all repeat takes
participate over training while a batch never holds two items of the same
sentence (no same-sentence false negatives). Eval always uses take 0.

Only sentences with at least one CORE-COMPLETE take are emitted, and the same
sentence set is used by every arm except E2 (see below). A take is
core-complete when it has: the FRONT real view + all unreal renders
(2 characters x 3 cams) + all 4 Flux variants. E2 additionally needs the other
real views; sentences whose complete takes never carry them are dropped from
the E2 manifest only, with a loud count (disclose in the thesis if non-zero).

File naming expected (produced by extract_logos_features*.py):
  real:    piotr_anims_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25.npy
  unreal:  ngt_aug_palmer_M20260227_5998_260316_0_cam4.npy
  flux:    ngt_flux_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25_signer_swap.npy

Usage (on Snellius):
    python make_ngt_singleview_manifest.py \
        --piotr_dir   /scratch-shared/psobecki/piotrAnims/logos_features \
        --ngt_aug_dir /scratch-shared/psobecki/NGT_Aug/logos_features \
        --flux_dir    /scratch-shared/psobecki/NGT_Flux/logos_features \
        --bushuis_dir /scratch-shared/psobecki/Bushuis/logos_features \
        --output_dir  /home/psobecki
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# real:  piotr_anims_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25.npy
_REAL_RE = re.compile(
    r'^(M\d{8}_\d+)_(\d{6})_(\d+)_(LEFT|MIDDLE|RIGHT)_'
    r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
# unreal: ngt_aug_palmer_M20260227_5998_260316_0_cam4.npy
_UNREAL_RE = re.compile(r'^(M\d{8}_\d+)_(\d{6})_(\d+)_cam(\d+)$')
# flux:  ngt_flux_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25_signer_swap.npy
_FLUX_RE = re.compile(
    r'^ngt_flux_(M\d{8}_\d+)_(\d{6})_(\d+)_(LEFT|MIDDLE|RIGHT)_'
    r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$')

FLUX_VARIANTS = ['signer_swap', 'skin_mst_diffusion', 'shirt_1', 'glasses']
FLUX_IDENTITY = ['signer_swap', 'skin_mst_diffusion']          # E4
FLUX_NO_GLASSES = ['signer_swap', 'skin_mst_diffusion', 'shirt_1']  # E6/E8/E10


def index_takes(args):
    """→ takes[sentence][(session, take_idx)] = {front, others{view: path},
    unreal{(char, cam): path}, flux{variant: path}}"""
    takes = defaultdict(lambda: defaultdict(
        lambda: {'front': None, 'others': {}, 'unreal': {}, 'flux': {}}))

    n = 0
    for p in sorted(Path(args.piotr_dir).glob(f'{args.real_prefix}*.npy')):
        m = _REAL_RE.match(p.stem[len(args.real_prefix):])
        if not m:
            continue
        sent, sess, take, view = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        slot = takes[sent][(sess, take)]
        if view == args.front_view_label:
            slot['front'] = str(p.resolve())
        else:
            slot['others'][view] = str(p.resolve())
        n += 1
    print(f'Indexed {n} real view files from {args.piotr_dir}')

    n = 0
    for char in args.unreal_characters:
        prefix = f'ngt_aug_{char}_'
        for p in sorted(Path(args.ngt_aug_dir).glob(f'{prefix}*.npy')):
            m = _UNREAL_RE.match(p.stem[len(prefix):])
            if not m:
                continue
            sent, sess, take, cam = m.group(1), m.group(2), int(m.group(3)), f'cam{m.group(4)}'
            takes[sent][(sess, take)]['unreal'][(char, cam)] = str(p.resolve())
            n += 1
    print(f'Indexed {n} unreal render files from {args.ngt_aug_dir}')

    n = 0
    for p in sorted(Path(args.flux_dir).glob('ngt_flux_*.npy')):
        m = _FLUX_RE.match(p.stem)
        if not m:
            continue
        sent, sess, take, view, variant = (m.group(1), m.group(2), int(m.group(3)),
                                           m.group(4), m.group(5))
        if view != args.front_view_label or variant not in FLUX_VARIANTS:
            continue
        takes[sent][(sess, take)]['flux'][variant] = str(p.resolve())
        n += 1
    print(f'Indexed {n} flux variant files from {args.flux_dir}')
    return takes


def core_complete(slot, args):
    """front + all unreal (chars x cams) + all 4 flux variants present."""
    if slot['front'] is None:
        return False
    for char in args.unreal_characters:
        for cam in args.unreal_cams:
            if (char, cam) not in slot['unreal']:
                return False
    return all(v in slot['flux'] for v in FLUX_VARIANTS)


def build_take_dict(slot, arm, args):
    """→ {"real": [...], "unreal": [...]} for one take under one arm."""
    front = slot['front']
    fcam = args.front_cam
    chars = args.unreal_characters
    unreal_full = [slot['unreal'][(c, cam)] for c in chars for cam in args.unreal_cams]
    flux = slot['flux']

    if arm == 'real':
        return {'real': [front], 'unreal': []}
    if arm == 'real_multiview':
        others = [slot['others'][v] for v in sorted(slot['others'])]
        return {'real': [front] + others, 'unreal': []}
    if arm == 'unreal':
        return {'real': [front],
                'unreal': [slot['unreal'][(c, fcam)] for c in chars]}
    if arm == 'flux':
        return {'real': [front], 'unreal': [flux[v] for v in FLUX_IDENTITY]}
    if arm == 'unreal_full':
        return {'real': [front], 'unreal': unreal_full}
    if arm == 'flux_full':
        return {'real': [front], 'unreal': [flux[v] for v in FLUX_NO_GLASSES]}
    if arm == 'flux_glasses':
        return {'real': [front], 'unreal': [flux[v] for v in FLUX_VARIANTS]}
    if arm == 'combined':
        return {'real': [front],
                'unreal': unreal_full + [flux[v] for v in FLUX_NO_GLASSES]}
    if arm == 'unreal_only':
        c = chars[0]
        return {'real': [],
                'unreal': [slot['unreal'][(c, cam)] for cam in args.unreal_cams]}
    if arm == 'flux_only':
        return {'real': [], 'unreal': [flux[v] for v in FLUX_NO_GLASSES]}
    raise ValueError(arm)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--piotr_dir', required=True,
                        help='real Logos features, all sessions')
    parser.add_argument('--ngt_aug_dir', required=True,
                        help='Unreal render Logos features (ngt_aug_{char}_*.npy)')
    parser.add_argument('--flux_dir', required=True,
                        help='Flux-augmented Logos features (ngt_flux_*.npy)')
    parser.add_argument('--bushuis_dir', required=True,
                        help='Bushuis test features (bushuis_M*.npy) — defines the '
                             'subset tier and lets us report test coverage')
    parser.add_argument('--real_prefix', default='piotr_anims_')
    parser.add_argument('--front_view_label', default='RIGHT',
                        help='label of the FRONT camera in real/flux filenames '
                             '(the recordings labeled RIGHT are the front view)')
    parser.add_argument('--front_cam', default='cam4',
                        help='Unreal render camera matching the front view')
    parser.add_argument('--unreal_characters', nargs='+', default=['palmer', 'digits'])
    parser.add_argument('--unreal_cams', nargs='+', default=['cam2', 'cam3', 'cam4'])
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    if args.front_cam not in args.unreal_cams:
        raise SystemExit(f'--front_cam {args.front_cam} not in --unreal_cams')

    takes = index_takes(args)

    # Core-complete takes per sentence, sorted (session, take) → take 0 stable.
    sentences = {}
    n_incomplete = 0
    for sent, slots in takes.items():
        complete = {k: v for k, v in slots.items() if core_complete(v, args)}
        n_incomplete += len(slots) - len(complete)
        if complete:
            sentences[sent] = [complete[k] for k in sorted(complete)]
    if not sentences:
        raise SystemExit('No sentence has a core-complete take — check dirs, '
                         '--front_view_label, --front_cam and that extraction '
                         'covered all sessions for real, unreal AND flux.')
    print(f'\nSentences with >=1 core-complete take: {len(sentences)} '
          f'({sum(len(v) for v in sentences.values())} takes; '
          f'{n_incomplete} incomplete takes dropped)')

    bushuis_ids = {p.stem[len('bushuis_'):] for p in
                   Path(args.bushuis_dir).glob('bushuis_M*.npy')}
    subset_ids = sorted(set(sentences) & bushuis_ids)
    print(f'Bushuis features found for {len(bushuis_ids)} sentences; '
          f'test-matched subset: {len(subset_ids)} sentences')

    keys = sorted(sentences)
    arms = ['real', 'real_multiview', 'unreal', 'flux', 'unreal_full',
            'flux_full', 'flux_glasses', 'combined', 'unreal_only', 'flux_only']

    os.makedirs(args.output_dir, exist_ok=True)

    def write(name, manifest):
        out = os.path.join(args.output_dir, name)
        with open(out, 'w') as f:
            json.dump(manifest, f, indent=2)
        ks = {len(t['real']) + len(t['unreal'])
              for takes_ in manifest.values() for t in takes_}
        print(f'  Wrote {len(manifest):4d} sentences (K={sorted(ks)}) → {out}')

    for arm in arms:
        if arm == 'real_multiview':
            # E2: keep only takes that carry extra real views; a sentence with
            # no such take drops out of THIS arm only. Require a consistent
            # view count so K is fixed within the arm.
            counts = defaultdict(int)
            for sent in keys:
                for slot in sentences[sent]:
                    counts[len(slot['others'])] += 1
            n_extra = max((c for c in counts if c > 0), default=0)
            manifest = {}
            for sent in keys:
                tl = [build_take_dict(s, arm, args) for s in sentences[sent]
                      if len(s['others']) == n_extra]
                if tl:
                    manifest[sent] = tl
            dropped = len(keys) - len(manifest)
            if n_extra == 0 or dropped:
                print(f'  WARNING real_multiview: {dropped} sentences lack the '
                      f'extra real views (K=1+{n_extra}); disclose the subset '
                      f'size in the thesis if this arm is reported.')
        else:
            manifest = {sent: [build_take_dict(s, arm, args)
                               for s in sentences[sent]] for sent in keys}
        write(f'ngt_sv_{arm}_manifest.json', manifest)

    # Subset tier: test-matched sentences, ONE fixed take (take 0).
    for arm in ['real', 'unreal_full', 'flux_full']:
        manifest = {sent: [build_take_dict(sentences[sent][0], arm, args)]
                    for sent in subset_ids}
        write(f'ngt_subset_{arm}_manifest.json', manifest)

    # Shared eval manifest: real views of take 0, FRONT first.
    eval_manifest = {}
    for sent in keys:
        slot = sentences[sent][0]
        others = [slot['others'][v] for v in sorted(slot['others'])]
        eval_manifest[sent] = [{'real': [slot['front']] + others, 'unreal': []}]
    write('ngt_sv_eval_manifest.json', eval_manifest)

    print('\nDone. All arms except real_multiview share the identical sentence '
          'set; the subset tier is the test-matched restriction of it.')


if __name__ == '__main__':
    main()
