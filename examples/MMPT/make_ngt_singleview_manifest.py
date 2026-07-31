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

View naming: the real front-camera files were originally mislabeled "RIGHT";
on 2026-07-20 the videos AND their .npy features were renamed to "MIDDLE"
(the true LEFT/RIGHT side cameras arrived as ngt_train_left/right .mkv, so
the label had to be freed). The 2026-07 re-rendered Unreal sequences name
their cameras by view (left/middle/right) instead of the purged cam2/3/4.
Flux frames/features were generated BEFORE the rename and keep the legacy
"RIGHT" label for the front view — hence the separate --flux_front_view_label.
Thesis text just says "front view".

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
  real:    piotr_anims_M20260227_5998_260316_1_MIDDLE_2026-03-16_12-55-25.npy
  unreal:  ngt_aug_palmer_M20260227_5998_260316_0_middle.npy
  flux:    ngt_flux_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25_signer_swap.npy
           (legacy RIGHT = front; see --flux_front_view_label)

Usage (on Snellius):
    python make_ngt_singleview_manifest.py \
        --piotr_dir   /scratch-shared/psobecki/piotrAnims/logos_features \
        --ngt_aug_dir /scratch-shared/psobecki/NGT_Aug/logos_features \
        --flux_dir    /scratch-shared/psobecki/NGT_Flux/logos_features \
        --bushuis_dir /scratch-shared/psobecki/Bushuis/logos_features \
        --output_dir  /home/psobecki

Real-signer-count mode (--sessions): instead of the E1-E10 ladder, emit ONE
fixed-data condition — the paired real + unreal_full arms restricted to the
given recording sessions (= signers), exactly --sentence_budget sentences split
evenly across sessions (largest remainder, CLI order), ONE take per sentence
(the first core-complete take in its session). #sentences == #videos, so every
condition trains on the same number of front-view videos and the same epoch
size regardless of signer count; both arms share identical sentences and takes.
A coverage sidecar records the Bushuis-matched count per session (reported as a
covariate — sessions are near sentence-disjoint, so eval coverage cannot be
equalised across signer counts).
    ... --sessions 251126 260129 --sentence_budget 200 --budget_seed 0 --suffix R2

Covered-only equalized mode (--covered_equal): the 2026-07 replacement for the
--sessions design above. Training sentences are restricted to the COVERED pool
(every training sentence has a counterpart in the frozen front-view eval set
= set(eval manifest) & set(bushuis features)), so eval coverage no longer
drifts with signer count. One run emits every condition S=1..len(--sessions):
condition S uses the first S sessions, draws C=--covered_per_condition videos
split evenly over them (largest remainder, CLI order: C=100 → 100 / 50+50 /
34+33+33 / 25*4), ONE take per sentence, and the C sentences are UNIQUE within
the condition. Sessions' covered pools overlap slightly, so a condition fills
its quotas scarcest-session-first, skipping sentences already taken by another
session of the SAME condition. Each session's covered pool is shuffled once
into a fixed priority order (seeded by --budget_seed and the session id, so it
does not depend on which condition is being built); conditions take prefixes of
that order, which makes the draws nested/overlap-maximizing across S and keeps
the union of all conditions' seen sentences — and hence the common-unseen eval
set (eval set minus that union) — small and identical for every condition.
Emits the same per-arm manifests + coverage sidecars as --sessions mode, plus a
design JSON with the cross-condition gate report.
    ... --covered_equal --sessions 251126 251217 260129 260316 \
        --covered_per_condition 100 --budget_seed 0 --suffix eq \
        --eval_manifest /home/psobecki/ngt_sv_eval_manifest.json
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

# real:  piotr_anims_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25.npy
_REAL_RE = re.compile(
    r'^(M\d{8}_\d+)_(\d{6})_(\d+)_(LEFT|MIDDLE|RIGHT)_'
    r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
# unreal: ngt_aug_palmer_M20260227_5998_260316_0_middle.npy
_UNREAL_RE = re.compile(r'^(M\d{8}_\d+)_(\d{6})_(\d+)_(left|middle|right)$')
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
            sent, sess, take, cam = m.group(1), m.group(2), int(m.group(3)), m.group(4)
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
        if view != args.flux_front_view_label or variant not in FLUX_VARIANTS:
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
    if arm == 'mv_unreal_full':
        # real multiview (front + real side views) PLUS the full Unreal set:
        # 3 real camera views + 2 avatars x 3 cams = K=9.
        others = [slot['others'][v] for v in sorted(slot['others'])]
        return {'real': [front] + others, 'unreal': unreal_full}
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


def write_session_subset(sentences, sentence_keys, bushuis_ids, args):
    """--sessions mode: one fixed-data signer-count condition (docstring top)."""
    sess_sel = list(dict.fromkeys(args.sessions))
    budget, suffix = args.sentence_budget, args.suffix
    mv = args.multiview

    # Multiview mode pairs real-multiview (K=3) with mv_unreal_full (K=9), so the
    # pinned take must additionally carry the real side views (>=2 'others').
    def take_ok(sent, i):
        return (not mv) or len(sentences[sent][i]['others']) >= 2

    # Assign each sentence to the first selected session holding a usable
    # take, and pin its single take = first usable take in that session.
    eligible = {s: [] for s in sess_sel}
    chosen = {}  # sent -> (session, take_key_index_into_sentence_lists)
    for sent in sorted(sentences):
        for i, (sess, take) in enumerate(sentence_keys[sent]):
            if sess in sess_sel and take_ok(sent, i):
                eligible[sess].append(sent)
                chosen[sent] = (sess, i, take)
                break

    # Largest-remainder split of the budget, CLI session order.
    base, rem = divmod(budget, len(sess_sel))
    quotas = {s: base + (1 if j < rem else 0) for j, s in enumerate(sess_sel)}
    for s in sess_sel:
        if quotas[s] > len(eligible[s]):
            raise SystemExit(f'Session {s}: quota {quotas[s]} > '
                             f'{len(eligible[s])} eligible sentences — lower '
                             f'--sentence_budget or change --sessions.')

    rng = random.Random(args.budget_seed)
    selected = []
    for s in sess_sel:
        selected += rng.sample(eligible[s], quotas[s])
    selected = sorted(selected)
    assert len(selected) == budget

    # Single-view: real (K=1) + unreal_full (K=7).  Multiview: real_multiview
    # (K=3) + mv_unreal_full (K=9).  Manifest filename stem per arm.
    if args.e3_arm:
        arms = ['unreal']
    elif mv:
        arms = ['real_multiview', 'mv_unreal_full']
    else:
        arms = ['real', 'unreal_full']
    stem = {'real': 'real', 'unreal_full': 'unreal_full', 'unreal': 'unreal',
            'real_multiview': 'mv_real', 'mv_unreal_full': 'mv_unreal_full'}
    manifests = {}
    for arm in arms:
        manifests[arm] = {
            sent: [build_take_dict(sentences[sent][chosen[sent][1]], arm, args)]
            for sent in selected}
        out = os.path.join(args.output_dir,
                           f'ngt_sv_{stem[arm]}_{suffix}_manifest.json')
        with open(out, 'w') as f:
            json.dump(manifests[arm], f, indent=2)
        ks = {len(t['real']) + len(t['unreal'])
              for tl in manifests[arm].values() for t in tl}
        print(f'  Wrote {len(manifests[arm]):4d} sentences (K={sorted(ks)}) → {out}')

    # Paired-arms gate: identical real slots (same sentence, same take).
    if len(arms) > 1:
        for sent in selected:
            assert (manifests[arms[0]][sent][0]['real']
                    == manifests[arms[1]][sent][0]['real']), sent

    per_session = {}
    for s in sess_sel:
        sel_s = [x for x in selected if chosen[x][0] == s]
        per_session[s] = {
            'eligible': len(eligible[s]), 'quota': quotas[s],
            'selected': len(sel_s),
            'bushuis_matched_selected': len(set(sel_s) & bushuis_ids)}
    coverage = {
        'suffix': suffix, 'sessions': sess_sel, 'sentence_budget': budget,
        'budget_seed': args.budget_seed, 'per_session': per_session,
        'total_selected': len(selected),
        'total_bushuis_matched': len(set(selected) & bushuis_ids),
        'sentences': {sent: {'session': chosen[sent][0],
                             'take': chosen[sent][2]} for sent in selected}}
    cov_stem = (f'ngt_sv_e3_{suffix}_coverage.json' if args.e3_arm else
                f'ngt_sv_mv_{suffix}_coverage.json' if mv else f'ngt_sv_{suffix}_coverage.json')
    out = os.path.join(args.output_dir, cov_stem)
    with open(out, 'w') as f:
        json.dump(coverage, f, indent=2)
    print(f'  Coverage sidecar → {out}')
    print(f'\nCondition {suffix}: {len(sess_sel)} signer(s) {sess_sel}, '
          f'{budget} sentences = {budget} front videos '
          f'(per-session {[quotas[s] for s in sess_sel]}), '
          f'Bushuis-matched {coverage["total_bushuis_matched"]}.')


_ARM_STEM = {'real': 'real', 'unreal_full': 'unreal_full', 'unreal': 'unreal',
             'real_multiview': 'mv_real', 'mv_unreal_full': 'mv_unreal_full'}


def covered_pools(sentences, sentence_keys, eval_ids, sess_list, mv):
    """→ pools[session][sentence] = (take_list_index, take_idx).

    A sentence is in a session's covered pool when it is in the frozen eval set
    (eval_ids) AND that session holds a usable take; the pinned take is the
    first usable one in that session (multiview additionally needs the real
    side views, i.e. >= 2 'others')."""
    pools = {s: {} for s in sess_list}
    for sent in sorted(eval_ids):
        if sent not in sentence_keys:
            continue
        for i, (sess, take) in enumerate(sentence_keys[sent]):
            if sess not in pools or sent in pools[sess]:
                continue
            if mv and len(sentences[sent][i]['others']) < 2:
                continue
            pools[sess][sent] = (i, take)
    return pools


def write_covered_equal(sentences, sentence_keys, bushuis_ids, eval_ids, args):
    """--covered_equal mode: all S conditions, covered-only, C videos each."""
    sess_all = list(dict.fromkeys(args.sessions))
    mv, C = args.multiview, args.covered_per_condition
    arms = (['real_multiview', 'mv_unreal_full'] if mv else
            ['real', 'unreal_full', 'unreal'])

    pools = covered_pools(sentences, sentence_keys, eval_ids, sess_all, mv)
    sv_pools = pools if not mv else covered_pools(
        sentences, sentence_keys, eval_ids, sess_all, False)
    print(f'\nCovered pools ({"mv" if mv else "sv"}), frozen eval set '
          f'{len(eval_ids)} sentences:')
    for s in sess_all:
        print(f'  {s}: {len(pools[s]):4d} covered'
              + ('' if not mv else f'   (sv pool {len(sv_pools[s])})'))
    if mv:
        broke = {s: sorted(set(sv_pools[s]) - set(pools[s])) for s in sess_all}
        if any(broke.values()):
            print('  MV GATE FAILED — sessions/sentences without a '
                  'side-view-complete covered take:')
            for s in sess_all:
                if broke[s]:
                    print(f'    {s}: {len(broke[s])} e.g. {broke[s][:8]}')
            raise SystemExit('mv covered pools != sv covered pools; '
                             'multiview family must stay out of this design.')
        print('  MV GATE PASSED: mv covered pools == sv covered pools.')

    # One fixed priority order per session, independent of the condition.
    prio = {}
    for s in sess_all:
        lst = sorted(pools[s])
        random.Random(f'{args.budget_seed}:{s}').shuffle(lst)
        prio[s] = lst

    conditions = {}          # S -> {'sessions', 'quotas', 'per_session', 'chosen'}
    for S in range(1, len(sess_all) + 1):
        sess_sel = sess_all[:S]
        base, rem = divmod(C, S)
        quotas = {s: base + (1 if j < rem else 0) for j, s in enumerate(sess_sel)}
        taken, chosen, per_session = set(), {}, {}
        # Scarcest session first: it has the least room to dodge collisions.
        for s in sorted(sess_sel, key=lambda x: (len(pools[x]), x)):
            pick = []
            for sent in prio[s]:
                if len(pick) == quotas[s]:
                    break
                if sent in taken:
                    continue
                pick.append(sent)
                taken.add(sent)
                chosen[sent] = (s,) + pools[s][sent]
            if len(pick) < quotas[s]:
                raise SystemExit(
                    f'S={S} session {s}: only {len(pick)} of quota {quotas[s]} '
                    f'available after within-condition exclusions (pool '
                    f'{len(pools[s])}) — lower --covered_per_condition.')
            per_session[s] = sorted(pick)
        assert len(taken) == C and len(chosen) == C
        conditions[S] = {'sessions': sess_sel, 'quotas': quotas,
                         'per_session': per_session, 'chosen': chosen,
                         'selected': sorted(taken)}

    # ---- per-condition manifests + coverage sidecars -----------------------
    for S, cond in conditions.items():
        suffix = f'{args.suffix}S{S}'
        selected, chosen = cond['selected'], cond['chosen']
        assert len(selected) == C, S
        assert len(set(selected)) == C, S                 # unique sentences
        assert set(selected) <= eval_ids, S               # covered-only
        manifests = {}
        for arm in arms:
            manifests[arm] = {
                sent: [build_take_dict(sentences[sent][chosen[sent][1]], arm, args)]
                for sent in selected}
            out = os.path.join(args.output_dir,
                               f'ngt_sv_{_ARM_STEM[arm]}_{suffix}_manifest.json')
            if os.path.exists(out) and not args.overwrite:
                raise SystemExit(f'{out} exists — pass --overwrite to replace.')
            with open(out, 'w') as f:
                json.dump(manifests[arm], f, indent=2)
            ks = {len(t['real']) + len(t['unreal'])
                  for tl in manifests[arm].values() for t in tl}
            print(f'  Wrote {len(manifests[arm]):4d} sentences (K={sorted(ks)}) '
                  f'→ {out}')
        # take identity: every arm anchors on the same real front video/take.
        for sent in selected:
            fronts = {manifests[a][sent][0]['real'][0] for a in arms}
            assert len(fronts) == 1, (sent, fronts)

        per_session = {
            s: {'eligible': len(pools[s]), 'quota': cond['quotas'][s],
                'selected': len(cond['per_session'][s]),
                'bushuis_matched_selected': len(
                    set(cond['per_session'][s]) & bushuis_ids),
                'sentences': cond['per_session'][s]}
            for s in cond['sessions']}
        coverage = {
            'suffix': suffix, 'sessions': cond['sessions'],
            'sentence_budget': C, 'budget_seed': args.budget_seed,
            'covered_only': True, 'design': 'covered_equal',
            'per_session': per_session, 'total_selected': len(selected),
            'total_bushuis_matched': len(set(selected) & bushuis_ids),
            'sentences': {sent: {'session': chosen[sent][0],
                                 'take': chosen[sent][2]} for sent in selected}}
        cov_stem = (f'ngt_sv_mv_{suffix}_coverage.json' if mv else
                    f'ngt_sv_{suffix}_coverage.json')
        out = os.path.join(args.output_dir, cov_stem)
        with open(out, 'w') as f:
            json.dump(coverage, f, indent=2)
        print(f'  Coverage sidecar → {out}')
        print(f'Condition {suffix}: {S} signer(s) {cond["sessions"]}, {C} '
              f'covered sentences = {C} front videos (per-session '
              f'{[cond["quotas"][s] for s in cond["sessions"]]}).\n')

    # ---- cross-condition gates --------------------------------------------
    union = set().union(*(set(c['selected']) for c in conditions.values()))
    common_unseen = sorted(eval_ids - union)
    print(f'Union of all conditions\' seen sentences: {len(union)}')
    print(f'Common-unseen eval set: {len(eval_ids)} - {len(union)} = '
          f'{len(common_unseen)} (identical for every condition by '
          f'construction).')

    nesting = []
    for s in sess_all:
        got = {S: set(c['per_session'][s]) for S, c in conditions.items()
               if s in c['per_session']}
        for a, b in zip(sorted(got), sorted(got)[1:]):
            extra = sorted(got[b] - got[a])
            nesting.append({'session': s, 'S_small': a, 'S_large': b,
                            'nested': not extra, 'n_extra': len(extra),
                            'extra': extra})
    bad = [n for n in nesting if not n['nested']]
    print('Nesting across S per session: '
          + ('OK (every larger-S draw is a subset of the smaller-S draw)'
             if not bad else
             f'{len(bad)} of {len(nesting)} pairs not strictly nested '
             + str([(n["session"], n["S_small"], n["S_large"], n["n_extra"])
                    for n in bad])))

    old_cmp = None
    if args.old_coverage:
        old_union = set()
        for f in sorted(args.old_coverage):
            old_union |= set(json.load(open(f)).get('sentences', {}))
        old_unseen = eval_ids - old_union
        leak = sorted(union & old_unseen)
        by_sess = {s: sorted(set(prio[s]) & set(leak)) for s in sess_all}
        old_cmp = {'old_coverage_files': sorted(args.old_coverage),
                   'old_common_unseen': len(old_unseen),
                   'new_common_unseen_is_superset': not leak,
                   'leaked_from_old_common_unseen': len(leak),
                   'leaked_sentences': leak,
                   'leaked_by_session': {s: len(v) for s, v in by_sess.items()}}
        print(f'Old common-unseen set: {len(old_unseen)} sentences; new seen '
              f'union covers {len(leak)} of them '
              f'(superset gate {"PASSED" if not leak else "FAILED"}); '
              f'by session {old_cmp["leaked_by_session"]}')

    design = {
        'design': 'covered_equal', 'multiview': mv, 'arms': arms,
        'covered_per_condition': C, 'budget_seed': args.budget_seed,
        'sessions': sess_all, 'eval_set_size': len(eval_ids),
        'covered_pool_sizes': {s: len(pools[s]) for s in sess_all},
        'sv_covered_pool_sizes': {s: len(sv_pools[s]) for s in sess_all},
        'mv_gate_passed': True if mv else None,
        'priority_order': prio,
        'conditions': {f'{args.suffix}S{S}': {
            'sessions': c['sessions'],
            'quotas': {s: c['quotas'][s] for s in c['sessions']},
            'per_session_sentences': c['per_session'],
            'selected': c['selected']} for S, c in conditions.items()},
        'seen_union': sorted(union), 'seen_union_size': len(union),
        'common_unseen': common_unseen,
        'common_unseen_size': len(common_unseen),
        'nesting': nesting, 'old_design_comparison': old_cmp}
    out = os.path.join(args.output_dir,
                       f'ngt_sv_{"mv_" if mv else ""}{args.suffix}_design.json')
    with open(out, 'w') as f:
        json.dump(design, f, indent=2)
    print(f'Design/gate report → {out}')


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
    parser.add_argument('--front_view_label', default='MIDDLE',
                        help='label of the FRONT camera in real filenames '
                             '(after the 2026-07-20 RIGHT→MIDDLE rename)')
    parser.add_argument('--flux_front_view_label', default='RIGHT',
                        help='label of the FRONT camera in flux filenames — '
                             'flux frames predate the rename and keep RIGHT')
    parser.add_argument('--front_cam', default='middle',
                        help='Unreal render camera matching the front view')
    parser.add_argument('--unreal_characters', nargs='+', default=['palmer', 'digits'])
    parser.add_argument('--unreal_cams', nargs='+', default=['left', 'middle', 'right'])
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sessions', nargs='+', default=None,
                        help='real-signer-count mode: 6-digit session ids '
                             '(=signers) to include; emits only the paired '
                             'real + unreal_full arms for this condition')
    parser.add_argument('--sentence_budget', type=int, default=200,
                        help='(--sessions mode) total sentences = total front '
                             'videos, split evenly across sessions')
    parser.add_argument('--budget_seed', type=int, default=0,
                        help='(--sessions mode) seed for the per-session '
                             'sentence sample')
    parser.add_argument('--suffix', default=None,
                        help='(--sessions mode) manifest name suffix, e.g. R2')
    parser.add_argument('--e3_arm', action='store_true',
                        help='(--sessions mode) emit only the E3-style arm '
                             '(front + avatars at the front cam, K=3), '
                             'take-matched to the non-mv selection')
    parser.add_argument('--multiview', action='store_true',
                        help='(--sessions mode) emit the real-multiview (K=3) and '
                             'mv_unreal_full (K=9) arm pair instead of real (K=1) / '
                             'unreal_full (K=7); pins side-view-complete takes')
    parser.add_argument('--covered_equal', action='store_true',
                        help='covered-only equalized design (see docstring): '
                             'with --sessions s1..sN emit every condition '
                             'S=1..N at once, training only on sentences that '
                             'have a counterpart in the frozen eval set')
    parser.add_argument('--covered_per_condition', type=int, default=100,
                        help='(--covered_equal) videos per condition, split '
                             'evenly across that condition\'s sessions')
    parser.add_argument('--eval_manifest', default=None,
                        help='(--covered_equal) frozen eval manifest; the '
                             'covered set is its sentences intersected with '
                             'the Bushuis features (defaults to all '
                             'core-complete sentences & Bushuis)')
    parser.add_argument('--old_coverage', nargs='*', default=None,
                        help='(--covered_equal) old coverage sidecars; report '
                             'whether the new common-unseen set is a superset '
                             'of the old one')
    parser.add_argument('--overwrite', action='store_true',
                        help='(--covered_equal) allow replacing existing '
                             'manifests (default: refuse, non-clobbering)')
    args = parser.parse_args()

    if args.covered_equal and not (args.sessions and args.suffix):
        raise SystemExit('--covered_equal requires --sessions and --suffix')

    if args.sessions and not args.suffix:
        raise SystemExit('--sessions mode requires --suffix (e.g. R2)')

    if args.front_cam not in args.unreal_cams:
        raise SystemExit(f'--front_cam {args.front_cam} not in --unreal_cams')

    takes = index_takes(args)

    # Core-complete takes per sentence, sorted (session, take) → take 0 stable.
    sentences = {}
    sentence_keys = {}   # sent -> [(session, take_idx), ...] parallel to lists
    n_incomplete = 0
    for sent, slots in takes.items():
        complete = {k: v for k, v in slots.items() if core_complete(v, args)}
        n_incomplete += len(slots) - len(complete)
        if complete:
            sentence_keys[sent] = sorted(complete)
            sentences[sent] = [complete[k] for k in sentence_keys[sent]]
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

    if args.covered_equal:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.eval_manifest:
            with open(args.eval_manifest) as f:
                eval_ids = set(json.load(f)) & bushuis_ids
        else:
            eval_ids = set(subset_ids)
        print(f'Frozen eval (covered) set: {len(eval_ids)} sentences')
        write_covered_equal(sentences, sentence_keys, bushuis_ids,
                            eval_ids, args)
        return

    if args.sessions:
        os.makedirs(args.output_dir, exist_ok=True)
        write_session_subset(sentences, sentence_keys, bushuis_ids, args)
        return

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
