"""Clone the R1 signer-count arm configs to the covered-only eqS variants.

Same hyperparameters (batch 64, lr 5e-5, 50 epochs, VidNCE for real K=1,
SupCon tau=0.07 elsewhere); only the manifest, the run dirs and the LR
schedule change.  Schedule convention in the existing configs:
    total_num_update = max_epoch * ceil(N_videos / batch_size)
    warmup_updates   = 10% of that
180 videos -> 50 * ceil(180/64) = 50*3 = 150, warmup 15   (existing R runs)
100 videos -> 50 * ceil(100/64) = 50*2 = 100, warmup 10   (eqS runs)
"""
import math, os, re, sys

D = os.path.expanduser('~/Fork_SignCLIP/examples/MMPT/projects/retri/signclip_ngt')
N, BATCH, EPOCHS = 100, 64, 50
TOTAL = EPOCHS * math.ceil(N / BATCH)
WARMUP = TOTAL // 10
assert (TOTAL, WARMUP) == (100, 10), (TOTAL, WARMUP)

SESS = ['251126', '251217', '260129', '260316']
SPLIT = {1: [100], 2: [50, 50], 3: [34, 33, 33], 4: [25, 25, 25, 25]}
# experiment stem -> (source config, manifest stem, arm blurb)
ARMS = {
    'ngt_sv_real':        ('ngt_sv_real_R{S}.yaml',        'ngt_sv_real',
                           'E1-style: real FRONT view only, K=1, VidNCELoss.'),
    'ngt_sv_unreal':      ('ngt_sv_unreal_R{S}.yaml',      'ngt_sv_unreal',
                           'E3-style: FRONT + 2 Unreal identities at the front '
                           'cam, K=3, SupCon.'),
    'ngt_sv_unreal_full': ('ngt_sv_unreal_full_R{S}.yaml', 'ngt_sv_unreal_full',
                           'E5-style: FRONT + 2 Unreal identities x 3 cams, '
                           'K=7, SupCon.'),
    'ngt_mv_real':        ('ngt_mv_real_R{S}.yaml',        'ngt_sv_mv_real',
                           'E2-style: 3 real camera views, K=3, SupCon.'),
    'ngt_mv_unreal_full': ('ngt_mv_unreal_full_R{S}.yaml', 'ngt_sv_mv_unreal_full',
                           '3 real camera views + 2 Unreal identities x 3 cams, '
                           'K=9, SupCon.'),
}
FAMILY = sys.argv[1] if len(sys.argv) > 1 else 'sv'
stems = [a for a in ARMS if a.startswith(f'ngt_{FAMILY}_')]

RUNROOT = '/scratch-shared/psobecki/runs/retri_ngt'
written = []
for stem in stems:
    src_t, man_stem, blurb = ARMS[stem]
    for S in (1, 2, 3, 4):
        exp = f'{stem}_eqS{S}'
        src = os.path.join(D, src_t.format(S=S))
        txt = open(src).read()
        # strip the old 2-line "Real-signer-count ablation R.." header
        lines = txt.splitlines(True)
        assert lines[0].startswith('# Real-signer-count ablation'), src
        body = ''.join(lines[2:])
        head = (f'# Covered-only equalized signer-count ablation, condition S={S}\n'
                f'# (sessions {" ".join(SESS[:S])}; {N} covered videos = '
                f'{"+".join(str(x) for x in SPLIT[S])} per session, 1 take each,\n'
                f'# every training sentence has a front-view eval counterpart).\n'
                f'# {blurb}\n'
                f'# Schedule retuned for {N} videos: {EPOCHS} epochs x '
                f'ceil({N}/{BATCH}) = {TOTAL} updates, {WARMUP} warmup.\n')
        out_txt = head + body
        subs = [
            (re.compile(r'^(  pair_manifest: ).*$', re.M),
             rf'\g<1>/home/psobecki/{man_stem}_eqS{S}_manifest.json'),
            (re.compile(r'^(    save_dir: ).*$', re.M), rf'\g<1>{RUNROOT}/{exp}'),
            (re.compile(r'^(  save_path: ).*$', re.M), rf'\g<1>{RUNROOT}/{exp}'),
            (re.compile(r'^(    total_num_update: ).*$', re.M), rf'\g<1>{TOTAL}'),
            (re.compile(r'^(    warmup_updates: ).*$', re.M), rf'\g<1>{WARMUP}'),
        ]
        for rx, rep in subs:
            out_txt, n = rx.subn(rep, out_txt)
            assert n == 1, (src, rx.pattern, n)
        dst = os.path.join(D, f'{exp}.yaml')
        assert not os.path.exists(dst), f'{dst} exists'
        open(dst, 'w').write(out_txt)
        written.append(dst)

import yaml
for p in written:
    c = yaml.safe_load(open(p))
    exp = os.path.basename(p)[:-5]
    o = c['fairseq']['optimization']
    assert o['total_num_update'] == TOTAL and o['warmup_updates'] == WARMUP
    assert o['max_epoch'] == EPOCHS and c['fairseq']['dataset']['batch_size'] == BATCH
    assert c['fairseq']['checkpoint']['save_dir'].endswith(exp)
    assert c['eval']['save_path'].endswith(exp)
    assert f'_eqS{exp[-1]}_manifest.json' in c['dataset']['pair_manifest']
    assert os.path.exists(c['dataset']['pair_manifest']), c['dataset']['pair_manifest']
    print(f'{exp:34s} loss={c["loss"]["loss_cls"]:12s} '
          f'lr={c["fairseq"]["optimization"]["lr"]} '
          f'batch={c["fairseq"]["dataset"]["batch_size"]} '
          f'upd={o["total_num_update"]}/{o["warmup_updates"]} '
          f'manifest={os.path.basename(c["dataset"]["pair_manifest"])}')
print(f'\nWrote {len(written)} configs.')
