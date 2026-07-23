# Prepared jobs: SignCLIP heads on fine-tuned Logos backbones (run when Snellius is up)

Purpose: fill the remaining rows of the ASL-Citizen SignCLIP-head retrieval table
(`tab:asl_citizen_retrieval`). This session already ran the **scratch recipe on
CE/aug/SDDA-FT** backbones (V2T R@1 0.772 / 0.781 / 0.782). These prepared jobs add
the two missing pieces.

## What's prepared
Train + test configs under `projects/retri/signclip_asl/`, jobs under `jobs/`:

| Job | Head recipe | Backbone (vfeat_dir) | Restores from |
|---|---|---|---|
| `sc_on_native.job` | scratch (nowd) | `logos_features_native` (frozen) | — (from scratch) |
| `sc_augft_on_ce.job` | aug-ft (+aug, lr1e-5) | `logos_features_run2` (CE-FT) | `signclip_scratch_on_ce/checkpoint_last.pt` |
| `sc_augft_on_aug.job` | aug-ft | `logos_features_run3` (aug-FT) | `signclip_scratch_on_aug/checkpoint_last.pt` |
| `sc_augft_on_sdda.job` | aug-ft | `logos_features_run_sdda` (SDDA-FT) | `signclip_scratch_on_sdda/checkpoint_last.pt` |
| `sc_augft_on_native.job` | aug-ft | `logos_features_native` (frozen) | `signclip_scratch_on_native/checkpoint_last.pt` |

## Why
- **`sc_on_native`** is the load-bearing control: the scratch-on-FT heads use
  `logos_native` preprocessing while the table's 0.750 baseline uses `direct224`.
  This row isolates preprocessing from the fine-tuning effect (rules out that the
  +2--3 pts of the FT-backbone heads is just direct224-vs-native).
- **`sc_augft_on_*`** applies the successful `aug_logos_ft` recipe (the +1.1
  headline head) on top of each fine-tuned backbone, to see if backbone-FT and the
  best head recipe compound.

## Submission order
1. `sbatch jobs/sc_on_native.job`   (independent; scratch, no restore, no aug)
2. `sbatch jobs/sc_augft_on_ce.job jobs/sc_augft_on_aug.job jobs/sc_augft_on_sdda.job`
   — the `signclip_scratch_on_{ce,aug,sdda}/checkpoint_last.pt` they restore from
   **already exist** from this session, so these can run immediately.
3. `sc_augft_on_native` restores from `signclip_scratch_on_native/checkpoint_last.pt`,
   which step 1 produces — submit it **after** `sc_on_native` finishes, e.g.
   `sbatch --dependency=afterok:<sc_on_native_jobid> jobs/sc_augft_on_native.job`.

## VERIFY before running the `sc_augft_on_*` jobs
The `aug_logos_ft` recipe uses `splits_dir: /home/psobecki/ASL_Citizen/splits_aug`
(augmented splits that reference augmented-video IDs). The chosen `vfeat_dir` (each
FT backbone dir) **must contain features for those augmented videos**, extracted with
the same fine-tuned backbone. CHECK:
```
awk -F, 'NR>1{print $2}' /home/psobecki/ASL_Citizen/splits_aug/train.csv | head -5   # sample aug video ids
ls /scratch-shared/psobecki/ASL_Citizen/logos_features_run2/ | grep -c <one of those ids>
```
If the augmented-video features are **absent** from the FT dirs, either (a) re-extract
the augmented videos with each FT backbone first, or (b) fall back to the plain
**fine-tuned (non-aug)** recipe on the FT backbones (drop `splits_aug`, use `splits`).

## After running
Pull V2T R@1/R@5/R@10/Med.R from each `runs/retri_asl/<run>/` eval log (the second
`R@1:` line is V2T), fill the corresponding rows in `tab:asl_citizen_retrieval`, and
resolve the `\todome` note there.
