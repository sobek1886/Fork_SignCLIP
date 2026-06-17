# ASL-Citizen backbone fine-tuning — experiment checklist

Goal: test whether fine-tuning the Logos MViTv2-S backbone with **appearance invariance**
(pulling a sign's original and appearance-augmented copies together) improves ISLR on the
signer-disjoint ASL-Citizen test set. Evaluate every feature set two ways:
- **Eval A** — raw video-to-video NN retrieval (no training; SignRep Table 4 protocol)
- **Eval B** — linear probe on frozen features (SignRep Table 3 protocol)

Both report DCG / MRR / Rec@{1,5,10,20}. Headline comparison: each fine-tuned run's A/B vs
**`baseline_native`** (original weights, same `logos_native` preprocessing = the matched reference).

Two execution sites:
- **Fork_Logos jobs** → run from `$HOME/fork_mmaction2` (mmaction venv).
- **eval jobs** → run from `$HOME/Fork_SignCLIP/examples/MMPT` (MMPT venv).

Preprocessing everywhere: **`logos_native`** (resize long→300, pad 300² grey-114, center-crop 224;
aspect-preserving) + end-anchor clip coverage at extraction. ASL-Citizen frames are non-square, so the
old `direct224` path (baseline/endanchor dirs) aspect-distorts — those are legacy references only.

Results live in MLflow:
- `asl-citizen-backbone-ft`  — training curves (train_ce / consist / lr) per FT run.
- `asl-citizen-feature-eval` — final A/B metrics, one run per feature set (evalA_* + evalB_*).

---

## 0. Pre-flight (once)

- [ ] Sync updated files to Snellius:
  - → `fork_mmaction2`: `train_logos_asl_citizen.py`, `extract_logos_features.py` (now `--preproc`),
    `ft_asl_citizen_run2_baseline.job`, `ft_asl_citizen_run3_augdata.job`,
    `ft_asl_citizen_run1_consistency.job`, `ft_asl_citizen_run1_consistency_glasses.job`,
    `ft_asl_citizen_run1_consistency_shirt1.job`, `extract_logos_features_native_baseline.job`
  - → `MMPT`: `eval_asl_citizen_features.py`, `eval_asl_citizen_linear_probe.py`,
    `eval_asl_citizen_retrieval.py`, `mlflow_eval_logging.py`, `export_signclip_features.py`,
    `jobs/eval_asl_citizen_features.job`, `jobs/eval_signclip_features.job`
- [ ] Regenerate `splits_aug` with all 4 variants (needed by run3 / aug-as-data):
  ```
  python generate_asl_citizen_aug_splits.py \
    --train_csv $HOME/ASL_Citizen/splits/train.csv \
    --val_csv   $HOME/ASL_Citizen/splits/val.csv \
    --test_csv  $HOME/ASL_Citizen/splits/test.csv \
    --aug_video_list $HOME/ASL_Citizen/aug_selection/aug_video_list.txt \
    --variants glasses shirt_1 signer_swap skin_mst_diffusion \
    --output_dir $HOME/ASL_Citizen/splits_aug
  ```
- [x] `mlflow` present in fork_mmaction2 venv — confirmed (3.13.0).
- [x] **numpy pinned `<2`** in fork_mmaction2 venv — mlflow install had bumped it to 2.4.6, which broke
      torch's numpy bridge (`RuntimeError: Numpy is not available`). `pip install "numpy<2"` fixes it.
- [x] Smoke test PASSED (`sbatch ft_asl_citizen_smoke.job`): CE + paired (consistency fired) +
      train→extract round-trip all green.
- [ ] (reference) Smoke-test trainer manually (interactive GPU, ~2 min each), CE + paired paths:
  ```
  python train_logos_asl_citizen.py --smoke --mlflow \
    --train_csv $HOME/ASL_Citizen/splits/train.csv \
    --video_dir /scratch-shared/psobecki/ASL_Citizen/videos \
    --checkpoint $HOME/fork_mmaction2/data/model/logos_autsl_wlasl_model.pth \
    --save_dir /scratch-shared/psobecki/runs/asl_ft_smoke
  # paired (run1 shape): add
  #   --paired --aug_frames_dir /scratch-shared/psobecki/ASL_Citizen/augmented_frames
  ```
  Check: backbone loads (no missing non-head keys), loss drops over 2 steps, checkpoint saved.

## 1. Baselines & controls (frozen features, no training — independent, run anytime)

- [x] `baseline` + `baseline_endanchor` A/B  (done — old direct224 features)
- [ ] `baseline_native` — `sbatch extract_logos_features_native_baseline.job` (Fork_Logos)
      → `/scratch-shared/psobecki/ASL_Citizen/logos_features_native`
- [ ] SignCLIP control — `sbatch jobs/eval_signclip_features.job` (MMPT)
      → exports pooled_video for `signclip_scratch` + `signclip_aug_ft`, then A+B

## 1b. Preprocessing comparison (decide the input pipeline before the full experiment)

ASL-Citizen frames are 640×480 (4:3), not square. Three ways to map → 224×224:
- **direct224** — resize whole frame to 224² (keeps all content, mild horizontal stretch).
- **letterbox224** — resize long side→224 + grey-pad (keeps all, no distortion, smaller signer).
- **logos_native** — resize-300 + pad + center-crop-224 (aspect-preserving but **clips hands at the
  sides** on 4:3 — confirmed visually + worst in eval).

Eval each both frozen (original ckpt) and after a baseline FT (CE, no aug). 3 of 6 cells already done.

|              | frozen (no FT)                       | baseline FT (CE)              |
|--------------|--------------------------------------|-------------------------------|
| logos_native | `baseline_native`  ✅ (A81.4/B84.3)  | `run2`            ✅ (A85.5/B86.1) |
| direct224    | `baseline_endanchor` ✅ (A83.0/B85.9)| `run2_direct224`    ← new      |
| letterbox224 | `baseline_letterbox` ← new extract   | `run2_letterbox224` ← new      |

- [ ] `sbatch extract_logos_features_letterbox_baseline.job`  (Fork_Logos) → `logos_features_letterbox`
- [ ] `sbatch ft_asl_citizen_run2_direct224.job`              (Fork_Logos)
- [ ] `sbatch ft_asl_citizen_run2_letterbox224.job`           (Fork_Logos)
- [ ] `sbatch jobs/eval_asl_citizen_preproc.job`              (MMPT) → all 6, skips not-ready
- [ ] **Decide the winner**, then make it the pipeline for the full experiment (defaults are still
      `logos_native`; the comparison jobs set preproc explicitly so nothing else is disturbed).

## 2. Backbone fine-tuning — partial FT first (Fork_Logos)

Each job: train (→MLflow) then re-extract features with `--preproc logos_native`.

- [ ] `sbatch ft_asl_citizen_run2_baseline.job`              → CE, original           → `logos_features_run2`
- [ ] `sbatch ft_asl_citizen_run3_augdata.job`               → CE, original+aug        → `logos_features_run3`
- [ ] `sbatch ft_asl_citizen_run1_consistency.job`           → CE + consist (glasses+shirt1) → `logos_features_run1`
- [ ] `sbatch ft_asl_citizen_run1_consistency_glasses.job`   → consist (glasses)       → `logos_features_run1_glasses`
- [ ] `sbatch ft_asl_citizen_run1_consistency_shirt1.job`    → consist (shirt_1)       → `logos_features_run1_shirt1`
- [ ] `sbatch ft_asl_citizen_run1_consistency_signerswap.job`→ consist (signer_swap)   → `logos_features_run1_signerswap`
- [ ] `sbatch ft_asl_citizen_run1_consistency_skin.job`      → consist (skin_mst_diffusion) → `logos_features_run1_skin`
- [ ] (optional) run4 supcon: add `--lambda_supcon 0.1` (needs class-balanced sampling to help)

4 augmentation variants now: glasses, shirt_1, signer_swap, skin_mst_diffusion. Combined run1 + run3
use all 4 (`--aug_names`); the four `run1_*` jobs isolate each.

Note: each is an A100 job up to 36 h — check the concurrent-GPU/QOS limit; stagger if needed.

## 3. Evaluate everything (MMPT; re-runnable, skips dirs not ready)

- [ ] `sbatch jobs/eval_asl_citizen_features.job`   → headline set: baseline / baseline_endanchor /
      **baseline_native** / run2 / run3 / run1  (A+B, one MLflow run each)
- [ ] `sbatch jobs/eval_asl_citizen_ablations.job` → ablations: run1_glasses / run1_shirt1 /
      run1_signerswap / run1_skin  (A+B, one MLflow run each)
- Both use parallel `.npy` loading (`--load_workers 32`, bump to 64 if scratch-shared is slow).

## 4. Escalate to full FT (only runs that show signal in step 3)

- [ ] Edit the relevant `ft_*` job header vars → `UNFREEZE=16 LR=1e-5 LLRD=0.75 EPOCHS=15 TAG=<run>_full`,
      resubmit, then re-run step 3.

---

## Decomposition (how to read the comparison)

| Step | Isolates |
|------|----------|
| `baseline_native` → `run2`        | effect of fine-tuning the backbone at all (CE) |
| `run2` → `run3`                   | augmentations as extra training data |
| `run3` → `run1`                   | explicit appearance-consistency (beyond aug-as-data) |
| `run1_glasses` vs `run1_shirt1`   | which appearance factor the invariance buys |
| SignCLIP A vs raw Logos A         | did SignCLIP training degrade or just re-task the features |
