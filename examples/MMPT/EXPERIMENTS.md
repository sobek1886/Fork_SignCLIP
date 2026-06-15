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

## 2. Backbone fine-tuning — partial FT first (Fork_Logos)

Each job: train (→MLflow) then re-extract features with `--preproc logos_native`.

- [ ] `sbatch ft_asl_citizen_run2_baseline.job`              → CE, original           → `logos_features_run2`
- [ ] `sbatch ft_asl_citizen_run3_augdata.job`               → CE, original+aug        → `logos_features_run3`
- [ ] `sbatch ft_asl_citizen_run1_consistency.job`           → CE + consist (glasses+shirt1) → `logos_features_run1`
- [ ] `sbatch ft_asl_citizen_run1_consistency_glasses.job`   → CE + consist (glasses)  → `logos_features_run1_glasses`
- [ ] `sbatch ft_asl_citizen_run1_consistency_shirt1.job`    → CE + consist (shirt1)   → `logos_features_run1_shirt1`
- [ ] (later) skin tone: sync `skin_mst_diffusion` frames, add to AUG_NAMES (combined) + a `run1_skin` job
- [ ] (optional) run4 supcon: add `--lambda_supcon 0.1` (needs class-balanced sampling to help)

Note: each is an A100 job up to 36 h — check the concurrent-GPU/QOS limit; stagger if needed.

## 3. Evaluate everything (MMPT; re-runnable, skips dirs not ready)

- [ ] `sbatch jobs/eval_asl_citizen_features.job`
      → A+B (one MLflow run each) for: baseline / baseline_endanchor / **baseline_native** /
        run2 / run3 / run1 / run1_glasses / run1_shirt1

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
