#!/bin/bash
# Submit the CORRECTED I3D SignCLIP augmentation experiment (WLASL-ft features)
# with the right SLURM dependencies.
#
#   [scratch, only if ckpt missing] ─┬─→ i3d_wlasl_ft        (control, no aug)
#   aug-extract (WLASL weights) ─────┼─→ aug_i3d_wlasl_ft    (aug-as-data)
#                                    └─→ asl_inv_i3d         (SupCon-on-pairs K=5)
#
# Why this replaces jobs/submit_i3d_signclip.sh (2026-07-12 debug):
#   The June batch was internally consistent but built end-to-end on BSL-5K
#   I3D features (i3d_features, bsl5k.pth.tar — and the aug extraction used
#   the script's num_classes default of 2000, which matches WLASL not
#   BSL-5K's 5383). The reported baseline row ("From scratch, RGB (I3D,
#   WLASL)" 0.32 R@1) uses the WLASL-fine-tuned I3D features
#   (i3d_wlasl_features, I3D_wlasl.tar, num_classes 2000) adopted in April
#   (commit ae3eb0ac) — so the June results were incomparably bad. This chain
#   reruns everything on the WLASL-ft features against the April baseline.
#
# Goal unchanged: does appearance augmentation help the WEAKER I3D backbone
# (more headroom than Logos, per the L2 experiments)? Headline comparisons:
#   aug_i3d_wlasl_ft vs i3d_wlasl_ft   (aug-as-data delta)
#   asl_inv_i3d      vs i3d_wlasl_ft   (contrastive-with-augments delta)
#
# Run from examples/MMPT:  bash jobs/submit_i3d_wlasl_signclip.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # → examples/MMPT

SCRATCH_CKPT=runs/retri_asl/asl_citizen_cnn_scratch_wlasl_nowd/checkpoint_best.pt

# 0. The April scratch baseline: reuse its checkpoint if still on disk,
#    otherwise resubmit it first (identical April recipe).
SCRATCH_DEP=""
if [ -f "$SCRATCH_CKPT" ]; then
    echo "scratch baseline : reusing existing $SCRATCH_CKPT"
else
    SCRATCH=$(sbatch --parsable jobs/asl_citizen_cnn_scratch_wlasl_nowd.job)
    SCRATCH_DEP="--dependency=afterok:$SCRATCH"
    echo "scratch baseline : resubmitted as $SCRATCH (checkpoint was missing)"
fi

# 1. WLASL-ft I3D features for the augmented frame dirs (aug + pair runs need
#    these). Independent of the scratch run.
AUG_EXTRACT=$(sbatch --parsable jobs/extract_i3d_wlasl_features_aug.job)
echo "aug I3D extract  : $AUG_EXTRACT"

# 2. No-aug FT control — needs only the scratch checkpoint.
FT=$(sbatch --parsable $SCRATCH_DEP jobs/asl_citizen_cnn_i3d_wlasl_ft.job)
echo "i3d_wlasl_ft     : $FT ${SCRATCH_DEP:+(after scratch)}"

# 3. +aug FT — needs scratch AND the augmented features.
if [ -n "$SCRATCH_DEP" ]; then
    AUGFT=$(sbatch --parsable --dependency=afterok:${SCRATCH}:${AUG_EXTRACT} jobs/asl_citizen_cnn_aug_i3d_wlasl_ft.job)
    INV=$(sbatch --parsable --dependency=afterok:${SCRATCH}:${AUG_EXTRACT} jobs/asl_citizen_cnn_appearance_invariant_i3d.job)
else
    AUGFT=$(sbatch --parsable --dependency=afterok:${AUG_EXTRACT} jobs/asl_citizen_cnn_aug_i3d_wlasl_ft.job)
    INV=$(sbatch --parsable --dependency=afterok:${AUG_EXTRACT} jobs/asl_citizen_cnn_appearance_invariant_i3d.job)
fi
echo "aug_i3d_wlasl_ft : $AUGFT   (after aug extract${SCRATCH_DEP:+ + scratch})"
echo "asl_inv_i3d      : $INV   (after aug extract${SCRATCH_DEP:+ + scratch})"

echo ""
echo "Submitted. Test metrics land in runs/retri_asl/asl_citizen_cnn_{i3d_wlasl_ft,aug_i3d_wlasl_ft,appearance_invariant_i3d}/eval"
echo "and in MLflow experiment 'signclip-cnn'."
echo "Headline = aug_i3d_wlasl_ft and asl_inv_i3d vs i3d_wlasl_ft (R@1/R@5),"
echo "all against the April scratch baseline (0.32 R@1 table row)."
