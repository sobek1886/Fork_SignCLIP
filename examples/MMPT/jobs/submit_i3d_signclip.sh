#!/bin/bash
# Submit the full I3D SignCLIP +augmentation experiment with the right SLURM dependencies.
#
#   aug-extract ─┐
#   scratch ─────┼─→ i3d_ft   (control, no aug)
#                └─→ aug_i3d_ft (needs scratch + aug-extract)
#
# Goal: does appearance augmentation help the WEAKER I3D backbone (more headroom than
# Logos, per the user's L2 experiments)?  Compare test_asl_citizen_cnn_i3d_ft (control)
# vs test_asl_citizen_cnn_aug_i3d_ft on the original ASL-Citizen test set.
#
# Run from examples/MMPT:  bash jobs/submit_i3d_signclip.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # → examples/MMPT

# 1. I3D features for the augmented frame dirs (only aug_i3d_ft needs these). Runs in
#    parallel with the scratch baseline — neither depends on the other.
AUG_EXTRACT=$(sbatch --parsable jobs/extract_i3d_features_aug.job)
echo "aug I3D extract  : $AUG_EXTRACT"

# 2. Scratch I3D SignCLIP baseline (original i3d_features, already present).
SCRATCH=$(sbatch --parsable jobs/asl_citizen_cnn_scratch_i3d.job)
echo "scratch I3D      : $SCRATCH"

# 3. No-aug FT control — resumes the scratch baseline.
FT=$(sbatch --parsable --dependency=afterok:$SCRATCH jobs/asl_citizen_cnn_i3d_ft.job)
echo "i3d_ft (control) : $FT   (after $SCRATCH)"

# 4. +aug FT — resumes scratch AND needs the augmented features.
AUGFT=$(sbatch --parsable --dependency=afterok:${SCRATCH}:${AUG_EXTRACT} jobs/asl_citizen_cnn_aug_i3d_ft.job)
echo "aug_i3d_ft       : $AUGFT   (after $SCRATCH + $AUG_EXTRACT)"

echo ""
echo "Submitted. Test metrics land in runs/retri_asl/asl_citizen_cnn_{i3d_ft,aug_i3d_ft}/eval"
echo "and in MLflow experiment 'signclip-cnn'. Headline = aug_i3d_ft vs i3d_ft (R@1/R@5)."
