#!/bin/bash
# Submit ASL-Citizen appearance-augmentation experiment.
# Run from: $HOME/Fork_SignCLIP/examples/MMPT on Snellius.
#
# Prerequisites (complete before running this script):
#   1. analyze_asl_citizen_for_aug.py  → aug_video_list.txt
#   2. SkyPilot augmentation jobs      → augmented_videos/{video_id}/{aug_name}.mp4
#   3. rsync augmented videos to Snellius:
#        rsync -avz output/augmented_videos/ \
#          psobecki@snellius.nl:/scratch-shared/psobecki/asl_citizen/augmented_videos/
#   4. extract_logos_features_asl_citizen_aug.job (in mmaction2/)
#   5. generate_asl_citizen_aug_splits.py → splits_aug/
#
# This script only submits steps 5 and the training + test jobs.

set -euo pipefail

SPLITS_SRC=/home/psobecki/ASL_Citizen/splits
SPLITS_AUG=/home/psobecki/ASL_Citizen/splits_aug
AUG_LIST=/home/psobecki/ASL_Citizen/aug_selection/aug_video_list.txt

source .venv/bin/activate
export PYTHONPATH=$HOME/Fork_SignCLIP:$PYTHONPATH

# ── Generate augmented splits ──────────────────────────────────────────────────

if [ ! -f "${SPLITS_AUG}/train.csv" ]; then
    echo "Generating splits_aug ..."
    python generate_asl_citizen_aug_splits.py \
        --train_csv      "${SPLITS_SRC}/train.csv" \
        --val_csv        "${SPLITS_SRC}/val.csv" \
        --test_csv       "${SPLITS_SRC}/test.csv" \
        --aug_video_list "${AUG_LIST}" \
        --variants       glasses shirt_1 \
        --output_dir     "${SPLITS_AUG}"
else
    echo "splits_aug already exists, skipping."
fi

# ── Submit training and test jobs ──────────────────────────────────────────────

echo ""
JID1=$(sbatch --parsable jobs/asl_citizen_cnn_aug_logos.job)
echo "Submitted training job: $JID1"

JID2=$(sbatch --parsable --dependency=afterok:${JID1} jobs/test_asl_citizen_cnn_aug_logos.job)
echo "Submitted test job:     $JID2 (depends on $JID1)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs at:       jobs/output/slurm_<name>_<jobid>.txt"
echo "MLflow:        signclip-asl-citizen"
