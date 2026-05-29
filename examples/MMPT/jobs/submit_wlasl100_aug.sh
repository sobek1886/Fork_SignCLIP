#!/bin/bash
# Submit all WLASL100 augmentation experiments.
# Run from: $HOME/Fork_SignCLIP/examples/MMPT on Snellius.
#
# Generates the three split directories if they are absent, then submits
# training+test jobs for aug (original+glasses+shirt_1), shirt1-only, and
# glasses-only variants.

set -euo pipefail

NSLT_JSON=/home/psobecki/wlasl100/nslt_100.json
CLASS_LIST=/home/psobecki/wlasl100/wlasl_class_list.txt
SPLITS_BASE=/home/psobecki/wlasl100

source .venv/bin/activate
export PYTHONPATH=$HOME/Fork_SignCLIP:$PYTHONPATH

# ── Generate splits ────────────────────────────────────────────────────────────

if [ ! -f "${SPLITS_BASE}/splits_aug/train.csv" ]; then
    echo "Generating splits_aug ..."
    python generate_wlasl100_splits.py \
        --nslt_json "$NSLT_JSON" \
        --class_list "$CLASS_LIST" \
        --train_variants original glasses shirt_1 \
        --output_dir "${SPLITS_BASE}/splits_aug"
else
    echo "splits_aug already exists, skipping."
fi

if [ ! -f "${SPLITS_BASE}/splits_shirt1/train.csv" ]; then
    echo "Generating splits_shirt1 ..."
    python generate_wlasl100_splits.py \
        --nslt_json "$NSLT_JSON" \
        --class_list "$CLASS_LIST" \
        --train_variants shirt_1 \
        --output_dir "${SPLITS_BASE}/splits_shirt1"
else
    echo "splits_shirt1 already exists, skipping."
fi

if [ ! -f "${SPLITS_BASE}/splits_glasses/train.csv" ]; then
    echo "Generating splits_glasses ..."
    python generate_wlasl100_splits.py \
        --nslt_json "$NSLT_JSON" \
        --class_list "$CLASS_LIST" \
        --train_variants glasses \
        --output_dir "${SPLITS_BASE}/splits_glasses"
else
    echo "splits_glasses already exists, skipping."
fi

# ── Submit jobs ────────────────────────────────────────────────────────────────

echo ""
JID1=$(sbatch --parsable jobs/wlasl100_cnn_aug_logos_nowd.job)
echo "Submitted aug      job: $JID1"

JID2=$(sbatch --parsable jobs/wlasl100_cnn_shirt1_logos_nowd.job)
echo "Submitted shirt1   job: $JID2"

JID3=$(sbatch --parsable jobs/wlasl100_cnn_glasses_logos_nowd.job)
echo "Submitted glasses  job: $JID3"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs at:       jobs/output/slurm_<name>_<jobid>.txt"
echo "MLflow:        https://mlflow.ai.mytkhgroup.com/ (experiment: signclip-wlasl100)"
