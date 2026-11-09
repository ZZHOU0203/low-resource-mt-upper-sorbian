#!/usr/bin/env bash
# Fine-tune the extended NLLB-200 600M model in both directions on one training split.
#
#   sbatch slurm/train_nllb.sh <split_dir> <tag>
#   sbatch slurm/train_nllb.sh data/splits_cleaned cleaned
#
# Writes models/nllb-600M-{de-hsb,hsb-de}-<tag>.
#SBATCH --job-name=nllb_train
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

SPLIT_DIR=${1:?usage: train_nllb.sh <split_dir> <tag>}
TAG=${2:?usage: train_nllb.sh <split_dir> <tag>}

for direction in de-hsb hsb-de; do
  python3 -m hsbmt.train.nllb \
    --direction "$direction" \
    --split_dir "$SPLIT_DIR" \
    --output_dir "models/nllb-600M-${direction}-${TAG}" \
    --num_train_epochs 5 --learning_rate 5e-5 \
    --batch_size 4 --gradient_accumulation_steps 4 --max_length 128
done
