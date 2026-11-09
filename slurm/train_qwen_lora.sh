#!/usr/bin/env bash
# QLoRA fine-tuning of Qwen2.5-7B-Instruct in one direction.
#
#   sbatch slurm/train_qwen_lora.sh <direction> [split_dir]
#   sbatch slurm/train_qwen_lora.sh de-hsb data/splits
#
# Writes models/qwen-7b-lora-<direction>.
#SBATCH --job-name=qwen_lora_train
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

DIRECTION=${1:?usage: train_qwen_lora.sh <direction> [split_dir]}
SPLIT_DIR=${2:-data/splits}

python3 -m hsbmt.train.qwen_lora \
  --direction "$DIRECTION" \
  --split_dir "$SPLIT_DIR" \
  --output_dir "models/qwen-7b-lora-${DIRECTION}" \
  --num_train_epochs 3 --learning_rate 2e-4 \
  --batch_size 1 --gradient_accumulation_steps 16 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --max_length 256
