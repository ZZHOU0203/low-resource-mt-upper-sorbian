#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=eval_nllb_100k
#SBATCH --output=logs/eval_nllb_100k_%j.out
#SBATCH --error=logs/eval_nllb_100k_%j.err

set -euo pipefail

cd "${PROJECT_ROOT:-$HOME/NLP-26SS}/src"
mkdir -p ../logs ../outputs

hostname
nvidia-smi
date

python3 evaluate_extended_nllb_baseline.py \
  --direction de-hsb \
  --split_dir ../data/splits_kde4_wmt22_lid_100k \
  --model_path ../models/nllb-600M-de-hsb-kde4-wmt22-lid-100k \
  --output_path ../outputs/nllb_de-hsb_kde4_wmt22_lid_100k_predictions.txt \
  --num_test 2000

python3 evaluate_extended_nllb_baseline.py \
  --direction hsb-de \
  --split_dir ../data/splits_kde4_wmt22_lid_100k \
  --model_path ../models/nllb-600M-hsb-de-kde4-wmt22-lid-100k \
  --output_path ../outputs/nllb_hsb-de_kde4_wmt22_lid_100k_predictions.txt \
  --num_test 2000
