#!/usr/bin/env bash
# Pipeline 2: translate the Czech/English/Polish side of X-hsb corpora into German with NLLB.
#
#   sbatch slurm/pivot_translate.sh
#SBATCH --job-name=pivot_translate
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

python3 -m hsbmt.data.pivot translate --batch_size 16 --max_length 128
