#!/usr/bin/env bash
# GlotLID language-ID check of the rebuilt WMT22 bitext (CPU-bound, ~451k pairs).
#
#   sbatch slurm/wmt22_lid.sh
#SBATCH --job-name=wmt22_lid
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

python3 -m hsbmt.data.wmt22 lid
python3 -m hsbmt.data.wmt22 categorize
