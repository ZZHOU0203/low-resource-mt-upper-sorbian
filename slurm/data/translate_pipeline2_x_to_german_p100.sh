#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=pipe2_mt_de
#SBATCH --output=logs/pipe2_mt_de_%j.out
#SBATCH --error=logs/pipe2_mt_de_%j.err

set -euo pipefail

cd "${PROJECT_ROOT:-$HOME/NLP-26SS}/src"
mkdir -p ../data/pipeline2/mt_german_hsb ../logs

hostname
date
nvidia-smi

python3 translate_pipeline2_x_to_german.py \
  --src_path ../data/pipeline2/multilingual_source/pipeline2_x.src \
  --lang_path ../data/pipeline2/multilingual_source/pipeline2_x.lang \
  --hsb_path ../data/pipeline2/multilingual_source/pipeline2_x.hsb \
  --output_de ../data/pipeline2/mt_german_hsb/pipeline2_mt_de.de \
  --output_hsb ../data/pipeline2/mt_german_hsb/pipeline2_mt_de.hsb \
  --output_tsv ../data/pipeline2/mt_german_hsb/pipeline2_mt_de.tsv \
  --model_name facebook/nllb-200-distilled-600M \
  --batch_size 16 \
  --max_length 128 \
  --overwrite
