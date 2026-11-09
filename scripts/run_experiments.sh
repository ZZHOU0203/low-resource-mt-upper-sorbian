#!/usr/bin/env bash
# Submit every training and evaluation job behind the results table in README.md.
# Run scripts/prepare_data.sh and `python -m hsbmt.models.nllb extend` first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# Zero-shot baselines
sbatch slurm/evaluate.sh nllb zero-shot
sbatch slurm/evaluate.sh qwen zero-shot

# NLLB fine-tuning: original KDE4, cleaned KDE4, Pipeline 1, Pipeline 2
for run in "data/splits original" "data/splits_cleaned cleaned" \
           "data/splits_kde4_wmt22_lid_100k wmt22-100k" "data/splits_kde4_pivot_mt pivot-mt"; do
  set -- $run
  train=$(sbatch --parsable slurm/train_nllb.sh "$1" "$2")
  sbatch --dependency=afterok:"$train" slurm/evaluate.sh nllb "$2"
  # Leakage-controlled re-evaluation: original test minus pairs seen in this training set
  case "$2" in
    original) sbatch --dependency=afterok:"$train" slurm/evaluate.sh nllb original \
                data/splits_original_test_dedup_vs_original_train ;;
    cleaned)  sbatch --dependency=afterok:"$train" slurm/evaluate.sh nllb cleaned \
                data/splits_original_test_dedup_vs_cleaned_train ;;
  esac
done

# Qwen2.5-7B + LoRA on original KDE4
de_hsb=$(sbatch --parsable slurm/train_qwen_lora.sh de-hsb)
hsb_de=$(sbatch --parsable slurm/train_qwen_lora.sh hsb-de)
sbatch --dependency=afterok:"$de_hsb":"$hsb_de" slurm/evaluate.sh qwen-lora lora
sbatch --dependency=afterok:"$de_hsb":"$hsb_de" \
  slurm/evaluate.sh qwen-lora lora data/splits_original_test_dedup_vs_original_train
