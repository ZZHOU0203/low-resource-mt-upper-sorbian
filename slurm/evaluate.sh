#!/usr/bin/env bash
# Translate a test set in both directions, then score with BLEU/chrF/chrF++ and xCOMET.
#
#   sbatch slurm/evaluate.sh nllb      <tag>  [test_dir]   # models/nllb-600M-<dir>-<tag>
#   sbatch slurm/evaluate.sh nllb      zero-shot           # models/nllb-600M-hsb-init
#   sbatch slurm/evaluate.sh qwen      zero-shot
#   sbatch slurm/evaluate.sh qwen-lora lora      [test_dir]   # models/qwen-7b-lora-<dir>
#
# test_dir defaults to data/splits (the original 2,000-pair KDE4 test set).
# Outputs: outputs/predictions/<system>-<tag>_<direction>_<test>.{txt,scores.json,xcomet.json}
#SBATCH --job-name=evaluate
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

SYSTEM=${1:?usage: evaluate.sh <nllb|qwen|qwen-lora> <tag> [test_dir]}
TAG=${2:?usage: evaluate.sh <nllb|qwen|qwen-lora> <tag> [test_dir]}
TEST_DIR=${3:-data/splits}
TEST_NAME=$(basename "$TEST_DIR")

for direction in de-hsb hsb-de; do
  case "$SYSTEM" in
    nllb)
      if [[ "$TAG" == zero-shot ]]; then model=models/nllb-600M-hsb-init; else model="models/nllb-600M-${direction}-${TAG}"; fi
      model_args=(--model_path "$model") ;;
    qwen)      model_args=() ;;
    qwen-lora) model_args=(--adapter_path "models/qwen-7b-lora-${direction}") ;;
    *) echo "unknown system: $SYSTEM" >&2; exit 1 ;;
  esac

  out="outputs/predictions/${SYSTEM}-${TAG}_${direction}_${TEST_NAME}.txt"
  python3 -m hsbmt.evaluate.translate "$SYSTEM" --direction "$direction" \
    --split_dir "$TEST_DIR" --output "$out" ${model_args[@]+"${model_args[@]}"}
  python3 -m hsbmt.evaluate.xcomet --direction "$direction" --split_dir "$TEST_DIR" --mt "$out"
done
