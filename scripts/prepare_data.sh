#!/usr/bin/env bash
# Build every dataset used in the experiments from the raw downloads (see data/README.md).
# All steps run on CPU except the two marked [GPU/slow], which have SLURM wrappers in slurm/.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. KDE4: original split, rule-based cleaning, cleaned split
python -m hsbmt.data.kde4 split
python -m hsbmt.data.kde4 clean
python -m hsbmt.data.kde4 split --src data/cleaned/KDE4.cleaned.de --tgt data/cleaned/KDE4.cleaned.hsb \
  --output_dir data/splits_cleaned

# 2. Leakage control
python -m hsbmt.data.leakage check
python -m hsbmt.data.leakage diff
python -m hsbmt.data.leakage dedup

# 3. Pipeline 1: WMT22 + GlotLID filter, then KDE4 train + 100k WMT22 pairs
python -m hsbmt.data.wmt22 rebuild
python -m hsbmt.data.wmt22 lid            # [slow] slurm/wmt22_lid.sh
python -m hsbmt.data.wmt22 categorize
python -m hsbmt.data.wmt22 filter
python -m hsbmt.data.combine \
  --extra_de data/wmt22_parallel/wmt22_lid_filtered.de \
  --extra_hsb data/wmt22_parallel/wmt22_lid_filtered.hsb \
  --max_extra_pairs 100000 --output_dir data/splits_kde4_wmt22_lid_100k

# 4. Pipeline 2: cs/en/pl-hsb -> machine-translated German, then KDE4 train + synthetic pairs
python -m hsbmt.data.pivot build
python -m hsbmt.data.pivot translate      # [GPU] slurm/pivot_translate.sh
python -m hsbmt.data.combine \
  --extra_de data/pipeline2/mt_german_hsb/pipeline2_mt_de.de \
  --extra_hsb data/pipeline2/mt_german_hsb/pipeline2_mt_de.hsb \
  --output_dir data/splits_kde4_pivot_mt
