# Data

This repository does not include any datasets. Download them yourself and place them as shown below. Each dataset keeps its own license.

```
data/
├── KDE4.de-hsb.de              # OPUS KDE4 German–Upper Sorbian (Moses format)
├── KDE4.de-hsb.hsb
├── cs-hsb.txt.zip              # OPUS KDE4 Czech/English/Polish–Upper Sorbian (Pipeline 2 sources)
├── en-hsb.txt.zip
├── hsb-pl.txt.zip
└── wmt22_parallel/             # WMT22 Upper Sorbian–German parallel data
    ├── train.hsb-de.{de,hsb}.gz
    ├── train2021.hsb-de.{de,hsb}.gz
    └── HSB-DE_{train,dev}.tsv.gz
```

## Sources

| Data | Where to get it |
|---|---|
| KDE4 de–hsb, cs–hsb, en–hsb, hsb–pl | [OPUS KDE4](https://opus.nlpl.eu/KDE4/corpus/version/KDE4). Download the "Moses" format for each language pair |
| WMT22 hsb–de parallel data | [WMT22 Unsupervised and Very Low Resource MT](https://www2.statmt.org/wmt22/unsup_and_very_low_res.html). The files are mirrored at [mariondimarco/WMT22_UnsupVeryLowResMT_Data](https://github.com/mariondimarco/WMT22_UnsupVeryLowResMT_Data) |

## Derived data

The scripts in `src/` write all derived data back into `data/`:

| Directory | Script | Content |
|---|---|---|
| `splits/` | `split_data.py` | Original KDE4 train / dev (2,000) / test (2,000) |
| `cleaned/`, `splits_cleaned/` | `clean_parallel_data.py`, `split_cleaned_data.py` | Rule-cleaned KDE4 (20,213 → 15,035 pairs) |
| `splits_original_test_dedup_vs_*` | `create_dedup_original_tests.py` | Leakage-controlled test sets |
| `wmt22_parallel/wmt22_lid_filtered.*` | `rebuild_wmt22_parallel.py`, `check_wmt22_lid_quality.py`, `filter_wmt22_by_lid_category.py` | GlotLID-filtered WMT22 pairs |
| `splits_kde4_wmt22_lid_100k/` | `create_combined_train_100k_wmt22.py` | KDE4 train + 100k WMT22 pairs (Pipeline 1) |
| `splits_kde4_pipeline2_mt_de_51k/` | `create_pipeline2_mt_de_splits.py` | KDE4 train + pivot-MT synthetic pairs (Pipeline 2) |
