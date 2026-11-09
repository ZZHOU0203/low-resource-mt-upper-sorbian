# Data

This repository does not include any datasets. Download them yourself and place them as shown below. Each dataset keeps its own license.

```
data/
├── KDE4.de-hsb.de              # OPUS KDE4 German–Upper Sorbian (Moses format)
├── KDE4.de-hsb.hsb
├── KDE4.cs-hsb.{cs,hsb}        # OPUS KDE4 Czech/English/Polish–Upper Sorbian (Pipeline 2 sources),
├── KDE4.en-hsb.{en,hsb}        #   unzipped from the OPUS Moses downloads
├── KDE4.hsb-pl.{hsb,pl}
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

`bash scripts/prepare_data.sh` writes all derived data back into `data/`:

| Directory | Command | Content |
|---|---|---|
| `splits/` | `hsbmt.data.kde4 split` | Original KDE4 train (16,213) / dev (2,000) / test (2,000) |
| `cleaned/`, `splits_cleaned/` | `hsbmt.data.kde4 clean`, `split` | Rule-cleaned KDE4 (20,213 → 15,035 pairs) |
| `splits_original_test_dedup_vs_*_train/` | `hsbmt.data.leakage dedup` | Original test set without pairs seen in training |
| `wmt22_parallel/wmt22_all_rebuilt.*` | `hsbmt.data.wmt22 rebuild` | All WMT22 sources merged (451,059 pairs) |
| `wmt22_parallel/wmt22_lid_filtered.*` | `hsbmt.data.wmt22 lid`, `categorize`, `filter` | GlotLID-filtered WMT22 pairs |
| `splits_kde4_wmt22_lid_100k/` | `hsbmt.data.combine --max_extra_pairs 100000` | KDE4 train + 100k WMT22 pairs (Pipeline 1) |
| `pipeline2/multilingual_source/` | `hsbmt.data.pivot build` | cs/en/pl–hsb merged (51,473 pairs) |
| `pipeline2/mt_german_hsb/` | `hsbmt.data.pivot translate` | Machine-translated German + original Upper Sorbian |
| `splits_kde4_pivot_mt/` | `hsbmt.data.combine` | KDE4 train + pivot-MT synthetic pairs (Pipeline 2) |
