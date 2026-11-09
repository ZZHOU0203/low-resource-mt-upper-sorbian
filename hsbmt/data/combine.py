"""Build a training split = KDE4 train + extra parallel data, keeping KDE4 dev/test unchanged.

Extra pairs are de-duplicated and every pair that also occurs anywhere in KDE4
(train, dev or test) is dropped, so the augmented data cannot leak into evaluation.

    # Pipeline 1: KDE4 + first 100k GlotLID-filtered WMT22 pairs
    python -m hsbmt.data.combine \\
        --extra_de data/wmt22_parallel/wmt22_lid_filtered.de \\
        --extra_hsb data/wmt22_parallel/wmt22_lid_filtered.hsb \\
        --max_extra_pairs 100000 --output_dir data/splits_kde4_wmt22_lid_100k

    # Pipeline 2: KDE4 + pivot-MT synthetic pairs
    python -m hsbmt.data.combine \\
        --extra_de data/pipeline2/mt_german_hsb/pipeline2_mt_de.de \\
        --extra_hsb data/pipeline2/mt_german_hsb/pipeline2_mt_de.hsb \\
        --output_dir data/splits_kde4_pipeline2_mt_de
"""

import argparse
import shutil
from pathlib import Path

from hsbmt.config import DATA_DIR
from hsbmt.io import Pair, read_parallel, read_split, write_report, write_split


def select_extra(extra: list[Pair], exclude: set[Pair], max_pairs: int | None = None):
    """De-duplicate `extra`, drop pairs in `exclude`, and keep at most `max_pairs` (in order)."""
    seen, selected = set(), []
    duplicates = overlaps = 0
    for pair in extra:
        if pair in seen:
            duplicates += 1
            continue
        seen.add(pair)
        if pair in exclude:
            overlaps += 1
            continue
        selected.append(pair)
        if max_pairs and len(selected) == max_pairs:
            break
    if max_pairs and len(selected) < max_pairs:
        raise ValueError(f"Only {len(selected)} usable extra pairs, fewer than --max_extra_pairs={max_pairs}")
    return selected, duplicates, overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kde4_split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--extra_de", type=Path, required=True)
    parser.add_argument("--extra_hsb", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_extra_pairs", type=int, default=None, help="Use only the first N usable extra pairs.")
    args = parser.parse_args()

    kde4 = {s: read_split(args.kde4_split_dir, s) for s in ("train", "dev", "test")}
    extra = read_parallel(args.extra_de, args.extra_hsb, encoding="utf-8-sig")
    exclude = set(kde4["train"]) | set(kde4["dev"]) | set(kde4["test"])
    selected, duplicates, overlaps = select_extra(extra, exclude, args.max_extra_pairs)

    write_split(args.output_dir, "train", kde4["train"] + selected)
    write_split(args.output_dir / "extra_only", "train", selected)
    for split in ("dev", "test"):
        for lang in ("de", "hsb"):
            shutil.copyfile(args.kde4_split_dir / f"{split}.{lang}", args.output_dir / f"{split}.{lang}")

    write_report(
        args.output_dir / "report.txt",
        [
            ("kde4_train", len(kde4["train"])),
            ("kde4_dev", len(kde4["dev"])),
            ("kde4_test", len(kde4["test"])),
            ("extra_input", len(extra)),
            ("extra_duplicates_removed", duplicates),
            ("extra_overlapping_kde4_removed", overlaps),
            ("extra_selected", len(selected)),
            ("final_train", len(kde4["train"]) + len(selected)),
        ],
    )


if __name__ == "__main__":
    main()
