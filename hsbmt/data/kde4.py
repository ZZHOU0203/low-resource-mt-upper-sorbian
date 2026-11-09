"""KDE4 German-Upper Sorbian corpus: rule-based cleaning and deterministic splits.

    python -m hsbmt.data.kde4 split                       # data/splits (original)
    python -m hsbmt.data.kde4 clean                       # data/cleaned
    python -m hsbmt.data.kde4 split --src data/cleaned/KDE4.cleaned.de \\
        --tgt data/cleaned/KDE4.cleaned.hsb --output_dir data/splits_cleaned
"""

import argparse
from collections import Counter
from pathlib import Path

from hsbmt.config import DATA_DIR
from hsbmt.io import Pair, read_parallel, write_report, write_split, write_tsv, write_parallel

DEV_SIZE = 2000
TEST_SIZE = 2000
MAX_TOKENS = 128
MAX_LENGTH_RATIO = 3.0
REASONS = ["empty", "duplicate", "too_long", "length_ratio", "identical"]


def removal_reason(de: str, hsb: str, seen: set[Pair]) -> str | None:
    """Return why a (stripped) pair should be dropped, or None to keep it."""
    if not de or not hsb:
        return "empty"
    if (de, hsb) in seen:
        return "duplicate"
    de_tokens, hsb_tokens = len(de.split()), len(hsb.split())
    if de_tokens > MAX_TOKENS or hsb_tokens > MAX_TOKENS:
        return "too_long"
    if max(de_tokens / hsb_tokens, hsb_tokens / de_tokens) > MAX_LENGTH_RATIO:
        return "length_ratio"
    if de == hsb and len(de) > 3:  # untranslated UI string copied to the target side
        return "identical"
    return None


def clean_pairs(pairs: list[Pair]):
    """Apply the cleaning rules. Returns kept pairs, removed rows and per-reason counts."""
    kept, removed, seen = [], [], set()
    counts = Counter({reason: 0 for reason in REASONS})
    for index, (raw_de, raw_hsb) in enumerate(pairs):
        de, hsb = raw_de.strip(), raw_hsb.strip()
        reason = removal_reason(de, hsb, seen)
        if reason:
            counts[reason] += 1
            removed.append((index, reason, de, hsb))
            continue
        seen.add((de, hsb))
        kept.append((de, hsb))
    return kept, removed, counts


def split_pairs(pairs: list[Pair], dev_size: int = DEV_SIZE, test_size: int = TEST_SIZE) -> dict[str, list[Pair]]:
    """Use the last `test_size` pairs as test, the `dev_size` before them as dev, the rest as train."""
    needed = dev_size + test_size
    if len(pairs) < needed:
        raise ValueError(f"Corpus has {len(pairs)} pairs, but dev+test requires {needed}.")
    return {"train": pairs[:-needed], "dev": pairs[-needed:-test_size], "test": pairs[-test_size:]}


def cmd_clean(args):
    pairs = read_parallel(args.src, args.tgt)
    kept, removed, counts = clean_pairs(pairs)
    out = args.output_dir
    write_parallel(out / "KDE4.cleaned.de", out / "KDE4.cleaned.hsb", kept)
    write_tsv(out / "removed_pairs.tsv", ["index", "reason", "de", "hsb"], removed)
    items = [("total input pairs", len(pairs)), ("kept pairs", len(kept)), ("removed pairs", len(removed))]
    items += [(f"removed {reason}", counts[reason]) for reason in REASONS]
    items.append(("percentage kept", f"{100.0 * len(kept) / max(len(pairs), 1):.2f}%"))
    write_report(out / "cleaning_report.txt", items)


def cmd_split(args):
    pairs = read_parallel(args.src, args.tgt)
    print(f"total\t{len(pairs)}")
    for name, split in split_pairs(pairs, args.dev_size, args.test_size).items():
        write_split(args.output_dir, name, split)
        print(f"{name}\t{len(split)}")
    print(f"output_dir\t{args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    clean = sub.add_parser("clean", help="Remove empty, duplicate, over-long, misaligned and untranslated pairs.")
    clean.add_argument("--output_dir", type=Path, default=DATA_DIR / "cleaned")
    clean.set_defaults(func=cmd_clean)

    split = sub.add_parser("split", help="Deterministic train/dev/test split (last 4,000 pairs held out).")
    split.add_argument("--output_dir", type=Path, default=DATA_DIR / "splits")
    split.add_argument("--dev_size", type=int, default=DEV_SIZE)
    split.add_argument("--test_size", type=int, default=TEST_SIZE)
    split.set_defaults(func=cmd_split)

    for p in (clean, split):
        p.add_argument("--src", type=Path, default=DATA_DIR / "KDE4.de-hsb.de", help="German side")
        p.add_argument("--tgt", type=Path, default=DATA_DIR / "KDE4.de-hsb.hsb", help="Upper Sorbian side")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
