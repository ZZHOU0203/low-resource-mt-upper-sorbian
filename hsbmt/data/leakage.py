"""Train/test leakage control for the KDE4 splits.

    python -m hsbmt.data.leakage check   # exact pair / source / target overlap between splits
    python -m hsbmt.data.leakage diff    # how the original and cleaned test sets differ
    python -m hsbmt.data.leakage dedup   # original test set minus pairs seen in training

Results go to outputs/data_leakage_check and outputs/testset_diff; deduplicated
test sets go to data/splits_original_test_dedup_vs_{original,cleaned}_train.
"""

import argparse
from collections import Counter
from pathlib import Path

from hsbmt.config import DATA_DIR, OUTPUTS_DIR
from hsbmt.data.kde4 import MAX_LENGTH_RATIO
from hsbmt.io import Pair, read_split, write_split, write_tsv

ORIGINAL = DATA_DIR / "splits"
CLEANED = DATA_DIR / "splits_cleaned"
LEAKAGE_DIR = OUTPUTS_DIR / "data_leakage_check"
DIFF_DIR = OUTPUTS_DIR / "testset_diff"


def overlap(name: str, kind: str, left_label: str, left, right_label: str, right) -> dict:
    """Exact-match overlap of two collections (pairs or single sentences)."""
    left, right = set(left), set(right)
    common = sorted(left & right)
    smaller = min(len(left), len(right))
    return {
        "check_name": name,
        "check_type": kind,
        "left": left_label,
        "right": right_label,
        "left_n": len(left),
        "right_n": len(right),
        "overlap_n": len(common),
        "percent_of_smaller": 100.0 * len(common) / smaller if smaller else 0.0,
        "examples": common[:50],
    }


def leakage_checks(splits: dict[str, dict[str, list[Pair]]]) -> list[dict]:
    """All checks reported in the project, for splits = {"original": {...}, "cleaned": {...}}."""
    results = []
    for corpus, s in splits.items():
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            results.append(
                overlap(f"{corpus}_{left}_vs_{right}_pair_overlap", "pair", f"{corpus}/{left}", s[left], f"{corpus}/{right}", s[right])
            )
        for side, kind, idx in (("de", "source", 0), ("hsb", "target", 1)):
            results.append(
                overlap(
                    f"{corpus}_train_{side}_vs_test_{side}_{kind}_overlap",
                    kind,
                    f"{corpus}/train.{side}",
                    [p[idx] for p in s["train"]],
                    f"{corpus}/test.{side}",
                    [p[idx] for p in s["test"]],
                )
            )
    o, c = splits["original"], splits["cleaned"]
    for left_name, left, right_name, right in (
        ("original_train", o["train"], "cleaned_test", c["test"]),
        ("cleaned_train", c["train"], "original_test", o["test"]),
        ("original_test", o["test"], "cleaned_test", c["test"]),
        ("original_train", o["train"], "cleaned_train", c["train"]),
    ):
        results.append(
            overlap(f"{left_name}_vs_{right_name}_pair_overlap", "pair", left_name.replace("_", "/"), left, right_name.replace("_", "/"), right)
        )
    return results


def cmd_check(args):
    splits = {
        "original": {s: read_split(args.original, s, strip=True) for s in ("train", "dev", "test")},
        "cleaned": {s: read_split(args.cleaned, s, strip=True) for s in ("train", "dev", "test")},
    }
    results = leakage_checks(splits)
    out = args.output_dir
    for r in results:
        if r["overlap_n"]:
            header = {"pair": ["de", "hsb"], "source": ["de"], "target": ["hsb"]}[r["check_type"]]
            rows = r["examples"] if r["check_type"] == "pair" else [(x,) for x in r["examples"]]
            write_tsv(out / f"examples_{r['check_name']}.tsv", header, rows)
    columns = ["check_name", "check_type", "left", "right", "left_n", "right_n", "overlap_n"]
    write_tsv(
        out / "leakage_summary.tsv",
        columns + ["percent_of_smaller"],
        [[r[c] for c in columns] + [f"{r['percent_of_smaller']:.4f}"] for r in results],
    )
    lines = [f"{r['check_name']}\t{r['overlap_n']}\t{r['percent_of_smaller']:.2f}%" for r in results]
    (out / "leakage_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("check\toverlap\tpercent_of_smaller")
    print("\n".join(lines))


def likely_reason(de: str, hsb: str) -> str:
    """Best guess why an original test pair is missing from the cleaned test set."""
    if de.strip() == hsb.strip():
        return "identical"
    de_n, hsb_n = len(de.split()), len(hsb.split())
    if de_n > 200 or hsb_n > 200:
        return "too_long"
    if de_n == 0 or hsb_n == 0:
        return "length_ratio" if de_n != hsb_n else "other_or_resplit"
    if max(de_n / hsb_n, hsb_n / de_n) > MAX_LENGTH_RATIO:
        return "length_ratio"
    return "other_or_resplit"


def multiset_in_order(pairs: list[Pair], counts: Counter) -> list[Pair]:
    """Elements of `pairs` (in order) that are still available in the multiset `counts`."""
    remaining, out = Counter(counts), []
    for pair in pairs:
        if remaining[pair] > 0:
            remaining[pair] -= 1
            out.append(pair)
    return out


def cmd_diff(args):
    original = read_split(args.original, "test")
    cleaned = read_split(args.cleaned, "test")
    o_counts, c_counts = Counter(original), Counter(cleaned)
    common = multiset_in_order(original, o_counts & c_counts)
    original_only = [(de, hsb, likely_reason(de, hsb)) for de, hsb in multiset_in_order(original, o_counts - c_counts)]
    cleaned_only = multiset_in_order(cleaned, c_counts - o_counts)

    out = args.output_dir
    write_tsv(out / "common_test_pairs.tsv", ["de", "hsb"], common)
    write_tsv(out / "original_test_only.tsv", ["de", "hsb", "likely_reason"], original_only)
    write_tsv(out / "cleaned_test_only.tsv", ["de", "hsb"], cleaned_only)
    for reason in ("identical", "too_long", "length_ratio", "other_or_resplit"):
        rows = [row for row in original_only if row[2] == reason][:30]
        write_tsv(out / f"examples_{reason}.tsv", ["de", "hsb", "likely_reason"], rows)

    summary = [
        f"original_test_pairs\t{len(original)}",
        f"cleaned_test_pairs\t{len(cleaned)}",
        f"exact_common_pairs\t{len(common)}",
        f"only_in_original\t{len(original_only)}",
        f"only_in_cleaned\t{len(cleaned_only)}",
        f"overlap_percent_of_original\t{100.0 * len(common) / max(len(original), 1):.2f}",
        "# The cleaned corpus is re-split after cleaning, so 'only_in_original' pairs were not",
        "# necessarily removed by cleaning; some just moved to another split.",
    ]
    (out / "testset_diff_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


def remove_seen(test: list[Pair], train: list[Pair]) -> tuple[list[Pair], list[Pair]]:
    train_set = set(train)
    kept = [p for p in test if p not in train_set]
    removed = [p for p in test if p in train_set]
    return kept, removed


def cmd_dedup(args):
    test = read_split(args.original, "test", strip=True)
    for name, split_dir in (("original", args.original), ("cleaned", args.cleaned)):
        kept, removed = remove_seen(test, read_split(split_dir, "train", strip=True))
        out_dir = DATA_DIR / f"splits_original_test_dedup_vs_{name}_train"
        write_split(out_dir, "test", kept)
        write_tsv(args.output_dir / f"dedup_removed_vs_{name}_train.tsv", ["de", "hsb"], removed)
        print(f"vs {name} train: removed {len(removed)}, kept {len(kept)} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, out, help_text in (
        ("check", cmd_check, LEAKAGE_DIR, "Overlap between train/dev/test of the original and cleaned splits."),
        ("diff", cmd_diff, DIFF_DIR, "Compare the original and the cleaned test sets."),
        ("dedup", cmd_dedup, LEAKAGE_DIR, "Build leakage-free versions of the original test set."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--original", type=Path, default=ORIGINAL)
        p.add_argument("--cleaned", type=Path, default=CLEANED)
        p.add_argument("--output_dir", type=Path, default=out)
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
