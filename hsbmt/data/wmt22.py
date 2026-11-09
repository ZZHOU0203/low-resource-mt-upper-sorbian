"""WMT22 Upper Sorbian-German bitext: rebuild, language-ID check with GlotLID, filter.

    python -m hsbmt.data.wmt22 rebuild      # merge the gzip sources -> wmt22_all_rebuilt.{de,hsb}
    python -m hsbmt.data.wmt22 lid          # GlotLID top-3 predictions for both sides (slow)
    python -m hsbmt.data.wmt22 categorize   # strict_pass / relaxed_pass / review per pair
    python -m hsbmt.data.wmt22 filter       # keep *_pass pairs -> wmt22_lid_filtered.{de,hsb}
"""

import argparse
import csv
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from hsbmt.config import DATA_DIR, OUTPUTS_DIR
from hsbmt.io import open_text, write_tsv

WMT_DIR = DATA_DIR / "wmt22_parallel"
LID_DIR = OUTPUTS_DIR / "lid_reports"
REBUILT = (WMT_DIR / "wmt22_all_rebuilt.de", WMT_DIR / "wmt22_all_rebuilt.hsb")
FILTERED = (WMT_DIR / "wmt22_lid_filtered.de", WMT_DIR / "wmt22_lid_filtered.hsb")
LID_FULL = LID_DIR / "wmt22_rebuilt_lid_full.tsv"
LID_CATEGORIZED = LID_DIR / "wmt22_rebuilt_lid_categorized.tsv"

# Line-aligned (de, hsb) gzip pairs and (hsb<TAB>de) gzip TSVs from the WMT22 release.
PARALLEL_GZ = [("train.hsb-de.de.gz", "train.hsb-de.hsb.gz"), ("train2021.hsb-de.de.gz", "train2021.hsb-de.hsb.gz")]
TSV_GZ = ["HSB-DE_train.tsv.gz", "HSB-DE_dev.tsv.gz"]

GLOTLID_REPO = "cis-lmu/glotlid"
DE_LABEL = "__label__deu_Latn"
HSB_LABEL = "__label__hsb_Latn"
# GlotLID often labels short Upper Sorbian text as a close Slavic relative.
RELATED_SLAVIC = {f"__label__{c}_Latn" for c in ("dsb", "ces", "pol", "slk", "slv", "hrv", "bos", "srp")}
KEEP = {"strict_pass", "relaxed_pass"}

LID_HEADER = ["line_no", "status", "de_text", "hsb_text"] + [
    f"{side}_pred{k}_{field}" for side in ("de", "hsb") for k in (1, 2, 3) for field in ("label", "prob")
]


def aligned(*iterables):
    """zip() that fails loudly if the inputs have different lengths."""
    for line_no, items in enumerate(zip_longest(*iterables), start=1):
        if any(item is None for item in items):
            raise ValueError(f"Inputs are not line-aligned (first mismatch at line {line_no})")
        yield line_no, items


# ---------------------------------------------------------------- rebuild
def iter_wmt22_pairs(input_dir: Path):
    """Yield non-empty (de, hsb) pairs from all WMT22 source files, in a fixed order."""
    for de_name, hsb_name in PARALLEL_GZ:
        kept = 0
        with open_text(input_dir / de_name, newline="") as de_f, open_text(input_dir / hsb_name, newline="") as hsb_f:
            for _, (de, hsb) in aligned(de_f, hsb_f):
                de, hsb = de.rstrip("\r\n"), hsb.rstrip("\r\n")
                if de.strip() and hsb.strip():
                    kept += 1
                    yield de, hsb
        print(f"{de_name}: {kept} pairs")
    for tsv_name in TSV_GZ:
        kept = 0
        with open_text(input_dir / tsv_name, newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    kept += 1
                    yield row[1].strip(), row[0].strip()  # TSV columns are hsb, de
        print(f"{tsv_name}: {kept} pairs")


def cmd_rebuild(args):
    total = 0
    args.output_de.parent.mkdir(parents=True, exist_ok=True)
    with args.output_de.open("w", encoding="utf-8", newline="\n") as de_out, args.output_hsb.open(
        "w", encoding="utf-8", newline="\n"
    ) as hsb_out:
        for de, hsb in iter_wmt22_pairs(args.input_dir):
            de_out.write(de + "\n")
            hsb_out.write(hsb + "\n")
            total += 1
    print(f"total\t{total}\nde\t{args.output_de}\nhsb\t{args.output_hsb}")


# ---------------------------------------------------------------- lid
def load_glotlid():
    import fasttext
    from huggingface_hub import hf_hub_download

    return fasttext.load_model(hf_hub_download(repo_id=GLOTLID_REPO, filename="model.bin"))


def top3(model, text: str) -> list[tuple[str, float]]:
    labels, probs = model.predict(text.replace("\n", " "), k=3)
    preds = list(zip(labels, probs))
    return (preds + [("", 0.0)] * 3)[:3]


def lid_status(de_preds, hsb_preds, min_prob: float) -> str:
    de_bad = de_preds[0][0] != DE_LABEL or de_preds[0][1] < min_prob
    hsb_bad = hsb_preds[0][0] != HSB_LABEL or hsb_preds[0][1] < min_prob
    return {(False, False): "ok", (True, False): "bad_de", (False, True): "bad_hsb", (True, True): "bad_both"}[
        (de_bad, hsb_bad)
    ]


def cmd_lid(args):
    model = load_glotlid()
    counts = Counter()
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.de.open(encoding="utf-8", newline="") as de_f, args.hsb.open(encoding="utf-8", newline="") as hsb_f, args.output_tsv.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(LID_HEADER)
        for line_no, (de, hsb) in aligned(de_f, hsb_f):
            if args.max_examples and line_no > args.max_examples:
                break
            de, hsb = de.rstrip("\r\n"), hsb.rstrip("\r\n")
            de_preds, hsb_preds = top3(model, de), top3(model, hsb)
            status = lid_status(de_preds, hsb_preds, args.min_prob)
            counts[status] += 1
            row = [line_no, status, de, hsb]
            for label, prob in de_preds + hsb_preds:
                row += [label, f"{float(prob):.6f}"]
            writer.writerow(row)
    for status in ("ok", "bad_de", "bad_hsb", "bad_both"):
        print(f"{status}\t{counts[status]}")
    print(f"output_tsv\t{args.output_tsv}")


# ---------------------------------------------------------------- categorize
def categorize(row: dict) -> str:
    """strict_pass: both top-1 labels correct. relaxed_pass: German in top-3 and
    Upper Sorbian in top-3 or top-1 is a closely related Slavic language. review: rest."""
    de_top3 = [row[f"de_pred{k}_label"] for k in (1, 2, 3)]
    hsb_top3 = [row[f"hsb_pred{k}_label"] for k in (1, 2, 3)]
    if de_top3[0] == DE_LABEL and hsb_top3[0] == HSB_LABEL:
        return "strict_pass"
    if DE_LABEL in de_top3 and (HSB_LABEL in hsb_top3 or hsb_top3[0] in RELATED_SLAVIC):
        return "relaxed_pass"
    return "review"


def cmd_categorize(args):
    categories, hsb_top1 = Counter(), Counter()
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.lid_tsv.open(encoding="utf-8", newline="") as src, args.output_tsv.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.DictWriter(dst, ["category"] + reader.fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            category = categorize(row)
            categories[category] += 1
            hsb_top1[row["hsb_pred1_label"]] += 1
            writer.writerow({"category": category, **row})
    rows = [("category", c, categories[c]) for c in ("strict_pass", "relaxed_pass", "review")]
    rows += [("category", "total", sum(categories.values()))]
    rows += [("hsb_pred1_label", label, n) for label, n in hsb_top1.most_common()]
    write_tsv(args.summary_tsv, ["section", "label", "count"], rows)
    for row in rows[:4] + rows[4:14]:
        print("\t".join(map(str, row)))
    print(f"categorized_tsv\t{args.output_tsv}\nsummary_tsv\t{args.summary_tsv}")


# ---------------------------------------------------------------- filter
def cmd_filter(args):
    kept = removed = 0
    for path in (args.output_de, args.removed_tsv):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.de.open(encoding="utf-8", newline="") as de_f, args.hsb.open(encoding="utf-8", newline="") as hsb_f, args.categorized_tsv.open(
        encoding="utf-8-sig", newline=""
    ) as cat_f, args.output_de.open("w", encoding="utf-8", newline="\n") as de_out, args.output_hsb.open(
        "w", encoding="utf-8", newline="\n"
    ) as hsb_out, args.removed_tsv.open("w", encoding="utf-8", newline="") as rm_f:
        reader = csv.DictReader(cat_f, delimiter="\t")
        rm_writer = csv.writer(rm_f, delimiter="\t", lineterminator="\n")
        rm_writer.writerow(["line_no", "de", "hsb"] + [c for c in reader.fieldnames if c.endswith("_label")])
        for line_no, (de, hsb, row) in aligned(de_f, hsb_f, reader):
            de, hsb = de.rstrip("\r\n"), hsb.rstrip("\r\n")
            if row["category"] in KEEP:
                de_out.write(de + "\n")
                hsb_out.write(hsb + "\n")
                kept += 1
            else:
                rm_writer.writerow([line_no, de, hsb] + [v for k, v in row.items() if k.endswith("_label")])
                removed += 1
    print(f"total\t{kept + removed}\nkept\t{kept}\nremoved\t{removed}")
    print(f"output\t{args.output_de} / {args.output_hsb.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rebuild", help="Merge all WMT22 hsb-de sources into one line-aligned pair of files.")
    p.add_argument("--input_dir", type=Path, default=WMT_DIR)
    p.add_argument("--output_de", type=Path, default=REBUILT[0])
    p.add_argument("--output_hsb", type=Path, default=REBUILT[1])
    p.set_defaults(func=cmd_rebuild)

    p = sub.add_parser("lid", help="Run GlotLID on both sides and write top-3 predictions per line.")
    p.add_argument("--de", type=Path, default=REBUILT[0])
    p.add_argument("--hsb", type=Path, default=REBUILT[1])
    p.add_argument("--output_tsv", type=Path, default=LID_FULL)
    p.add_argument("--min_prob", type=float, default=0.80, help="Top-1 probability needed for status 'ok'.")
    p.add_argument("--max_examples", type=int, default=None)
    p.set_defaults(func=cmd_lid)

    p = sub.add_parser("categorize", help="Label each LID row strict_pass / relaxed_pass / review.")
    p.add_argument("--lid_tsv", type=Path, default=LID_FULL)
    p.add_argument("--output_tsv", type=Path, default=LID_CATEGORIZED)
    p.add_argument("--summary_tsv", type=Path, default=LID_DIR / "wmt22_rebuilt_lid_summary.tsv")
    p.set_defaults(func=cmd_categorize)

    p = sub.add_parser("filter", help="Keep pairs whose category is strict_pass or relaxed_pass.")
    p.add_argument("--de", type=Path, default=REBUILT[0])
    p.add_argument("--hsb", type=Path, default=REBUILT[1])
    p.add_argument("--categorized_tsv", type=Path, default=LID_CATEGORIZED)
    p.add_argument("--output_de", type=Path, default=FILTERED[0])
    p.add_argument("--output_hsb", type=Path, default=FILTERED[1])
    p.add_argument("--removed_tsv", type=Path, default=LID_DIR / "wmt22_lid_filtered_removed.tsv")
    p.set_defaults(func=cmd_filter)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
