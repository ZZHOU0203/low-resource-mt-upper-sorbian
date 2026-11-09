"""Pipeline 2: synthetic German-Upper Sorbian data via pivot translation.

Upper Sorbian has more parallel data with Czech, English and Polish than with German.
We translate the non-German side into German with NLLB and pair the machine-translated
German with the original (human) Upper Sorbian side.

    python -m hsbmt.data.pivot inventory  --input_dir DIR   # list files that look like X-hsb corpora
    python -m hsbmt.data.pivot candidates --input_dir DIR   # count usable X-hsb pairs per corpus
    python -m hsbmt.data.pivot build      # merge cs/en/pl-hsb into one source file + language file
    python -m hsbmt.data.pivot translate  # NLLB X -> German (GPU)
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from hsbmt.config import DATA_DIR, NLLB_BASE_MODEL, NLLB_LANG, OUTPUTS_DIR
from hsbmt.io import open_text, read_lines, write_lines, write_tsv

PIVOT_DIR = DATA_DIR / "pipeline2"
SOURCE_DIR = PIVOT_DIR / "multilingual_source"
MT_DIR = PIVOT_DIR / "mt_german_hsb"
PIVOT_LANGS = ["cs", "en", "pl"]

LANG_ALIASES = {
    "hsb": "hsb", "dsb": "dsb", "en": "en", "eng": "en", "cs": "cs", "ces": "cs", "cz": "cs",
    "pl": "pl", "pol": "pl", "sk": "sk", "slk": "sk", "de": "de", "deu": "de", "ger": "de",
}  # fmt: skip
NON_DE = {"dsb", "en", "cs", "pl", "sk"}
TEXT_SUFFIXES = {".gz", ".xz", ".zip", ".tsv", ".txt"} | {f".{alias}" for alias in LANG_ALIASES}
LANG_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(sorted(LANG_ALIASES, key=len, reverse=True)) + r")(?![A-Za-z0-9])", re.IGNORECASE
)


# ---------------------------------------------------------------- discovery
def detect_languages(filename: str) -> list[str]:
    """Language codes mentioned in a file name, e.g. KDE4.cs-hsb.cs -> ['cs', 'hsb']."""
    langs = {LANG_ALIASES[m.group(1).lower()] for m in LANG_PATTERN.finditer(filename)}
    langs |= {LANG_ALIASES[s.lstrip(".").lower()] for s in Path(filename).suffixes if s.lstrip(".").lower() in LANG_ALIASES}
    return sorted(langs)


def group_key(path: Path) -> str:
    """Corpus name with language codes and extensions removed, so that the files of
    one parallel corpus (KDE4.cs-hsb.cs, KDE4.cs-hsb.hsb) share a key."""
    name = path.name.lower()
    while Path(name).suffix.lower() in TEXT_SUFFIXES:
        name = Path(name).stem
    parts = [p for p in re.split(r"([._\-\s]+)", name) if p.lower() not in LANG_ALIASES]
    key = re.sub(r"[._\-\s]+", "_", "".join(parts).strip("._- "))
    return key or path.stem.lower()


def scan(input_dir: Path) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for path in sorted(input_dir.rglob("*")):
        if path.is_file():
            langs = detect_languages(path.name)
            if langs:
                groups[group_key(path)].append({"path": path, "languages": langs})
    return groups


def cmd_inventory(args):
    groups = scan(args.input_dir)
    rows = [
        [f["path"], f["path"].name, f["path"].stat().st_size, ",".join(f["languages"]), key, len(files) > 1]
        for key, files in sorted(groups.items())
        for f in files
    ]
    write_tsv(args.output_tsv, ["file_path", "filename", "size_bytes", "languages", "group_key", "in_parallel_group"], rows)
    combos = Counter(row[3] for row in rows)
    for combo, n in sorted(combos.items()):
        print(f"{combo}\t{n}")
    print(f"files\t{len(rows)}\noutput_tsv\t{args.output_tsv}")


def count_lines_any(path: Path) -> int:
    with open_text(path, errors="replace", newline="") as handle:
        return sum(1 for _ in handle)


def cmd_candidates(args):
    rows = []
    for key, files in sorted(scan(args.input_dir).items()):
        text_files = [f for f in files if Path(f["path"]).suffix.lower() not in {".zip", ".xz", ".xml"}]
        hsb_files = sorted((f for f in text_files if "hsb" in f["languages"]), key=lambda f: len(f["languages"]))
        for lang in sorted(NON_DE):
            src_files = sorted(
                (f for f in text_files if lang in f["languages"] and "de" not in f["languages"]),
                key=lambda f: len(f["languages"]),
            )
            if not hsb_files or not src_files or hsb_files[0]["path"] == src_files[0]["path"]:
                continue
            n_hsb, n_src = count_lines_any(hsb_files[0]["path"]), count_lines_any(src_files[0]["path"])
            rows.append([key, f"{lang}-hsb", hsb_files[0]["path"], src_files[0]["path"], n_hsb, n_src, min(n_hsb, n_src)])
    rows.sort(key=lambda r: -r[-1])
    write_tsv(args.output_tsv, ["group_key", "lang_pair", "hsb_file", "src_file", "hsb_lines", "src_lines", "min_pairs"], rows)
    for row in rows:
        print(f"{row[0]}\t{row[1]}\t{row[-1]}")
    print(f"output_tsv\t{args.output_tsv}")


# ---------------------------------------------------------------- build
def read_x_hsb(src_path: Path, hsb_path: Path) -> list[tuple[str, str]]:
    """Non-empty aligned (x, hsb) pairs; fail if the two files differ in length."""
    with open_text(src_path, encoding="utf-8-sig", errors="replace", newline="") as s, open_text(
        hsb_path, encoding="utf-8-sig", errors="replace", newline=""
    ) as h:
        lines = list(zip_longest(s, h))
    if any(a is None or b is None for a, b in lines):
        raise SystemExit(f"Line count mismatch: {src_path} vs {hsb_path}")
    pairs = [(a.rstrip("\r\n"), b.rstrip("\r\n")) for a, b in lines]
    return [(a, b) for a, b in pairs if a.strip() and b.strip()]


def merge_sources(corpora: dict[str, list[tuple[str, str]]], max_pairs: int = -1) -> list[tuple[str, str, str]]:
    """Concatenate per-language corpora as (nllb_lang, x, hsb) rows, dropping exact duplicates."""
    seen, rows = set(), []
    for lang, pairs in corpora.items():
        for x, hsb in pairs:
            if (lang, x, hsb) not in seen:
                seen.add((lang, x, hsb))
                rows.append((NLLB_LANG[lang], x, hsb))
    return rows[:max_pairs] if max_pairs > 0 else rows


def cmd_build(args):
    corpora = {}
    for lang in PIVOT_LANGS:
        pair = f"{lang}-hsb" if lang < "hsb" else f"hsb-{lang}"
        corpora[lang] = read_x_hsb(args.input_dir / f"KDE4.{pair}.{lang}", args.input_dir / f"KDE4.{pair}.hsb")
        print(f"{lang}\t{len(corpora[lang])} non-empty pairs")
    rows = merge_sources(corpora, args.max_pairs)
    out = args.output_dir
    write_lines(out / "pipeline2_x.lang", (r[0] for r in rows))
    write_lines(out / "pipeline2_x.src", (r[1] for r in rows))
    write_lines(out / "pipeline2_x.hsb", (r[2] for r in rows))
    for lang, n in Counter(r[0] for r in rows).items():
        print(f"{lang}\t{n}")
    print(f"total\t{len(rows)}\noutput_dir\t{out}")


# ---------------------------------------------------------------- translate
def cmd_translate(args):
    from hsbmt.models import nllb

    src, langs, hsb = (read_lines(args.input_dir / f"pipeline2_x.{ext}", encoding="utf-8-sig") for ext in ("src", "lang", "hsb"))
    if not len(src) == len(langs) == len(hsb):
        raise SystemExit("pipeline2_x.{src,lang,hsb} have different lengths")
    rows = [(l.strip(), s, h) for s, l, h in zip(src, langs, hsb) if s.strip() and l.strip() and h.strip()]
    if args.num_examples > 0:
        rows = rows[: args.num_examples]

    tokenizer, model = nllb.load(args.model_name)
    german = [""] * len(rows)
    by_lang = defaultdict(list)
    for i, (lang, _, _) in enumerate(rows):
        by_lang[lang].append(i)
    for lang, idx in sorted(by_lang.items()):
        outputs = nllb.translate(
            [rows[i][1] for i in idx], tokenizer, model, lang, NLLB_LANG["de"],
            batch_size=args.batch_size, num_beams=None, max_new_tokens=args.max_length,
            truncate_to=args.max_length, progress=True,
        )  # fmt: skip
        for i, text in zip(idx, outputs):
            german[i] = text

    out = args.output_dir
    write_lines(out / "pipeline2_mt_de.de", german)
    write_lines(out / "pipeline2_mt_de.hsb", (r[2] for r in rows))
    write_tsv(out / "pipeline2_mt_de.tsv", ["src_lang", "src_sentence", "mt_german", "hsb_sentence"],
              [(l, s, d, h) for (l, s, h), d in zip(rows, german)])  # fmt: skip
    print(f"translated\t{len(rows)}\noutput_dir\t{out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, default_out in (
        ("inventory", cmd_inventory, OUTPUTS_DIR / "pipeline2" / "hsb_source_inventory.tsv"),
        ("candidates", cmd_candidates, OUTPUTS_DIR / "pipeline2" / "hsb_non_de_candidates.tsv"),
    ):
        p = sub.add_parser(name, help=f"{name} of X-hsb corpora under --input_dir")
        p.add_argument("--input_dir", type=Path, default=DATA_DIR)
        p.add_argument("--output_tsv", type=Path, default=default_out)
        p.set_defaults(func=func)

    p = sub.add_parser("build", help="Merge KDE4 cs/en/pl-hsb into pipeline2_x.{src,lang,hsb}.")
    p.add_argument("--input_dir", type=Path, default=DATA_DIR, help="Directory with the unzipped KDE4.X-hsb.* files")
    p.add_argument("--output_dir", type=Path, default=SOURCE_DIR)
    p.add_argument("--max_pairs", type=int, default=-1)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("translate", help="Translate pipeline2_x.src into German with NLLB.")
    p.add_argument("--input_dir", type=Path, default=SOURCE_DIR)
    p.add_argument("--output_dir", type=Path, default=MT_DIR)
    p.add_argument("--model_name", default=NLLB_BASE_MODEL)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--num_examples", type=int, default=-1)
    p.set_defaults(func=cmd_translate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
