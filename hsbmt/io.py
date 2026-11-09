"""Small helpers for line-aligned parallel text files."""

import csv
import gzip
from pathlib import Path
from typing import Iterable, Sequence

Pair = tuple[str, str]


def open_text(path: Path, mode: str = "rt", encoding: str = "utf-8", **kwargs):
    """Open a plain or .gz text file."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=encoding, **kwargs)
    return path.open(mode.replace("t", ""), encoding=encoding, **kwargs)


def read_lines(path: Path, strip: bool = False, encoding: str = "utf-8") -> list[str]:
    """Read a file into a list of lines without line breaks.

    With strip=True surrounding whitespace is removed as well.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open_text(path, encoding=encoding) as handle:
        if strip:
            return [line.strip() for line in handle]
        return [line.rstrip("\r\n") for line in handle]


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def count_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def read_parallel(src_path: Path, tgt_path: Path, strip: bool = False, encoding: str = "utf-8") -> list[Pair]:
    """Read two line-aligned files into a list of pairs; fail on length mismatch."""
    src = read_lines(src_path, strip=strip, encoding=encoding)
    tgt = read_lines(tgt_path, strip=strip, encoding=encoding)
    if len(src) != len(tgt):
        raise ValueError(f"Line count mismatch: {src_path} has {len(src)}, {tgt_path} has {len(tgt)}")
    return list(zip(src, tgt))


def write_parallel(src_path: Path, tgt_path: Path, pairs: Iterable[Pair]) -> None:
    pairs = list(pairs)
    write_lines(src_path, (src for src, _ in pairs))
    write_lines(tgt_path, (tgt for _, tgt in pairs))


def read_split(split_dir: Path, split: str, strip: bool = False) -> list[Pair]:
    """Read `<split>.de` / `<split>.hsb` from a KDE4-style split directory as (de, hsb) pairs."""
    split_dir = Path(split_dir)
    return read_parallel(split_dir / f"{split}.de", split_dir / f"{split}.hsb", strip=strip)


def write_split(split_dir: Path, split: str, pairs: Iterable[Pair]) -> None:
    split_dir = Path(split_dir)
    write_parallel(split_dir / f"{split}.de", split_dir / f"{split}.hsb", pairs)


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_report(path: Path, items: Sequence[tuple[str, object]]) -> None:
    """Write `key<TAB>value` lines to `path` and print them."""
    lines = [f"{key}\t{value}" for key, value in items]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"report\t{path}")
