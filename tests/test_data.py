"""Unit tests for the CPU-only data-processing logic (no models or downloads needed)."""

from pathlib import Path

import pytest

from hsbmt.config import Direction
from hsbmt.data.combine import select_extra
from hsbmt.data.kde4 import clean_pairs, removal_reason, split_pairs
from hsbmt.data.leakage import leakage_checks, remove_seen
from hsbmt.data.pivot import detect_languages, group_key, merge_sources
from hsbmt.data.wmt22 import categorize, lid_status
from hsbmt.io import read_parallel, write_parallel


def test_removal_rules():
    seen = {("Datei", "Dataja")}
    assert removal_reason("", "x", seen) == "empty"
    assert removal_reason("Datei", "Dataja", seen) == "duplicate"
    assert removal_reason("a " * 129, "b", seen) == "too_long"
    assert removal_reason("ein zwei drei vier", "jedyn", seen) == "length_ratio"
    assert removal_reason("KDE Plasma", "KDE Plasma", seen) == "identical"
    assert removal_reason("OK", "OK", seen) is None  # short identical strings are kept
    assert removal_reason("Datei öffnen", "Dataju wočinić", seen) is None


def test_clean_pairs_strips_and_deduplicates():
    kept, removed, counts = clean_pairs([(" Datei ", "Dataja"), ("Datei", "Dataja "), ("", "x")])
    assert kept == [("Datei", "Dataja")]
    assert counts["duplicate"] == 1 and counts["empty"] == 1
    assert [r[0] for r in removed] == [1, 2]


def test_split_sizes_and_order():
    pairs = [(str(i), str(i)) for i in range(10)]
    splits = split_pairs(pairs, dev_size=2, test_size=3)
    assert [len(splits[s]) for s in ("train", "dev", "test")] == [5, 2, 3]
    assert splits["test"][-1] == ("9", "9")
    with pytest.raises(ValueError):
        split_pairs(pairs[:4], dev_size=2, test_size=3)


def test_select_extra_removes_duplicates_and_kde4_overlap():
    extra = [("a", "1"), ("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]
    selected, dups, overlaps = select_extra(extra, exclude={("b", "2")}, max_pairs=2)
    assert selected == [("a", "1"), ("c", "3")]
    assert (dups, overlaps) == (1, 1)
    with pytest.raises(ValueError):
        select_extra(extra, exclude=set(), max_pairs=10)


def test_leakage_checks_and_dedup():
    o = {"train": [("a", "1"), ("b", "2")], "dev": [("c", "3")], "test": [("a", "1"), ("d", "4")]}
    c = {"train": [("b", "2")], "dev": [], "test": [("d", "4")]}
    results = {r["check_name"]: r["overlap_n"] for r in leakage_checks({"original": o, "cleaned": c})}
    assert results["original_train_vs_test_pair_overlap"] == 1
    assert results["cleaned_train_vs_test_pair_overlap"] == 0
    assert remove_seen(o["test"], o["train"]) == ([("d", "4")], [("a", "1")])


def lid_row(de_labels, hsb_labels):
    row = {}
    for side, labels in (("de", de_labels), ("hsb", hsb_labels)):
        for k, label in enumerate(labels, start=1):
            row[f"{side}_pred{k}_label"] = f"__label__{label}"
    return row


def test_lid_categories():
    assert categorize(lid_row(["deu_Latn", "x", "y"], ["hsb_Latn", "x", "y"])) == "strict_pass"
    assert categorize(lid_row(["eng_Latn", "deu_Latn", "y"], ["ces_Latn", "x", "y"])) == "relaxed_pass"
    assert categorize(lid_row(["deu_Latn", "x", "y"], ["eng_Latn", "x", "y"])) == "review"
    assert lid_status([("__label__deu_Latn", 0.9)], [("__label__hsb_Latn", 0.5)], 0.8) == "bad_hsb"


def test_pivot_file_discovery_and_merge():
    assert detect_languages("KDE4.cs-hsb.cs") == ["cs", "hsb"]
    assert group_key(Path("KDE4.cs-hsb.cs")) == group_key(Path("KDE4.cs-hsb.hsb"))
    rows = merge_sources({"cs": [("x", "h"), ("x", "h")], "pl": [("x", "h")]})
    assert rows == [("ces_Latn", "x", "h"), ("pol_Latn", "x", "h")]


def test_direction_and_io_roundtrip(tmp_path):
    d = Direction.parse("hsb-de")
    assert (d.nllb_src, d.nllb_tgt, d.src_name) == ("hsb_Latn", "deu_Latn", "Upper Sorbian")
    assert d.files(tmp_path, "test") == (tmp_path / "test.hsb", tmp_path / "test.de")
    write_parallel(tmp_path / "a.de", tmp_path / "a.hsb", [("Hallo", "Witaj")])
    assert read_parallel(tmp_path / "a.de", tmp_path / "a.hsb") == [("Hallo", "Witaj")]
    with pytest.raises(ValueError):
        Direction.parse("de-en")
