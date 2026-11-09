"""Collect all *.scores.json and *.xcomet.json files under outputs/ into one table.

    python -m hsbmt.evaluate.summarize            # writes outputs/results_summary.tsv
"""

import argparse
import json
from pathlib import Path

from hsbmt.config import OUTPUTS_DIR
from hsbmt.io import write_tsv

COLUMNS = ["predictions", "system", "direction", "test", "n", "BLEU", "chrF", "chrF++", "xCOMET"]


def collect(root: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    for path in sorted(root.rglob("*.scores.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        key = str(path).removesuffix(".scores.json")
        rows[key] = {"predictions": key + ".txt", **data}
    for path in sorted(root.rglob("*.xcomet.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        key = str(path).removesuffix(".xcomet.json")
        row = rows.setdefault(key, {"predictions": data["mt"], "direction": data["direction"], "test": data["test"], "n": data["n"]})
        row["xCOMET"] = data["system_score"]
    return list(rows.values())


def fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value if value is not None else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "results_summary.tsv")
    args = parser.parse_args()

    rows = [[fmt(row.get(c)) for c in COLUMNS] for row in collect(args.root)]
    write_tsv(args.output, COLUMNS, rows)
    print("\t".join(COLUMNS[1:]))
    for row in rows:
        print("\t".join(row[1:]))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
