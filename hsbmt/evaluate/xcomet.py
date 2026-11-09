"""Score translations with xCOMET (Unbabel/XCOMET-XL).

    python -m hsbmt.evaluate.xcomet --direction hsb-de --split_dir data/splits \\
        --mt outputs/predictions/nllb_hsb-de.txt

Writes <mt>.xcomet.json (system score + per-sentence scores) next to the MT file.
"""

import argparse
import json
from pathlib import Path

from hsbmt.config import DATA_DIR, DIRECTIONS, XCOMET_MODEL, Direction
from hsbmt.io import read_lines, read_parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--mt", type=Path, required=True, help="One hypothesis per line, aligned with test.<src>")
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--model_name", default=XCOMET_MODEL)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    from comet import download_model, load_from_checkpoint

    direction = Direction.parse(args.direction)
    pairs = read_parallel(*direction.files(args.split_dir, "test"))
    mt = read_lines(args.mt)
    n = min(len(pairs), len(mt))
    data = [{"src": s, "mt": m, "ref": r} for (s, r), m in zip(pairs[:n], mt[:n])]

    model = load_from_checkpoint(download_model(args.model_name))
    result = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
    scores = [float(x) for x in result.scores]
    system_score = float(result.system_score) if result.system_score is not None else sum(scores) / len(scores)

    output = args.output_json or args.mt.with_suffix(".xcomet.json")
    output.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "direction": direction.name,
                "test": str(args.split_dir),
                "mt": str(args.mt),
                "n": n,
                "system_score": system_score,
                "sentences": [{**row, "score": s} for row, s in zip(data, scores)],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"xCOMET: {system_score:.4f} ({n} sentences) -> {output}")


if __name__ == "__main__":
    main()
