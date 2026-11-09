"""Translate a test set with one of the systems and report BLEU / chrF / chrF++.

    # NLLB (zero-shot extended model or any fine-tuned checkpoint)
    python -m hsbmt.evaluate.translate nllb --direction de-hsb \\
        --model_path models/nllb-600M-de-hsb-cleaned --split_dir data/splits

    # Qwen2.5-7B-Instruct zero-shot
    python -m hsbmt.evaluate.translate qwen --direction hsb-de

    # Qwen2.5-7B-Instruct + LoRA adapter
    python -m hsbmt.evaluate.translate qwen-lora --direction hsb-de --adapter_path models/qwen-7b-lora-hsb-de

Predictions are written one per line to --output (default: outputs/predictions/<system>_<direction>.txt).
"""

import argparse
import json
from pathlib import Path

from hsbmt.config import DATA_DIR, DIRECTIONS, NLLB_HSB_INIT, OUTPUTS_DIR, QWEN_BASE_MODEL, Direction
from hsbmt.io import read_parallel, write_lines
from hsbmt.metrics import print_scores, score


def run_nllb(args, direction: Direction, sources: list[str]) -> list[str]:
    from hsbmt.models import nllb

    tokenizer, model = nllb.load(args.model_path)
    return nllb.translate(
        sources, tokenizer, model, direction.nllb_src, direction.nllb_tgt,
        batch_size=args.batch_size, num_beams=5, max_new_tokens=128, progress=True,
    )  # fmt: skip


def run_qwen(args, direction: Direction, sources: list[str]) -> list[str]:
    import torch
    from tqdm import tqdm

    from hsbmt.models import qwen

    if args.system == "qwen":
        tokenizer = qwen.load_tokenizer(args.base_model)
        model = qwen.load_model(args.base_model, load_in_4bit=False, dtype=torch.bfloat16).eval()
        outputs = []
        for src in tqdm(sources):
            raw = qwen.generate(model, tokenizer, qwen.zero_shot_prompt(src, direction.src_name, direction.tgt_name), 30)
            outputs.append(qwen.clean_zero_shot_output(raw, direction.src_name, direction.tgt_name))
        return outputs

    from peft import PeftModel

    tokenizer = qwen.load_tokenizer(args.adapter_path)
    model = PeftModel.from_pretrained(qwen.load_model(args.base_model, load_in_4bit=True), args.adapter_path).eval()
    return [
        qwen.generate(model, tokenizer, qwen.lora_prompt(src, direction.src_name, direction.tgt_name), 128, max_prompt_tokens=512).replace("\n", " ")
        for src in tqdm(sources)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=["nllb", "qwen", "qwen-lora"])
    parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits", help="Reads test.de / test.hsb from here")
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model_path", type=Path, default=NLLB_HSB_INIT, help="[nllb] checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=1, help="[nllb] reported numbers use 1")
    parser.add_argument("--base_model", default=QWEN_BASE_MODEL, help="[qwen, qwen-lora]")
    parser.add_argument("--adapter_path", type=Path, help="[qwen-lora] LoRA adapter directory")
    args = parser.parse_args()
    if args.system == "qwen-lora" and not args.adapter_path:
        parser.error("qwen-lora needs --adapter_path")

    direction = Direction.parse(args.direction)
    pairs = read_parallel(*direction.files(args.split_dir, "test"), strip=True)[: args.num_test]
    sources, references = [s for s, _ in pairs], [r for _, r in pairs]
    print(f"{args.system} | {direction.name} | {len(pairs)} sentences from {args.split_dir}")

    predictions = run_nllb(args, direction, sources) if args.system == "nllb" else run_qwen(args, direction, sources)

    output = args.output or OUTPUTS_DIR / "predictions" / f"{args.system}_{direction.name}.txt"
    write_lines(output, (p.strip() for p in predictions))
    scores = score(predictions, references)
    print_scores(scores)
    output.with_suffix(".scores.json").write_text(
        json.dumps({"system": args.system, "direction": direction.name, "test": str(args.split_dir), "n": len(pairs), **scores}, indent=2),
        encoding="utf-8",
    )
    print("Predictions saved to:", output)


if __name__ == "__main__":
    main()
