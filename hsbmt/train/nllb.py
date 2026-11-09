"""Full fine-tuning of the extended NLLB-200 model in one direction.

    python -m hsbmt.train.nllb --direction de-hsb --split_dir data/splits_cleaned \\
        --output_dir models/nllb-600M-de-hsb-cleaned
"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from hsbmt.config import DATA_DIR, DIRECTIONS, NLLB_HSB_INIT, Direction
from hsbmt.io import read_parallel
from hsbmt.models.nllb import HSB_TAG, lang_id


def build_dataset(direction: Direction, split_dir: Path, split: str, tokenizer, max_length: int, limit: int | None):
    pairs = read_parallel(*direction.files(split_dir, split), strip=True)[:limit]
    print(f"{split}: {len(pairs)} pairs from {split_dir}")
    dataset = Dataset.from_dict({"source": [s for s, _ in pairs], "target": [t for _, t in pairs]})
    return dataset.map(
        lambda batch: tokenizer(batch["source"], text_target=batch["target"], max_length=max_length, truncation=True),
        batched=True,
        remove_columns=["source", "target"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits", help="Directory with train/dev .de/.hsb")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, default=NLLB_HSB_INIT)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--debug", action="store_true", help="10 pairs, 1 epoch, batch size 1")
    args = parser.parse_args()

    direction = Direction.parse(args.direction)
    if args.debug:
        args.num_train_epochs, args.batch_size, args.gradient_accumulation_steps = 1, 1, 1
    limit = 10 if args.debug else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    lang_id(tokenizer, HSB_TAG)  # fails early if the model was not extended
    tokenizer.src_lang, tokenizer.tgt_lang = direction.nllb_src, direction.nllb_tgt
    model.generation_config.forced_bos_token_id = lang_id(tokenizer, direction.nllb_tgt)
    model.config.forced_bos_token_id = None  # keep generation settings only in generation_config

    train = build_dataset(direction, args.split_dir, "train", tokenizer, args.max_length, limit)
    dev = build_dataset(direction, args.split_dir, "dev", tokenizer, args.max_length, limit)

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(args.output_dir),
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=1 if args.debug else 50,
            save_strategy="no",
            report_to="none",
            fp16=torch.cuda.is_available(),
        ),
        train_dataset=train,
        eval_dataset=dev,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    print(f"Fine-tuning {args.model_path} for {direction.name} ...")
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
