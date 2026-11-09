"""QLoRA fine-tuning of Qwen2.5-7B-Instruct for translation in one direction.

Only the target tokens contribute to the loss; the prompt tokens are masked with -100.

    python -m hsbmt.train.qwen_lora --direction de-hsb --split_dir data/splits \\
        --output_dir models/qwen-7b-lora-de-hsb
"""

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from hsbmt.config import DATA_DIR, DIRECTIONS, QWEN_BASE_MODEL, Direction
from hsbmt.io import read_parallel
from hsbmt.models.qwen import LORA_TARGET_MODULES, load_model, load_tokenizer, lora_prompt


class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, direction: Direction, max_length: int):
        self.pairs, self.tokenizer, self.direction, self.max_length = pairs, tokenizer, direction, max_length

    def __len__(self):
        return len(self.pairs)

    def _encode(self, text):
        return list(self.tokenizer(text, add_special_tokens=True, truncation=True, max_length=self.max_length)["input_ids"])

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        prompt = lora_prompt(src, self.direction.src_name, self.direction.tgt_name)
        prompt_ids = self._encode(prompt)
        input_ids = self._encode(prompt + " " + tgt + self.tokenizer.eos_token)
        n_prompt = min(len(prompt_ids), len(input_ids))
        labels = [-100] * n_prompt + input_ids[n_prompt:]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features):
        width = max(len(f["input_ids"]) for f in features)
        pad = lambda seq, value: seq + [value] * (width - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(f["input_ids"], self.pad_token_id) for f in features]),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features]),
            "labels": torch.tensor([pad(f["labels"], -100) for f in features]),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default=QWEN_BASE_MODEL)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--no_4bit", action="store_true", help="Load the base model in fp16 instead of 4-bit")
    parser.add_argument("--debug", action="store_true", help="Use only 10 pairs")
    args = parser.parse_args()

    direction = Direction.parse(args.direction)
    load_in_4bit = not args.no_4bit
    limit = 10 if args.debug else None

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, load_in_4bit=load_in_4bit)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        ),
    )
    model.print_trainable_parameters()

    datasets = {
        split: TranslationDataset(
            read_parallel(*direction.files(args.split_dir, split))[:limit], tokenizer, direction, args.max_length
        )
        for split in ("train", "dev")
    }

    kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to=[],
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
    )
    # The argument was renamed in transformers 4.41.
    eval_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kwargs[eval_key] = "epoch"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**kwargs),
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved LoRA adapter to:", args.output_dir)


if __name__ == "__main__":
    main()
