"""DPO training scaffold for RLAIF.

Reads a preferences JSONL (produced by `ale rlaif collect`) and DPO-trains an
open-weight base model against those pairs. Heavy deps (`trl`, `transformers`,
`peft`, `datasets`, `torch`) are imported lazily so the rest of the package
works without them.

Install:
    pip install 'ale[train]'   # installs trl, transformers, peft, datasets, torch

Typical invocation on a single GPU:
    ale rlaif train --pairs traces/pairs.jsonl \\
        --base-model Qwen/Qwen2.5-0.5B-Instruct \\
        --out runs/dpo-qwen
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DPOArgs:
    pairs_path: Path
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: Path = Path("runs/dpo")
    epochs: int = 1
    lr: float = 5e-6
    batch_size: int = 1
    grad_accum: int = 8
    max_len: int = 2048
    beta: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32


def train(args: DPOArgs) -> None:
    try:
        import torch  # noqa: F401
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        raise ImportError(
            "DPO training requires trl, transformers, peft, datasets, and torch.\n"
            "Install with: pip install 'ale[train]'"
        ) from e

    if not args.pairs_path.exists():
        raise FileNotFoundError(f"pairs file not found: {args.pairs_path}")

    dataset = load_dataset("json", data_files=str(args.pairs_path), split="train")
    dataset = dataset.map(lambda x: {"prompt": x["prompt"], "chosen": x["chosen"], "rejected": x["rejected"]})

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    dpo_config = DPOConfig(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_len,
        beta=args.beta,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.out_dir))
    print(f"\nDPO adapter saved to {args.out_dir}")
    print("Serve it with vLLM and point the OpenAI provider at it to close the loop.")
