"""Train a sentence-level reward model with TRL's PRMTrainer.

Qwen2.5 token classifier (0.5B or 3B), lr 1e-5 linear with 3% warm-up,
batch 2 x 8 accumulation (effective 16), max length 1024, step separator
"\\n", 6 epochs with a checkpoint per epoch. A 90/10 study split (seed 42)
is held out for monitoring only; the deployed epoch was chosen on a separate
selection pool by the downstream Best-of-N metric, not by eval loss.

Usage:  09_train_prm.py --base Qwen/Qwen2.5-0.5B --dataset <hf-dir from 08> --out <dir>
"""
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer
from trl import PRMConfig, PRMTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()

    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForTokenClassification.from_pretrained(a.base, num_labels=2)
    split = load_from_disk(a.dataset).train_test_split(test_size=0.10, seed=42)
    print(f"train {len(split['train'])} studies | dev {len(split['test'])} studies")

    cfg = PRMConfig(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=1e-5,
        lr_scheduler_type="linear", warmup_ratio=0.03,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        gradient_checkpointing=True, max_length=1024, max_prompt_length=512,
        step_separator="\n", eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=a.epochs, logging_steps=10, seed=42, report_to=[],
        bf16=torch.cuda.is_available(), dataset_num_proc=4,
    )
    trainer = PRMTrainer(model=model, args=cfg, processing_class=tok,
                         train_dataset=split["train"], eval_dataset=split["test"])
    resume = any(Path(a.out).glob("checkpoint-*")) if Path(a.out).exists() else False
    trainer.train(resume_from_checkpoint=resume)
    for h in trainer.state.log_history:
        if "eval_loss" in h:
            print(f"epoch {h['epoch']:.0f}: eval_loss={h['eval_loss']:.4f}")
    trainer.save_model(a.out + "/final")


if __name__ == "__main__":
    main()
