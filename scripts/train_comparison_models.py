#!/usr/bin/env python3
"""
Train ADDITIONAL EN→VI models for comparison report.

Models to train:
  1. M2M-100 (418M) — Facebook's Many-to-Many multilingual model
  2. MarianMT EN→VI — Different hyperparameters (higher LR, more epochs)

Usage:
    PYTHONUNBUFFERED=1 python3 scripts/train_comparison_models.py --model m2m100
    PYTHONUNBUFFERED=1 python3 scripts/train_comparison_models.py --model marian-lr
    PYTHONUNBUFFERED=1 python3 scripts/train_comparison_models.py --model all
"""

import argparse
import json
import sys
import time
import gc
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE_DIR / "training_runs"
TRAINING_DIR.mkdir(exist_ok=True)


def load_en_vi_data(max_samples=5000):
    """Load EN→VI parallel data from OPUS-100."""
    from datasets import load_dataset
    
    pairs = []
    print(f"  Loading OPUS-100 en-vi (max {max_samples})...", flush=True)
    ds = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="train")
    for item in ds:
        if "translation" in item:
            src = item["translation"].get("en", "").strip()
            tgt = item["translation"].get("vi", "").strip()
            if src and tgt and len(src) > 3 and len(tgt) > 3:
                pairs.append({"src": src, "tgt": tgt})
                if len(pairs) >= max_samples:
                    break
    print(f"  → {len(pairs)} pairs loaded", flush=True)
    return pairs


# ══════════════════════════════════════════════════════════════
# MODEL 1: M2M-100 (418M) — Facebook Many-to-Many
# ══════════════════════════════════════════════════════════════

def train_m2m100(pairs, epochs=3, batch_size=8, lr=1e-5, max_length=64):
    """Fine-tune M2M-100 (418M) for EN→VI."""
    import torch
    from transformers import (
        M2M100ForConditionalGeneration, M2M100Tokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, TrainerCallback,
    )
    from datasets import Dataset

    output_dir = TRAINING_DIR / "m2m100-en-vi-finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = "facebook/m2m100_418M"

    print(f"\n{'='*60}", flush=True)
    print(f"Training M2M-100 (418M): EN → VI", flush=True)
    print(f"Data: {len(pairs)} pairs, {epochs} epochs, batch={batch_size}", flush=True)
    print(f"{'='*60}", flush=True)

    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = "vi"

    # Split data
    split_idx = int(len(pairs) * 0.95)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    # Get target language token ID
    tgt_token_id = tokenizer.get_lang_id("vi")

    def tokenize_fn(examples):
        tokenizer.src_lang = "en"
        inputs = tokenizer(
            examples["src"], max_length=max_length, truncation=True, padding="max_length"
        )
        tokenizer.src_lang = "vi"
        labels = tokenizer(
            examples["tgt"], max_length=max_length, truncation=True, padding="max_length"
        )
        tokenizer.src_lang = "en"
        inputs["labels"] = labels["input_ids"]
        return inputs

    print("Tokenizing...", flush=True)
    train_ds = Dataset.from_list(train_pairs).map(tokenize_fn, batched=True, batch_size=200,
                                                   remove_columns=["src", "tgt"])
    eval_ds = Dataset.from_list(eval_pairs).map(tokenize_fn, batched=True, batch_size=200,
                                                 remove_columns=["src", "tgt"])
    print(f"Tokenized: {len(train_ds)} train, {len(eval_ds)} eval", flush=True)

    loss_history = []
    start_time = time.time()

    class LogCb(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                elapsed = (time.time() - start_time) / 60
                loss_history.append({
                    "step": state.global_step,
                    "epoch": round(state.epoch, 2),
                    "elapsed_min": round(elapsed, 1),
                    "train_loss": round(logs["loss"], 4),
                })
                print(f"  Step {state.global_step}: loss={logs['loss']:.4f}, epoch={state.epoch:.2f}, time={elapsed:.1f}m", flush=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        weight_decay=0.01,
        warmup_steps=200,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        predict_with_generate=False,
        fp16=False,
        use_cpu=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCb()],
    )

    # Resume from checkpoint if available
    checkpoint_dir = output_dir / "checkpoints"
    last_checkpoint = None
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"📌 Resuming from checkpoint: {last_checkpoint}", flush=True)

    print(f"\n🚀 Starting M2M-100 training...", flush=True)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    total_time = (time.time() - start_time) / 60
    log = {
        "type": "m2m100_finetune",
        "model": model_name,
        "dataset": "Helsinki-NLP/opus-100/en-vi",
        "lang_pair": "en→vi",
        "trained_at": datetime.now().isoformat(),
        "config": {
            "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
            "max_length": max_length,
            "train_samples": len(train_pairs), "eval_samples": len(eval_pairs),
        },
        "result": {
            "train_loss": loss_history[-1]["train_loss"] if loss_history else None,
            "total_steps": loss_history[-1]["step"] if loss_history else 0,
            "total_time_minutes": round(total_time, 1),
        },
        "loss_history": loss_history,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ M2M-100 training complete!", flush=True)
    print(f"   Final loss: {log['result']['train_loss']}", flush=True)
    print(f"   Time: {total_time:.1f} minutes", flush=True)
    print(f"   Model: {final_dir}", flush=True)

    del model, tokenizer, trainer
    gc.collect()
    return str(final_dir)


# ══════════════════════════════════════════════════════════════
# MODEL 2: MarianMT EN→VI — Different hyperparameters
# ══════════════════════════════════════════════════════════════

def train_marian_variant(pairs, epochs=5, batch_size=8, lr=5e-5, max_length=128):
    """Fine-tune MarianMT EN→VI with different hyperparameters.
    
    Compared to existing v1 (lr=2e-5, epochs=3, samples=5K):
      - Higher LR: 5e-5 (more aggressive learning)
      - More epochs: 5 (longer training)
      - Same data: 5K samples
    """
    import torch
    from transformers import (
        MarianMTModel, MarianTokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, TrainerCallback,
    )
    from datasets import Dataset

    output_dir = TRAINING_DIR / "opus-mt-en-vi-highLR"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = "Helsinki-NLP/opus-mt-en-vi"

    print(f"\n{'='*60}", flush=True)
    print(f"Training MarianMT EN→VI (High LR variant)", flush=True)
    print(f"Data: {len(pairs)} pairs, {epochs} epochs, LR={lr}, batch={batch_size}", flush=True)
    print(f"{'='*60}", flush=True)

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    split_idx = int(len(pairs) * 0.95)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    def tokenize_fn(examples):
        inputs = tokenizer(
            examples["src"], max_length=max_length, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            text_target=examples["tgt"], max_length=max_length, truncation=True, padding="max_length"
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    print("Tokenizing...", flush=True)
    train_ds = Dataset.from_list(train_pairs).map(tokenize_fn, batched=True, batch_size=200,
                                                   remove_columns=["src", "tgt"])
    eval_ds = Dataset.from_list(eval_pairs).map(tokenize_fn, batched=True, batch_size=200,
                                                 remove_columns=["src", "tgt"])
    print(f"Tokenized: {len(train_ds)} train, {len(eval_ds)} eval", flush=True)

    loss_history = []
    start_time = time.time()

    class LogCb(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                elapsed = (time.time() - start_time) / 60
                loss_history.append({
                    "step": state.global_step,
                    "epoch": round(state.epoch, 2),
                    "elapsed_min": round(elapsed, 1),
                    "train_loss": round(logs["loss"], 4),
                })
                print(f"  Step {state.global_step}: loss={logs['loss']:.4f}, epoch={state.epoch:.2f}, time={elapsed:.1f}m", flush=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="no",
        predict_with_generate=False,
        fp16=False,
        use_cpu=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCb()],
    )

    print(f"\n🚀 Starting MarianMT High-LR training...", flush=True)
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    total_time = (time.time() - start_time) / 60
    log = {
        "type": "marian_finetune_highLR",
        "model": model_name,
        "dataset": "Helsinki-NLP/opus-100/en-vi",
        "lang_pair": "en→vi",
        "trained_at": datetime.now().isoformat(),
        "config": {
            "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
            "max_length": max_length,
            "train_samples": len(train_pairs), "eval_samples": len(eval_pairs),
        },
        "result": {
            "train_loss": loss_history[-1]["train_loss"] if loss_history else None,
            "total_steps": loss_history[-1]["step"] if loss_history else 0,
            "total_time_minutes": round(total_time, 1),
        },
        "loss_history": loss_history,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ MarianMT High-LR training complete!", flush=True)
    print(f"   Final loss: {log['result']['train_loss']}", flush=True)
    print(f"   Time: {total_time:.1f} minutes", flush=True)
    print(f"   Model: {final_dir}", flush=True)

    del model, tokenizer, trainer
    gc.collect()
    return str(final_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                       choices=["m2m100", "marian-lr", "all"])
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()

    pairs = load_en_vi_data(args.samples)
    if len(pairs) < 100:
        print("❌ Not enough data!", flush=True)
        sys.exit(1)

    if args.model in ("marian-lr", "all"):
        print("\n" + "#" * 60, flush=True)
        print("# Training MarianMT EN→VI (High LR: 5e-5, 5 epochs)", flush=True)
        print("#" * 60, flush=True)
        train_marian_variant(pairs, epochs=5, batch_size=8, lr=5e-5)
        gc.collect()

    if args.model in ("m2m100", "all"):
        print("\n" + "#" * 60, flush=True)
        print("# Training M2M-100 (418M) EN→VI", flush=True)
        print("#" * 60, flush=True)
        train_m2m100(pairs, epochs=3, batch_size=8, lr=1e-5)
        gc.collect()

    print("\n" + "=" * 60, flush=True)
    print("All comparison training complete! 🎉", flush=True)
    print("=" * 60, flush=True)
    print("\nRun evaluation:", flush=True)
    print("  python3 scripts/evaluate_models.py", flush=True)


if __name__ == "__main__":
    main()
