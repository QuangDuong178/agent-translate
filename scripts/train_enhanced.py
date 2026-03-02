#!/usr/bin/env python3
"""Enhanced training script for translation models.
Downloads larger datasets and trains with more data for better quality.

Usage:
    python3 scripts/train_enhanced.py --lang en-vi --samples 20000 --epochs 5
    python3 scripts/train_enhanced.py --lang ja-vi --samples 10000 --epochs 5
    python3 scripts/train_enhanced.py --lang all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE_DIR / "training_runs"
TRAINING_DIR.mkdir(exist_ok=True)

# Training configs per language pair
TRAIN_CONFIGS = {
    "en-vi": {
        "model": "Helsinki-NLP/opus-mt-en-vi",
        "datasets": [
            {"name": "Helsinki-NLP/opus-100", "config": "en-vi", "src": "en", "tgt": "vi"},
            {"name": "Helsinki-NLP/tatoeba_mt", "config": "eng-vie", "src": "sourceString", "tgt": "targetString"},
        ],
        "default_samples": 20000,
        "default_epochs": 5,
        "output_dir": "opus-mt-en-vi-enhanced",
    },
    "ja-vi": {
        "model": "facebook/nllb-200-distilled-600M",
        "datasets": [
            {"name": "Helsinki-NLP/tatoeba_mt", "config": "jpn-vie", "src": "sourceString", "tgt": "targetString"},
            {"name": "Helsinki-NLP/opus-100", "config": "en-vi", "src": "en", "tgt": "vi", "pivot": True},
        ],
        "default_samples": 10000,
        "default_epochs": 5,
        "output_dir": "nllb-ja-vi-finetuned",
        "nllb_src": "jpn_Jpan",
        "nllb_tgt": "vie_Latn",
    },
    "en-ja": {
        "model": "Helsinki-NLP/opus-mt-en-jap",
        "datasets": [
            {"name": "Helsinki-NLP/opus-100", "config": "en-ja", "src": "en", "tgt": "ja"},
        ],
        "default_samples": 15000,
        "default_epochs": 5,
        "output_dir": "opus-mt-en-ja-enhanced",
    },
}


def load_dataset_pairs(dataset_configs, max_samples):
    """Load parallel sentence pairs from multiple datasets."""
    from datasets import load_dataset
    
    all_pairs = []
    
    for ds_config in dataset_configs:
        if ds_config.get("pivot"):
            continue  # Skip pivot datasets for now
            
        name = ds_config["name"]
        config = ds_config.get("config")
        src_key = ds_config["src"]
        tgt_key = ds_config["tgt"]
        
        print(f"  Loading {name} ({config})...")
        try:
            ds = load_dataset(name, config, split="train", trust_remote_code=True)
            
            count = 0
            for item in ds:
                # Handle different dataset structures
                if "translation" in item:
                    src = item["translation"].get(src_key, "")
                    tgt = item["translation"].get(tgt_key, "")
                elif src_key in item:
                    src = item[src_key]
                    tgt = item[tgt_key]
                else:
                    continue
                    
                if src and tgt and len(src.strip()) > 3 and len(tgt.strip()) > 3:
                    all_pairs.append({"src": src.strip(), "tgt": tgt.strip()})
                    count += 1
                    
                if count >= max_samples:
                    break
                    
            print(f"    → Loaded {count} pairs from {name}")
            
        except Exception as e:
            print(f"    ⚠ Failed to load {name}: {e}")
            continue
    
    return all_pairs[:max_samples]


def train_opus_mt(config, pairs, epochs, batch_size=4, lr=2e-5, max_length=128):
    """Fine-tune an OpusMT model."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset
    
    model_name = config["model"]
    output_dir = TRAINING_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Data: {len(pairs)} pairs, {epochs} epochs")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    
    # Prepare dataset
    print("Preparing dataset...")
    split_idx = int(len(pairs) * 0.95)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]
    
    def tokenize_fn(examples):
        inputs = tokenizer(
            examples["src"], max_length=max_length, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            examples["tgt"], max_length=max_length, truncation=True, padding="max_length"
        )
        inputs["labels"] = labels["input_ids"]
        return inputs
    
    train_ds = Dataset.from_list(train_pairs).map(tokenize_fn, batched=True, batch_size=100)
    eval_ds = Dataset.from_list(eval_pairs).map(tokenize_fn, batched=True, batch_size=100)
    
    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=False,
        report_to="none",
        dataloader_num_workers=0,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Training log
    loss_history = []
    start_time = time.time()
    
    from transformers import TrainerCallback
    
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
                print(f"  Step {state.global_step}: loss={logs['loss']:.4f}, epoch={state.epoch:.2f}, time={elapsed:.1f}m")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCb()],
    )
    
    print(f"\nStarting training... ({len(train_pairs)} train, {len(eval_pairs)} eval)")
    trainer.train()
    
    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    total_time = (time.time() - start_time) / 60
    
    # Save training log
    log = {
        "model": model_name,
        "trained_at": datetime.utcnow().isoformat(),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_length": max_length,
            "train_samples": len(train_pairs),
            "eval_samples": len(eval_pairs),
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
    
    print(f"\n✅ Training complete!")
    print(f"   Final loss: {log['result']['train_loss']}")
    print(f"   Time: {total_time:.1f} minutes")
    print(f"   Model saved to: {final_dir}")
    
    # Cleanup
    del model, tokenizer, trainer
    import gc; gc.collect()
    
    return str(final_dir)


def train_nllb(config, pairs, epochs, batch_size=2, lr=1e-5, max_length=128):
    """Fine-tune NLLB-200 for a specific language pair."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, TrainerCallback,
    )
    from datasets import Dataset
    
    model_name = config["model"]
    output_dir = TRAINING_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    src_lang = config["nllb_src"]
    tgt_lang = config["nllb_tgt"]
    
    print(f"\n{'='*60}")
    print(f"Training NLLB: {src_lang} → {tgt_lang}")
    print(f"Data: {len(pairs)} pairs, {epochs} epochs")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    tokenizer.src_lang = src_lang
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    split_idx = int(len(pairs) * 0.95)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]
    
    def tokenize_fn(examples):
        inputs = tokenizer(
            examples["src"], max_length=max_length, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            examples["tgt"], max_length=max_length, truncation=True, padding="max_length"
        )
        inputs["labels"] = labels["input_ids"]
        return inputs
    
    train_ds = Dataset.from_list(train_pairs).map(tokenize_fn, batched=True, batch_size=50)
    eval_ds = Dataset.from_list(eval_pairs).map(tokenize_fn, batched=True, batch_size=50)
    
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
                print(f"  Step {state.global_step}: loss={logs['loss']:.4f}, epoch={state.epoch:.2f}, time={elapsed:.1f}m")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=200,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        fp16=False,
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
    
    print(f"Starting training... ({len(train_pairs)} train, {len(eval_pairs)} eval)")
    trainer.train()
    
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    total_time = (time.time() - start_time) / 60
    
    log = {
        "model": model_name,
        "lang_pair": f"{src_lang}→{tgt_lang}",
        "trained_at": datetime.utcnow().isoformat(),
        "config": {
            "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
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
    
    print(f"\n✅ NLLB Training complete!")
    print(f"   Final loss: {log['result']['train_loss']}")
    print(f"   Time: {total_time:.1f} minutes")
    print(f"   Model saved to: {final_dir}")
    
    del model, tokenizer, trainer
    import gc; gc.collect()
    
    return str(final_dir)


def main():
    parser = argparse.ArgumentParser(description="Enhanced translation model training")
    parser.add_argument("--lang", type=str, default="en-vi", 
                        help="Language pair (en-vi, ja-vi, en-ja, all)")
    parser.add_argument("--samples", type=int, default=0,
                        help="Number of training samples (0=use default)")
    parser.add_argument("--epochs", type=int, default=0,
                       help="Number of epochs (0=use default)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    pairs_to_train = []
    if args.lang == "all":
        pairs_to_train = list(TRAIN_CONFIGS.keys())
    elif args.lang in TRAIN_CONFIGS:
        pairs_to_train = [args.lang]
    else:
        print(f"Unknown language pair: {args.lang}")
        print(f"Available: {', '.join(TRAIN_CONFIGS.keys())}")
        sys.exit(1)
    
    for lang_pair in pairs_to_train:
        config = TRAIN_CONFIGS[lang_pair]
        samples = args.samples or config["default_samples"]
        epochs = args.epochs or config["default_epochs"]
        
        print(f"\n{'#'*60}")
        print(f"# Training: {lang_pair} ({samples} samples, {epochs} epochs)")
        print(f"{'#'*60}")
        
        # Load data
        pairs = load_dataset_pairs(config["datasets"], samples)
        if not pairs:
            print(f"⚠ No data loaded for {lang_pair}, skipping")
            continue
        
        print(f"Total pairs: {len(pairs)}")
        
        # Train
        if "nllb" in config["model"].lower():
            train_nllb(config, pairs, epochs, args.batch_size)
        else:
            train_opus_mt(config, pairs, epochs, args.batch_size)
    
    print(f"\n{'='*60}")
    print("All training complete! 🎉")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
