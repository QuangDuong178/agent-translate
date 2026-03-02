#!/usr/bin/env python3
"""
Auto Training Script — Agent-Translate
Downloads model, dataset, and runs fine-tuning automatically.

Model : Helsinki-NLP/opus-mt-en-vi  (English → Vietnamese, ~300MB)
Dataset: Helsinki-NLP/opus-100       (en-vi pair, 1M+ sentence pairs)
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto-train")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
TRAINING_DIR = BASE_DIR / "training_runs"
OUTPUT_DIR = TRAINING_DIR / "opus-mt-en-vi-finetuned"

for d in [MODELS_DIR, DATASETS_DIR, TRAINING_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Configuration ──────────────────────────────────────────────────
MODEL_NAME = "Helsinki-NLP/opus-mt-en-vi"
DATASET_NAME = "Helsinki-NLP/opus-100"
LANG_PAIR = "en-vi"
SOURCE_LANG = "en"
TARGET_LANG = "vi"

# Training hyperparameters (optimized for Mac / CPU)
NUM_EPOCHS = 3
BATCH_SIZE = 4            # small batch for CPU/MPS
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
WARMUP_STEPS = 200
LOGGING_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 500
MAX_TRAIN_SAMPLES = 5000  # limit for faster training demo
MAX_EVAL_SAMPLES = 500

def main():
    log.info("=" * 60)
    log.info("  Agent-Translate — Automatic Training Pipeline")
    log.info("=" * 60)
    log.info(f"Model    : {MODEL_NAME}")
    log.info(f"Dataset  : {DATASET_NAME} ({LANG_PAIR})")
    log.info(f"Epochs   : {NUM_EPOCHS}")
    log.info(f"Batch    : {BATCH_SIZE}")
    log.info(f"LR       : {LEARNING_RATE}")
    log.info(f"Max Len  : {MAX_LENGTH}")
    log.info(f"Samples  : train={MAX_TRAIN_SAMPLES}, eval={MAX_EVAL_SAMPLES}")
    log.info(f"Output   : {OUTPUT_DIR}")
    log.info("=" * 60)

    # ── Step 1: Import libraries ─────────────────────────────────────
    log.info("\n📦 Step 1/6: Importing libraries...")
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        TrainerCallback,
        EarlyStoppingCallback,
    )
    from datasets import load_dataset

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        log.info(f"  ✅ GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        log.info("  ✅ Apple MPS detected")
    else:
        log.info("  ⚠️  No GPU detected, using CPU (training will be slower)")

    log.info(f"  Device: {device}")
    log.info(f"  PyTorch: {torch.__version__}")

    # ── Step 2: Download / Load model ────────────────────────────────
    log.info("\n🧠 Step 2/6: Downloading model...")
    model_path = MODELS_DIR / MODEL_NAME.replace("/", "_")

    if (model_path / "config.json").exists():
        log.info(f"  Model already cached at {model_path}")
    else:
        log.info(f"  Downloading {MODEL_NAME} from HuggingFace...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(model_path),
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(model_path),
    )
    log.info(f"  ✅ Model loaded: {MODEL_NAME}")
    log.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"  Trainable : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Step 3: Download / Load dataset ──────────────────────────────
    log.info("\n📊 Step 3/6: Downloading dataset...")
    ds_cache = DATASETS_DIR / DATASET_NAME.replace("/", "_")

    log.info(f"  Loading {DATASET_NAME} ({LANG_PAIR})...")
    try:
        raw_dataset = load_dataset(
            DATASET_NAME,
            LANG_PAIR,
            cache_dir=str(ds_cache),
            trust_remote_code=True,
        )
    except Exception as e:
        log.warning(f"  Failed with lang pair '{LANG_PAIR}', trying without: {e}")
        raw_dataset = load_dataset(
            DATASET_NAME,
            cache_dir=str(ds_cache),
            trust_remote_code=True,
        )

    log.info(f"  ✅ Dataset loaded!")
    for split_name, split_data in raw_dataset.items():
        log.info(f"    {split_name}: {len(split_data):,} samples")

    # ── Step 4: Preprocess / Tokenize ────────────────────────────────
    log.info("\n🔧 Step 4/6: Preprocessing and tokenizing...")

    # Take a subset for faster training
    train_ds = raw_dataset["train"]
    if MAX_TRAIN_SAMPLES and len(train_ds) > MAX_TRAIN_SAMPLES:
        train_ds = train_ds.shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
        log.info(f"  Using {MAX_TRAIN_SAMPLES} training samples (from {len(raw_dataset['train']):,})")

    # Use test split or validation split for eval
    if "test" in raw_dataset:
        eval_ds = raw_dataset["test"]
    elif "validation" in raw_dataset:
        eval_ds = raw_dataset["validation"]
    else:
        # Split train into train+eval
        splits = train_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = splits["train"]
        eval_ds = splits["test"]

    if MAX_EVAL_SAMPLES and len(eval_ds) > MAX_EVAL_SAMPLES:
        eval_ds = eval_ds.shuffle(seed=42).select(range(MAX_EVAL_SAMPLES))
        log.info(f"  Using {MAX_EVAL_SAMPLES} eval samples")

    # Show sample
    sample = train_ds[0]
    log.info(f"  Sample data keys: {list(sample.keys())}")
    if "translation" in sample:
        log.info(f"  Sample: {sample['translation']}")
    else:
        log.info(f"  Sample: {sample}")

    # Tokenize function
    prefix = ""  # MarianMT models don't need a prefix

    def preprocess_function(examples):
        if "translation" in examples:
            inputs = [ex[SOURCE_LANG] for ex in examples["translation"]]
            targets = [ex[TARGET_LANG] for ex in examples["translation"]]
        elif SOURCE_LANG in examples and TARGET_LANG in examples:
            inputs = examples[SOURCE_LANG]
            targets = examples[TARGET_LANG]
        else:
            cols = list(examples.keys())
            inputs = examples[cols[0]]
            targets = examples[cols[1]]

        inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            text_target=targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    log.info("  Tokenizing training set...")
    tokenized_train = train_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )

    log.info("  Tokenizing eval set...")
    tokenized_eval = eval_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval",
    )

    log.info(f"  ✅ Tokenization complete!")
    log.info(f"    Train: {len(tokenized_train):,} samples")
    log.info(f"    Eval : {len(tokenized_eval):,} samples")

    # ── Step 5: Train ────────────────────────────────────────────────
    log.info("\n🚀 Step 5/6: Starting training...")
    log.info(f"  Output directory: {OUTPUT_DIR}")

    total_steps = (len(tokenized_train) // BATCH_SIZE) * NUM_EPOCHS
    log.info(f"  Total estimated steps: {total_steps}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=False,  # CPU/MPS – no fp16
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        use_cpu=(device == "cpu"),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",
    )

    # Progress callback
    class LiveProgressCallback(TrainerCallback):
        def __init__(self):
            self.start_time = time.time()
            self.training_log = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.global_step > 0:
                elapsed = time.time() - self.start_time
                entry = {
                    "step": state.global_step,
                    "epoch": round(state.epoch, 2) if state.epoch else 0,
                    "elapsed_min": round(elapsed / 60, 1),
                }
                if "loss" in logs:
                    entry["train_loss"] = round(logs["loss"], 4)
                    log.info(
                        f"  📈 Step {state.global_step}/{total_steps} | "
                        f"Epoch {entry['epoch']} | "
                        f"Loss: {logs['loss']:.4f} | "
                        f"Time: {entry['elapsed_min']}min"
                    )
                if "eval_loss" in logs:
                    entry["eval_loss"] = round(logs["eval_loss"], 4)
                    log.info(
                        f"  📉 Eval Loss: {logs['eval_loss']:.4f}"
                    )
                self.training_log.append(entry)

        def get_log(self):
            return self.training_log

    progress_cb = LiveProgressCallback()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[progress_cb],
    )

    log.info("  Training started! This may take a while...")
    log.info("  " + "─" * 50)

    train_result = trainer.train()

    log.info("  " + "─" * 50)
    log.info(f"  ✅ Training completed!")
    log.info(f"  Train Loss     : {train_result.training_loss:.4f}")
    log.info(f"  Total Steps    : {train_result.global_step}")
    total_time = time.time() - progress_cb.start_time
    log.info(f"  Total Time     : {total_time/60:.1f} minutes")

    # ── Step 6: Save model ───────────────────────────────────────────
    log.info("\n💾 Step 6/6: Saving fine-tuned model...")
    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"  ✅ Model saved to: {final_dir}")

    # Save training log
    log_file = OUTPUT_DIR / "training_log.json"
    with open(log_file, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "dataset": f"{DATASET_NAME}/{LANG_PAIR}",
            "config": {
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_length": MAX_LENGTH,
                "train_samples": len(tokenized_train),
                "eval_samples": len(tokenized_eval),
            },
            "result": {
                "train_loss": round(train_result.training_loss, 4),
                "total_steps": train_result.global_step,
                "total_time_minutes": round(total_time / 60, 1),
            },
            "loss_history": progress_cb.get_log(),
            "completed_at": datetime.utcnow().isoformat(),
        }, f, indent=2)
    log.info(f"  Training log saved: {log_file}")

    # ── Test translation ─────────────────────────────────────────────
    log.info("\n🌐 Testing translation with fine-tuned model...")

    test_sentences = [
        "Hello, how are you today?",
        "Machine learning is the future of technology.",
        "I love programming and building AI systems.",
        "The weather is beautiful today.",
        "Can you help me translate this sentence?",
    ]

    # Load fine-tuned model
    ft_tokenizer = AutoTokenizer.from_pretrained(str(final_dir))
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(str(final_dir))
    ft_model.eval()

    import torch

    log.info("  " + "─" * 50)
    for sent in test_sentences:
        inputs = ft_tokenizer(sent, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        with torch.no_grad():
            generated = ft_model.generate(**inputs, max_length=MAX_LENGTH, num_beams=4)
        translated = ft_tokenizer.decode(generated[0], skip_special_tokens=True)
        log.info(f"  EN: {sent}")
        log.info(f"  VI: {translated}")
        log.info("")

    log.info("=" * 60)
    log.info("  🎉 ALL DONE! Fine-tuned model ready for use.")
    log.info(f"  Model path: {final_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
