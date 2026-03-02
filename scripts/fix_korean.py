#!/usr/bin/env python3
"""
Fix Korean Training — Use a different approach.
Download and test opus-mt-tc-big-en-ko with proper preprocessing.
"""

import time
import json
import gc
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fix-korean")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
TRAINING_DIR = BASE_DIR / "training_runs"
OUTPUT_DIR = TRAINING_DIR / "opus-mt-en-ko-finetuned"

NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
WARMUP_STEPS = 200
LOGGING_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 500
MAX_TRAIN_SAMPLES = 5000
MAX_EVAL_SAMPLES = 500


def main():
    log.info("═" * 60)
    log.info("  🇰🇷 Fix Korean Training — opus-mt-tc-big-en-ko")
    log.info("═" * 60)

    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        TrainerCallback,
    )
    from datasets import load_dataset

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    log.info(f"  Device: {device}")

    # Load model
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-ko"
    model_cache = MODELS_DIR / model_name.replace("/", "_")

    log.info(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(model_cache))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=str(model_cache))
    
    # Check the tokenizer and model config
    log.info(f"  Vocab size: {tokenizer.vocab_size}")
    log.info(f"  Model config decoder_start_token_id: {model.config.decoder_start_token_id}")
    log.info(f"  Model config pad_token_id: {model.config.pad_token_id}")
    
    # Test base model first
    log.info("\n  📋 Testing base model before training...")
    test_sentences = [
        "Hello, how are you today?",
        "Machine learning is the future of technology.",
        "I love programming and building AI systems.",
        "The weather is beautiful today.",
        "Can you help me translate this sentence?",
        "Artificial intelligence is transforming the world.",
    ]

    model.eval()
    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=MAX_LENGTH, num_beams=4)
        translated = tokenizer.decode(generated[0], skip_special_tokens=True)
        log.info(f"  EN: {sent}")
        log.info(f"  KO: {translated}")
        log.info("")

    # Load dataset
    log.info("\n  📊 Loading dataset...")
    ds_cache = DATASETS_DIR / "Helsinki-NLP_opus-100"

    raw_dataset = load_dataset(
        "Helsinki-NLP/opus-100", "en-ko",
        cache_dir=str(ds_cache),
        trust_remote_code=True,
    )

    log.info(f"  Dataset loaded!")
    for split_name, split_data in raw_dataset.items():
        log.info(f"    {split_name}: {len(split_data):,} samples")

    # Check sample
    sample = raw_dataset["train"][0]
    log.info(f"\n  Sample: {sample}")

    # Prepare data
    train_ds = raw_dataset["train"].shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
    eval_ds = raw_dataset["test"].shuffle(seed=42).select(range(min(MAX_EVAL_SAMPLES, len(raw_dataset["test"]))))

    def preprocess(examples):
        inputs = []
        targets = []
        for ex in examples["translation"]:
            en_text = ex.get("en", "")
            ko_text = ex.get("ko", "")
            if en_text and ko_text:
                inputs.append(en_text)
                targets.append(ko_text)
            else:
                inputs.append("")
                targets.append("")

        # For tc-big models, we need to handle the tokenization carefully
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

        # Use text_target for target tokenization (modern API)
        labels = tokenizer(
            text_target=targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

        # Replace padding token id's of the labels by -100 so they are not counted in the loss
        label_ids = labels["input_ids"]
        label_ids = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in label_ids
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    log.info("  🔧 Tokenizing...")
    tokenized_train = train_ds.map(
        preprocess, batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenize train (Korean)",
    )
    tokenized_eval = eval_ds.map(
        preprocess, batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenize eval (Korean)",
    )

    log.info(f"  ✅ Train: {len(tokenized_train)}, Eval: {len(tokenized_eval)}")

    # Verify tokenization
    sample_enc = tokenized_train[0]
    labels = [l for l in sample_enc["labels"] if l != -100]
    log.info(f"  Sample labels (non-pad): {len(labels)} tokens")
    decoded_labels = tokenizer.decode([l for l in sample_enc["labels"] if l >= 0], skip_special_tokens=True)
    log.info(f"  Decoded labels sample: {decoded_labels[:100]}")

    # Train
    log.info("\n  🚀 Starting training...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_steps = (len(tokenized_train) // BATCH_SIZE) * NUM_EPOCHS

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
        fp16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        use_cpu=(device == "cpu"),
        label_smoothing_factor=0.1,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    class ProgressCB(TrainerCallback):
        def __init__(self):
            self.start = time.time()
            self.entries = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.global_step > 0:
                elapsed = (time.time() - self.start) / 60
                entry = {"step": state.global_step, "epoch": round(state.epoch, 2), "elapsed_min": round(elapsed, 1)}
                if "loss" in logs:
                    entry["train_loss"] = round(logs["loss"], 4)
                    log.info(f"  🇰🇷 Step {state.global_step}/{total_steps} | Epoch {entry['epoch']} | Loss: {logs['loss']:.4f} | {entry['elapsed_min']}min")
                if "eval_loss" in logs:
                    entry["eval_loss"] = round(logs["eval_loss"], 4)
                    log.info(f"  🇰🇷 Eval Loss: {logs['eval_loss']:.4f}")
                self.entries.append(entry)

    cb = ProgressCB()
    model.train()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[cb],
    )

    train_result = trainer.train()
    total_time = time.time() - cb.start

    log.info(f"\n  ✅ Training complete!")
    log.info(f"     Loss: {train_result.training_loss:.4f}")
    log.info(f"     Steps: {train_result.global_step}")
    log.info(f"     Time: {total_time/60:.1f}min")

    # Save
    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"  💾 Saved: {final_dir}")

    # Save log
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump({
            "model": model_name,
            "dataset": "Helsinki-NLP/opus-100/en-ko",
            "language": "Korean",
            "language_code": "ko",
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
            "loss_history": cb.entries,
            "completed_at": datetime.utcnow().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    # Test fine-tuned model
    log.info("\n  🌐 Testing fine-tuned model...")
    ft_tokenizer = AutoTokenizer.from_pretrained(str(final_dir))
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(str(final_dir))
    ft_model.eval()

    results = []
    for sent in test_sentences:
        inputs = ft_tokenizer(sent, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        with torch.no_grad():
            generated = ft_model.generate(**inputs, max_length=MAX_LENGTH, num_beams=4)
        translated = ft_tokenizer.decode(generated[0], skip_special_tokens=True)
        log.info(f"  EN: {sent}")
        log.info(f"  KO: {translated}")
        log.info("")
        results.append({"en": sent, "ko": translated})

    with open(OUTPUT_DIR / "test_translations.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("═" * 60)
    log.info("  🎉 Korean training complete!")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
