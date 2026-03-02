#!/usr/bin/env python3
"""
Multi-Language Training Script — Agent-Translate
Trains translation models for Japanese, Chinese, and Korean from English.

Languages:
  🇯🇵 English → Japanese  (Helsinki-NLP/opus-mt-en-jap)
  🇨🇳 English → Chinese   (Helsinki-NLP/opus-mt-en-zh)
  🇰🇷 English → Korean    (Helsinki-NLP/opus-mt-en-ko)

Datasets: Helsinki-NLP/opus-100 for each language pair
"""

import os
import sys
import time
import json
import logging
import gc
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multi-train")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
TRAINING_DIR = BASE_DIR / "training_runs"

for d in [MODELS_DIR, DATASETS_DIR, TRAINING_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Language configurations
# ═══════════════════════════════════════════════════════════════════
LANGUAGES = [
    {
        "name": "Japanese",
        "flag": "🇯🇵",
        "code": "ja",
        "model_name": "Helsinki-NLP/opus-mt-en-jap",
        "dataset_name": "Helsinki-NLP/opus-100",
        "dataset_pair": "en-ja",
        "src_lang": "en",
        "tgt_lang": "ja",
        "test_sentences": [
            "Hello, how are you today?",
            "Machine learning is the future of technology.",
            "I love programming and building AI systems.",
            "The weather is beautiful today.",
            "Can you help me translate this sentence?",
            "Artificial intelligence is transforming the world.",
        ],
    },
    {
        "name": "Chinese",
        "flag": "🇨🇳",
        "code": "zh",
        "model_name": "Helsinki-NLP/opus-mt-en-zh",
        "dataset_name": "Helsinki-NLP/opus-100",
        "dataset_pair": "en-zh",
        "src_lang": "en",
        "tgt_lang": "zh",
        "test_sentences": [
            "Hello, how are you today?",
            "Machine learning is the future of technology.",
            "I love programming and building AI systems.",
            "The weather is beautiful today.",
            "Can you help me translate this sentence?",
            "Artificial intelligence is transforming the world.",
        ],
    },
    {
        "name": "Korean",
        "flag": "🇰🇷",
        "code": "ko",
        "model_name": "Helsinki-NLP/opus-mt-en-ko",
        "dataset_name": "Helsinki-NLP/opus-100",
        "dataset_pair": "en-ko",
        "src_lang": "en",
        "tgt_lang": "ko",
        "test_sentences": [
            "Hello, how are you today?",
            "Machine learning is the future of technology.",
            "I love programming and building AI systems.",
            "The weather is beautiful today.",
            "Can you help me translate this sentence?",
            "Artificial intelligence is transforming the world.",
        ],
    },
]

# Training hyperparameters
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


def train_single_language(lang_config):
    """Train a single language pair."""
    name = lang_config["name"]
    flag = lang_config["flag"]
    code = lang_config["code"]
    model_name = lang_config["model_name"]
    dataset_name = lang_config["dataset_name"]
    dataset_pair = lang_config["dataset_pair"]
    src_lang = lang_config["src_lang"]
    tgt_lang = lang_config["tgt_lang"]
    test_sentences = lang_config["test_sentences"]

    output_dir = TRAINING_DIR / f"opus-mt-en-{code}-finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("═" * 64)
    log.info(f"  {flag}  Training: English → {name}")
    log.info(f"  Model  : {model_name}")
    log.info(f"  Dataset: {dataset_name} ({dataset_pair})")
    log.info(f"  Output : {output_dir}")
    log.info("═" * 64)

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

    # ── Download model ───────────────────────────────────────────────
    log.info(f"\n  📦 Downloading model {model_name}...")
    model_cache = MODELS_DIR / model_name.replace("/", "_")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(model_cache))
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=str(model_cache))
    except Exception as e:
        # Try alternative model names
        alt_names = [
            f"Helsinki-NLP/opus-mt-en-{code}",
            f"Helsinki-NLP/opus-mt-tc-big-en-{code}",
        ]
        loaded = False
        for alt in alt_names:
            if alt == model_name:
                continue
            try:
                log.info(f"  ⚠️  {model_name} not found, trying {alt}...")
                model_cache = MODELS_DIR / alt.replace("/", "_")
                tokenizer = AutoTokenizer.from_pretrained(alt, cache_dir=str(model_cache))
                model = AutoModelForSeq2SeqLM.from_pretrained(alt, cache_dir=str(model_cache))
                model_name = alt
                lang_config["model_name"] = alt
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            log.error(f"  ❌ Could not find a model for English → {name}. Skipping.")
            return None

    params = sum(p.numel() for p in model.parameters())
    log.info(f"  ✅ Model loaded: {model_name}")
    log.info(f"     Parameters: {params:,}")

    # ── Download dataset ─────────────────────────────────────────────
    log.info(f"\n  📊 Downloading dataset {dataset_name} ({dataset_pair})...")
    ds_cache = DATASETS_DIR / dataset_name.replace("/", "_")

    try:
        raw_dataset = load_dataset(
            dataset_name, dataset_pair,
            cache_dir=str(ds_cache),
            trust_remote_code=True,
        )
    except Exception as e:
        log.warning(f"  ⚠️  Failed with pair '{dataset_pair}': {e}")
        try:
            # Try reversed pair
            rev_pair = f"{tgt_lang}-{src_lang}"
            log.info(f"  Trying reversed pair: {rev_pair}")
            raw_dataset = load_dataset(
                dataset_name, rev_pair,
                cache_dir=str(ds_cache),
                trust_remote_code=True,
            )
        except Exception as e2:
            log.error(f"  ❌ Could not load dataset for {name}: {e2}")
            return None

    log.info(f"  ✅ Dataset loaded!")
    for split_name, split_data in raw_dataset.items():
        log.info(f"     {split_name}: {len(split_data):,} samples")

    # ── Tokenize ─────────────────────────────────────────────────────
    log.info(f"\n  🔧 Tokenizing...")

    train_ds = raw_dataset["train"]
    if MAX_TRAIN_SAMPLES and len(train_ds) > MAX_TRAIN_SAMPLES:
        train_ds = train_ds.shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
        log.info(f"     Using {MAX_TRAIN_SAMPLES} train samples")

    if "test" in raw_dataset:
        eval_ds = raw_dataset["test"]
    elif "validation" in raw_dataset:
        eval_ds = raw_dataset["validation"]
    else:
        splits = train_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = splits["train"]
        eval_ds = splits["test"]

    if MAX_EVAL_SAMPLES and len(eval_ds) > MAX_EVAL_SAMPLES:
        eval_ds = eval_ds.shuffle(seed=42).select(range(MAX_EVAL_SAMPLES))

    # Detect data format
    sample = train_ds[0]
    log.info(f"     Sample keys: {list(sample.keys())}")
    if "translation" in sample:
        log.info(f"     Sample: {sample['translation']}")

    def preprocess_function(examples):
        if "translation" in examples:
            inputs = []
            targets = []
            for ex in examples["translation"]:
                # Handle different key names
                src_text = ex.get(src_lang, ex.get("en", ""))
                tgt_text = ex.get(tgt_lang, ex.get(code, ""))
                inputs.append(src_text)
                targets.append(tgt_text)
        elif src_lang in examples and tgt_lang in examples:
            inputs = examples[src_lang]
            targets = examples[tgt_lang]
        else:
            cols = list(examples.keys())
            inputs = examples[cols[0]]
            targets = examples[cols[1]]

        model_inputs = tokenizer(
            inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length",
        )
        labels = tokenizer(
            text_target=targets, max_length=MAX_LENGTH, truncation=True, padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(
        preprocess_function, batched=True,
        remove_columns=train_ds.column_names, desc=f"Tokenize train ({name})",
    )
    tokenized_eval = eval_ds.map(
        preprocess_function, batched=True,
        remove_columns=eval_ds.column_names, desc=f"Tokenize eval ({name})",
    )

    log.info(f"  ✅ Tokenization complete: {len(tokenized_train)} train, {len(tokenized_eval)} eval")

    # ── Train ────────────────────────────────────────────────────────
    log.info(f"\n  🚀 Starting training for {flag} {name}...")

    total_steps = (len(tokenized_train) // BATCH_SIZE) * NUM_EPOCHS

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
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
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.start_time = time.time()
            self.log_entries = []

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
                        f"  {flag} Step {state.global_step}/{total_steps} | "
                        f"Epoch {entry['epoch']} | "
                        f"Loss: {logs['loss']:.4f} | "
                        f"Time: {entry['elapsed_min']}min"
                    )
                if "eval_loss" in logs:
                    entry["eval_loss"] = round(logs["eval_loss"], 4)
                    log.info(f"  {flag} Eval Loss: {logs['eval_loss']:.4f}")
                self.log_entries.append(entry)

    progress_cb = ProgressCallback()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[progress_cb],
    )

    train_result = trainer.train()

    train_loss = train_result.training_loss
    total_time = time.time() - progress_cb.start_time

    log.info(f"\n  ✅ {flag} Training completed!")
    log.info(f"     Train Loss: {train_loss:.4f}")
    log.info(f"     Steps: {train_result.global_step}")
    log.info(f"     Time: {total_time/60:.1f} minutes")

    # ── Save model ───────────────────────────────────────────────────
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"  💾 Model saved: {final_dir}")

    # Save training log
    log_data = {
        "model": model_name,
        "dataset": f"{dataset_name}/{dataset_pair}",
        "language": name,
        "language_code": code,
        "config": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "train_samples": len(tokenized_train),
            "eval_samples": len(tokenized_eval),
        },
        "result": {
            "train_loss": round(train_loss, 4),
            "total_steps": train_result.global_step,
            "total_time_minutes": round(total_time / 60, 1),
        },
        "loss_history": progress_cb.log_entries,
        "completed_at": datetime.utcnow().isoformat(),
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # ── Test translation ─────────────────────────────────────────────
    log.info(f"\n  🌐 Testing {flag} translations...")
    import torch as th

    ft_tokenizer = AutoTokenizer.from_pretrained(str(final_dir))
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(str(final_dir))
    ft_model.eval()

    results = []
    log.info(f"  {'─' * 50}")
    for sent in test_sentences:
        inputs = ft_tokenizer(sent, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        with th.no_grad():
            generated = ft_model.generate(**inputs, max_length=MAX_LENGTH, num_beams=4)
        translated = ft_tokenizer.decode(generated[0], skip_special_tokens=True)
        log.info(f"  EN: {sent}")
        log.info(f"  {code.upper()}: {translated}")
        log.info("")
        results.append({"en": sent, code: translated})

    # Save test results
    with open(output_dir / "test_translations.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Clean up memory
    del model, trainer, ft_model, tokenizer, ft_tokenizer
    del tokenized_train, tokenized_eval, train_ds, eval_ds
    gc.collect()
    if device == "mps":
        th.mps.empty_cache()
    elif device == "cuda":
        th.cuda.empty_cache()

    return log_data


def main():
    log.info("╔" + "═" * 62 + "╗")
    log.info("║  Agent-Translate — Multi-Language Training Pipeline         ║")
    log.info("║  Languages: 🇯🇵 Japanese  🇨🇳 Chinese  🇰🇷 Korean             ║")
    log.info("╚" + "═" * 62 + "╝")

    all_results = {}
    start_all = time.time()

    for i, lang in enumerate(LANGUAGES):
        log.info(f"\n{'▓' * 64}")
        log.info(f"  [{i+1}/{len(LANGUAGES)}] Training {lang['flag']} {lang['name']}...")
        log.info(f"{'▓' * 64}")

        result = train_single_language(lang)
        if result:
            all_results[lang["code"]] = result
            log.info(f"\n  ✅ {lang['flag']} {lang['name']} — DONE! "
                     f"(Loss: {result['result']['train_loss']:.4f}, "
                     f"Time: {result['result']['total_time_minutes']:.1f}min)")
        else:
            log.error(f"\n  ❌ {lang['flag']} {lang['name']} — FAILED!")

    total_time = time.time() - start_all

    # Summary
    log.info("\n")
    log.info("╔" + "═" * 62 + "╗")
    log.info("║              TRAINING COMPLETE — SUMMARY                    ║")
    log.info("╠" + "═" * 62 + "╣")
    for code, result in all_results.items():
        lang_name = result["language"]
        loss = result["result"]["train_loss"]
        t = result["result"]["total_time_minutes"]
        status = "✅"
        log.info(f"║  {status} {lang_name:12s} | Loss: {loss:.4f} | Time: {t:.1f}min")
    
    for lang in LANGUAGES:
        if lang["code"] not in all_results:
            log.info(f"║  ❌ {lang['name']:12s} | FAILED")

    log.info(f"╠{'═' * 62}╣")
    log.info(f"║  Total Time: {total_time/60:.1f} minutes")
    log.info(f"║  Models saved in: {TRAINING_DIR}")
    log.info("╚" + "═" * 62 + "╝")

    # Save overall summary
    summary = {
        "total_time_minutes": round(total_time / 60, 1),
        "completed_at": datetime.utcnow().isoformat(),
        "results": all_results,
    }
    with open(TRAINING_DIR / "multilang_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"\n  Summary saved: {TRAINING_DIR / 'multilang_training_summary.json'}")


if __name__ == "__main__":
    main()
