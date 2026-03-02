#!/usr/bin/env python3
"""Train 3 DIRECT translation models using NLLB-200 fine-tuning.

FAST version - No synthetic generation. Uses only real parallel data:
  1. Bridge matching: JA→EN + EN→VI datasets share same EN → create JA→VI pairs
  2. Tatoeba: Direct parallel sentences
  3. OPUS-100 for EN→VI

Usage:
    PYTHONUNBUFFERED=1 python3 scripts/train_direct_models.py --lang ja-vi
    PYTHONUNBUFFERED=1 python3 scripts/train_direct_models.py --lang zh-vi
    PYTHONUNBUFFERED=1 python3 scripts/train_direct_models.py --lang en-vi
    PYTHONUNBUFFERED=1 python3 scripts/train_direct_models.py --lang all
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

# NLLB language codes
NLLB_CODES = {
    "en": "eng_Latn",
    "vi": "vie_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
}


def load_en_vi_data(max_samples=30000):
    """Load EN→VI parallel data from OPUS-100."""
    from datasets import load_dataset
    
    pairs = []
    print(f"  [EN→VI] Loading OPUS-100 en-vi...", flush=True)
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="train")
        count = 0
        for item in ds:
            if "translation" in item:
                src = item["translation"].get("en", "")
                tgt = item["translation"].get("vi", "")
                if src and tgt and len(src.strip()) > 3 and len(tgt.strip()) > 3:
                    pairs.append({"src": src.strip(), "tgt": tgt.strip()})
                    count += 1
            if count >= max_samples:
                break
        print(f"    → {count} EN→VI pairs loaded", flush=True)
    except Exception as e:
        print(f"    ✗ Failed: {e}", flush=True)
    
    return pairs


def _build_en_vi_lookup():
    """Build EN→VI lookup from OPUS-100."""
    from datasets import load_dataset
    
    print(f"  Loading EN→VI lookup from OPUS-100...", flush=True)
    en_vi = {}
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="train")
        for item in ds:
            if "translation" in item:
                en = item["translation"].get("en", "").strip()
                vi = item["translation"].get("vi", "").strip()
                if en and vi and len(en) > 3:
                    # Use normalized key for matching
                    key = en.lower().strip()
                    en_vi[key] = vi
        print(f"    → {len(en_vi)} EN→VI entries in lookup", flush=True)
    except Exception as e:
        print(f"    ✗ EN→VI lookup failed: {e}", flush=True)
    return en_vi


def load_ja_vi_data(max_samples=15000):
    """Load JA→VI data using bridge matching (no synthetic generation).
    
    Strategy: OPUS-100 has en-ja and en-vi. Find shared EN sentences
    to create JA→VI pairs directly.
    """
    from datasets import load_dataset
    
    pairs = []
    
    # Build EN→VI lookup
    en_vi = _build_en_vi_lookup()
    
    # Load JA→EN and match
    print(f"  [JA→VI] Loading OPUS-100 en-ja for bridge matching...", flush=True)
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train")
        matched = 0
        total = 0
        for item in ds:
            if "translation" in item:
                en = item["translation"].get("en", "").strip()
                ja = item["translation"].get("ja", "").strip()
                if en and ja and len(ja) > 2:
                    total += 1
                    key = en.lower().strip()
                    if key in en_vi:
                        pairs.append({"src": ja, "tgt": en_vi[key]})
                        matched += 1
                        if matched % 1000 == 0:
                            print(f"    → Matched {matched} JA→VI pairs (scanned {total})...", flush=True)
            if matched >= max_samples:
                break
        print(f"    → Total: {matched} JA→VI pairs from {total} JA→EN scanned", flush=True)
    except Exception as e:
        print(f"    ✗ JA→EN bridge failed: {e}", flush=True)
    
    # Try Tatoeba for additional JA→VI pairs
    print(f"  [JA→VI] Trying Tatoeba jpn-vie...", flush=True)
    try:
        ds = load_dataset("Helsinki-NLP/tatoeba_mt", "jpn-vie", split="test", trust_remote_code=True)
        tatoeba_count = 0
        for item in ds:
            src = item.get("sourceString", "").strip()
            tgt = item.get("targetString", "").strip()
            if src and tgt and len(src) > 2 and len(tgt) > 2:
                pairs.append({"src": src, "tgt": tgt})
                tatoeba_count += 1
        print(f"    → {tatoeba_count} JA→VI from Tatoeba", flush=True)
    except Exception as e:
        print(f"    → Tatoeba jpn-vie not available: {e}", flush=True)
    
    print(f"  [JA→VI] Total: {len(pairs)} training pairs", flush=True)
    return pairs[:max_samples]


def load_zh_vi_data(max_samples=15000):
    """Load ZH→VI data using bridge matching."""
    from datasets import load_dataset
    
    pairs = []
    
    # Build EN→VI lookup
    en_vi = _build_en_vi_lookup()
    
    # Load ZH→EN and match
    print(f"  [ZH→VI] Loading OPUS-100 en-zh for bridge matching...", flush=True)
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train")
        matched = 0
        total = 0
        for item in ds:
            if "translation" in item:
                en = item["translation"].get("en", "").strip()
                zh = item["translation"].get("zh", "").strip()
                if en and zh and len(zh) > 1:
                    total += 1
                    key = en.lower().strip()
                    if key in en_vi:
                        pairs.append({"src": zh, "tgt": en_vi[key]})
                        matched += 1
                        if matched % 1000 == 0:
                            print(f"    → Matched {matched} ZH→VI pairs (scanned {total})...", flush=True)
            if matched >= max_samples:
                break
        print(f"    → Total: {matched} ZH→VI pairs from {total} ZH→EN scanned", flush=True)
    except Exception as e:
        print(f"    ✗ ZH→EN bridge failed: {e}", flush=True)
    
    # Try Tatoeba for ZH→VI
    print(f"  [ZH→VI] Trying Tatoeba cmn-vie...", flush=True)
    try:
        ds = load_dataset("Helsinki-NLP/tatoeba_mt", "cmn-vie", split="test", trust_remote_code=True)
        tatoeba_count = 0
        for item in ds:
            src = item.get("sourceString", "").strip()
            tgt = item.get("targetString", "").strip()
            if src and tgt:
                pairs.append({"src": src, "tgt": tgt})
                tatoeba_count += 1
        print(f"    → {tatoeba_count} ZH→VI from Tatoeba", flush=True)
    except Exception as e:
        print(f"    → Tatoeba cmn-vie not available: {e}", flush=True)
    
    print(f"  [ZH→VI] Total: {len(pairs)} training pairs", flush=True)
    return pairs[:max_samples]


def train_nllb_direct(lang_pair, pairs, epochs=5, batch_size=4, lr=2e-5, max_length=64):
    """Fine-tune NLLB-200 for a direct language pair."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, TrainerCallback,
    )
    from datasets import Dataset
    
    src_lang, tgt_lang = lang_pair.split("-")
    src_nllb = NLLB_CODES[src_lang]
    tgt_nllb = NLLB_CODES[tgt_lang]
    
    output_dir = TRAINING_DIR / f"nllb-{lang_pair}-direct"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nllb_name = "facebook/nllb-200-distilled-600M"
    
    print(f"\n{'='*60}", flush=True)
    print(f"Training NLLB Direct: {src_nllb} → {tgt_nllb}", flush=True)
    print(f"Data: {len(pairs)} pairs, {epochs} epochs, batch={batch_size}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(nllb_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name)
    
    tokenizer.src_lang = src_nllb
    
    # Split data
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
    
    print("Tokenizing dataset...", flush=True)
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
        warmup_steps=300,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        predict_with_generate=False,
        fp16=False,
        use_cpu=True,  # Force CPU - NLLB too large for MPS
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
    
    print(f"\n🚀 Starting training...", flush=True)
    trainer.train()
    
    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    total_time = (time.time() - start_time) / 60
    
    log = {
        "type": "nllb_direct_finetune",
        "model": nllb_name,
        "lang_pair": f"{src_nllb}→{tgt_nllb}",
        "lang_pair_short": lang_pair,
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
    
    print(f"\n✅ Training complete: {lang_pair}!", flush=True)
    print(f"   Final loss: {log['result']['train_loss']}", flush=True)
    print(f"   Time: {total_time:.1f} minutes", flush=True)
    print(f"   Model: {final_dir}", flush=True)
    
    del model, tokenizer, trainer
    gc.collect()
    
    return str(final_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="all")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    CONFIGS = {
        "en-vi": {"loader": load_en_vi_data, "default_samples": 30000, "lr": 2e-5},
        "ja-vi": {"loader": load_ja_vi_data, "default_samples": 15000, "lr": 1e-5},
        "zh-vi": {"loader": load_zh_vi_data, "default_samples": 15000, "lr": 1e-5},
    }
    
    if args.lang == "all":
        pairs_to_train = list(CONFIGS.keys())
    elif args.lang in CONFIGS:
        pairs_to_train = [args.lang]
    else:
        print(f"Unknown: {args.lang}")
        sys.exit(1)
    
    for lang_pair in pairs_to_train:
        cfg = CONFIGS[lang_pair]
        samples = args.samples or cfg["default_samples"]
        
        print(f"\n{'#'*60}", flush=True)
        print(f"# Data: {lang_pair} (max {samples})", flush=True)
        print(f"{'#'*60}", flush=True)
        
        pairs = cfg["loader"](samples)
        if len(pairs) < 100:
            print(f"⚠ Only {len(pairs)} pairs for {lang_pair}, skipping", flush=True)
            continue
        
        train_nllb_direct(lang_pair, pairs, args.epochs, args.batch_size, cfg["lr"])
        gc.collect()
    
    print(f"\n{'='*60}", flush=True)
    print("All training complete! 🎉", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
