#!/usr/bin/env python3
"""Real-time learning system for translation models.

This system allows the model to learn from user corrections:
1. User corrects a translation in the UI
2. Correction is saved as a training pair (original → corrected)
3. When enough corrections accumulate, the model fine-tunes on them
4. The improved model is hot-swapped into the running system

This implements "online learning" / "continuous improvement".
"""

import json
import os
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CORRECTIONS_DIR = BASE_DIR / "corrections"
CORRECTIONS_DIR.mkdir(exist_ok=True)
TRAINING_DIR = BASE_DIR / "training_runs"


class RealtimeLearner:
    """Manages real-time learning from user corrections."""
    
    # Minimum corrections before triggering re-training
    MIN_CORRECTIONS_FOR_TRAIN = 50
    
    def __init__(self):
        self.corrections_file = CORRECTIONS_DIR / "corrections.jsonl"
        self.stats_file = CORRECTIONS_DIR / "stats.json"
        self._lock = threading.Lock()
        self._load_stats()
    
    def _load_stats(self):
        """Load correction statistics."""
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                self.stats = json.load(f)
        else:
            self.stats = {
                "total_corrections": 0,
                "corrections_since_last_train": 0,
                "last_train_at": None,
                "train_count": 0,
                "lang_pair_counts": {},
            }
    
    def _save_stats(self):
        """Save stats to disk."""
        with open(self.stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)
    
    def add_correction(self, 
                       original_text: str, 
                       machine_translation: str,
                       corrected_text: str, 
                       source_lang: str, 
                       target_lang: str,
                       job_id: str = None,
                       segment_idx: int = None) -> Dict:
        """Add a user correction.
        
        Returns status dict with correction count and whether training is triggered.
        """
        with self._lock:
            correction = {
                "timestamp": datetime.utcnow().isoformat(),
                "original": original_text,
                "machine_translation": machine_translation,
                "corrected": corrected_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "job_id": job_id,
                "segment_idx": segment_idx,
            }
            
            # Append to JSONL file
            with open(self.corrections_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(correction, ensure_ascii=False) + "\n")
            
            # Update stats
            lang_pair = f"{source_lang}-{target_lang}"
            self.stats["total_corrections"] += 1
            self.stats["corrections_since_last_train"] += 1
            self.stats["lang_pair_counts"][lang_pair] = \
                self.stats["lang_pair_counts"].get(lang_pair, 0) + 1
            self._save_stats()
            
            # Check if we should trigger training
            should_train = self.stats["corrections_since_last_train"] >= self.MIN_CORRECTIONS_FOR_TRAIN
            
            return {
                "total_corrections": self.stats["total_corrections"],
                "since_last_train": self.stats["corrections_since_last_train"],
                "train_threshold": self.MIN_CORRECTIONS_FOR_TRAIN,
                "should_train": should_train,
                "lang_pair": lang_pair,
            }
    
    def get_corrections(self, lang_pair: str = None, limit: int = None) -> List[Dict]:
        """Get stored corrections, optionally filtered by language pair."""
        corrections = []
        if not self.corrections_file.exists():
            return corrections
        
        with open(self.corrections_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                    if lang_pair:
                        pair = f"{c['source_lang']}-{c['target_lang']}"
                        if pair != lang_pair:
                            continue
                    corrections.append(c)
                except json.JSONDecodeError:
                    continue
        
        if limit:
            corrections = corrections[-limit:]
        
        return corrections
    
    def get_stats(self) -> Dict:
        """Get current learning statistics."""
        return {
            **self.stats,
            "corrections_file": str(self.corrections_file),
            "file_exists": self.corrections_file.exists(),
        }
    
    def trigger_training(self, lang_pair: str = None) -> Dict:
        """Trigger incremental training on accumulated corrections.
        
        This runs in background and returns immediately.
        """
        corrections = self.get_corrections(lang_pair)
        if len(corrections) < 10:
            return {"status": "skipped", "reason": f"Only {len(corrections)} corrections, need at least 10"}
        
        # Mark training started
        self.stats["last_train_at"] = datetime.utcnow().isoformat()
        self.stats["train_count"] += 1
        self.stats["corrections_since_last_train"] = 0
        self._save_stats()
        
        # Run training in background
        thread = threading.Thread(
            target=self._run_incremental_training,
            args=(corrections, lang_pair),
            daemon=True,
        )
        thread.start()
        
        return {
            "status": "training_started",
            "corrections_count": len(corrections),
            "lang_pair": lang_pair,
            "train_number": self.stats["train_count"],
        }
    
    def _run_incremental_training(self, corrections: List[Dict], lang_pair: str = None):
        """Run incremental training on corrections (background thread)."""
        try:
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForSeq2SeqLM,
                Seq2SeqTrainer, Seq2SeqTrainingArguments,
                DataCollatorForSeq2Seq,
            )
            from datasets import Dataset
            
            # Group corrections by language pair
            pairs_by_lang = {}
            for c in corrections:
                pair = f"{c['source_lang']}-{c['target_lang']}"
                if pair not in pairs_by_lang:
                    pairs_by_lang[pair] = []
                pairs_by_lang[pair].append({
                    "src": c["original"],
                    "tgt": c["corrected"],
                })
            
            # Train each language pair
            for pair, train_pairs in pairs_by_lang.items():
                if len(train_pairs) < 5:
                    continue
                
                src, tgt = pair.split("-")
                print(f"\n[RealtimeLearner] Training {pair} with {len(train_pairs)} corrections...")
                
                # Determine which model to use
                if src == "en":
                    # Use OpusMT
                    model_dir = TRAINING_DIR / f"opus-mt-{pair}-finetuned" / "final"
                    if not model_dir.exists():
                        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
                    else:
                        model_name = str(model_dir)
                else:
                    # Use NLLB
                    model_name = "facebook/nllb-200-distilled-600M"
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
                
                # Duplicate corrections to amplify signal (small dataset trick)
                amplified = train_pairs * 3  # Repeat 3x
                
                def tokenize_fn(examples):
                    inputs = tokenizer(
                        examples["src"], max_length=128, truncation=True, padding="max_length"
                    )
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(
                            examples["tgt"], max_length=128, truncation=True, padding="max_length"
                        )
                    inputs["labels"] = labels["input_ids"]
                    return inputs
                
                ds = Dataset.from_list(amplified).map(tokenize_fn, batched=True, batch_size=50)
                
                output_dir = TRAINING_DIR / f"realtime-{pair}-v{self.stats['train_count']}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                training_args = Seq2SeqTrainingArguments(
                    output_dir=str(output_dir / "checkpoints"),
                    num_train_epochs=3,  # Quick training
                    per_device_train_batch_size=4,
                    learning_rate=5e-6,  # Low LR to not overfit
                    weight_decay=0.01,
                    logging_steps=10,
                    save_total_limit=1,
                    report_to="none",
                    dataloader_num_workers=0,
                )
                
                trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=ds,
                    tokenizer=tokenizer,
                    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                )
                
                trainer.train()
                
                final_dir = output_dir / "final"
                trainer.save_model(str(final_dir))
                tokenizer.save_pretrained(str(final_dir))
                
                # Save log
                with open(output_dir / "training_log.json", "w") as f:
                    json.dump({
                        "type": "realtime_learning",
                        "lang_pair": pair,
                        "corrections_used": len(train_pairs),
                        "trained_at": datetime.utcnow().isoformat(),
                        "base_model": model_name,
                    }, f, indent=2)
                
                print(f"[RealtimeLearner] ✅ {pair} model updated at {final_dir}")
                
                del model, tokenizer, trainer
                import gc; gc.collect()
                
        except Exception as e:
            print(f"[RealtimeLearner] ❌ Training failed: {e}")
            import traceback
            traceback.print_exc()


# Global instance
realtime_learner = RealtimeLearner()
