"""
Startup script: loads ALL trained models into registry and starts FastAPI server.
Supports: Vietnamese, Japanese, Chinese, Korean
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app, models_registry, datasets_registry, training_jobs

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE_DIR / "training_runs"
MODELS_DIR = BASE_DIR / "models"


# ═══════════════════════════════════════════════════════════════════
# Pre-registered fine-tuned models
TRAINED_MODELS = [
    # ── NLLB Direct models (best quality for direct translation) ──
    {
        "id": "nllb-en-vi-direct",
        "alias": "NLLB EN→VI Direct (30K)",
        "hf_name": "fine-tuned/nllb-en-vi-direct",
        "dir": "nllb-en-vi-direct/final",
        "type": "nllb",
    },
    {
        "id": "nllb-ja-vi-direct",
        "alias": "NLLB JA→VI Direct (20K)",
        "hf_name": "fine-tuned/nllb-ja-vi-direct",
        "dir": "nllb-ja-vi-direct/final",
        "type": "nllb",
    },
    {
        "id": "nllb-zh-vi-direct",
        "alias": "NLLB ZH→VI Direct (20K)",
        "hf_name": "fine-tuned/nllb-zh-vi-direct",
        "dir": "nllb-zh-vi-direct/final",
        "type": "nllb",
    },
    # ── M2M-100 Direct models ──
    {
        "id": "m2m100-en-vi-direct",
        "alias": "M2M-100 EN→VI Direct (30K)",
        "hf_name": "fine-tuned/m2m100-en-vi",
        "dir": "m2m100-en-vi-finetuned/final",
        "type": "m2m100",
    },
    # ── OpusMT EN→X fine-tuned models ──
    {
        "id": "ft-opus-en-vi-v2",
        "alias": "FT-OpusMT EN→VI v2 (20K)",
        "hf_name": "fine-tuned/opus-mt-en-vi-enhanced",
        "dir": "opus-mt-en-vi-enhanced/final",
    },
    {
        "id": "ft-opus-en-vi",
        "alias": "FT-OpusMT EN→VI",
        "hf_name": "fine-tuned/opus-mt-en-vi",
        "dir": "opus-mt-en-vi-finetuned/final",
    },
    {
        "id": "ft-opus-en-ja",
        "alias": "FT-OpusMT EN→JA",
        "hf_name": "fine-tuned/opus-mt-en-jap",
        "dir": "opus-mt-en-ja-finetuned/final",
    },
    {
        "id": "ft-opus-en-zh",
        "alias": "FT-OpusMT EN→ZH",
        "hf_name": "fine-tuned/opus-mt-en-zh",
        "dir": "opus-mt-en-zh-finetuned/final",
    },
]

# Base model for Korean (tc-big model already translates well without fine-tuning)
BASE_MODELS = [
    {
        "id": "base-opus-en-ko",
        "alias": "OpusMT-tc-big EN→KO",
        "hf_name": "Helsinki-NLP/opus-mt-tc-big-en-ko",
        "cache_dir": "Helsinki-NLP_opus-mt-tc-big-en-ko",
    },
]


def preload():
    """Pre-load all trained and base models into the registry."""

    # ── Fine-tuned models ────────────────────────────────────────────
    for m in TRAINED_MODELS:
        ft_dir = TRAINING_DIR / m["dir"]
        if ft_dir.exists() and (ft_dir / "config.json").exists():
            models_registry[m["id"]] = {
                "id": m["id"],
                "hf_name": m["hf_name"],
                "alias": m["alias"],
                "status": "ready",
                "progress": 100,
                "local_path": str(ft_dir),
                "created_at": "2026-02-25T13:45:28Z",
                "size_gb": None,
                "error": None,
            }
            print(f"✅ Loaded fine-tuned: {m['alias']} → {ft_dir}")
        else:
            print(f"⚠️  Not found: {m['alias']} at {ft_dir}")

    # ── Base models (pre-trained, no fine-tuning needed) ─────────────
    for m in BASE_MODELS:
        cache_dir = MODELS_DIR / m["cache_dir"]
        if cache_dir.exists():
            models_registry[m["id"]] = {
                "id": m["id"],
                "hf_name": m["hf_name"],
                "alias": m["alias"],
                "status": "ready",
                "progress": 100,
                "local_path": m["hf_name"],  # Use HF name for AutoModel loading
                "created_at": "2026-02-25T14:00:00Z",
                "size_gb": None,
                "error": None,
            }
            print(f"✅ Loaded base model: {m['alias']}")
        else:
            print(f"⚠️  Not cached: {m['alias']}")

    # ── Training jobs ────────────────────────────────────────────────
    # Load from training_log.json files
    job_configs = [
        ("auto-train-vi", "opus-mt-en-vi-finetuned", "ft-opus-en-vi", "opus-mt-en-vi", "en", "vi"),
        ("auto-train-ja", "opus-mt-en-ja-finetuned", "ft-opus-en-ja", "opus-mt-en-jap", "en", "ja"),
        ("auto-train-zh", "opus-mt-en-zh-finetuned", "ft-opus-en-zh", "opus-mt-en-zh", "en", "zh"),
    ]

    for job_id, dirname, model_id, model_name, src, tgt in job_configs:
        log_file = TRAINING_DIR / dirname / "training_log.json"
        if log_file.exists():
            with open(log_file) as f:
                tlog = json.load(f)
            training_jobs[job_id] = {
                "id": job_id,
                "model_id": model_id,
                "model_name": model_name,
                "dataset_id": f"opus-100-{src}{tgt}",
                "dataset_name": tlog.get("dataset", f"Helsinki-NLP/opus-100"),
                "source_lang": src,
                "target_lang": tgt,
                "config": tlog.get("config", {}),
                "status": "completed",
                "progress": 100,
                "current_epoch": tlog["config"].get("epochs", 3),
                "current_step": tlog["result"]["total_steps"],
                "total_steps": tlog["result"]["total_steps"],
                "train_loss": tlog["result"]["train_loss"],
                "eval_loss": None,
                "eval_bleu": None,
                "loss_history": tlog.get("loss_history", []),
                "started_at": "auto",
                "finished_at": tlog.get("completed_at"),
                "output_dir": str(TRAINING_DIR / dirname),
                "error": None,
            }
            # Extract last eval_loss from loss_history
            for entry in reversed(tlog.get("loss_history", [])):
                if "eval_loss" in entry:
                    training_jobs[job_id]["eval_loss"] = entry["eval_loss"]
                    break
            print(f"✅ Loaded training job: {job_id} ({src}→{tgt})")

    print(f"\n📊 Summary: {len(models_registry)} models, {len(training_jobs)} training jobs registered")


preload()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
