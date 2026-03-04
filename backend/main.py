"""
Agent-Translate Backend — FastAPI Server
Handles model management, dataset management, training pipeline, and translation.
"""

import os
import sys
import json
import time
import uuid
import shutil
import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import psutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent-translate")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
TRAINING_DIR = BASE_DIR / "training_runs"
LOGS_DIR = BASE_DIR / "logs"

for d in [MODELS_DIR, DATASETS_DIR, TRAINING_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agent-Translate API",
    description="Translation Model Training & Inference System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state (replace with DB in production)
# ---------------------------------------------------------------------------
models_registry: Dict[str, Dict] = {}
datasets_registry: Dict[str, Dict] = {}
training_jobs: Dict[str, Dict] = {}

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ModelDownloadRequest(BaseModel):
    model_name: str = Field(..., description="HuggingFace model ID, e.g. 'facebook/mbart-large-50-many-to-many-mmt'")
    alias: Optional[str] = Field(None, description="Friendly name for the model")

class DatasetDownloadRequest(BaseModel):
    dataset_name: str = Field(..., description="HuggingFace dataset ID, e.g. 'opus100'")
    source_lang: str = Field("en", description="Source language code")
    target_lang: str = Field("vi", description="Target language code")
    split: str = Field("train", description="Dataset split to download")
    max_samples: Optional[int] = Field(None, description="Max number of samples to download")

class TrainingRequest(BaseModel):
    model_id: str = Field(..., description="ID of the model to fine-tune")
    dataset_id: str = Field(..., description="ID of the dataset to use")
    source_lang: str = Field("en", description="Source language code")
    target_lang: str = Field("vi", description="Target language code")
    num_epochs: int = Field(3, description="Number of training epochs")
    batch_size: int = Field(8, description="Training batch size")
    learning_rate: float = Field(5e-5, description="Learning rate")
    max_length: int = Field(128, description="Max sequence length")
    warmup_steps: int = Field(500, description="Number of warmup steps")
    save_steps: int = Field(1000, description="Save checkpoint every N steps")
    use_lora: bool = Field(True, description="Use LoRA for efficient fine-tuning")
    lora_r: int = Field(16, description="LoRA rank")
    lora_alpha: int = Field(32, description="LoRA alpha")

class TranslateRequest(BaseModel):
    model_id: str = Field(..., description="ID of the model to use for translation")
    text: str = Field(..., description="Text to translate")
    source_lang: str = Field("en", description="Source language code")
    target_lang: str = Field("vi", description="Target language code")
    max_length: int = Field(512, description="Max generation length")

class CustomDatasetUpload(BaseModel):
    name: str = Field(..., description="Dataset name")
    source_lang: str = Field("en")
    target_lang: str = Field("vi")

# ---------------------------------------------------------------------------
# Helper: system info
# ---------------------------------------------------------------------------
def get_system_info() -> Dict:
    cpu_pct = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(BASE_DIR))
    gpu_info = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(torch.cuda.get_device_properties(i).total_mem / 1e9, 2),
                    "memory_used_gb": round(torch.cuda.memory_allocated(i) / 1e9, 2),
                })
        elif torch.backends.mps.is_available():
            gpu_info.append({
                "id": 0,
                "name": "Apple MPS",
                "memory_total_gb": round(mem.total / 1e9, 2),
                "memory_used_gb": "N/A",
            })
    except Exception:
        pass
    return {
        "cpu_percent": cpu_pct,
        "memory_total_gb": round(mem.total / 1e9, 2),
        "memory_used_gb": round(mem.used / 1e9, 2),
        "memory_percent": mem.percent,
        "disk_total_gb": round(disk.total / 1e9, 2),
        "disk_used_gb": round(disk.used / 1e9, 2),
        "disk_percent": round(disk.used / disk.total * 100, 1),
        "gpu": gpu_info,
    }

# ---------------------------------------------------------------------------
# Routes — Health / System
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/system")
async def system_info():
    return get_system_info()

# ---------------------------------------------------------------------------
# Routes — Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    return {"models": list(models_registry.values())}

@app.post("/api/models/download")
async def download_model(req: ModelDownloadRequest, background_tasks: BackgroundTasks):
    model_id = str(uuid.uuid4())[:8]
    alias = req.alias or req.model_name.split("/")[-1]
    entry = {
        "id": model_id,
        "hf_name": req.model_name,
        "alias": alias,
        "status": "downloading",
        "progress": 0,
        "local_path": str(MODELS_DIR / model_id),
        "created_at": datetime.utcnow().isoformat(),
        "size_gb": None,
        "error": None,
    }
    models_registry[model_id] = entry
    background_tasks.add_task(_download_model_task, model_id, req.model_name)
    return {"model_id": model_id, "message": f"Started downloading {req.model_name}"}


def _download_model_task(model_id: str, hf_name: str):
    try:
        from huggingface_hub import snapshot_download
        local = str(MODELS_DIR / model_id)
        logger.info(f"Downloading model {hf_name} → {local}")
        models_registry[model_id]["status"] = "downloading"
        models_registry[model_id]["progress"] = 10
        snapshot_download(
            repo_id=hf_name,
            local_dir=local,
            ignore_patterns=["*.bin", "*.safetensors"] if False else None,
        )
        # Calculate size
        total_size = sum(f.stat().st_size for f in Path(local).rglob("*") if f.is_file())
        models_registry[model_id]["size_gb"] = round(total_size / 1e9, 2)
        models_registry[model_id]["status"] = "ready"
        models_registry[model_id]["progress"] = 100
        logger.info(f"Model {hf_name} downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download model {hf_name}: {e}")
        models_registry[model_id]["status"] = "error"
        models_registry[model_id]["error"] = str(e)


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    if model_id not in models_registry:
        raise HTTPException(404, "Model not found")
    local_path = Path(models_registry[model_id]["local_path"])
    if local_path.exists():
        shutil.rmtree(local_path)
    del models_registry[model_id]
    return {"message": "Model deleted"}

# ---------------------------------------------------------------------------
# Routes — Datasets
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
async def list_datasets():
    return {"datasets": list(datasets_registry.values())}

@app.post("/api/datasets/download")
async def download_dataset(req: DatasetDownloadRequest, background_tasks: BackgroundTasks):
    ds_id = str(uuid.uuid4())[:8]
    entry = {
        "id": ds_id,
        "hf_name": req.dataset_name,
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
        "split": req.split,
        "max_samples": req.max_samples,
        "status": "downloading",
        "progress": 0,
        "num_samples": None,
        "local_path": str(DATASETS_DIR / ds_id),
        "created_at": datetime.utcnow().isoformat(),
        "error": None,
    }
    datasets_registry[ds_id] = entry
    background_tasks.add_task(_download_dataset_task, ds_id, req)
    return {"dataset_id": ds_id, "message": f"Started downloading {req.dataset_name}"}


def _download_dataset_task(ds_id: str, req: DatasetDownloadRequest):
    try:
        from datasets import load_dataset
        logger.info(f"Downloading dataset {req.dataset_name}")
        datasets_registry[ds_id]["progress"] = 20

        lang_pair = f"{req.source_lang}-{req.target_lang}"
        try:
            ds = load_dataset(req.dataset_name, lang_pair, split=req.split, trust_remote_code=True)
        except Exception:
            try:
                ds = load_dataset(req.dataset_name, split=req.split, trust_remote_code=True)
            except Exception:
                ds = load_dataset(req.dataset_name, f"{req.target_lang}-{req.source_lang}", split=req.split, trust_remote_code=True)

        datasets_registry[ds_id]["progress"] = 60

        if req.max_samples and len(ds) > req.max_samples:
            ds = ds.select(range(req.max_samples))

        out_dir = Path(DATASETS_DIR / ds_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out_dir))

        datasets_registry[ds_id]["num_samples"] = len(ds)
        datasets_registry[ds_id]["status"] = "ready"
        datasets_registry[ds_id]["progress"] = 100
        logger.info(f"Dataset {req.dataset_name} downloaded: {len(ds)} samples")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        datasets_registry[ds_id]["status"] = "error"
        datasets_registry[ds_id]["error"] = str(e)


@app.post("/api/datasets/upload")
async def upload_dataset(
    name: str = Form(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("vi"),
    file: UploadFile = File(...),
):
    ds_id = str(uuid.uuid4())[:8]
    out_dir = DATASETS_DIR / ds_id
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Try to parse the file
    num_samples = 0
    try:
        if file.filename.endswith(".json") or file.filename.endswith(".jsonl"):
            with open(file_path, "r") as f:
                if file.filename.endswith(".jsonl"):
                    num_samples = sum(1 for _ in f)
                else:
                    data = json.load(f)
                    num_samples = len(data) if isinstance(data, list) else 0
        elif file.filename.endswith(".csv") or file.filename.endswith(".tsv"):
            with open(file_path, "r") as f:
                num_samples = sum(1 for _ in f) - 1  # minus header
    except Exception:
        pass

    entry = {
        "id": ds_id,
        "hf_name": f"custom/{name}",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "split": "custom",
        "max_samples": None,
        "status": "ready",
        "progress": 100,
        "num_samples": num_samples,
        "local_path": str(out_dir),
        "created_at": datetime.utcnow().isoformat(),
        "error": None,
    }
    datasets_registry[ds_id] = entry
    return {"dataset_id": ds_id, "message": f"Dataset '{name}' uploaded successfully"}


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    if dataset_id not in datasets_registry:
        raise HTTPException(404, "Dataset not found")
    local_path = Path(datasets_registry[dataset_id]["local_path"])
    if local_path.exists():
        shutil.rmtree(local_path)
    del datasets_registry[dataset_id]
    return {"message": "Dataset deleted"}

# ---------------------------------------------------------------------------
# Routes — Training
# ---------------------------------------------------------------------------

@app.get("/api/training")
async def list_training_jobs():
    return {"jobs": list(training_jobs.values())}

@app.get("/api/training/{job_id}")
async def get_training_job(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(404, "Training job not found")
    return training_jobs[job_id]

@app.post("/api/training/start")
async def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    if req.model_id not in models_registry:
        raise HTTPException(404, "Model not found")
    if req.dataset_id not in datasets_registry:
        raise HTTPException(404, "Dataset not found")

    model = models_registry[req.model_id]
    dataset = datasets_registry[req.dataset_id]

    if model["status"] != "ready":
        raise HTTPException(400, "Model is not ready")
    if dataset["status"] != "ready":
        raise HTTPException(400, "Dataset is not ready")

    job_id = str(uuid.uuid4())[:8]
    job = {
        "id": job_id,
        "model_id": req.model_id,
        "model_name": model["alias"],
        "dataset_id": req.dataset_id,
        "dataset_name": dataset["hf_name"],
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
        "config": {
            "num_epochs": req.num_epochs,
            "batch_size": req.batch_size,
            "learning_rate": req.learning_rate,
            "max_length": req.max_length,
            "warmup_steps": req.warmup_steps,
            "save_steps": req.save_steps,
            "use_lora": req.use_lora,
            "lora_r": req.lora_r,
            "lora_alpha": req.lora_alpha,
        },
        "status": "initializing",
        "progress": 0,
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "train_loss": None,
        "eval_loss": None,
        "eval_bleu": None,
        "loss_history": [],
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
        "output_dir": str(TRAINING_DIR / job_id),
        "error": None,
    }
    training_jobs[job_id] = job
    background_tasks.add_task(_training_task, job_id, req)
    return {"job_id": job_id, "message": "Training job started"}


def _training_task(job_id: str, req: TrainingRequest):
    """Background training task."""
    job = training_jobs[job_id]
    try:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq,
        )
        from datasets import load_from_disk

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Training on device: {device}")

        # Load model & tokenizer
        job["status"] = "loading_model"
        job["progress"] = 5
        model_path = models_registry[req.model_id]["local_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)

        if req.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=req.lora_r,
                    lora_alpha=req.lora_alpha,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj"],
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            except Exception as e:
                logger.warning(f"LoRA failed, training full model: {e}")

        # Load dataset
        job["status"] = "loading_dataset"
        job["progress"] = 15
        ds_path = datasets_registry[req.dataset_id]["local_path"]
        dataset = load_from_disk(ds_path)

        # Tokenize
        job["status"] = "tokenizing"
        job["progress"] = 20
        src_lang = req.source_lang
        tgt_lang = req.target_lang

        def preprocess(examples):
            if "translation" in examples:
                inputs = [ex.get(src_lang, "") for ex in examples["translation"]]
                targets = [ex.get(tgt_lang, "") for ex in examples["translation"]]
            elif src_lang in examples and tgt_lang in examples:
                inputs = examples[src_lang]
                targets = examples[tgt_lang]
            else:
                # Fallback: use first two columns
                cols = list(examples.keys())
                inputs = examples[cols[0]]
                targets = examples[cols[1]]

            model_inputs = tokenizer(inputs, max_length=req.max_length, truncation=True, padding="max_length")
            labels = tokenizer(targets, max_length=req.max_length, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

        # Split 90/10
        split = tokenized.train_test_split(test_size=0.1, seed=42)

        # Training arguments
        output_dir = str(TRAINING_DIR / job_id)
        total_steps = (len(split["train"]) // req.batch_size) * req.num_epochs
        job["total_steps"] = total_steps

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=req.num_epochs,
            per_device_train_batch_size=req.batch_size,
            per_device_eval_batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            warmup_steps=req.warmup_steps,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=req.save_steps,
            save_steps=req.save_steps,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Custom callback for progress
        from transformers import TrainerCallback

        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if state.global_step > 0:
                    pct = min(95, 25 + int(70 * state.global_step / max(total_steps, 1)))
                    job["progress"] = pct
                    job["current_step"] = state.global_step
                    job["current_epoch"] = state.epoch or 0
                    if logs:
                        if "loss" in logs:
                            job["train_loss"] = round(logs["loss"], 4)
                            job["loss_history"].append({
                                "step": state.global_step,
                                "loss": round(logs["loss"], 4),
                            })
                        if "eval_loss" in logs:
                            job["eval_loss"] = round(logs["eval_loss"], 4)

        job["status"] = "training"
        job["progress"] = 25

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[ProgressCallback()],
        )

        trainer.train()

        # Save final model
        job["status"] = "saving"
        job["progress"] = 95
        trainer.save_model(output_dir + "/final")
        tokenizer.save_pretrained(output_dir + "/final")

        # Register the fine-tuned model
        ft_model_id = f"ft-{job_id}"
        models_registry[ft_model_id] = {
            "id": ft_model_id,
            "hf_name": f"fine-tuned/{models_registry[req.model_id]['alias']}",
            "alias": f"FT-{models_registry[req.model_id]['alias']}-{src_lang}-{tgt_lang}",
            "status": "ready",
            "progress": 100,
            "local_path": output_dir + "/final",
            "created_at": datetime.utcnow().isoformat(),
            "size_gb": None,
            "error": None,
        }

        job["status"] = "completed"
        job["progress"] = 100
        job["finished_at"] = datetime.utcnow().isoformat()
        logger.info(f"Training job {job_id} completed")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
        job["status"] = "failed"
        job["error"] = str(e)


@app.post("/api/training/{job_id}/stop")
async def stop_training(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(404, "Training job not found")
    training_jobs[job_id]["status"] = "stopped"
    return {"message": "Stop signal sent to training job"}

# ---------------------------------------------------------------------------
# Routes — Translation
# ---------------------------------------------------------------------------

@app.post("/api/translate")
async def translate(req: TranslateRequest):
    if req.model_id not in models_registry:
        raise HTTPException(404, "Model not found")
    model_entry = models_registry[req.model_id]
    if model_entry["status"] != "ready":
        raise HTTPException(400, "Model is not ready")

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_path = model_entry["local_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)

        # Try to set source/target lang for mBART-like models
        try:
            tokenizer.src_lang = req.source_lang
            tokenizer.tgt_lang = req.target_lang
        except Exception:
            pass

        inputs = tokenizer(req.text, return_tensors="pt", max_length=req.max_length, truncation=True).to(device)

        # Generate translation
        with torch.no_grad():
            try:
                generated = model.generate(
                    **inputs,
                    max_length=req.max_length,
                    num_beams=5,
                    forced_bos_token_id=tokenizer.lang_code_to_id.get(req.target_lang),
                )
            except Exception:
                generated = model.generate(
                    **inputs,
                    max_length=req.max_length,
                    num_beams=5,
                )

        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        return {
            "original": req.text,
            "translated": translated,
            "source_lang": req.source_lang,
            "target_lang": req.target_lang,
            "model_used": model_entry["alias"],
        }
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Translation failed: {str(e)}")

# ---------------------------------------------------------------------------
# Routes — Settings / API Keys
# ---------------------------------------------------------------------------
_settings_file = BASE_DIR / "settings.json"
_settings: Dict[str, str] = {}

# Load saved settings on startup
if _settings_file.exists():
    try:
        import json as _json
        _settings = _json.loads(_settings_file.read_text())
        if _settings.get("gemini_api_key"):
            logger.info("✅ Loaded Gemini API key from settings.json")
    except Exception:
        pass

class SettingsUpdate(BaseModel):
    gemini_api_key: Optional[str] = None

@app.get("/api/settings")
async def get_settings():
    return {"gemini_api_key": bool(_settings.get("gemini_api_key"))}

@app.post("/api/settings")
async def update_settings(s: SettingsUpdate):
    if s.gemini_api_key is not None:
        _settings["gemini_api_key"] = s.gemini_api_key
        # Save to file for persistence
        import json as _json
        _settings_file.write_text(_json.dumps(_settings))
        logger.info("✅ Gemini API key saved to settings.json")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes — Real-time Learning (User Corrections)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR / "scripts"))
try:
    from realtime_learner import realtime_learner
    logger.info("✅ Real-time learning system loaded")
except Exception as e:
    realtime_learner = None
    logger.warning(f"⚠ Real-time learning not available: {e}")

class CorrectionRequest(BaseModel):
    job_id: str = Field(..., description="Subtitle job ID")
    segment_idx: int = Field(..., description="Segment index")
    original_text: str = Field(..., description="Original text")
    machine_translation: str = Field(..., description="Machine translation")
    corrected_text: str = Field(..., description="User-corrected translation")
    source_lang: str = Field("auto", description="Source language")
    target_lang: str = Field("vi", description="Target language")

@app.post("/api/corrections")
async def add_correction(req: CorrectionRequest):
    """Save a user correction for real-time learning."""
    if not realtime_learner:
        raise HTTPException(500, "Real-time learning not available")
    
    result = realtime_learner.add_correction(
        original_text=req.original_text,
        machine_translation=req.machine_translation,
        corrected_text=req.corrected_text,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        job_id=req.job_id,
        segment_idx=req.segment_idx,
    )
    
    # Also update the job's translated segments if job exists
    if req.job_id in subtitle_jobs:
        job = subtitle_jobs[req.job_id]
        segs = job.get("translated_segments", [])
        if 0 <= req.segment_idx < len(segs):
            segs[req.segment_idx]["text"] = req.corrected_text
            segs[req.segment_idx]["user_corrected"] = True
    
    return result

@app.get("/api/corrections/stats")
async def get_correction_stats():
    """Get real-time learning statistics."""
    if not realtime_learner:
        return {"error": "Real-time learning not available"}
    return realtime_learner.get_stats()

@app.post("/api/corrections/train")
async def trigger_correction_training(background_tasks: BackgroundTasks):
    """Trigger incremental training on accumulated corrections."""
    if not realtime_learner:
        raise HTTPException(500, "Real-time learning not available")
    result = realtime_learner.trigger_training()
    return result

# ---------------------------------------------------------------------------
# Routes — Subtitle Pipeline
# ---------------------------------------------------------------------------

SUBTITLES_DIR = BASE_DIR / "subtitles"
SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)

subtitle_jobs: Dict[str, Dict] = {}

PIPELINE_NODES = [
    {"id": "input",      "label": "📥 Input",       "desc": "YouTube URL or video file"},
    {"id": "download",   "label": "⬇️ Download",    "desc": "Download & extract audio"},
    {"id": "transcribe", "label": "🎤 Transcribe",  "desc": "Speech-to-text (Whisper)"},
    {"id": "detect",     "label": "🔍 Detect Lang", "desc": "Language detection (Whisper)"},
    {"id": "translate",  "label": "🔄 Translate",   "desc": "Multi-model local translation"},
    {"id": "output",     "label": "📄 Output",      "desc": "Generate SRT files"},
]

# NLLB language code mapping
NLLB_LANG_CODES = {
    "en": "eng_Latn", "vi": "vie_Latn", "ja": "jpn_Jpan",
    "zh": "zho_Hans", "ko": "kor_Hang", "fr": "fra_Latn",
    "de": "deu_Latn", "es": "spa_Latn", "th": "tha_Thai",
    "id": "ind_Latn", "ru": "rus_Cyrl", "pt": "por_Latn",
    "ar": "arb_Arab", "hi": "hin_Deva", "it": "ita_Latn",
}


def _make_pipeline():
    """Create a fresh pipeline state."""
    return {n["id"]: {"status": "pending", "started_at": None, "ended_at": None, "meta": {}} for n in PIPELINE_NODES}


def _pipeline_set(job, node_id, status, **meta):
    """Update a pipeline node's status."""
    node = job["pipeline"][node_id]
    node["status"] = status
    if status == "running" and not node["started_at"]:
        node["started_at"] = datetime.utcnow().isoformat()
    if status in ("completed", "failed", "skipped"):
        node["ended_at"] = datetime.utcnow().isoformat()
    node["meta"].update(meta)
    # Also set top-level status to the current active node
    if status == "running":
        job["status"] = node_id
    elif status == "failed":
        job["status"] = "failed"


def _find_best_model(source_lang: str, target_lang: str) -> Optional[str]:
    """Auto-find the best translation model for a language pair.
    STRICT matching: source AND target must both match.
    Priority: NLLB direct fine-tuned > OpusMT fine-tuned > base models
    """
    candidates = []
    for mid, m in models_registry.items():
        if m["status"] != "ready":
            continue
        alias = (m.get("alias") or "").lower()
        hf = (m.get("hf_name") or "").lower()
        src = source_lang.lower()
        tgt = target_lang.lower()

        # STRICT: Both source AND target must match
        src_match = (f"{src}→" in alias or f"{src}-" in hf or
                     f"/{src}-" in hf or hf.startswith(f"{src}-"))
        tgt_match = (f"→{tgt}" in alias or f"-{tgt}" in hf or
                     hf.endswith(f"-{tgt}"))

        if src_match and tgt_match:
            # Priority: 0=NLLB direct, 1=fine-tuned, 2=base
            if "nllb" in mid and "direct" in mid:
                priority = 0
            elif "ft-" in mid or "fine" in hf:
                priority = 1
            else:
                priority = 2
            candidates.append((priority, mid))

    if candidates:
        candidates.sort()
        logger.info(f"_find_best_model({source_lang}→{target_lang}): found {candidates[0][1]} (from {len(candidates)} candidates)")
        return candidates[0][1]

    logger.info(f"_find_best_model({source_lang}→{target_lang}): no model found")
    return None


class SubtitleExtractRequest(BaseModel):
    youtube_url: Optional[str] = Field(None, description="YouTube video URL")
    source_lang: str = Field("auto", description="Source language (auto for auto-detect)")
    target_lang: str = Field("vi", description="Target language for translation")
    model_id: Optional[str] = Field(None, description="Translation model ID (auto-selected if empty)")
    whisper_model: str = Field("base", description="Whisper model size")
    enable_refine: bool = Field(False, description="Enable LLM refinement step")

class SubtitleTranslateRequest(BaseModel):
    job_id: str = Field(..., description="Subtitle job ID to translate")
    target_lang: str = Field("vi", description="Target language")
    model_id: Optional[str] = Field(None, description="Translation model ID (auto if empty)")
    enable_refine: bool = Field(False, description="Enable LLM refinement step")


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: list) -> str:
    """Convert whisper segments to SRT format."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_lines)


def _parse_srt(srt_content: str) -> list:
    """Parse SRT content into a list of segments."""
    import re
    segments = []
    blocks = re.split(r'\n\n+', srt_content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                idx = int(lines[0])
                time_parts = lines[1].strip()
                text = '\n'.join(lines[2:])
                # Parse timestamps
                match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', time_parts)
                if match:
                    segments.append({
                        "index": idx,
                        "start_time": match.group(1),
                        "end_time": match.group(2),
                        "text": text,
                    })
            except (ValueError, IndexError):
                continue
    return segments


@app.get("/api/subtitles/pipeline-config")
async def get_pipeline_config():
    return {"nodes": PIPELINE_NODES}

@app.get("/api/subtitles")
async def list_subtitle_jobs():
    return {"jobs": list(subtitle_jobs.values())}


@app.get("/api/subtitles/{job_id}")
async def get_subtitle_job(job_id: str):
    if job_id not in subtitle_jobs:
        raise HTTPException(404, "Subtitle job not found")
    return subtitle_jobs[job_id]


@app.post("/api/subtitles/extract")
async def extract_subtitles(req: SubtitleExtractRequest, background_tasks: BackgroundTasks):
    """Extract subtitles from YouTube URL and optionally translate."""
    if not req.youtube_url:
        raise HTTPException(400, "YouTube URL is required")

    job_id = str(uuid.uuid4())[:8]
    job_dir = SUBTITLES_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    job = {
        "id": job_id,
        "youtube_url": req.youtube_url,
        "video_title": None,
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
        "model_id": req.model_id,
        "whisper_model": req.whisper_model,
        "enable_refine": req.enable_refine,
        "status": "input",
        "progress": 0,
        "original_srt": None,
        "translated_srt": None,
        "refined_srt": None,
        "segments": [],
        "translated_segments": [],
        "duration": None,
        "created_at": datetime.utcnow().isoformat(),
        "error": None,
        "pipeline": _make_pipeline(),
    }
    _pipeline_set(job, "input", "completed", source="youtube", url=req.youtube_url)
    subtitle_jobs[job_id] = job
    background_tasks.add_task(_subtitle_extract_task, job_id, req)
    return {"job_id": job_id, "message": "Subtitle extraction started"}


@app.post("/api/subtitles/upload")
async def upload_video_for_subtitles(
    source_lang: str = Form("auto"),
    target_lang: str = Form("vi"),
    model_id: str = Form(""),
    whisper_model: str = Form("base"),
    enable_refine: bool = Form(False),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Upload a video file for subtitle extraction."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = SUBTITLES_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_path = job_dir / file.filename
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    job = {
        "id": job_id,
        "youtube_url": None,
        "video_title": file.filename,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model_id": model_id or None,
        "whisper_model": whisper_model,
        "enable_refine": enable_refine,
        "status": "input",
        "progress": 5,
        "original_srt": None,
        "translated_srt": None,
        "refined_srt": None,
        "segments": [],
        "translated_segments": [],
        "duration": None,
        "created_at": datetime.utcnow().isoformat(),
        "error": None,
        "pipeline": _make_pipeline(),
    }
    _pipeline_set(job, "input", "completed", source="upload", filename=file.filename)
    _pipeline_set(job, "download", "completed")
    subtitle_jobs[job_id] = job

    req = SubtitleExtractRequest(
        source_lang=source_lang,
        target_lang=target_lang,
        model_id=model_id or None,
        whisper_model=whisper_model,
        enable_refine=enable_refine,
    )
    background_tasks.add_task(_subtitle_process_video, job_id, str(video_path), req)
    return {"job_id": job_id, "message": "Video uploaded, processing started"}


def _subtitle_extract_task(job_id: str, req: SubtitleExtractRequest):
    """Background: download YouTube video, extract audio, transcribe, translate, refine."""
    job = subtitle_jobs[job_id]
    job_dir = SUBTITLES_DIR / job_id

    try:
        # ── Node: download ──
        _pipeline_set(job, "download", "running")
        job["progress"] = 5
        logger.info(f"[Subtitle {job_id}] Downloading: {req.youtube_url}")

        import yt_dlp
        import imageio_ffmpeg
        import subprocess

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(job_dir / 'audio_raw.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.youtube_url, download=True)
            job["video_title"] = info.get("title", "Unknown")
            job["duration"] = info.get("duration")

        job["progress"] = 15

        # Convert to WAV using bundled ffmpeg
        raw_audio = None
        for f in job_dir.iterdir():
            if f.name.startswith('audio_raw'):
                raw_audio = str(f)
                break
        if not raw_audio:
            raise Exception("Could not find downloaded audio file")

        audio_path = str(job_dir / "audio.wav")
        result = subprocess.run(
            [ffmpeg_exe, '-i', raw_audio, '-ar', '16000', '-ac', '1', '-y', audio_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise Exception(f"ffmpeg conversion failed: {result.stderr[:500]}")

        _pipeline_set(job, "download", "completed", title=job["video_title"], duration=job["duration"])
        job["progress"] = 25
        duration_str = f"{job['duration']//60}:{job['duration']%60:02d}" if job.get('duration') else 'N/A'
        logger.info(f"[Subtitle {job_id}] ✅ Download complete: \"{job['video_title']}\" ({duration_str})")

        # Continue with transcription pipeline
        _subtitle_process_video(job_id, audio_path, req)

    except Exception as e:
        logger.error(f"[Subtitle {job_id}] Failed: {e}", exc_info=True)
        _pipeline_set(job, "download", "failed", error=str(e))
        job["status"] = "failed"
        job["error"] = str(e)


def _subtitle_process_video(job_id: str, audio_path: str, req: SubtitleExtractRequest):
    """Process video/audio: transcribe → detect → translate → refine."""
    job = subtitle_jobs[job_id]
    job_dir = SUBTITLES_DIR / job_id

    try:
        # ── Node: transcribe ──
        _pipeline_set(job, "transcribe", "running")
        job["progress"] = 30
        logger.info(f"══════════════════════════════════════════════")
        logger.info(f"[Subtitle {job_id}] 🎤 Transcribing with Whisper model: {req.whisper_model}")
        logger.info(f"══════════════════════════════════════════════")

        from faster_whisper import WhisperModel

        whisper = WhisperModel(
            req.whisper_model,
            device="cpu",
            compute_type="int8",
        )

        segments_iter, info = whisper.transcribe(
            audio_path,
            language=req.source_lang if req.source_lang != "auto" else None,
            beam_size=5,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        )

        detected_lang = info.language
        job["detected_lang"] = detected_lang
        job["source_lang"] = detected_lang
        job["progress"] = 45

        segments = []
        job["live_transcriptions"] = []  # Realtime transcription display
        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            # Send to frontend in realtime
            job["live_transcriptions"].append({
                "idx": len(segments) - 1,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            job["segment_count"] = len(segments)
            # Log each segment as it's transcribed
            start_t = f"{int(seg.start)//60}:{int(seg.start)%60:02d}"
            end_t = f"{int(seg.end)//60}:{int(seg.end)%60:02d}"
            logger.info(f"[Subtitle {job_id}] 🎤 [{len(segments):3d}] [{start_t}-{end_t}] {seg.text.strip()[:80]}")

        if not segments:
            raise Exception("No speech detected in the audio")

        job["segments"] = segments

        # Generate original SRT
        original_srt = _segments_to_srt(segments)
        srt_path = job_dir / "original.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(original_srt)
        job["original_srt"] = original_srt

        _pipeline_set(job, "transcribe", "completed",
                       detected_lang=detected_lang,
                       probability=round(info.language_probability, 2),
                       segment_count=len(segments))
        job["progress"] = 50
        logger.info(f"══════════════════════════════════════════════")
        logger.info(f"[Subtitle {job_id}] ✅ Transcribed {len(segments)} segments | Language: {detected_lang} ({info.language_probability:.0%})")
        logger.info(f"══════════════════════════════════════════════")

        # Clean up whisper
        del whisper
        import gc
        gc.collect()

        target_lang = req.target_lang

        if detected_lang == target_lang:
            _pipeline_set(job, "detect", "skipped", reason="source == target")
            _pipeline_set(job, "translate", "skipped", reason="source == target")
            _pipeline_set(job, "output", "completed")
            job["status"] = "completed"
            job["progress"] = 100
            logger.info(f"[Subtitle {job_id}] Source=target ({detected_lang}), done")
        else:
            # ── Node: detect + translate ──
            model_id = req.model_id
            if not model_id or model_id not in models_registry:
                model_id = _find_best_model(detected_lang, target_lang)
                if model_id:
                    job["model_id"] = model_id

            _translate_srt_segments(job_id, target_lang, fallback_model_id=model_id)

            # ── Node: refine (disabled by default) ──
            if req.enable_refine and job["status"] != "failed":
                _summarize_transcript(job_id, target_lang)
                _refine_with_context(job_id, target_lang)

            # ── Node: output ──
            if job["status"] != "failed":
                _generate_final_output(job_id, target_lang)

    except Exception as e:
        logger.error(f"[Subtitle {job_id}] Processing failed: {e}", exc_info=True)
        job["status"] = "failed"
        job["error"] = str(e)


def _summarize_transcript(job_id: str, target_lang: str):
    """Step 2: Summarize transcript to build context for better translation.
    Uses NLLB to translate full text → target language in chunks,
    then extracts key terms and context for the refine step.
    """
    job = subtitle_jobs[job_id]
    _pipeline_set(job, "summarize", "running")
    job["progress"] = 52

    try:
        segments = job["segments"]
        full_text = " ".join(seg["text"] for seg in segments)

        # Extract key information from transcript
        # 1. Get unique words/phrases (top frequency terms)
        import re
        from collections import Counter

        # Split into words, filter short ones
        words = re.findall(r'\b\w{3,}\b', full_text.lower())
        word_freq = Counter(words)
        key_terms = [w for w, c in word_freq.most_common(30) if c >= 2]

        # 2. Get first few sentences as context summary
        first_sentences = " ".join(seg["text"] for seg in segments[:min(10, len(segments))])

        # 3. Detect main topics from segment content
        total_chars = len(full_text)
        total_segments = len(segments)
        avg_len = total_chars / max(total_segments, 1)

        summary = {
            "total_segments": total_segments,
            "total_characters": total_chars,
            "avg_segment_length": round(avg_len, 1),
            "key_terms": key_terms[:20],
            "opening_context": first_sentences[:500],
            "detected_lang": job.get("detected_lang", "unknown"),
        }

        job["transcript_summary"] = summary
        _pipeline_set(job, "summarize", "completed",
                       key_terms_count=len(key_terms),
                       total_chars=total_chars,
                       output=summary)
        job["progress"] = 55
        logger.info(f"[Subtitle {job_id}] Summarized: {total_segments} segments, {len(key_terms)} key terms")

    except Exception as e:
        logger.warning(f"[Subtitle {job_id}] Summarize failed (non-critical): {e}")
        job["transcript_summary"] = {}
        _pipeline_set(job, "summarize", "completed", note="basic mode")


def _detect_segment_language(text: str, whisper_lang: str = None) -> str:
    """Detect language of a text segment.
    Uses character analysis for CJK scripts (more reliable than langdetect for short text),
    then falls back to langdetect, then to whisper_lang.
    """
    text = text.strip()
    if len(text) < 2:
        return whisper_lang or "unknown"

    # Character-based detection for CJK (more reliable for short subtitle segments)
    import unicodedata
    cjk_counts = {"ja": 0, "zh": 0, "ko": 0, "latin": 0}
    for ch in text:
        name = unicodedata.name(ch, "")
        if "HIRAGANA" in name or "KATAKANA" in name:
            cjk_counts["ja"] += 3  # Strong signal for Japanese
        elif "CJK" in name or "IDEOGRAPH" in name:
            cjk_counts["zh"] += 1  # Could be JA/ZH, ambiguous
            cjk_counts["ja"] += 0.5  # Japanese also uses kanji
        elif "HANGUL" in name:
            cjk_counts["ko"] += 3
        elif ch.isascii() and ch.isalpha():
            cjk_counts["latin"] += 1

    total_chars = sum(cjk_counts.values()) or 1

    # If significant CJK characters detected, use character-based detection
    if cjk_counts["ja"] > 2 or (cjk_counts["ja"] > 0 and cjk_counts["ja"] / total_chars > 0.3):
        return "ja"
    if cjk_counts["ko"] > 2:
        return "ko"
    if cjk_counts["zh"] > 2 and cjk_counts["ja"] == 0:
        return "zh"

    # For Latin-script text, use langdetect
    if cjk_counts["latin"] > 3:
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            return detect(text)
        except Exception:
            pass

    # Fallback: trust Whisper's language detection
    if whisper_lang:
        return whisper_lang

    # Last resort: try langdetect
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "unknown"


def _translate_srt_segments(job_id: str, target_lang: str, fallback_model_id: str = None):
    """Translate SRT segments — fully automatic.
    
    SIMPLE RULES:
      1. Detect language per segment (CJK-aware + Whisper fallback)
      2. EN segments + we have local EN→target model → local model (fast)
      3. ALL other languages → Gemini LLM (reliable)
      4. Same-as-target segments → keep as-is
    """
    job = subtitle_jobs[job_id]
    job_dir = SUBTITLES_DIR / job_id

    LANG_NAMES = {
        "en": "English", "vi": "Vietnamese", "ja": "Japanese",
        "zh": "Chinese", "ko": "Korean", "fr": "French",
        "de": "German", "es": "Spanish", "th": "Thai",
        "id": "Indonesian", "ru": "Russian", "pt": "Portuguese",
    }
    target_name = LANG_NAMES.get(target_lang, target_lang)

    try:
        segments = job["segments"]
        total = len(segments)

        # ═══ STEP 1: Language Detection ═══
        _pipeline_set(job, "detect", "running")
        whisper_lang = job.get("detected_lang") or job.get("source_lang", "en")
        logger.info(f"══════════════════════════════════════════════")
        logger.info(f"[Subtitle {job_id}] 🔍 Detecting language for {total} segments...")
        logger.info(f"[Subtitle {job_id}] 🔍 Whisper detected: {whisper_lang}")

        # Use Whisper's whole-file detection for all segments (faster & more accurate than per-segment)
        segment_langs = [whisper_lang] * total

        from collections import Counter
        lang_counts = Counter(segment_langs)
        logger.info(f"[Subtitle {job_id}] Language distribution: {dict(lang_counts)}")
        job["lang_distribution"] = dict(lang_counts)

        _pipeline_set(job, "detect", "completed",
                       distribution=dict(lang_counts),
                       note=f"Whisper detected: {whisper_lang}")
        job["progress"] = 55

        # ═══ STEP 2: Classify segments into LOCAL_EN vs NLLB vs SKIP ═══
        _pipeline_set(job, "translate", "running")

        # Find local EN→target model ONLY
        local_en_model_id = _find_best_model("en", target_lang) or fallback_model_id
        if local_en_model_id and local_en_model_id in models_registry:
            alias = (models_registry[local_en_model_id].get("alias") or "").lower()
            hf = (models_registry[local_en_model_id].get("hf_name") or "").lower()
            is_en_source = ("en→" in alias or "en-" in hf or "en→" in alias.replace("->", "→"))
            if not is_en_source:
                local_en_model_id = None

        local_indices = []    # EN segments → local OpusMT
        nllb_indices = []     # Non-EN segments → NLLB-200
        skip_indices = []     # Same-as-target → keep original

        for i, lang in enumerate(segment_langs):
            if lang == target_lang:
                skip_indices.append(i)
            elif lang == "en" and local_en_model_id:
                local_indices.append(i)
            else:
                nllb_indices.append(i)

        logger.info(f"══════════════════════════════════════════════")
        logger.info(f"[Subtitle {job_id}] 🔄 Starting translation: {len(local_indices)} local EN + {len(nllb_indices)} NLLB + {len(skip_indices)} skip")
        logger.info(f"══════════════════════════════════════════════")

        # ═══ STEP 3a: Local OpusMT for EN segments ═══
        all_translated = [""] * total

        for i in skip_indices:
            all_translated[i] = segments[i]["text"]

        # Initialize live_translations for realtime display
        job["live_translations"] = []
        for i, seg in enumerate(segments):
            status = "skip" if i in set(skip_indices) else "pending"
            job["live_translations"].append({
                "idx": i,
                "original": seg["text"],
                "translated": all_translated[i] if status == "skip" else "",
                "lang": segment_langs[i],
                "status": status,
            })

        if local_indices and local_en_model_id:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_entry = models_registry[local_en_model_id]
            model_path = model_entry["local_path"]
            alias = model_entry.get('alias', local_en_model_id)
            logger.info(f"[Subtitle {job_id}] Loading local model for EN→{target_lang}: {alias}")

            # Detect if model is NLLB or M2M-100 (both need forced_bos_token_id)
            is_nllb = 'nllb' in model_path.lower() or 'nllb' in alias.lower()
            is_m2m = 'm2m' in model_path.lower() or 'm2m' in alias.lower()
            needs_forced_bos = is_nllb or is_m2m
            logger.info(f"[Subtitle {job_id}] Model type: is_nllb={is_nllb}, is_m2m={is_m2m}, path={model_path}")
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
            mdl.eval()

            # NLLB/M2M requires forced_bos_token_id to generate the target language
            nllb_generate_kwargs = {}
            if needs_forced_bos:
                if is_nllb:
                    tok.src_lang = "eng_Latn"
                    target_code = NLLB_LANG_CODES.get(target_lang, "vie_Latn")
                    forced_bos_id = tok.convert_tokens_to_ids(target_code)
                else:  # M2M-100
                    tok.src_lang = "en"
                    target_code = target_lang
                    forced_bos_id = tok.lang_code_to_id.get(target_lang, tok.convert_tokens_to_ids(f"__{target_lang}__"))
                nllb_generate_kwargs = {"forced_bos_token_id": forced_bos_id}
                logger.info(f"[Subtitle {job_id}] {'NLLB' if is_nllb else 'M2M-100'} mode: target={target_code}, forced_bos_id={forced_bos_id}")

            # Batch translate EN segments
            EN_BATCH_SIZE = 8
            for batch_start in range(0, len(local_indices), EN_BATCH_SIZE):
                batch_end = min(batch_start + EN_BATCH_SIZE, len(local_indices))
                batch_idx = local_indices[batch_start:batch_end]
                batch_texts = [segments[i]["text"] for i in batch_idx]

                try:
                    inputs = tok(batch_texts, return_tensors="pt", max_length=256, truncation=True, padding=True)
                    with torch.no_grad():
                        generated = mdl.generate(**inputs, max_length=256, num_beams=2, length_penalty=1.0, **nllb_generate_kwargs)
                    translations = tok.batch_decode(generated, skip_special_tokens=True)

                    for j, idx in enumerate(batch_idx):
                        translated = translations[j].strip() if j < len(translations) else segments[idx]["text"]
                        all_translated[idx] = translated
                        job["live_translations"][idx]["translated"] = translated
                        job["live_translations"][idx]["status"] = "done"
                        logger.info(f"[Subtitle {job_id}] [{idx+1}/{total}] 🇬🇧→🇻🇳 \"{segments[idx]['text'][:60]}\" → \"{translated[:60]}\"")
                except Exception as e:
                    logger.warning(f"[Subtitle {job_id}] EN batch {batch_start} failed: {e}, falling back")
                    for idx in batch_idx:
                        try:
                            inp = tok(segments[idx]["text"], return_tensors="pt", max_length=256, truncation=True)
                            with torch.no_grad():
                                out = mdl.generate(**inp, max_length=256, num_beams=2)
                            all_translated[idx] = tok.decode(out[0], skip_special_tokens=True).strip()
                            job["live_translations"][idx]["translated"] = all_translated[idx]
                            job["live_translations"][idx]["status"] = "done"
                        except:
                            all_translated[idx] = segments[idx]["text"]
                            job["live_translations"][idx]["status"] = "error"

                job["progress"] = 55 + int(10 * batch_end / len(local_indices))

            del tok, mdl
            import gc; gc.collect()
            logger.info(f"[Subtitle {job_id}] Local model translated {len(local_indices)} EN segments (batched)")
        elif local_indices:
            nllb_indices.extend(local_indices)
            local_indices = []

        job["progress"] = 65

        # ═══ STEP 3b: Translation for non-EN segments ═══
        # Strategy: CJK (ja/zh/ko) → PIVOT via English (JA→EN→VI) for better quality
        #           Other langs → Direct NLLB translation
        if nllb_indices:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from collections import defaultdict

            nllb_model_name = "facebook/nllb-200-distilled-600M"
            logger.info(f"[Subtitle {job_id}] Loading NLLB-200 for {len(nllb_indices)} segments...")

            nllb_tok = AutoTokenizer.from_pretrained(nllb_model_name)
            nllb_mdl = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
            nllb_mdl.eval()

            target_nllb = NLLB_LANG_CODES.get(target_lang, "vie_Latn")
            target_token_id = nllb_tok.convert_tokens_to_ids(target_nllb)
            en_nllb = "eng_Latn"
            en_token_id = nllb_tok.convert_tokens_to_ids(en_nllb)

            # Separate segments by strategy:
            # 1. Direct NLLB fine-tuned model (best quality) → direct_model_indices
            # 2. CJK pivot JA→EN→VI (fallback) → pivot_indices
            # 3. Other languages → direct_indices (generic NLLB)
            CJK_LANGS = {"ja", "zh", "ko"}

            # Check which CJK langs have direct fine-tuned models
            direct_model_map = {}  # lang → (model_id, model_path)
            for lang in CJK_LANGS:
                dm = _find_best_model(lang, target_lang)
                if dm and dm in models_registry and "nllb" in dm:
                    direct_model_map[lang] = (dm, models_registry[dm]["local_path"])
                    logger.info(f"[Subtitle {job_id}] Found direct model {lang}→{target_lang}: {dm}")

            # Classify segments
            direct_model_indices = defaultdict(list)  # lang → [indices] (has direct model)
            pivot_indices = []  # CJK without direct model
            direct_indices = [i for i in nllb_indices if segment_langs[i] not in CJK_LANGS]

            for i in nllb_indices:
                lang = segment_langs[i]
                if lang in CJK_LANGS:
                    if lang in direct_model_map:
                        direct_model_indices[lang].append(i)
                    else:
                        pivot_indices.append(i)

            dm_total = sum(len(v) for v in direct_model_indices.values())
            logger.info(f"[Subtitle {job_id}] {dm_total} direct NLLB, {len(pivot_indices)} CJK pivot, {len(direct_indices)} other NLLB")

            BATCH_SIZE = 8
            completed_count = 0

            # ── DIRECT NLLB: Use fine-tuned JA→VI, ZH→VI models ──
            if direct_model_indices:
                for lang, indices in direct_model_indices.items():
                    dm_id, dm_path = direct_model_map[lang]
                    logger.info(f"[Subtitle {job_id}] Loading direct NLLB {lang}→{target_lang}: {dm_id}")

                    src_nllb = NLLB_LANG_CODES.get(lang, "eng_Latn")
                    dm_tok = AutoTokenizer.from_pretrained(dm_path)
                    dm_mdl = AutoModelForSeq2SeqLM.from_pretrained(dm_path)
                    dm_mdl.eval()
                    dm_tok.src_lang = src_nllb

                    for batch_start in range(0, len(indices), BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, len(indices))
                        batch_idx = indices[batch_start:batch_end]
                        batch_texts = [segments[i]["text"] for i in batch_idx]

                        try:
                            inputs = dm_tok(batch_texts, return_tensors="pt",
                                           max_length=256, truncation=True, padding=True)
                            with torch.no_grad():
                                generated = dm_mdl.generate(
                                    **inputs, forced_bos_token_id=target_token_id,
                                    max_length=256, num_beams=4,
                                )
                            translations = dm_tok.batch_decode(generated, skip_special_tokens=True)

                            for j, idx in enumerate(batch_idx):
                                translated = translations[j].strip() if j < len(translations) else segments[idx]["text"]
                                all_translated[idx] = translated
                                job["live_translations"][idx]["translated"] = translated
                                job["live_translations"][idx]["status"] = "done"
                                logger.info(f"[Subtitle {job_id}] [{idx+1}/{total}] 🔄 \"{segments[idx]['text'][:60]}\" → \"{translated[:60]}\"")
                        except Exception as e:
                            logger.warning(f"[Subtitle {job_id}] Direct NLLB {lang}→VI batch failed: {e}")
                            for idx in batch_idx:
                                all_translated[idx] = segments[idx]["text"]
                                job["live_translations"][idx]["status"] = "error"

                        completed_count += len(batch_idx)
                        pct = 65 + int(5 * completed_count / len(nllb_indices))
                        job["progress"] = min(pct, 70)

                    del dm_tok, dm_mdl
                    import gc; gc.collect()
                    logger.info(f"[Subtitle {job_id}] Direct NLLB {lang}→VI: {len(indices)} segments done")

            # ── PIVOT: CJK → English → Vietnamese (fallback for langs without direct model) ──
            if pivot_indices:
                # Group by CJK language
                cjk_groups = defaultdict(list)
                for i in pivot_indices:
                    cjk_groups[segment_langs[i]].append(i)

                # Step A: CJK → English (NLLB)
                en_translations = {}  # idx → english text
                for src_lang_code, indices in cjk_groups.items():
                    src_nllb = NLLB_LANG_CODES.get(src_lang_code, "eng_Latn")
                    nllb_tok.src_lang = src_nllb

                    for batch_start in range(0, len(indices), BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, len(indices))
                        batch_idx = indices[batch_start:batch_end]
                        batch_texts = [segments[i]["text"] for i in batch_idx]

                        try:
                            inputs = nllb_tok(batch_texts, return_tensors="pt",
                                            max_length=256, truncation=True, padding=True)
                            with torch.no_grad():
                                generated = nllb_mdl.generate(
                                    **inputs, forced_bos_token_id=en_token_id,
                                    max_length=256, num_beams=4,  # More beams for quality
                                    length_penalty=1.0,
                                )
                            en_texts = nllb_tok.batch_decode(generated, skip_special_tokens=True)
                            for j, idx in enumerate(batch_idx):
                                en_translations[idx] = en_texts[j].strip() if j < len(en_texts) else segments[idx]["text"]
                        except Exception as e:
                            logger.warning(f"[Subtitle {job_id}] CJK→EN batch failed: {e}")
                            for idx in batch_idx:
                                en_translations[idx] = segments[idx]["text"]

                        completed_count += len(batch_idx)
                        pct = 65 + int(5 * completed_count / len(nllb_indices))
                        job["progress"] = min(pct, 70)

                logger.info(f"[Subtitle {job_id}] CJK→EN pivot done: {len(en_translations)} segments")

                # Step B: English → Vietnamese (use local OpusMT if available, else NLLB)
                local_en_vi = _find_best_model("en", target_lang)
                if local_en_vi and local_en_vi in models_registry:
                    # Use fine-tuned OpusMT EN→VI
                    model_path = models_registry[local_en_vi]["local_path"]
                    logger.info(f"[Subtitle {job_id}] Pivot EN→VI via local model: {models_registry[local_en_vi].get('alias', local_en_vi)}")
                    en_tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    en_mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
                    en_mdl.eval()

                    pivot_list = list(en_translations.items())
                    for batch_start in range(0, len(pivot_list), BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, len(pivot_list))
                        batch = pivot_list[batch_start:batch_end]
                        batch_idx = [b[0] for b in batch]
                        batch_en = [b[1] for b in batch]

                        try:
                            inputs = en_tok(batch_en, return_tensors="pt", max_length=256,
                                           truncation=True, padding=True)
                            with torch.no_grad():
                                generated = en_mdl.generate(**inputs, max_length=256, num_beams=4)
                            vi_texts = en_tok.batch_decode(generated, skip_special_tokens=True)

                            for j, idx in enumerate(batch_idx):
                                translated = vi_texts[j].strip() if j < len(vi_texts) else en_translations[idx]
                                all_translated[idx] = translated
                                job["live_translations"][idx]["translated"] = translated
                                job["live_translations"][idx]["status"] = "done"
                                job["live_translations"][idx]["pivot_en"] = en_translations[idx]
                                logger.info(f"[Subtitle {job_id}] [{idx+1}/{total}] 🔄 \"{segments[idx]['text'][:50]}\" → 🇬🇧 \"{en_translations[idx][:40]}\" → 🇻🇳 \"{translated[:50]}\"")
                        except Exception as e:
                            logger.warning(f"[Subtitle {job_id}] EN→VI batch failed: {e}")
                            for idx in batch_idx:
                                all_translated[idx] = en_translations[idx]
                                job["live_translations"][idx]["translated"] = en_translations[idx]
                                job["live_translations"][idx]["status"] = "done"
                                job["live_translations"][idx]["pivot_en"] = en_translations[idx]

                    del en_tok, en_mdl
                    import gc; gc.collect()
                else:
                    # Fallback: NLLB EN→VI
                    nllb_tok.src_lang = en_nllb
                    pivot_list = list(en_translations.items())
                    for batch_start in range(0, len(pivot_list), BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, len(pivot_list))
                        batch = pivot_list[batch_start:batch_end]
                        batch_idx = [b[0] for b in batch]
                        batch_en = [b[1] for b in batch]

                        try:
                            inputs = nllb_tok(batch_en, return_tensors="pt",
                                            max_length=256, truncation=True, padding=True)
                            with torch.no_grad():
                                generated = nllb_mdl.generate(
                                    **inputs, forced_bos_token_id=target_token_id,
                                    max_length=256, num_beams=4,
                                )
                            vi_texts = nllb_tok.batch_decode(generated, skip_special_tokens=True)
                            for j, idx in enumerate(batch_idx):
                                translated = vi_texts[j].strip() if j < len(vi_texts) else en_translations[idx]
                                all_translated[idx] = translated
                                job["live_translations"][idx]["translated"] = translated
                                job["live_translations"][idx]["status"] = "done"
                                job["live_translations"][idx]["pivot_en"] = en_translations[idx]
                        except Exception as e:
                            for idx in batch_idx:
                                all_translated[idx] = en_translations[idx]
                                job["live_translations"][idx]["translated"] = en_translations[idx]
                                job["live_translations"][idx]["status"] = "done"

                completed_count = len(pivot_indices)
                logger.info(f"[Subtitle {job_id}] Pivot EN→VI done: {len(pivot_indices)} segments")

            # ── DIRECT: Non-CJK → target via NLLB ──
            if direct_indices:
                lang_groups = defaultdict(list)
                for i in direct_indices:
                    src_nllb = NLLB_LANG_CODES.get(segment_langs[i], "eng_Latn")
                    lang_groups[src_nllb].append(i)

                for src_nllb, indices in lang_groups.items():
                    nllb_tok.src_lang = src_nllb
                    for batch_start in range(0, len(indices), BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, len(indices))
                        batch_idx = indices[batch_start:batch_end]
                        batch_texts = [segments[i]["text"] for i in batch_idx]

                        try:
                            inputs = nllb_tok(batch_texts, return_tensors="pt",
                                            max_length=256, truncation=True, padding=True)
                            with torch.no_grad():
                                generated = nllb_mdl.generate(
                                    **inputs, forced_bos_token_id=target_token_id,
                                    max_length=256, num_beams=2,
                                )
                            translations = nllb_tok.batch_decode(generated, skip_special_tokens=True)
                            for j, idx in enumerate(batch_idx):
                                translated = translations[j].strip() if j < len(translations) else segments[idx]["text"]
                                all_translated[idx] = translated
                                job["live_translations"][idx]["translated"] = translated
                                job["live_translations"][idx]["status"] = "done"
                                logger.info(f"[Subtitle {job_id}] [{idx+1}/{total}] 🌐 \"{segments[idx]['text'][:60]}\" → \"{translated[:60]}\"")
                        except Exception as e:
                            for idx in batch_idx:
                                all_translated[idx] = segments[idx]["text"]
                                job["live_translations"][idx]["status"] = "error"
                                job["live_translations"][idx]["error"] = str(e)[:80]

                        completed_count += len(batch_idx)
                        pct = 65 + int(10 * completed_count / len(nllb_indices))
                        job["progress"] = min(pct, 75)

            del nllb_tok, nllb_mdl
            import gc; gc.collect()
            logger.info(f"[Subtitle {job_id}] NLLB translated {len(nllb_indices)} segments (batched)")

        # ═══ STEP 4: Build output ═══
        translated_srt_segments = []
        for i, seg in enumerate(segments):
            translated_srt_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": all_translated[i],
                "original_text": seg["text"],
                "detected_lang": segment_langs[i],
            })

        job["translated_segments"] = translated_srt_segments

        translated_srt = _segments_to_srt([
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in translated_srt_segments
        ])
        translated_srt_path = job_dir / f"translated_{target_lang}.srt"
        with open(translated_srt_path, "w", encoding="utf-8") as f:
            f.write(translated_srt)
        job["translated_srt"] = translated_srt

        methods_used = []
        if local_indices:
            alias = models_registry.get(local_en_model_id, {}).get('alias', local_en_model_id)
            methods_used.append(f"Local: {alias} ({len(local_indices)} seg)")
        if nllb_indices:
            methods_used.append(f"NLLB-200 ({len(nllb_indices)} seg)")
        _pipeline_set(job, "translate", "completed",
                       segments_translated=total, methods_used=methods_used,
                       local_count=len(local_indices), nllb_count=len(nllb_indices))
        job["progress"] = 78
        logger.info(f"[Subtitle {job_id}] ✅ Translation done: {len(local_indices)} local + {len(nllb_indices)} NLLB + {len(skip_indices)} skip = {total}")

    except Exception as e:
        logger.error(f"[Subtitle {job_id}] Translation failed: {e}", exc_info=True)
        _pipeline_set(job, "translate", "failed", error=str(e))
        job["status"] = "failed"
        job["error"] = str(e)


def _refine_with_context(job_id: str, target_lang: str):
    """Refine translation using NLLB cross-check (back-translation quality check).
    
    Strategy:
      1. For each translated segment, translate it BACK to source language
      2. Compare back-translation with original text (similarity check)
      3. If similarity is low → retranslate with adjusted parameters
      4. Uses transcript summary context for consistency
    """
    job = subtitle_jobs[job_id]
    job_dir = SUBTITLES_DIR / job_id

    try:
        _pipeline_set(job, "refine", "running")
        job["progress"] = 80
        logger.info(f"[Subtitle {job_id}] Refining translation with NLLB cross-check...")

        translated_segments = job.get("translated_segments", [])
        if not translated_segments:
            _pipeline_set(job, "refine", "skipped", reason="No translated segments")
            return

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from difflib import SequenceMatcher

        nllb_model_name = "facebook/nllb-200-distilled-600M"
        nllb_tok = AutoTokenizer.from_pretrained(nllb_model_name)
        nllb_mdl = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
        nllb_mdl.eval()

        target_nllb = NLLB_LANG_CODES.get(target_lang, "vie_Latn")
        refined_count = 0
        total = len(translated_segments)

        # Filter segments that need refinement & group by source lang
        from collections import defaultdict
        refine_groups = defaultdict(list)  # src_nllb → list of (index, seg)
        for i, seg in enumerate(translated_segments):
            src_lang = seg.get("detected_lang", "unknown")
            if src_lang == target_lang or src_lang == "unknown":
                continue
            if seg["text"] == seg["original_text"]:
                continue
            src_nllb = NLLB_LANG_CODES.get(src_lang, "eng_Latn")
            refine_groups[src_nllb].append(i)

        if not refine_groups:
            _pipeline_set(job, "refine", "skipped", reason="No segments to refine")
            job["progress"] = 95
            return

        REFINE_BATCH = 8
        processed = 0
        total_to_refine = sum(len(v) for v in refine_groups.values())

        for src_nllb, indices in refine_groups.items():
            # Phase 1: Back-translate in batches
            for batch_start in range(0, len(indices), REFINE_BATCH):
                batch_end = min(batch_start + REFINE_BATCH, len(indices))
                batch_indices = indices[batch_start:batch_end]
                batch_translated = [translated_segments[i]["text"] for i in batch_indices]
                batch_originals = [translated_segments[i]["original_text"] for i in batch_indices]

                try:
                    # Back-translate: target → source
                    nllb_tok.src_lang = target_nllb
                    src_token_id = nllb_tok.convert_tokens_to_ids(src_nllb)
                    inputs = nllb_tok(batch_translated, return_tensors="pt",
                                     max_length=256, truncation=True, padding=True)
                    with torch.no_grad():
                        back_out = nllb_mdl.generate(
                            **inputs, forced_bos_token_id=src_token_id,
                            max_length=256, num_beams=2,
                        )
                    back_texts = nllb_tok.batch_decode(back_out, skip_special_tokens=True)

                    # Check similarity & collect low-quality segments
                    low_quality_indices = []
                    for j, idx in enumerate(batch_indices):
                        if j < len(back_texts):
                            similarity = SequenceMatcher(
                                None, batch_originals[j].lower(), back_texts[j].strip().lower()
                            ).ratio()
                            if similarity < 0.35:
                                low_quality_indices.append(idx)

                    # Phase 2: Re-translate low-quality segments
                    if low_quality_indices:
                        retrans_texts = [translated_segments[i]["original_text"] for i in low_quality_indices]
                        nllb_tok.src_lang = src_nllb
                        target_token_id = nllb_tok.convert_tokens_to_ids(target_nllb)
                        inp2 = nllb_tok(retrans_texts, return_tensors="pt",
                                       max_length=512, truncation=True, padding=True)
                        with torch.no_grad():
                            re_out = nllb_mdl.generate(
                                **inp2, forced_bos_token_id=target_token_id,
                                max_length=512, num_beams=4,
                                length_penalty=0.8, no_repeat_ngram_size=3,
                            )
                        better_texts = nllb_tok.batch_decode(re_out, skip_special_tokens=True)

                        for k, idx in enumerate(low_quality_indices):
                            if k < len(better_texts):
                                better = better_texts[k].strip()
                                if better and better != translated_segments[idx]["text"]:
                                    translated_segments[idx]["raw_translation"] = translated_segments[idx]["text"]
                                    translated_segments[idx]["text"] = better
                                    refined_count += 1

                except Exception as e:
                    logger.debug(f"[Subtitle {job_id}] Refine batch error: {e}")

                processed += len(batch_indices)
                pct = 80 + int(15 * processed / total_to_refine)
                job["progress"] = min(pct, 95)

        del nllb_tok, nllb_mdl
        import gc; gc.collect()

        job["translated_segments"] = translated_segments

        # Generate refined SRT
        refined_srt = _segments_to_srt([
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in translated_segments
        ])
        refined_path = job_dir / f"refined_{target_lang}.srt"
        with open(refined_path, "w", encoding="utf-8") as f:
            f.write(refined_srt)
        job["refined_srt"] = refined_srt

        _pipeline_set(job, "refine", "completed", refined_count=refined_count)
        job["progress"] = 95
        logger.info(f"[Subtitle {job_id}] Refinement complete: {refined_count}/{total} segments improved")

    except Exception as e:
        logger.error(f"[Subtitle {job_id}] Refine failed: {e}", exc_info=True)
        _pipeline_set(job, "refine", "failed", error=str(e))
        logger.info(f"[Subtitle {job_id}] Continuing without refinement")


def _generate_final_output(job_id: str, target_lang: str):
    """Generate final SRT files (output node)."""
    job = subtitle_jobs[job_id]
    job_dir = SUBTITLES_DIR / job_id

    try:
        _pipeline_set(job, "output", "running")

        translated_segments = job.get("translated_segments", [])

        # Save bilingual SRT
        bilingual_lines = []
        for i, seg in enumerate(translated_segments, 1):
            start = _format_srt_time(seg["start"])
            end = _format_srt_time(seg["end"])
            lang_tag = f"[{seg.get('detected_lang', '?')}] " if seg.get('detected_lang') != 'unknown' else ""
            bilingual_lines.append(f"{i}\n{start} --> {end}\n{lang_tag}{seg['original_text']}\n{seg['text']}\n")
        bilingual_srt = "\n".join(bilingual_lines)
        bilingual_path = job_dir / f"bilingual_{target_lang}.srt"
        with open(bilingual_path, "w", encoding="utf-8") as f:
            f.write(bilingual_srt)

        # Update translated_srt with final (possibly refined) text
        final_srt = _segments_to_srt([
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in translated_segments
        ])
        final_path = job_dir / f"translated_{target_lang}.srt"
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(final_srt)
        job["translated_srt"] = final_srt

        files_generated = ["original.srt", f"translated_{target_lang}.srt", f"bilingual_{target_lang}.srt"]
        if job.get("refined_srt"):
            files_generated.append(f"refined_{target_lang}.srt")

        _pipeline_set(job, "output", "completed", files=files_generated)
        job["status"] = "completed"
        job["progress"] = 100
        logger.info(f"[Subtitle {job_id}] Pipeline complete! Files: {files_generated}")

    except Exception as e:
        logger.error(f"[Subtitle {job_id}] Output generation failed: {e}", exc_info=True)
        _pipeline_set(job, "output", "failed", error=str(e))
        job["status"] = "failed"
        job["error"] = str(e)


@app.post("/api/subtitles/{job_id}/translate")
async def translate_subtitles(job_id: str, req: SubtitleTranslateRequest, background_tasks: BackgroundTasks):
    """Translate an already-extracted subtitle to a different language."""
    if job_id not in subtitle_jobs:
        raise HTTPException(404, "Subtitle job not found")

    job = subtitle_jobs[job_id]
    if not job["segments"]:
        raise HTTPException(400, "No segments to translate. Extract subtitles first.")

    job["status"] = "translating"
    job["target_lang"] = req.target_lang
    job["model_id"] = req.model_id
    job["progress"] = 70

    background_tasks.add_task(_translate_srt_segments, job_id, req.target_lang, req.model_id or None)
    return {"message": "Subtitle translation started"}


@app.get("/api/subtitles/{job_id}/download/{file_type}")
async def download_subtitle_file(job_id: str, file_type: str):
    """Download SRT file. file_type: 'original', 'translated', 'bilingual'"""
    if job_id not in subtitle_jobs:
        raise HTTPException(404, "Subtitle job not found")

    job = subtitle_jobs[job_id]
    job_dir = SUBTITLES_DIR / job_id

    if file_type == "original":
        srt_path = job_dir / "original.srt"
    elif file_type == "translated":
        target = job.get("target_lang", "vi")
        srt_path = job_dir / f"translated_{target}.srt"
    elif file_type == "bilingual":
        target = job.get("target_lang", "vi")
        srt_path = job_dir / f"bilingual_{target}.srt"
    else:
        raise HTTPException(400, "Invalid file type. Use: original, translated, bilingual")

    if not srt_path.exists():
        raise HTTPException(404, "SRT file not found")

    from fastapi.responses import FileResponse
    return FileResponse(
        str(srt_path),
        media_type="text/plain",
        filename=srt_path.name,
    )



RECOMMENDED_MODELS = [
    {
        "id": "facebook/mbart-large-50-many-to-many-mmt",
        "name": "mBART-50 Many-to-Many",
        "description": "Multilingual model supporting translation between 50 languages. Best for general translation tasks.",
        "size": "~2.4GB",
        "languages": 50,
        "type": "seq2seq",
    },
    {
        "id": "Helsinki-NLP/opus-mt-en-vi",
        "name": "Opus-MT English→Vietnamese",
        "description": "Specialized model for English to Vietnamese translation. Fast and efficient.",
        "size": "~300MB",
        "languages": 2,
        "type": "seq2seq",
    },
    {
        "id": "Helsinki-NLP/opus-mt-vi-en",
        "name": "Opus-MT Vietnamese→English",
        "description": "Specialized model for Vietnamese to English translation.",
        "size": "~300MB",
        "languages": 2,
        "type": "seq2seq",
    },
    {
        "id": "facebook/nllb-200-distilled-600M",
        "name": "NLLB-200 Distilled 600M",
        "description": "No Language Left Behind — supports 200 languages. Distilled version for faster inference.",
        "size": "~1.2GB",
        "languages": 200,
        "type": "seq2seq",
    },
    {
        "id": "google/mt5-base",
        "name": "mT5-Base",
        "description": "Multilingual T5 model. Can be fine-tuned for any translation pair.",
        "size": "~1.2GB",
        "languages": 101,
        "type": "seq2seq",
    },
    {
        "id": "Helsinki-NLP/opus-mt-en-zh",
        "name": "Opus-MT English→Chinese",
        "description": "Specialized model for English to Chinese translation.",
        "size": "~300MB",
        "languages": 2,
        "type": "seq2seq",
    },
    {
        "id": "Helsinki-NLP/opus-mt-en-ja",
        "name": "Opus-MT English→Japanese",
        "description": "Specialized model for English to Japanese translation.",
        "size": "~300MB",
        "languages": 2,
        "type": "seq2seq",
    },
    {
        "id": "Helsinki-NLP/opus-mt-en-ko",
        "name": "Opus-MT English→Korean",
        "description": "Specialized model for English to Korean translation.",
        "size": "~300MB",
        "languages": 2,
        "type": "seq2seq",
    },
]

RECOMMENDED_DATASETS = [
    {
        "id": "Helsinki-NLP/opus-100",
        "name": "OPUS-100",
        "description": "English-centric multilingual corpus covering 100 languages. Great for general translation training.",
        "size": "Varies by language pair",
        "pairs": "100 language pairs",
    },
    {
        "id": "wmt16",
        "name": "WMT16",
        "description": "Workshop on Machine Translation 2016 dataset. High-quality parallel sentences.",
        "size": "~500MB",
        "pairs": "Multiple pairs",
    },
    {
        "id": "opus_books",
        "name": "OPUS Books",
        "description": "Parallel corpus from translated books. Good for literary translation.",
        "size": "Varies",
        "pairs": "Multiple pairs",
    },
    {
        "id": "ted_talks_iwslt",
        "name": "TED Talks IWSLT",
        "description": "TED talk transcripts with translations. Good for conversational translation.",
        "size": "~100MB",
        "pairs": "Multiple pairs",
    },
    {
        "id": "facebook/flores",
        "name": "FLORES",
        "description": "Facebook's evaluation benchmark covering 200 languages. Small but high quality.",
        "size": "~50MB",
        "pairs": "200 language pairs",
    },
]

@app.get("/api/catalog/models")
async def catalog_models():
    return {"models": RECOMMENDED_MODELS}

@app.get("/api/catalog/datasets")
async def catalog_datasets():
    return {"datasets": RECOMMENDED_DATASETS}

# ---------------------------------------------------------------------------
# Supported Languages
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English", "flag": "🇺🇸"},
    {"code": "vi", "name": "Vietnamese", "flag": "🇻🇳"},
    {"code": "zh", "name": "Chinese", "flag": "🇨🇳"},
    {"code": "ja", "name": "Japanese", "flag": "🇯🇵"},
    {"code": "ko", "name": "Korean", "flag": "🇰🇷"},
    {"code": "fr", "name": "French", "flag": "🇫🇷"},
    {"code": "de", "name": "German", "flag": "🇩🇪"},
    {"code": "es", "name": "Spanish", "flag": "🇪🇸"},
    {"code": "pt", "name": "Portuguese", "flag": "🇧🇷"},
    {"code": "ru", "name": "Russian", "flag": "🇷🇺"},
    {"code": "ar", "name": "Arabic", "flag": "🇸🇦"},
    {"code": "th", "name": "Thai", "flag": "🇹🇭"},
    {"code": "id", "name": "Indonesian", "flag": "🇮🇩"},
    {"code": "hi", "name": "Hindi", "flag": "🇮🇳"},
    {"code": "it", "name": "Italian", "flag": "🇮🇹"},
    {"code": "nl", "name": "Dutch", "flag": "🇳🇱"},
    {"code": "tr", "name": "Turkish", "flag": "🇹🇷"},
    {"code": "pl", "name": "Polish", "flag": "🇵🇱"},
]

@app.get("/api/languages")
async def list_languages():
    return {"languages": SUPPORTED_LANGUAGES}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
