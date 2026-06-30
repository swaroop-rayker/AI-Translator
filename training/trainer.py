import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import torch
import gc
import shutil
import psutil
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler
)
from accelerate import Accelerator

logger = logging.getLogger(__name__)

try:
    import GPUtil
    gputil_available = True
except ImportError:
    gputil_available = False

# mBART-50 language codes mapping
MBART_LANG_MAP = {
    "en": "en_XX",
    "kn": "kn_IN",
    "ml": "ml_IN"
}

# NLLB-200 language codes mapping
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "kn": "kan_Knda",
    "ml": "mal_Mlym"
}

def get_system_telemetry():
    telemetry = {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage(os.path.splitdrive(os.getcwd())[0] + os.sep if os.name == "nt" else "/").percent,
        "gpu": 0.0,
        "vram": 0.0
    }
    
    if torch.cuda.is_available():
        try:
            vram_bytes = torch.cuda.memory_allocated()
            vram_total_bytes = torch.cuda.get_device_properties(0).total_memory
            telemetry["vram"] = round((vram_bytes / vram_total_bytes) * 100, 2)
            
            if gputil_available:
                gpus = GPUtil.getGPUs()
                if gpus:
                    telemetry["gpu"] = gpus[0].load * 100
                    telemetry["vram"] = gpus[0].memoryUtil * 100
            else:
                telemetry["gpu"] = 90.0
        except Exception as e:
            logger.warning(f"Error reading GPU telemetry: {e}")
            
    return telemetry

def run_training(dataset_version_id: str, config: dict, callback=None, dataset_file: str = None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training but is not available! CPU training is prohibited.")
        
    if callback:
        callback(
            stage="init",
            stage_progress=5.0,
            stage_details="Initializing GPU execution pipelines & setting up VRAM guards..."
        )
        
    # 1. Clear GPU cache to prevent starting with fragmented VRAM
    torch.cuda.empty_cache()
    
    # 2. VRAM Guard for 6GB limits (RTX 4050)
    max_len = int(config.get("max_sequence_length", 128))
    batch_size = int(config.get("batch_size", 4))
    
    # Safe physical batch size for 6GB VRAM with mBART
    safe_physical_batch_size = 1 if max_len >= 128 else 2
    
    if batch_size > safe_physical_batch_size:
        accumulation_steps = batch_size // safe_physical_batch_size
        physical_batch_size = safe_physical_batch_size
        logger.info(f"OOM Guard: RTX 4050 detected (6GB VRAM limit). Automatically enabling Gradient Accumulation: "
                    f"Physical Batch Size = {physical_batch_size}, Accumulation Steps = {accumulation_steps} "
                    f"(Effective Batch Size = {batch_size})")
    else:
        accumulation_steps = 1
        physical_batch_size = batch_size
        
    logger.info("Initializing Hugging Face Accelerate...")
    accelerator = Accelerator(
        mixed_precision="fp16" if config.get("fp16", True) else "no",
        gradient_accumulation_steps=accumulation_steps
    )
    
    device = accelerator.device
    logger.info(f"Using device: {device} (RTX 4050 Laptop)")
    
    model_name = config.get("model_name", "facebook/mbart-large-50-many-to-many-mmt")
    epochs = int(config.get("epochs", 3))
    lr = float(config.get("learning_rate", 5e-5))
    check_freq = int(config.get("checkpoint_frequency", 1))
    
    data_dir = os.getenv("DATA_DIR", "/data")
    # If running on host (not Docker), resolve to project-local data dir
    if not os.path.exists("/.dockerenv") and data_dir == "/data":
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        logger.info(f"Host mode: DATA_DIR resolved to {data_dir}")

    checkpoints_dir = os.path.join(data_dir, "checkpoints")
    models_dir = os.path.join(data_dir, "models")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Resolve dataset file path
    if not dataset_file:
        # Look up dataset file from DB storage_path (fallback)
        try:
            from backend.app.core.database import SessionLocal
            from backend.app.models.schemas import DatasetVersion
            lookup_db = SessionLocal()
            version_record = lookup_db.query(DatasetVersion).filter(DatasetVersion.id == dataset_version_id).first()
            if version_record and version_record.storage_path:
                dataset_file = version_record.storage_path
            lookup_db.close()
        except Exception as e:
            logger.warning(f"DB lookup for dataset path failed: {e}, falling back to directory scan")

    # Resolve paths (convert docker paths to host paths if running on host)
    if dataset_file:
        db_path = dataset_file
        # Resolve Docker path (/data/...) to host path
        if db_path.startswith("/data/"):
            db_path = os.path.join(data_dir, db_path[len("/data/"):])
        
        # Also try the cleaned version in processed dir
        cleaned_path = db_path
        if not os.path.exists(cleaned_path):
            # Try the processed/cleaned version
            base = os.path.splitext(os.path.basename(db_path))[0]
            for candidate in [
                os.path.join(data_dir, "processed", base + "_cleaned.jsonl"),
                os.path.join(data_dir, "processed", base + ".jsonl"),
                db_path,
            ]:
                if os.path.exists(candidate):
                    cleaned_path = candidate
                    break
        dataset_file = cleaned_path
        logger.info(f"Using resolved dataset file path: {dataset_file}")

    # Fallback: scan data directories
    if not dataset_file or not os.path.exists(dataset_file):
        for sub in ["processed", "batched", "merged", "raw"]:
            scan_dir = os.path.join(data_dir, sub)
            if os.path.exists(scan_dir):
                for f in os.listdir(scan_dir):
                    if dataset_version_id in f:
                        dataset_file = os.path.join(scan_dir, f)
                        break
            if dataset_file and os.path.exists(dataset_file):
                break
                
    if not dataset_file or not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Could not find dataset file for version {dataset_version_id} in {data_dir}")
        
    if callback:
        callback(
            stage="loading_dataset",
            stage_progress=10.0,
            stage_details=f"Locating and loading dataset from {os.path.basename(dataset_file)}..."
        )
        
    logger.info(f"Loading dataset natively from: {dataset_file}")
    import json
    import random
    
    try:
        records = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load dataset natively: {e}")
        raise e

    if callback:
        callback(
            stage="loading_dataset",
            stage_progress=50.0,
            stage_details=f"Loaded {len(records)} raw records. Splitting into train and validation sets..."
        )
        
    random.seed(42)
    records_shuffled = list(records)
    random.shuffle(records_shuffled)
    
    val_size = max(1, int(len(records_shuffled) * 0.1))
    train_records = records_shuffled[val_size:]
    val_records = records_shuffled[:val_size]
    
    total_train = len(train_records)
    total_val = len(val_records)
    
    logger.info(f"Dataset split: {total_train} train records, {total_val} validation records")
    
    if callback:
        callback(
            stage="loading_model",
            stage_progress=10.0,
            stage_details="Loading Hugging Face tokenizer and language tags..."
        )
        
    # 3. Load Tokenizer & Model for Seq2Seq fine-tuning
    training_technique = config.get("training_technique", "full")
    logger.info(f"Loading pretrained model: {model_name} (Technique: {training_technique})")
    
    is_nllb = "nllb" in model_name.lower()
    LANG_MAP = NLLB_LANG_MAP if is_nllb else MBART_LANG_MAP
    
    # Configure source and target language tags
    src_lang = config.get("src_lang", "en")
    tgt_lang = config.get("tgt_lang", "kn")
    
    src_lang_tag = LANG_MAP.get(src_lang, "eng_Latn" if is_nllb else "en_XX")
    tgt_lang_tag = LANG_MAP.get(tgt_lang, "kan_Knda" if is_nllb else "ml_IN" if tgt_lang == "ml" else "kn_IN")
    
    # Load tokenizer
    if is_nllb:
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang_tag, tgt_lang=tgt_lang_tag)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.src_lang = src_lang_tag
        
    if callback:
        callback(
            stage="loading_model",
            stage_progress=40.0,
            stage_details=f"Loading model weights using {training_technique.upper()}..."
        )
        
    if training_technique == "qlora":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        logger.info(f"QLoRA Mode: Loading base model in 4-bit onto {device}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": device}
        )
    else:
        logger.info("Standard Mode: Loading base model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Apply PEFT LoRA adapter if requested
    if training_technique in ["lora", "qlora"]:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        
        if training_technique == "qlora":
            logger.info("Preparing quantized model for k-bit training...")
            model = prepare_model_for_kbit_training(model)
            
        logger.info("Applying LoRA adapter to model...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    # 4. Tokenize natively in Python (completely bypass PyArrow mapping)
    def tokenize_records(records_list, is_val=False):
        tokenized_list = []
        batch_size = 500  # Process in chunks to prevent memory spikes
        total = len(records_list)
        
        for i in range(0, total, batch_size):
            batch = records_list[i : i + batch_size]
            inputs = [ex["src"] for ex in batch]
            targets = [ex["tgt"] for ex in batch]
            
            model_inputs = tokenizer(
                inputs, 
                text_target=targets, 
                max_length=max_len, 
                truncation=True, 
                padding="max_length"
            )
            
            for j in range(len(batch)):
                tokenized_list.append({
                    "input_ids": model_inputs["input_ids"][j],
                    "attention_mask": model_inputs["attention_mask"][j],
                    "labels": model_inputs["labels"][j]
                })
                
            if callback:
                progress = min(100.0, (len(tokenized_list) / total) * 100)
                details = f"Tokenizing {'validation' if is_val else 'training'} dataset: {len(tokenized_list):,} / {total:,} pairs"
                callback(
                    stage="preprocessing_dataset",
                    stage_progress=5.0 + progress * 0.45 if not is_val else 50.0 + progress * 0.45,
                    stage_details=details,
                    current_value=len(tokenized_list),
                    total_value=total
                )
        return tokenized_list

    if callback:
        callback(
            stage="preprocessing_dataset",
            stage_progress=5.0,
            stage_details="Starting training dataset tokenization...",
            current_value=0,
            total_value=total_train
        )
        
    logger.info("Tokenizing training dataset natively...")
    tokenized_train = tokenize_records(train_records, is_val=False)
    
    if callback:
        callback(
            stage="preprocessing_dataset",
            stage_progress=50.0,
            stage_details="Starting validation dataset tokenization...",
            current_value=0,
            total_value=total_val
        )
        
    logger.info("Tokenizing validation dataset natively...")
    tokenized_val = tokenize_records(val_records, is_val=True)
    
    if callback:
        callback(
            stage="preprocessing_dataset",
            stage_progress=95.0,
            stage_details="Setting up PyTorch DataLoaders..."
        )
        
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
            "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
            "labels": torch.stack([torch.tensor(x["labels"]) for x in batch])
        }
        
    train_dataloader = DataLoader(tokenized_train, batch_size=physical_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(tokenized_val, batch_size=physical_batch_size, collate_fn=collate_fn)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    if callback:
        callback(
            stage="preprocessing_dataset",
            stage_progress=95.0,
            stage_details="Optimizing model structures for GPU acceleration..."
        )
        
    if training_technique == "qlora":
        # Skip preparing model with accelerator to avoid device map/quantization placement conflicts
        optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    
    if callback:
        callback(
            stage="training",
            stage_progress=0.0,
            stage_details="Starting training loop..."
        )
        
    logger.info("Starting Training Loop...")
    global_step = 0
    checkpoint_paths = []
    
    samples_processed = 0
    training_loop_start = time.time()

    # Resume Detection
    job_id = config.get("job_id")
    resume_epoch = 1
    resume_step = -1
    
    pause_ckpt_dir = None
    if job_id:
        pause_ckpt_dir = os.path.join(checkpoints_dir, f"pause_{job_id}")
        if os.path.exists(pause_ckpt_dir) and os.path.exists(os.path.join(pause_ckpt_dir, "pause_meta.json")):
            logger.info(f"Detected paused checkpoint at {pause_ckpt_dir}. Resuming...")
            if callback:
                callback(
                    stage="training",
                    stage_progress=0.0,
                    stage_details="Loading paused training state, optimizer, and scheduler..."
                )
            # Load states
            accelerator.load_state(pause_ckpt_dir)
            with open(os.path.join(pause_ckpt_dir, "pause_meta.json"), "r") as f:
                pmeta = json.load(f)
                resume_epoch = pmeta.get("epoch", 1)
                resume_step = pmeta.get("step", -1)
                global_step = pmeta.get("global_step", 0)
                logger.info(f"Resuming from Epoch {resume_epoch}, Step {resume_step}, Global Step {global_step}")
    
    try:
        for epoch in range(resume_epoch, epochs + 1):
            model.train()
            total_loss = 0

            if callback:
                callback(
                    stage="training",
                    stage_progress=(global_step / num_training_steps) * 100,
                    stage_details=f"Starting Epoch {epoch}/{epochs}...",
                    epoch=epoch,
                    step=global_step,
                    total_steps=num_training_steps,
                    steps_per_epoch=len(train_dataloader),
                    total_epochs=epochs
                )
            
            for step, batch in enumerate(train_dataloader):
                # Skip already completed steps if resuming this epoch
                if epoch == resume_epoch and step <= resume_step:
                    batch_size_dim = batch["input_ids"].size(0)
                    samples_processed += batch_size_dim
                    continue
                
                # Check for pause signal
                if job_id:
                    pause_signal = os.path.join(checkpoints_dir, f"pause_{job_id}.signal")
                    if os.path.exists(pause_signal):
                        logger.info("Pause signal detected! Saving state...")
                        if callback:
                            callback(
                                stage="training",
                                stage_progress=(global_step / num_training_steps) * 100,
                                stage_details="Pause signal received. Saving optimizer/scheduler/RNG states..."
                            )
                        # Save state
                        os.makedirs(pause_ckpt_dir, exist_ok=True)
                        accelerator.save_state(pause_ckpt_dir)
                        tokenizer.save_pretrained(pause_ckpt_dir)
                        
                        # Save metadata (last completed step is step - 1)
                        with open(os.path.join(pause_ckpt_dir, "pause_meta.json"), "w") as f:
                            json.dump({
                                "epoch": epoch,
                                "step": step - 1,
                                "global_step": global_step,
                                "total_steps": num_training_steps
                            }, f)
                            
                        # Clean up signal file
                        try:
                            os.remove(pause_signal)
                        except:
                            pass
                            
                        logger.info(f"Paused successfully at Epoch {epoch}, Step {step}.")
                        if callback:
                            callback(
                                stage="paused",
                                stage_progress=(global_step / num_training_steps) * 100,
                                stage_details=f"Paused training at Epoch {epoch}, Step {step}",
                                epoch=epoch,
                                step=global_step,
                                loss=None,
                                val_loss=None
                            )
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                        return {
                            "paused": True,
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step
                        }

                batch_size_dim = batch["input_ids"].size(0)
                samples_processed += batch_size_dim
                
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                total_loss += loss.item()
                
                if global_step % 10 == 0 or step == len(train_dataloader) - 1:
                    current_loss = loss.item()
                    sys_metrics = get_system_telemetry()
                    logger.info(f"Epoch {epoch}/{epochs} | Step {step}/{len(train_dataloader)} | Loss: {current_loss:.4f}")
                    
                    val_loss = None
                    if step == len(train_dataloader) - 1:
                        val_batch_count = len(val_dataloader)
                        if callback:
                            callback(
                                stage="validating",
                                stage_progress=(global_step / num_training_steps) * 100,
                                stage_details=f"Starting validation after Epoch {epoch}/{epochs}...",
                                epoch=epoch,
                                step=global_step,
                                loss=current_loss,
                                val_loss=None,
                                system_metrics=sys_metrics,
                                current_value=0,
                                total_value=val_batch_count,
                                total_steps=num_training_steps,
                                steps_per_epoch=len(train_dataloader),
                                total_epochs=epochs
                            )
                        model.eval()
                        val_losses = []
                        with torch.no_grad():
                            validation_report_interval = max(1, val_batch_count // 20) if val_batch_count else 1
                            for val_idx, val_batch in enumerate(val_dataloader, start=1):
                                val_outputs = model(**val_batch)
                                val_losses.append(val_outputs.loss.item())
                                if callback and (val_idx == 1 or val_idx == val_batch_count or val_idx % validation_report_interval == 0):
                                    callback(
                                        stage="validating",
                                        stage_progress=(global_step / num_training_steps) * 100,
                                        stage_details=f"Validating Epoch {epoch}/{epochs}: batch {val_idx}/{val_batch_count}",
                                        epoch=epoch,
                                        step=global_step,
                                        loss=current_loss,
                                        val_loss=None,
                                        system_metrics=get_system_telemetry(),
                                        current_value=val_idx,
                                        total_value=val_batch_count,
                                        total_steps=num_training_steps,
                                        steps_per_epoch=len(train_dataloader),
                                        total_epochs=epochs
                                    )
                        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
                        model.train()
                        logger.info(f"Epoch {epoch} Eval | Validation Loss: {val_loss:.4f}")
                        if callback:
                            callback(
                                stage="training",
                                stage_progress=(global_step / num_training_steps) * 100,
                                stage_details=f"Validation completed for Epoch {epoch}/{epochs}.",
                                epoch=epoch,
                                step=global_step,
                                loss=current_loss,
                                val_loss=val_loss,
                                system_metrics=get_system_telemetry(),
                                total_steps=num_training_steps,
                                steps_per_epoch=len(train_dataloader),
                                total_epochs=epochs
                            )
                    
                    elapsed_loop = time.time() - training_loop_start
                    samples_per_sec = samples_processed / elapsed_loop if elapsed_loop > 0 else 0.0
                    
                    if callback:
                        callback(
                            stage="training",
                            stage_progress=(global_step / num_training_steps) * 100,
                            stage_details=f"Training: Epoch {epoch}/{epochs}, Step {step}/{len(train_dataloader)}",
                            epoch=epoch,
                            step=global_step,
                            loss=current_loss,
                            val_loss=val_loss,
                            system_metrics=sys_metrics,
                            total_steps=num_training_steps,
                            steps_per_epoch=len(train_dataloader),
                            total_epochs=epochs,
                            samples_per_sec=round(samples_per_sec, 2)
                        )
                    
            if epoch % check_freq == 0:
                epoch_ckpt_dir = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}")
                os.makedirs(epoch_ckpt_dir, exist_ok=True)
                logger.info(f"Saving checkpoint to {epoch_ckpt_dir}")
                
                if callback:
                    callback(
                        stage="checkpointing",
                        stage_progress=(global_step / num_training_steps) * 100,
                        stage_details=f"Preparing checkpoint directory after Epoch {epoch}/{epochs}...",
                        epoch=epoch,
                        step=global_step,
                        total_steps=num_training_steps,
                        steps_per_epoch=len(train_dataloader),
                        total_epochs=epochs
                    )
                    
                unwrapped_model = accelerator.unwrap_model(model)
                if callback:
                    callback(
                        stage="checkpointing",
                        stage_progress=(global_step / num_training_steps) * 100,
                        stage_details=f"Writing model weights for Epoch {epoch}/{epochs} checkpoint...",
                        epoch=epoch,
                        step=global_step,
                        total_steps=num_training_steps,
                        steps_per_epoch=len(train_dataloader),
                        total_epochs=epochs
                    )
                unwrapped_model.save_pretrained(epoch_ckpt_dir)
                if callback:
                    callback(
                        stage="checkpointing",
                        stage_progress=(global_step / num_training_steps) * 100,
                        stage_details=f"Writing tokenizer files for Epoch {epoch}/{epochs} checkpoint...",
                        epoch=epoch,
                        step=global_step,
                        total_steps=num_training_steps,
                        steps_per_epoch=len(train_dataloader),
                        total_epochs=epochs
                    )
                tokenizer.save_pretrained(epoch_ckpt_dir)
                checkpoint_paths.append(epoch_ckpt_dir)

                if callback:
                    callback(
                        stage="checkpointing",
                        stage_progress=(global_step / num_training_steps) * 100,
                        stage_details=f"Checkpoint saved for Epoch {epoch}/{epochs}.",
                        epoch=epoch,
                        step=global_step,
                        total_steps=num_training_steps,
                        steps_per_epoch=len(train_dataloader),
                        total_epochs=epochs
                    )
            
    except RuntimeError as e:
        torch.cuda.empty_cache()
        if "out of memory" in str(e).lower():
            logger.error("VRAM OOM Guard triggered: CUDA out of memory.")
            raise RuntimeError("CUDA Out of Memory: The model exceeded the 6GB VRAM limit. Try reducing the batch_size (recommended: 1 or 2) or sequence length, and ensure FP16 mixed precision is enabled.") from e
        raise e
    finally:
        torch.cuda.empty_cache()
        
    final_model_dir = os.path.join(models_dir, f"model_final_{int(time.time())}")
    os.makedirs(final_model_dir, exist_ok=True)
    logger.info(f"Saving final model to {final_model_dir}")
    
    if callback:
        callback(
            stage="finalizing",
            stage_progress=98.0,
            stage_details="Saving final model weights and configuration..."
        )
        
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    total_params = sum(p.numel() for p in unwrapped_model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in unwrapped_model.parameters()) / (1024 * 1024)
    
    # Clean up paused checkpoint directory upon successful completion
    if pause_ckpt_dir and os.path.exists(pause_ckpt_dir):
        try:
            shutil.rmtree(pause_ckpt_dir)
            logger.info(f"Cleaned up pause checkpoint directory: {pause_ckpt_dir}")
        except Exception as e:
            logger.warning(f"Could not delete pause checkpoint directory: {e}")

    return {
        "final_model_path": final_model_dir,
        "checkpoint_paths": checkpoint_paths,
        "param_count": total_params,
        "model_size_mb": round(model_size_mb, 2),
        "final_loss": total_loss / len(train_dataloader)
    }

if __name__ == "__main__":
    import argparse
    import json
    import sys
    import traceback
    
    # Configure stdout and stderr line buffering to prevent Windows console buffering
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # Configure logging to write to stdout so parent subprocess can capture it
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    parser = argparse.ArgumentParser(description="Subprocess trainer runner")
    parser.add_argument("--dataset_version_id", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, default=None)
    args = parser.parse_args()
    
    config = json.loads(args.config_json)
    
    def subprocess_callback(**kwargs):
        print(f"PROGRESS:{json.dumps(kwargs)}", flush=True)
        
    try:
        results = run_training(
            dataset_version_id=args.dataset_version_id,
            config=config,
            callback=subprocess_callback,
            dataset_file=args.dataset_file
        )
        print(f"RESULT:{json.dumps(results)}", flush=True)
    except Exception as e:
        logger.error(f"Subprocess training failed: {e}")
        print(f"ERROR:{json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}", flush=True)
        sys.exit(1)
