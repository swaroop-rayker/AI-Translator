import os
import sys
import time
import traceback
import logging

# Ensure parent directory is in python path (project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Use normcase for Windows case-insensitive comparison and insert at front
normalized_paths = [os.path.normcase(p) for p in sys.path]
if os.path.normcase(parent_dir) not in normalized_paths:
    sys.path.insert(0, parent_dir)

from celery import Celery
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

# Import core modules
from backend.app.core.database import SessionLocal
from backend.app.models.schemas import Job, DatasetVersion, ExperimentRun, ModelRegistry
from backend.app.core.state_manager import StateManager
from backend.app.core.scheduler import gpu_scheduler
from dataset_service.app.routes.datasets import process_dataset, validate_dataset

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
# If running on Host, redis is on localhost. We adjust dynamically
if not os.path.exists("/.dockerenv") and "redis:6379" in REDIS_URL:
    REDIS_URL = REDIS_URL.replace("redis:6379", "localhost:6379")

celery_app = Celery("translation_tasks", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_routes={
        "workers.tasks.process_dataset_task": {"queue": "dataset"},
        "workers.tasks.train_model_task": {"queue": "training"},
        "workers.tasks.evaluate_model_task": {"queue": "training"},
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@celery_app.task(name="workers.tasks.process_dataset_task", bind=True)
def process_dataset_task(self, job_id: str, dataset_version_id: str, config: dict):
    """
    Celery task to run validation and processing of a dataset asynchronously.
    """
    db = SessionLocal()
    # Find job
    job_record = db.query(Job).filter(Job.id == job_id).first()
    if not job_record:
        # Create a tracker job if not already present
        job_record = Job(
            id=job_id,
            celery_task_id=self.request.id,
            job_type="dataset_processing",
            status="Running",
            config={"dataset_version_id": dataset_version_id, "config": config}
        )
        db.add(job_record)
        db.commit()
        db.refresh(job_record)
    else:
        job_record.celery_task_id = self.request.id
        job_record.status = "Running"
        db.commit()

    try:
        # 1. Run validation
        logger.info(f"Starting dataset validation for version {dataset_version_id}")
        val_res = validate_dataset(dataset_version_id, db=db)
        
        # 2. Run processing
        logger.info(f"Starting dataset cleaning/processing for version {dataset_version_id}")
        proc_res = process_dataset(
            dataset_version_id,
            min_length=config.get("min_length", 2),
            max_length=config.get("max_length", 300),
            db=db
        )
        
        job_record.status = "Completed"
        job_record.config = {
            "dataset_version_id": dataset_version_id,
            "validation": val_res,
            "processing": proc_res
        }
        db.commit()
        return proc_res
    except Exception as e:
        logger.error(f"Dataset processing task failed: {e}")
        from dataset_service.app.routes.datasets import redis_client
        
        dataset_id = None
        try:
            dataset_version = db.query(DatasetVersion).filter(DatasetVersion.id == dataset_version_id).first()
            if dataset_version:
                dataset_id = dataset_version.dataset_id
        except Exception:
            pass
            
        was_cancelled = False
        if dataset_id:
            was_cancelled = bool(redis_client.get(f"merge_cancel:{dataset_id}"))
            
        db.refresh(job_record)
        if was_cancelled or job_record.status == "Cancelled":
            job_record.status = "Cancelled"
        else:
            job_record.status = "Failed"
            
        job_record.error_log = traceback.format_exc()
        db.commit()
        raise e
    finally:
        db.close()

@celery_app.task(name="workers.tasks.train_model_task", bind=True)
def train_model_task(self, job_id: str, run_id: str, dataset_version_id: str, config: dict):
    """
    Celery task to run translation model training in a separate subprocess.
    """
    db = SessionLocal()
    
    # Find job
    job_record = db.query(Job).filter(Job.id == job_id).first()
    if not job_record:
        db.close()
        return {"error": f"Job {job_id} not found"}
        
    job_record.celery_task_id = self.request.id
    StateManager.transition_job(db, job_record, "Starting", "Worker started job execution")
        
    run_record = db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
    if run_record:
        run_record.status = "Starting"
        db.commit()
    
    # 1. Try to acquire GPU lock
    logger.info(f"Job {job_id} requesting GPU lock...")
    acquired = gpu_scheduler.acquire_gpu_lock(job_id, lease_seconds=1200) # 20 mins lock initially
    if not acquired:
        logger.info(f"GPU lock busy. Queueing job {job_id}.")
        # Put task back in queue or reschedule
        time.sleep(10)
        self.retry(countdown=20, max_retries=100) # Retry until lock is free
        db.close()
        return
        
    # Transition to Running
    StateManager.transition_job(db, job_record, "Running", "Acquired GPU lock and started training loop")
    
    # Resolve dataset storage path from DB before importing trainer (avoids CUDA/DB conflict)
    dataset_file_path = None
    dataset_version = db.query(DatasetVersion).filter(DatasetVersion.id == dataset_version_id).first()
    if dataset_version:
        dataset_file_path = dataset_version.storage_path
        logger.info(f"Resolved dataset storage path for task: {dataset_file_path}")
        
    try:
        import subprocess
        import json
        import queue
        import threading
        
        # Start training
        logger.info(f"Starting training session for Job: {job_id}, Run: {run_id}")
        training_start_time = None
        
        # We pass a callback to run_training to renew the GPU lock and report hardware/metrics
        def training_callback(epoch=None, step=None, loss=None, val_loss=None, system_metrics=None, stage="training", stage_progress=0.0, stage_details=None, samples_per_sec=None, current_value=None, total_value=None, **kwargs):
            nonlocal training_start_time
            # Refresh DB session inside callback to prevent stale transactions
            callback_db = SessionLocal()
            try:
                # Renew lock
                gpu_scheduler.renew_gpu_lock(job_id, lease_seconds=300)
                
                # Update run record if metrics/telemetry are provided
                r = callback_db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
                if r:
                    if loss is not None and step is not None:
                        metrics_history = r.metrics or {"loss": [], "val_loss": [], "epoch": [], "step": []}
                        metrics_history["loss"].append(float(loss))
                        metrics_history["step"].append(int(step))
                        if val_loss is not None:
                            metrics_history["val_loss"].append(float(val_loss))
                        if epoch is not None:
                            metrics_history["epoch"].append(int(epoch))
                        r.metrics = metrics_history
                        flag_modified(r, "metrics")
                    
                    if system_metrics:
                        hw_history = r.hardware_telemetry or {"cpu": [], "gpu": [], "vram": [], "ram": [], "disk": []}
                        hw_history["cpu"].append(system_metrics.get("cpu", 0))
                        hw_history["gpu"].append(system_metrics.get("gpu", 0))
                        hw_history["vram"].append(system_metrics.get("vram", 0))
                        hw_history["ram"].append(system_metrics.get("ram", 0))
                        hw_history["disk"].append(system_metrics.get("disk", 0))
                        r.hardware_telemetry = hw_history
                        flag_modified(r, "hardware_telemetry")
                    
                    callback_db.commit()
                    
                # Update job progress in config
                j = callback_db.query(Job).filter(Job.id == job_id).first()
                if j:
                    total_steps = kwargs.get("total_steps", 0)
                    steps_per_epoch = kwargs.get("steps_per_epoch", 0)
                    total_epochs = kwargs.get("total_epochs", 0)
                    
                    if stage == "training" and training_start_time is None:
                        training_start_time = time.time()
                        
                    elapsed = 0.0
                    if training_start_time is not None:
                        elapsed = time.time() - training_start_time
                        
                    eta = None
                    if step and total_steps and step > 0 and elapsed > 0:
                        time_per_step = elapsed / step
                        eta = max(0, int((total_steps - step) * time_per_step))
                        
                    progress_info = {
                        "stage": stage,
                        "stage_progress": stage_progress,
                        "stage_details": stage_details,
                        "samples_per_sec": samples_per_sec,
                        "current_value": current_value,
                        "total_value": total_value,
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "step": step,
                        "total_steps": total_steps,
                        "steps_per_epoch": steps_per_epoch,
                        "loss": float(loss) if loss is not None else None,
                        "val_loss": float(val_loss) if val_loss is not None else None,
                        "eta": eta
                    }
                    j.config = {**j.config, "progress": progress_info, "current_step": step or 0, "current_loss": loss}
                    callback_db.commit()
            except Exception as cb_err:
                logger.error(f"Error in training callback: {cb_err}")
            finally:
                callback_db.close()
        
        # Execute training script in a separate python subprocess (insulates from Celery thread locks)
        python_exe = sys.executable
        trainer_script = os.path.join(parent_dir, "training", "trainer.py")
        config_with_ids = {**config, "job_id": job_id, "run_id": run_id}
        config_json_str = json.dumps(config_with_ids)
        
        cmd = [
            python_exe,
            trainer_script,
            "--dataset_version_id", dataset_version_id,
            "--config_json", config_json_str
        ]
        if dataset_file_path:
            cmd.extend(["--dataset_file", dataset_file_path])
            
        logger.info(f"Launching training subprocess: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=parent_dir
        )
        
        results = None
        subprocess_error = None
        
        # Run a non-blocking queue reader thread to avoid blocking readline()
        q = queue.Queue()
        def enqueue_output(out, queue_obj):
            for line in iter(out.readline, ''):
                queue_obj.put(line)
            out.close()
            
        reader_thread = threading.Thread(target=enqueue_output, args=(process.stdout, q))
        reader_thread.daemon = True
        reader_thread.start()
        
        last_cancel_check = 0.0
        
        while True:
            # Break if process has completed and queue is completely consumed
            if process.poll() is not None and q.empty():
                break
                
            # Periodically poll database to check if the job has been cancelled or paused
            current_time = time.time()
            if current_time - last_cancel_check > 2.0:
                last_cancel_check = current_time
                check_db = SessionLocal()
                try:
                    j_check = check_db.query(Job).filter(Job.id == job_id).first()
                    if j_check:
                        if j_check.status in ["Cancelled", "Failed"]:
                            logger.info(f"Cancellation/failure state detected for Job {job_id}. Terminating subprocess...")
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            subprocess_error = "Job cancelled by user"
                            raise InterruptedError("Job cancelled by user")
                        elif j_check.status == "Paused":
                            logger.info(f"Pause state detected for Job {job_id}. Writing signal file...")
                            pause_signal_path = os.path.join(parent_dir, "data", "checkpoints", f"pause_{job_id}.signal")
                            os.makedirs(os.path.dirname(pause_signal_path), exist_ok=True)
                            with open(pause_signal_path, "w") as f:
                                f.write("pause")
                except InterruptedError:
                    raise
                except Exception as check_err:
                    logger.warning(f"Error checking job status: {check_err}")
                finally:
                    check_db.close()
            
            # Retrieve progress line from queue with a small timeout
            try:
                line = q.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if line:
                stripped = line.strip()
                if stripped.startswith("PROGRESS:"):
                    try:
                        progress_data = json.loads(stripped[len("PROGRESS:"):])
                        training_callback(**progress_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse progress telemetry line: {e}")
                elif stripped.startswith("RESULT:"):
                    try:
                        results = json.loads(stripped[len("RESULT:"):])
                    except Exception as e:
                        logger.warning(f"Failed to parse result line: {e}")
                elif stripped.startswith("ERROR:"):
                    try:
                        error_data = json.loads(stripped[len("ERROR:"):])
                        subprocess_error = error_data.get("error", "Unknown training error")
                    except Exception as e:
                        subprocess_error = stripped
                else:
                    logger.info(f"[Trainer] {stripped}")
                    
        return_code = process.wait()
        
        # Check if subprocess exited cleanly due to a pause request
        if results and results.get("paused"):
            logger.info(f"Training run for Job {job_id} paused cleanly.")
            if run_record:
                run_record.status = "Paused"
                db.commit()
            
            StateManager.transition_job(db, job_record, "Paused", "Training paused by user")
            gpu_scheduler.release_gpu_lock(job_id)
            db.close()
            return {"paused": True}

        if return_code != 0:
            err_msg = subprocess_error or f"Trainer subprocess exited with code {return_code}"
            raise RuntimeError(err_msg)
        
        # Finalize Experiment Run
        if run_record:
            run_record.status = "Completed"
            run_record.duration_seconds = time.time() - run_record.created_at.timestamp()
            
        StateManager.transition_job(db, job_record, "Completed", "Training completed successfully")
        
        # Trigger model registration job
        from backend.app.routes.registry import register_model_internal
        register_model_internal(db, job_id, run_id, results)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        
        db.refresh(job_record)
        if job_record.status == "Cancelled":
            logger.info("Job was cancelled by user. Transitioning experiment run to Cancelled.")
            if run_record:
                run_record.status = "Cancelled"
                run_record.metrics = {**(run_record.metrics or {}), "error": "Job cancelled by user"}
                db.commit()
        else:
            if run_record:
                run_record.status = "Failed"
                run_record.metrics = {**(run_record.metrics or {}), "error": str(e)}
            job_record.error_log = traceback.format_exc()
            try:
                StateManager.transition_job(db, job_record, "Failed", f"Training crashed: {str(e)}")
            except Exception as transition_err:
                logger.error(f"Could not transition job to Failed: {transition_err}")
                job_record.status = "Failed"
                db.commit()
        
        # Rollback dataset version state to TrainReady
        try:
            dataset_version = db.query(DatasetVersion).filter(DatasetVersion.id == dataset_version_id).first()
            if dataset_version and dataset_version.status == "TrainingUsed":
                dataset_version.status = "TrainReady"
                db.commit()
                logger.info(f"Dataset version {dataset_version_id} rolled back to TrainReady due to training failure/cancellation")
        except Exception as rollback_err:
            logger.error(f"Failed to rollback dataset version: {rollback_err}")
        
    finally:
        # Always release GPU lock
        gpu_scheduler.release_gpu_lock(job_id)
        db.close()

@celery_app.task(name="workers.tasks.evaluate_model_task")
def evaluate_model_task(model_id: str):
    import subprocess
    import json
    from sqlalchemy.orm.attributes import flag_modified
    
    db = SessionLocal()
    try:
        model_record = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
        if not model_record:
            logger.error(f"Model registry record {model_id} not found during evaluation task.")
            return
            
        dataset_ver = db.query(DatasetVersion).filter(DatasetVersion.id == model_record.dataset_version_id).first()
        if not dataset_ver:
            logger.error(f"Dataset version for model {model_id} not found.")
            return
            
        # Resolve dataset path
        dataset_path = dataset_ver.storage_path
        data_dir = os.getenv("DATA_DIR", "/data")
        
        # Translate to local path if not exists
        if not os.path.exists(dataset_path):
            rel_path = dataset_path.replace("/data", "")
            dataset_path = os.path.join(data_dir, rel_path.lstrip("/\\"))
            
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset validation file not found at: {dataset_path}")
            
        # Resolve checkpoint path
        checkpoint_path = model_record.checkpoint_path
        if not os.path.exists(checkpoint_path):
            rel_path = checkpoint_path.replace("/data", "")
            checkpoint_path = os.path.join(data_dir, rel_path.lstrip("/\\"))
            
        base_model_name = model_record.hyperparameters.get("model_name", "facebook/nllb-200-distilled-600M") if model_record.hyperparameters else "facebook/nllb-200-distilled-600M"
        tech = model_record.hyperparameters.get("training_technique", "full") if model_record.hyperparameters else "full"
        src_lang = dataset_ver.src_lang
        tgt_lang = dataset_ver.tgt_lang
        
        evaluator_script = os.path.join(parent_dir, "training", "evaluator.py")
        
        logger.info(f"Spawning evaluation subprocess for model {model_id}...")
        
        cmd = [
            sys.executable,
            evaluator_script,
            "--dataset-path", dataset_path,
            "--checkpoint-path", checkpoint_path,
            "--base-model", base_model_name,
            "--technique", tech,
            "--src-lang", src_lang,
            "--tgt-lang", tgt_lang
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Evaluator subprocess exited with code {process.returncode}. Error: {stderr}")
            
        # Parse JSON output from stdout
        try:
            lines = stdout.strip().split("\n")
            result_json = None
            for line in reversed(lines):
                if line.strip().startswith("{") and line.strip().endswith("}"):
                    result_json = json.loads(line)
                    break
            if not result_json:
                raise ValueError("Could not find JSON output in evaluator stdout.")
        except Exception as e:
            raise RuntimeError(f"Failed to parse evaluator output: {e}. Output was:\n{stdout}")
            
        # Save results to DB
        curr_metrics = dict(model_record.metrics or {})
        curr_metrics["bleu"] = result_json["bleu"]
        curr_metrics["chrf"] = result_json["chrf"]
        curr_metrics["evaluation_status"] = "Completed"
        
        model_record.metrics = curr_metrics
        flag_modified(model_record, "metrics")
        db.commit()
        logger.info(f"Successfully evaluated model {model_id} via subprocess: BLEU={result_json['bleu']}, ChrF={result_json['chrf']}")
        
    except Exception as e:
        logger.error(f"Failed to evaluate model {model_id}: {e}", exc_info=True)
        try:
            model_record = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
            if model_record:
                curr_metrics = dict(model_record.metrics or {})
                curr_metrics["evaluation_status"] = "Failed"
                curr_metrics["evaluation_error"] = str(e)
                model_record.metrics = curr_metrics
                flag_modified(model_record, "metrics")
                db.commit()
        except Exception as db_err:
            logger.error(f"Failed to write failure status to DB: {db_err}")
    finally:
        db.close()

