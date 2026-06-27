from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.schemas import Job, ExperimentRun, DatasetVersion, ModelRegistry, Dataset
from backend.app.core.scheduler import gpu_scheduler
from backend.app.core.state_manager import StateManager
from workers.tasks import train_model_task, process_dataset_task
import os
import uuid
import logging
import shutil

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["orchestration"])

@router.get("/status")
def get_system_status(db: Session = Depends(get_db)):
    """
    Returns the current host GPU lock ownership status plus real-time
    hardware telemetry (gpu%, vram%, cpu%) from the active training run.
    """
    status = gpu_scheduler.get_gpu_status()

    # Default telemetry values
    status["gpu_util"] = 0.0
    status["vram_util"] = 0.0
    status["cpu_util"] = 0.0

    if status["is_locked"] and status.get("active_job_id"):
        try:
            # Find the active run for this job and read latest telemetry
            run = (
                db.query(ExperimentRun)
                .filter(ExperimentRun.run_name.like(f"run-{status['active_job_id'][:8]}%"))
                .order_by(ExperimentRun.created_at.desc())
                .first()
            )
            if run and run.hardware_telemetry:
                hw = run.hardware_telemetry
                if hw.get("gpu"):
                    status["gpu_util"] = round(hw["gpu"][-1], 1)
                if hw.get("vram"):
                    status["vram_util"] = round(hw["vram"][-1], 1)
                if hw.get("cpu"):
                    status["cpu_util"] = round(hw["cpu"][-1], 1)
        except Exception as e:
            logger.warning(f"Could not read telemetry for active job: {e}")

    return status

@router.post("/dataset-process")
def submit_dataset_processing(
    dataset_version_id: str,
    min_length: int = 2,
    max_length: int = 300,
    db: Session = Depends(get_db)
):
    """
    Submits an asynchronous dataset validation & cleaning job to the Celery broker.
    """
    version_record = db.query(DatasetVersion).filter(DatasetVersion.id == dataset_version_id).first()
    if not version_record:
        raise HTTPException(status_code=404, detail="Dataset version not found")
        
    if version_record.status != "Validated":
        raise HTTPException(
            status_code=400,
            detail=f"Dataset version must be in 'Validated' state. Current: {version_record.status}"
        )
        
    # Create Job record
    new_job = Job(
        job_type="dataset_processing",
        status="Queued",
        config={"dataset_version_id": dataset_version_id, "min_length": min_length, "max_length": max_length}
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    
    # Dispatch Celery CPU task
    task = process_dataset_task.delay(
        job_id=new_job.id,
        dataset_version_id=dataset_version_id,
        config={"min_length": min_length, "max_length": max_length}
    )
    
    # Store celery task id
    new_job.celery_task_id = task.id
    db.commit()
    
    return {
        "job_id": new_job.id,
        "celery_task_id": task.id,
        "status": new_job.status
    }

@router.post("/train")
def submit_training_run(
    dataset_version_id: str,
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_sequence_length: int = 128,
    fp16: bool = True,
    checkpoint_frequency: int = 1,
    db: Session = Depends(get_db)
):
    """
    Creates and schedules a training run. Check GPU availability, creates an experiment log,
    and pushes the task to Celery's GPU queue.
    """
    # 1. Validate dataset is ready
    dataset_version = db.query(DatasetVersion).filter(DatasetVersion.id == dataset_version_id).first()
    if not dataset_version:
        raise HTTPException(status_code=404, detail="Dataset version not found")
        
    if dataset_version.status not in ["Processed", "TrainReady", "TrainingUsed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be clean & train-ready to start training (Current state: {dataset_version.status})"
        )
        
    # Mark dataset as TrainingUsed if currently TrainReady
    if dataset_version.status == "TrainReady":
        StateManager.transition_dataset(db, dataset_version, "TrainingUsed", "Submitted to Training Engine")
        
    # 2. Check if GPU is currently locked
    gpu_status = gpu_scheduler.get_gpu_status()
    gpu_warning = False
    gpu_message = "Training job successfully queued."
    
    if gpu_status["is_locked"]:
        gpu_warning = True
        gpu_message = f"GPU is currently busy with job {gpu_status['active_job_id']}. This job is queued and will start once the GPU becomes available."
        
    # 3. Create training job record
    hyperparams = {
        "dataset_version_id": dataset_version_id,
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_sequence_length": max_sequence_length,
        "fp16": fp16,
        "checkpoint_frequency": checkpoint_frequency,
        "src_lang": dataset_version.src_lang,
        "tgt_lang": dataset_version.tgt_lang
    }
    
    new_job = Job(
        job_type="training",
        status="Queued",
        config=hyperparams
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    
    # 4. Create Experiment Log (MLflow run equivalent)
    new_run = ExperimentRun(
        run_name=f"run-{new_job.id[:8]}-{model_name.split('/')[-1]}",
        dataset_version_id=dataset_version.id,
        status="Queued",
        hyperparameters=hyperparams
    )
    db.add(new_run)
    db.commit()
    db.refresh(new_run)
    
    # 5. Dispatch Celery task to the local GPU worker queue
    task = train_model_task.delay(
        job_id=new_job.id,
        dataset_version_id=dataset_version.id,
        run_id=new_run.id,
        config=hyperparams
    )
    
    new_job.celery_task_id = task.id
    db.commit()
    
    return {
        "job_id": new_job.id,
        "run_id": new_run.id,
        "celery_task_id": task.id,
        "status": new_job.status,
        "gpu_busy": gpu_warning,
        "message": gpu_message
    }

@router.get("")
def list_jobs(db: Session = Depends(get_db)):
    """
    List all platform jobs and tasks.
    """
    return db.query(Job).order_by(Job.created_at.desc()).all()

@router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    """
    Fetches status and progress details of a single job.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, db: Session = Depends(get_db)):
    """
    Cancels a running or queued job by revoking the Celery task,
    updating the job status to 'Cancelled',
    and releasing any associated locks (like GPU lock).
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.status in ["Completed", "Failed", "Cancelled"]:
        return {"message": f"Job already finished with status: {job.status}"}

    # 0. Mark job as Cancelled FIRST so the worker loop detects it immediately
    StateManager.transition_job(db, job, "Cancelled", "Cancelled by user")
        
    # 1. Revoke Celery task (best-effort; solo pool may not process this while busy)
    if job.celery_task_id:
        try:
            from workers.tasks import celery_app
            celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")
            logger.info(f"Revoked Celery task {job.celery_task_id} for Job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to revoke celery task {job.celery_task_id}: {e}")
            
    # 2. Release GPU lock if it's a training job
    if job.job_type == "training":
        try:
            gpu_scheduler.release_gpu_lock(job_id)
            logger.info(f"Released GPU lock for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to release GPU lock: {e}")
            
        # Clean up any paused checkpoint directory
        data_dir = os.getenv("DATA_DIR", "/data")
        if not os.path.exists("/.dockerenv") and data_dir == "/data":
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        pause_dir = os.path.join(data_dir, "checkpoints", f"pause_{job_id}")
        if os.path.exists(pause_dir):
            try:
                shutil.rmtree(pause_dir)
                logger.info(f"Deleted paused checkpoint directory on cancel: {pause_dir}")
            except Exception as pe:
                logger.warning(f"Failed to delete paused checkpoint: {pe}")
                
        # Update associated ExperimentRun status to Cancelled
        try:
            run = db.query(ExperimentRun).filter(ExperimentRun.job_id == job_id).first()
            if not run:
                run = db.query(ExperimentRun).filter(ExperimentRun.run_name.like(f"run-{job_id[:8]}%")).first()
            if run:
                run.status = "Cancelled"
                run.metrics = {**(run.metrics or {}), "error": "Job cancelled by user"}
                db.commit()
        except Exception as e:
            logger.warning(f"Failed to update experiment run: {e}")

    # 2b. Signal cancellation via Redis for dataset_processing jobs
    if job.job_type == "dataset_processing":
        try:
            dataset_ver_id = job.config.get("dataset_version_id")
            if dataset_ver_id:
                dv = db.query(DatasetVersion).filter(DatasetVersion.id == dataset_ver_id).first()
                if dv:
                    import redis
                    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
                    if not os.path.exists("/.dockerenv") and "redis:6379" in redis_url:
                        redis_url = redis_url.replace("redis:6379", "localhost:6379")
                    r_client = redis.Redis.from_url(redis_url, decode_responses=True)
                    r_client.set(f"merge_cancel:{dv.dataset_id}", "1", ex=300)
                    r_client.hset(f"merge_progress:{dv.dataset_id}", mapping={
                        "status": "cancelled",
                        "error": "Cancellation requested by user. Rolling back..."
                    })
                    logger.info(f"Signalled Redis cancellation for dataset {dv.dataset_id}")
        except Exception as e:
            logger.warning(f"Failed to signal dataset cancel in Redis: {e}")
            
    # 3. Rollback dataset status to allow reprocessing or training again
    try:
        dataset_ver_id = job.config.get("dataset_version_id")
        if dataset_ver_id:
            dataset_version = db.query(DatasetVersion).filter(DatasetVersion.id == dataset_ver_id).first()
            if dataset_version:
                if job.job_type == "training" and dataset_version.status == "TrainingUsed":
                    StateManager.transition_dataset(db, dataset_version, "TrainReady", "Training run cancelled. Rolled back state.")
                elif job.job_type == "dataset_processing" and dataset_version.status == "Processing":
                    StateManager.transition_dataset(db, dataset_version, "Validated", "Cleaning run cancelled. Rolled back state.")
    except Exception as e:
        logger.warning(f"Failed to rollback dataset version state: {e}")
    
    return {"status": "Cancelled", "job_id": job_id}

@router.post("/{job_id}/reset")
def reset_failed_job(job_id: str, db: Session = Depends(get_db)):
    """
    Resets a failed job from the UI by unlinking the dataset version association 
    and reverting the dataset status back to TrainReady.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # Revert dataset version state
    if job.config and "dataset_version_id" in job.config:
        ver_id = job.config["dataset_version_id"]
        dataset_version = db.query(DatasetVersion).filter(DatasetVersion.id == ver_id).first()
        if dataset_version:
            dataset_version.status = "TrainReady"
            
        # Remove dataset_version_id from config to unlink it
        new_config = {**job.config}
        new_config.pop("dataset_version_id", None)
        job.config = new_config
        
    db.commit()
    return {"status": "Reset completed", "job_id": job_id}

@router.post("/purge-failed")
def purge_failed_jobs(db: Session = Depends(get_db)):
    """
    Deletes all failed or cancelled jobs, experiment runs, AND dataset versions
    from the database, and reverts associated dataset versions' status back to
    'TrainReady' if they have no other active or completed jobs.
    
    Robust Validation:
    - Never deletes active jobs (Running, Starting, Queued, Paused).
    - Never deletes successful runs (Completed).
    - Never deletes experiment runs that are referenced in ModelRegistry.
    - Never deletes dataset versions referenced by active/successful runs or models.
    """
    # 1. Fetch failed/cancelled jobs and runs
    stale_jobs = db.query(Job).filter(Job.status.in_(["Failed", "Cancelled"])).all()
    
    # Exclude runs referenced in ModelRegistry to preserve foreign key constraints
    registered_run_ids = db.query(ModelRegistry.experiment_run_id).distinct()
    
    # Find all active jobs to match run prefixes
    active_jobs = db.query(Job).filter(Job.status.in_(["Running", "Starting", "Queued", "Paused"])).all()
    active_prefixes = {j.id[:8] for j in active_jobs}
    
    # Retrieve all runs that are not registered
    all_unregistered_runs = db.query(ExperimentRun).filter(~ExperimentRun.id.in_(registered_run_ids)).all()
    
    stale_runs = []
    for r in all_unregistered_runs:
        if r.status in ["Failed", "Cancelled"]:
            stale_runs.append(r)
        elif r.status in ["Running", "Starting"]:
            # Check if this run's job is active
            is_active = False
            if r.run_name and r.run_name.startswith("run-"):
                parts = r.run_name.split("-")
                if len(parts) > 1:
                    prefix = parts[1]
                    if prefix in active_prefixes:
                        is_active = True
            if not is_active:
                stale_runs.append(r)
                
    purged_job_ids = [j.id for j in stale_jobs]
    purged_run_ids = [r.id for r in stale_runs]
    
    # 2. Revert associated datasets back to TrainReady
    reverted_datasets = []
    for job in stale_jobs:
        if job.config and "dataset_version_id" in job.config:
            ver_id = job.config["dataset_version_id"]
            dataset_version = db.query(DatasetVersion).filter(DatasetVersion.id == ver_id).first()
            if dataset_version and dataset_version.status in ["TrainingUsed", "Processing"]:
                active_or_completed_jobs = db.query(Job).filter(
                    Job.status.in_(["Running", "Starting", "Queued", "Completed"])
                ).all()
                
                has_other_jobs = False
                for active_job in active_or_completed_jobs:
                    if active_job.id not in purged_job_ids and active_job.config and active_job.config.get("dataset_version_id") == ver_id:
                        has_other_jobs = True
                        break
                
                if not has_other_jobs:
                    dataset_version.status = "TrainReady"
                    reverted_datasets.append(ver_id)
                
    # 3. Perform deletes for jobs and runs
    if purged_run_ids:
        db.query(ExperimentRun).filter(ExperimentRun.id.in_(purged_run_ids)).delete(synchronize_session=False)
        
    if purged_job_ids:
        # Clean up paused checkpoints for purged jobs
        for job_id in purged_job_ids:
            pause_dir = os.path.join(data_dir, "checkpoints", f"pause_{job_id}")
            if os.path.exists(pause_dir):
                try:
                    shutil.rmtree(pause_dir)
                    logger.info(f"Purged pause checkpoint directory: {pause_dir}")
                except Exception as pe:
                    logger.warning(f"Failed to delete paused checkpoint {pause_dir}: {pe}")
        db.query(Job).filter(Job.id.in_(purged_job_ids)).delete(synchronize_session=False)

    # 4. Purge failed/cancelled dataset versions (robust validation)
    purged_dataset_versions = []
    stale_datasets = db.query(DatasetVersion).filter(
        DatasetVersion.status.in_(["Failed", "Cancelled"])
    ).all()

    for dv in stale_datasets:
        # Never delete if referenced by a successful experiment run
        has_successful_run = db.query(ExperimentRun).filter(
            ExperimentRun.dataset_version_id == dv.id,
            ExperimentRun.status == "Completed"
        ).first() is not None
        if has_successful_run:
            continue

        # Never delete if referenced by a model in the registry
        has_model = db.query(ModelRegistry).filter(
            ModelRegistry.dataset_version_id == dv.id
        ).first() is not None
        if has_model:
            continue

        # Safe to delete — nullify child references, clean up files
        db.query(DatasetVersion).filter(
            DatasetVersion.parent_version_id == dv.id
        ).update({DatasetVersion.parent_version_id: None})

        # Delete any associated failed experiment runs for this version
        db.query(ExperimentRun).filter(
            ExperimentRun.dataset_version_id == dv.id,
            ExperimentRun.status.in_(["Failed", "Cancelled"])
        ).delete(synchronize_session=False)

        file_path = dv.storage_path
        dataset_id = dv.dataset_id
        db.delete(dv)
        db.flush()

        # Remove file from disk
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        # Delete empty parent dataset
        sibling_count = db.query(DatasetVersion).filter(
            DatasetVersion.dataset_id == dataset_id
        ).count()
        if sibling_count == 0:
            parent_ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if parent_ds:
                db.delete(parent_ds)

        purged_dataset_versions.append(dv.id)
        
    db.commit()
    
    return {
        "status": "Purged",
        "purged_jobs_count": len(purged_job_ids),
        "purged_runs_count": len(purged_run_ids),
        "reverted_datasets_count": len(reverted_datasets),
        "purged_dataset_versions_count": len(purged_dataset_versions)
    }

@router.post("/{job_id}/pause")
def pause_training_run(job_id: str, db: Session = Depends(get_db)):
    """
    Pauses a running training job.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.job_type != "training":
        raise HTTPException(status_code=400, detail="Only training jobs can be paused")
        
    if job.status not in ["Running", "Starting", "Queued"]:
        raise HTTPException(status_code=400, detail=f"Job cannot be paused in status: {job.status}")
        
    # Transition to Paused
    StateManager.transition_job(db, job, "Paused", "User requested pause")
    
    # Write the pause signal file immediately so the trainer subprocess can see it at the next step
    data_dir = os.getenv("DATA_DIR", "/data")
    if not os.path.exists("/.dockerenv") and data_dir == "/data":
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    
    pause_signal_path = os.path.join(data_dir, "checkpoints", f"pause_{job_id}.signal")
    os.makedirs(os.path.dirname(pause_signal_path), exist_ok=True)
    try:
        with open(pause_signal_path, "w") as f:
            f.write("pause")
        logger.info(f"Pause signal file written at {pause_signal_path} from API")
    except Exception as e:
        logger.warning(f"Could not write pause signal file from API: {e}")
        
    return {"status": "Paused", "job_id": job_id}

@router.post("/{job_id}/resume")
def resume_training_run(job_id: str, db: Session = Depends(get_db)):
    """
    Resumes a paused training job.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.job_type != "training":
        raise HTTPException(status_code=400, detail="Only training jobs can be resumed")
        
    if job.status != "Paused":
        raise HTTPException(status_code=400, detail=f"Only Paused jobs can be resumed. Current status: {job.status}")
        
    # Transition back to Queued
    StateManager.transition_job(db, job, "Queued", "User requested resume")
    
    # Update associated ExperimentRun status back to Starting
    run = db.query(ExperimentRun).filter(ExperimentRun.run_name.like(f"run-{job_id[:8]}%")).first()
    if run:
        run.status = "Starting"
        db.commit()
        
    # Re-queue Celery task
    dataset_version_id = job.config.get("dataset_version_id")
    # Extract only hyperparameters (ignore dynamic status tracking keys)
    train_config = {k: v for k, v in job.config.items() if k not in ["progress", "current_step", "current_loss", "job_id", "run_id"]}
    
    # We must import workers.tasks to dispatch Celery CPU task
    from workers.tasks import train_model_task
    task = train_model_task.delay(
        job_id=job.id,
        run_id=run.id if run else job.id,
        dataset_version_id=dataset_version_id,
        config=train_config
    )
    
    # Update celery task id
    job.celery_task_id = task.id
    db.commit()
    
    return {"status": "Queued", "job_id": job_id, "celery_task_id": task.id}

