import os
import requests
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.schemas import ModelRegistry, ExperimentRun, Job, DatasetVersion
from backend.app.core.state_manager import StateManager

router = APIRouter(prefix="/models", tags=["model_registry"])

INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://inference:8080")

def _data_dir() -> str:
    data_dir = os.getenv("DATA_DIR", "/data")
    if not os.path.exists("/.dockerenv") and data_dir == "/data":
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
    return data_dir

def _docker_data_path(path: str) -> str:
    data_dir = _data_dir()
    try:
        rel_path = os.path.relpath(path, data_dir).replace("\\", "/")
        if not rel_path.startswith(".."):
            return f"/data/{rel_path}"
    except Exception:
        pass
    return path

def _resolve_data_path(path: str) -> str:
    if not path:
        return path
    normalized = path.replace("\\", "/")
    data_dir = _data_dir()
    if normalized.startswith("/data/"):
        return os.path.join(data_dir, normalized.replace("/data/", "", 1))
    return path

def _checkpoint_artifact_status(model: ModelRegistry) -> dict:
    checkpoint_path = model.checkpoint_path or ""
    resolved_path = _resolve_data_path(checkpoint_path)
    exists = bool(resolved_path and os.path.exists(resolved_path))
    required_files = ["tokenizer.json"]

    if exists:
        has_adapter = os.path.exists(os.path.join(resolved_path, "adapter_config.json"))
        full_weight_file = next(
            (
                name for name in ["model.safetensors", "pytorch_model.bin"]
                if os.path.exists(os.path.join(resolved_path, name))
            ),
            None
        )
        has_full_model = full_weight_file is not None
        if has_adapter:
            required_files.append("adapter_model.safetensors")
        else:
            required_files.append(full_weight_file or "model.safetensors")
    else:
        has_adapter = False
        has_full_model = False

    missing_files = [
        name for name in required_files
        if not resolved_path or not os.path.exists(os.path.join(resolved_path, name))
    ]
    ctranslate2_path = None
    ctranslate2_exists = False
    if model.metrics and model.metrics.get("ctranslate2_path"):
        ctranslate2_path = _resolve_data_path(model.metrics["ctranslate2_path"])
        ctranslate2_exists = os.path.exists(os.path.join(ctranslate2_path, "model.bin"))

    return {
        "checkpoint_exists": exists,
        "checkpoint_path": checkpoint_path,
        "resolved_checkpoint_path": resolved_path,
        "missing_files": missing_files,
        "is_peft_adapter": has_adapter,
        "has_full_model_weights": has_full_model,
        "ctranslate2_exists": ctranslate2_exists,
        "ctranslate2_path": ctranslate2_path,
        "status": "available" if exists and not missing_files else "missing"
    }

def _serialize_model_with_artifact_status(model: ModelRegistry) -> dict:
    return {
        "id": model.id,
        "model_name": model.model_name,
        "version": model.version,
        "experiment_run_id": model.experiment_run_id,
        "dataset_version_id": model.dataset_version_id,
        "hyperparameters": model.hyperparameters,
        "metrics": model.metrics,
        "checkpoint_path": model.checkpoint_path,
        "exported_model_path": model.exported_model_path,
        "approval_status": model.approval_status,
        "deployment_status": model.deployment_status,
        "created_at": model.created_at,
        "artifact_status": _checkpoint_artifact_status(model)
    }

def _artifact_status_for_path(path: str) -> dict:
    model_like = type("ModelLike", (), {"checkpoint_path": _docker_data_path(path), "metrics": {}})()
    return _checkpoint_artifact_status(model_like)

def _discover_model_artifacts(db: Session) -> list:
    data_dir = _data_dir()
    registered_paths = {
        (_resolve_data_path(model.checkpoint_path) or "").replace("\\", "/")
        for model in db.query(ModelRegistry).all()
    }

    def scan_root(root: str, source: str) -> list:
        found = []
        if not os.path.isdir(root):
            return found
        entries = sorted(
            [entry for entry in os.scandir(root) if entry.is_dir()],
            key=lambda entry: entry.stat().st_mtime
        )
        for idx, entry in enumerate(entries):
            if not entry.is_dir():
                continue
            resolved_path = entry.path
            if resolved_path.replace("\\", "/") in registered_paths:
                continue
            artifact_status = _artifact_status_for_path(resolved_path)
            if artifact_status["status"] != "available":
                continue
            version = f"v1.{idx}" if source == "models" else entry.name.replace("checkpoint_epoch_", "epoch ")
            found.append({
                "id": f"disk-{entry.name}",
                "model_name": "Recovered fine-tuned model" if source == "models" else "Recovered epoch checkpoint",
                "version": version,
                "checkpoint_path": _docker_data_path(resolved_path),
                "source": "disk",
                "artifact_kind": "fine_tuned_model" if source == "models" else "epoch_checkpoint",
                "folder_name": entry.name,
                "artifact_status": artifact_status,
            })
        return found

    final_models = scan_root(os.path.join(data_dir, "models"), "models")
    if final_models:
        return sorted(final_models, key=lambda item: item["version"], reverse=True)

    checkpoints = scan_root(os.path.join(data_dir, "checkpoints"), "checkpoints")
    return sorted(checkpoints, key=lambda item: item["version"], reverse=True)

def register_model_internal(db: Session, job_id: str, run_id: str, trainer_results: dict) -> ModelRegistry:
    """
    Internal helper used by training worker task to automatically register
    a model checkpoint after a completed training run.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    run = db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
    
    if not job or not run:
        raise ValueError("Job or Experiment Run not found during model registration.")
        
    dataset_version = db.query(DatasetVersion).filter(DatasetVersion.id == run.dataset_version_id).first()
    
    # 1. Determine model version number
    model_count = db.query(ModelRegistry).count()
    version_str = f"v1.{model_count}" # Example version scheme: v1.0, v1.1
    
    # 2. Register in Model Registry
    new_model = ModelRegistry(
        model_name=f"translation-model-{dataset_version.src_lang}-{dataset_version.tgt_lang}",
        version=version_str,
        experiment_run_id=run.id,
        dataset_version_id=dataset_version.id,
        hyperparameters=run.hyperparameters,
        metrics={
            "final_loss": trainer_results.get("final_loss", 0.0),
            "param_count": trainer_results.get("param_count", 0),
            "model_size_mb": trainer_results.get("model_size_mb", 0.0)
        },
        checkpoint_path=trainer_results.get("final_model_path", ""),
        exported_model_path=trainer_results.get("final_model_path", ""),
        approval_status="Pending",
        deployment_status="Undeployed"
    )
    
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    
    # Back-reference run
    run.model_version_id = new_model.id
    db.commit()
    
    # Transition model state to Ready
    StateManager.transition_model(db, new_model, "Ready", "Model generated and checkpoints validated")
    
    return new_model

@router.get("")
def list_models(db: Session = Depends(get_db)):
    """
    Lists all models in the registry.
    """
    models = db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).all()
    return [_serialize_model_with_artifact_status(model) for model in models]

@router.get("/discover-artifacts")
def discover_model_artifacts(db: Session = Depends(get_db)):
    """
    Lists valid model/checkpoint folders on disk that are not registered in DB.
    These are loadable in inference but are not registry records.
    """
    return _discover_model_artifacts(db)

@router.get("/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db)):
    """
    Gets details of a specific model.
    """
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return _serialize_model_with_artifact_status(model)

@router.post("/{model_id}/approve")
def approve_model(model_id: str, db: Session = Depends(get_db)):
    """
    Approves a model in the registry, moving it from Ready to Approved.
    """
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    StateManager.transition_model(db, model, "Approved", "Manual developer approval")
    return {
        "model_id": model.id,
        "approval_status": model.approval_status
    }

@router.post("/{model_id}/deploy")
def deploy_model(model_id: str, engine: str = "pytorch", db: Session = Depends(get_db)):
    """
    Deploys a model to the inference service.
    Updates the registry deployment statuses and triggers Inference reload.
    """
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    if model.approval_status != "Approved":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only manually 'Approved' models can be deployed to production inference service."
        )

    artifact_status = _checkpoint_artifact_status(model)
    if artifact_status["status"] != "available":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Model checkpoint files are missing or incomplete. "
                f"Missing: {artifact_status['missing_files'] or ['checkpoint folder']} "
                f"Path: {artifact_status['checkpoint_path']}"
            )
        )
        
    # If CTranslate2 engine is selected, ensure the model is converted
    if engine == "ctranslate2":
        ct_converted = False
        if model.metrics and "ctranslate2_path" in model.metrics:
            data_dir = os.getenv("DATA_DIR", "/data")
            ct_path = model.metrics["ctranslate2_path"]
            if ct_path.startswith("/data"):
                # Translate docker path to local path for verification if running on host
                if not os.path.exists(ct_path):
                    rel_path = ct_path.replace("/data", "")
                    ct_path = os.path.join(data_dir, rel_path.lstrip("/\\"))
            if os.path.exists(os.path.join(ct_path, "model.bin")):
                ct_converted = True
                
        if not ct_converted:
            from workers.tasks import convert_to_ctranslate2_task
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Model {model_id} not converted to CTranslate2 yet. Triggering conversion task on Celery worker...")
            try:
                task = convert_to_ctranslate2_task.delay(model_id)
                task.get(timeout=60.0) # Wait for Celery worker to finish
                db.refresh(model)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"CTranslate2 conversion failed on task execution: {str(e)}"
                )
                
    # Undeploy all currently deployed models (excluding the one being redeployed/switched)
    currently_deployed = db.query(ModelRegistry).filter(
        ModelRegistry.deployment_status.like("Deployed%"),
        ModelRegistry.id != model_id
    ).all()
    for m in currently_deployed:
        StateManager.transition_model(db, m, "Approved", "Undeployed in favor of new model")
        
    # Deploy this model
    deployment_label = f"Deployed ({engine.upper()})"
    StateManager.transition_model(db, model, deployment_label, f"Production Deployment using {engine.upper()} engine")
    
    # Notify Inference service to reload weights
    try:
        payload = {
            "model_path": model.checkpoint_path,
            "model_version": model.version,
            "engine": engine
        }
        res = requests.post(f"{INFERENCE_SERVICE_URL}/reload", json=payload, timeout=15)
        inference_status = res.json() if res.status_code == 200 else {"error": res.text}
    except Exception as e:
        inference_status = {"error": f"Failed to notify inference service: {str(e)}"}
        
    return {
        "model_id": model.id,
        "deployment_status": model.deployment_status,
        "inference_reload_status": inference_status
    }

@router.post("/{model_id}/evaluate")
def evaluate_model_route(model_id: str, db: Session = Depends(get_db)):
    """
    Triggers an asynchronous accuracy evaluation (sacreBLEU and ChrF)
    for the specified model on its validation dataset.
    """
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    artifact_status = _checkpoint_artifact_status(model)
    if artifact_status["status"] != "available":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Model checkpoint files are missing or incomplete. "
                f"Missing: {artifact_status['missing_files'] or ['checkpoint folder']} "
                f"Path: {artifact_status['checkpoint_path']}"
            )
        )
        
    # Mark status as Evaluating
    curr_metrics = dict(model.metrics or {})
    curr_metrics["evaluation_status"] = "Evaluating"
    curr_metrics.pop("evaluation_error", None)
    model.metrics = curr_metrics
    
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(model, "metrics")
    db.commit()
    
    # Trigger Celery evaluation task
    from workers.tasks import evaluate_model_task
    evaluate_model_task.delay(model_id)
    
    return {"status": "Evaluation started", "model_id": model_id}
