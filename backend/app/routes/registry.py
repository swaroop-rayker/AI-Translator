import os
import requests
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.schemas import ModelRegistry, ExperimentRun, Job, DatasetVersion
from backend.app.core.state_manager import StateManager

router = APIRouter(prefix="/models", tags=["model_registry"])

INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://inference:8080")

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
    return db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).all()

@router.get("/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db)):
    """
    Gets details of a specific model.
    """
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

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
def deploy_model(model_id: str, db: Session = Depends(get_db)):
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
        
    # Undeploy all currently deployed models
    currently_deployed = db.query(ModelRegistry).filter(ModelRegistry.deployment_status == "Deployed").all()
    for m in currently_deployed:
        m.deployment_status = "Undeployed"
        StateManager.transition_model(db, m, "Archived", "Undeployed in favor of new model")
        
    # Deploy this model
    StateManager.transition_model(db, model, "Deployed", "Production Deployment")
    
    # Notify Inference service to reload weights
    try:
        # Convert container path to relative or shared host volume path if applicable
        # Our containers share /data volume directly, so the path "/data/models/..." is identical!
        payload = {
            "model_path": model.checkpoint_path,
            "model_version": model.version
        }
        res = requests.post(f"{INFERENCE_SERVICE_URL}/reload", json=payload, timeout=10)
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

