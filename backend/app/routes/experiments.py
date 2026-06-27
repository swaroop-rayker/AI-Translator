from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.schemas import ExperimentRun, DatasetVersion
from typing import List

router = APIRouter(prefix="/experiments", tags=["experiment_tracker"])

@router.get("")
def list_runs(db: Session = Depends(get_db)):
    """
    List all experiment runs.
    """
    return db.query(ExperimentRun).order_by(ExperimentRun.created_at.desc()).all()

@router.get("/compare")
def compare_runs(ids: List[str] = Query(...), db: Session = Depends(get_db)):
    """
    Compares details and curves of multiple runs.
    """
    runs = db.query(ExperimentRun).filter(ExperimentRun.id.in_(ids)).all()
    if not runs:
        raise HTTPException(status_code=404, detail="No runs found matching the provided IDs")
        
    comparison_data = []
    for r in runs:
        # Get dataset info
        dataset_ver = db.query(DatasetVersion).filter(DatasetVersion.id == r.dataset_version_id).first()
        dataset_name = f"{dataset_ver.dataset.name if dataset_ver and dataset_ver.dataset else 'Unknown'} ({dataset_ver.version if dataset_ver else ''})"
        
        comparison_data.append({
            "id": r.id,
            "name": r.run_name or f"run-{r.id[:8]}",
            "status": r.status,
            "dataset_version": dataset_name,
            "created_at": r.created_at,
            "duration_seconds": r.duration_seconds,
            "hyperparameters": r.hyperparameters,
            "metrics": r.metrics or {},
            "hardware_telemetry": r.hardware_telemetry or {}
        })
        
    return comparison_data

@router.get("/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)):
    """
    Retrieves detailed logs and metrics for a single run.
    """
    run = db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Experiment run not found")
    return run
