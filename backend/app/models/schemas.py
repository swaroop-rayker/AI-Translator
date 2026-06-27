import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, DateTime, Text, JSON, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class StateTransition(Base):
    __tablename__ = "state_transitions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    entity_type = Column(String(50), nullable=False)  # 'dataset', 'training_job', 'model'
    entity_id = Column(String(36), nullable=False)
    from_state = Column(String(50), nullable=False)
    to_state = Column(String(50), nullable=False)
    trigger_action = Column(String(500), nullable=False)
    user_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version = Column(String(50), nullable=False)  # 'v1_raw', 'v2_cleaned', etc.
    parent_version_id = Column(String(36), ForeignKey("dataset_versions.id"), nullable=True)
    status = Column(String(50), nullable=False)  # Uploaded, Validated, Processing, Processed, TrainReady, TrainingUsed, Archived, Deprecated
    src_lang = Column(String(10), nullable=False)
    tgt_lang = Column(String(10), nullable=False)
    record_count = Column(Integer, nullable=False, default=0)
    file_hash = Column(String(64), nullable=False, unique=True)
    storage_path = Column(String(512), nullable=False)
    validation_report = Column(JSON, nullable=True)
    processing_history = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="versions")
    experiment_runs = relationship("ExperimentRun", back_populates="dataset_version")
    models = relationship("ModelRegistry", back_populates="dataset_version")

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    celery_task_id = Column(String(255), unique=True, nullable=True)
    job_type = Column(String(50), nullable=False)  # 'dataset_processing', 'training', 'model_registration', 'cleanup'
    status = Column(String(50), nullable=False)  # Queued, Running, Completed, Failed, Cancelled
    config = Column(JSON, nullable=False)
    error_log = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_name = Column(String(255), nullable=True)
    dataset_version_id = Column(String(36), ForeignKey("dataset_versions.id"), nullable=False)
    status = Column(String(50), nullable=False)  # Queued, Starting, Running, Completed, Failed, Cancelled
    hyperparameters = Column(JSON, nullable=False)  # learning_rate, batch_size, epochs, optimizer, etc.
    metrics = Column(JSON, nullable=True)  # train_loss_history, val_loss_history, lr_history, bleu_scores
    hardware_telemetry = Column(JSON, nullable=True)  # cpu_util, gpu_util, vram_util, etc.
    duration_seconds = Column(Float, nullable=True)
    model_version_id = Column(String(36), nullable=True)  # Set after model registration
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset_version = relationship("DatasetVersion", back_populates="experiment_runs")
    models = relationship("ModelRegistry", back_populates="experiment_run")

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)  # e.g. 'v1.0'
    experiment_run_id = Column(String(36), ForeignKey("experiment_runs.id"), nullable=False)
    dataset_version_id = Column(String(36), ForeignKey("dataset_versions.id"), nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=False)  # Final evaluation losses, BLEU, model size
    checkpoint_path = Column(String(512), nullable=False)
    exported_model_path = Column(String(512), nullable=True)
    approval_status = Column(String(50), nullable=False, default="Pending")  # Pending, Approved, Archived
    deployment_status = Column(String(50), nullable=False, default="Undeployed")  # Undeployed, Deployed
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment_run = relationship("ExperimentRun", back_populates="models")
    dataset_version = relationship("DatasetVersion", back_populates="models")
