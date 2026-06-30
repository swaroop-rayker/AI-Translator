from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from backend.app.models.schemas import StateTransition, DatasetVersion, Job, ModelRegistry
from datetime import datetime

class StateMachineError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

class StateManager:
    # Allowed transitions for each entity type
    VALID_TRANSITIONS = {
        "dataset": {
            "Uploaded": {"Validated", "Archived"},
            "Validated": {"Processing", "Archived"},
            "Processing": {"Processed", "Failed", "Validated"},
            "Processed": {"TrainReady", "Archived"},
            "TrainReady": {"TrainingUsed", "Archived"},
            "TrainingUsed": {"Archived", "Deprecated", "TrainReady"},
            "Archived": {"Deprecated", "TrainReady"},  # Can unarchive if needed
            "Deprecated": set()
        },
        "training": {
            "Queued": {"Starting", "Cancelled"},
            "Starting": {"Running", "Failed", "Cancelled"},
            "Running": {"Completed", "Failed", "Cancelled", "Paused"},
            "Paused": {"Running", "Cancelled", "Queued", "Starting"},
            "Completed": set(),
            "Failed": set(),
            "Cancelled": set()
        },
        "model": {
            "Created": {"Training"},
            "Training": {"Ready", "Failed"},
            "Ready": {"Approved", "Archived"},
            "Approved": {"Deployed", "Deployed (CTRANSLATE2)", "Deployed (PYTORCH)", "Archived"},
            "Deployed": {"Archived", "Deployed (CTRANSLATE2)", "Deployed (PYTORCH)"},
            "Deployed (CTRANSLATE2)": {"Archived", "Deployed (PYTORCH)", "Deployed (CTRANSLATE2)"},
            "Deployed (PYTORCH)": {"Archived", "Deployed (CTRANSLATE2)", "Deployed (PYTORCH)"},
            "Archived": set()
        }
    }

    @classmethod
    def validate_transition(cls, entity_type: str, from_state: str, to_state: str):
        if entity_type not in cls.VALID_TRANSITIONS:
            raise StateMachineError(f"Unknown entity type: '{entity_type}'")
        
        allowed_states = cls.VALID_TRANSITIONS[entity_type].get(from_state, set())
        if to_state not in allowed_states:
            raise StateMachineError(
                f"Invalid transition for '{entity_type}': '{from_state}' -> '{to_state}'. "
                f"Allowed target states: {list(allowed_states) or 'None'}"
            )

    @classmethod
    def transition_dataset(cls, db: Session, dataset_version: DatasetVersion, target_state: str, action: str, user: str = "system") -> DatasetVersion:
        current_state = dataset_version.status
        if current_state == target_state:
            return dataset_version
        cls.validate_transition("dataset", current_state, target_state)
        
        # Update record
        dataset_version.status = target_state
        if target_state == "Processed":
            dataset_version.processed_at = datetime.utcnow()
        
        # Log transition
        transition = StateTransition(
            entity_type="dataset",
            entity_id=dataset_version.id,
            from_state=current_state,
            to_state=target_state,
            trigger_action=action[:500] if action else "",
            user_id=user
        )
        db.add(transition)
        db.commit()
        db.refresh(dataset_version)
        return dataset_version

    @classmethod
    def transition_job(cls, db: Session, job: Job, target_state: str, action: str, user: str = "system") -> Job:
        current_state = job.status
        if current_state == target_state:
            return job
        cls.validate_transition("training", current_state, target_state)
        
        # Update record
        job.status = target_state
        
        # Log transition
        transition = StateTransition(
            entity_type="training_job",
            entity_id=job.id,
            from_state=current_state,
            to_state=target_state,
            trigger_action=action[:500] if action else "",
            user_id=user
        )
        db.add(transition)
        db.commit()
        db.refresh(job)
        return job

    @classmethod
    def transition_model(cls, db: Session, model: ModelRegistry, target_state: str, action: str, user: str = "system") -> ModelRegistry:
        current_state = model.approval_status if target_state in ["Approved", "Archived"] else model.deployment_status
        # Wait, the model transitions can affect approval_status or deployment_status. Let's generalize.
        # Let's map model registry states:
        # model.approval_status: Pending (Created/Training/Ready), Approved, Archived
        # model.deployment_status: Undeployed, Deployed
        # Let's check which state we are changing.
        # If target_state in ['Created', 'Training', 'Ready', 'Approved', 'Archived']: we modify approval_status (or custom mapping).
        # Let's define target states mapping:
        # In our DB schemas, model.approval_status is 'Pending', 'Approved', 'Archived'.
        # model.deployment_status is 'Undeployed', 'Deployed'.
        # Let's map target states to columns:
        
        # To match the implementation plan's Model states: Created, Training, Ready, Approved, Deployed, Archived
        # Let's translate model state transition rules using model.approval_status and deployment_status:
        # Created -> approval_status = 'Pending', deployment_status = 'Undeployed'
        # Training -> approval_status = 'Pending'
        # Ready -> approval_status = 'Pending' (but trained)
        # Approved -> approval_status = 'Approved'
        # Deployed -> deployment_status = 'Deployed'
        # Archived -> approval_status = 'Archived' (and deployment_status = 'Undeployed')

        # Let's track a virtual state mapping for transitions, but update DB columns accordingly.
        # We can store the current virtual state of the model. Where? We can derive it:
        # If approval_status == 'Archived': 'Archived'
        # Else if deployment_status == 'Deployed': 'Deployed'
        # Else if approval_status == 'Approved': 'Approved'
        # Else if model has exported_model_path: 'Ready'
        # Else if model has checkpoints but training: 'Training'
        # Else: 'Created'
        
        # Let's write a simple resolver for current virtual state:
        virtual_state = "Created"
        if model.approval_status == "Archived":
            virtual_state = "Archived"
        elif model.deployment_status and model.deployment_status.startswith("Deployed"):
            virtual_state = model.deployment_status
        elif model.approval_status == "Approved":
            virtual_state = "Approved"
        elif model.exported_model_path or model.metrics:
            virtual_state = "Ready"
        elif model.checkpoint_path:
            virtual_state = "Training"
            
        if virtual_state == target_state:
            return model
            
        cls.validate_transition("model", virtual_state, target_state)
        
        # Update columns based on target virtual state
        if target_state == "Training":
            pass # virtual state transition
        elif target_state == "Ready":
            pass
        elif target_state == "Approved":
            model.approval_status = "Approved"
        elif target_state.startswith("Deployed"):
            model.deployment_status = target_state
        elif target_state == "Archived":
            model.approval_status = "Archived"
            model.deployment_status = "Undeployed"
            
        # Log transition
        transition = StateTransition(
            entity_type="model",
            entity_id=model.id,
            from_state=virtual_state,
            to_state=target_state,
            trigger_action=action[:500] if action else "",
            user_id=user
        )
        db.add(transition)
        db.commit()
        db.refresh(model)
        return model
