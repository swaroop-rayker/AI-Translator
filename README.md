# AI Translator: End-to-End Multilingual Translation Lifecycle Platform

AI Translator is a production-grade, microservice-based lifecycle automation platform for multilingual translation model development. It provides full pipeline orchestration—from raw dataset ingestion, character-level filtering, deduplication, and immutable version lineage, to asynchronous model training on hardware nodes, real-time telemetry monitoring, and model serving with dynamic weight reloading.

---

## 🏗️ System Architecture

The platform is designed around a decoupled, microservices-oriented architecture:

```mermaid
graph TD
    User([User UI / Browser]) <--> |HTTP / JSON| Orchestrator[API Gateway & Orchestrator: Port 8000]
    Orchestrator <--> |PostgreSQL| DB[(PostgreSQL DB)]
    Orchestrator <--> |Redis / Celery| Redis[(Redis Broker & Cache)]
    
    subgraph Background Task Workers
        Redis <--> DatasetWorker[Dataset Worker CPU: Port 8001]
        Redis <--> GPUWorker[GPU Trainer Worker Node]
    end

    DatasetWorker <--> |HTTP| DatasetService[Dataset Processing Service]
    GPUWorker --> |Spawns Subprocess| PyTorchTrainer[PyTorch MBart-50 Trainer]
    PyTorchTrainer --> |Reads/Writes Checkpoints| LocalStorage[(Local SSD Storage /data)]

    Orchestrator <--> |HTTP| InferenceService[Inference Service: Port 8002]
    InferenceService --> |Dynamic Weight Reload| LocalStorage
```

### Core Architecture Components:
1. **API Gateway & Orchestrator (FastAPI)**: Central command center. Manages experiments, queues tasks via Celery, coordinates GPU lock reservations, and updates the state ledger.
2. **Dataset Service (FastAPI)**: An isolated service dedicated to string normalizations, NFKC cleaning, language detection (Kannada/Malayalam character range matching), duplicate filtering, and lineage versioning.
3. **Queue & Lock Manager (Redis & Celery)**: Manages background task distribution. A custom distributed GPU scheduler protects the local VRAM from concurrent training spikes while routing serving threads safely.
4. **PyTorch Training Engine**: An accelerated trainer supporting FP16 mixed precision, loss tracking, and state persistence.
5. **Inference Service (FastAPI)**: A stateless translator decoupled from training. It runs on CPU by default to keep the GPU dedicated to training and supports active, zero-downtime model weight reloads.
6. **Frontend Dashboard (React / Vite)**: Real-timeOutfit-themed dashboard with live hardware telemetry, train charts, interactive translation sandbox, and control endpoints (Pause, Resume, Abort).

---

## 📁 Directory Structure

```
.
├── backend/                   # Central API Gateway & State Management (FastAPI)
│   ├── app/
│   │   ├── core/              # DB Session, GPU Scheduler, State Manager
│   │   ├── models/            # SQLAlchemy database schemas
│   │   └── routes/            # API endpoints (Experiments, Registry, Jobs)
│   └── Dockerfile
├── dataset_service/           # Unicode cleaning and dataset versioning service
│   ├── app/
│   │   ├── cleaning/          # Character-level NFKC clean pipelines
│   │   ├── validation/        # Language matchers (Kannada/Malayalam)
│   │   └── routes/            # Datasets endpoints
│   └── Dockerfile
├── inference/                 # Translation translation wrapper (served on CPU)
│   ├── app/
│   │   └── translator.py      # Translation models loading & generation
│   └── Dockerfile
├── training/                  # Accelerated PyTorch Trainer Engine
│   └── trainer.py             # Training loop, telemetry metrics & checkpointing
├── workers/                   # Background Celery task definitions
│   └── tasks.py               # Dataset processing & training wrappers
├── monitoring/                # Prometheus & Grafana configs
├── docker-compose.yml         # Dev environment container orchestrator
└── README.md
```

---

## ⚡ Training Pause & Resume Mechanics

The platform features a robust **Pause & Resume** mechanism designed for absolute VRAM safety and zero-bloat state persistence:

1. **Signal Handlers**: During training, the trainer checks for a `.signal` file. If detected, it pauses the loop.
2. **State Persistence**: The script serializes the model weights, optimizer buffers, learning rate scheduler state, gradient scaling values, and random number generator (RNG) seeds using PyTorch Accelerate.
3. **Hardware Flushing**: It calls `gc.collect()` and `torch.cuda.empty_cache()` to completely clear the GPU's memory footprint, preventing memory leaks during idle time.
4. **dataloader Skip Loop**: Upon resume, the trainer reloads the complete serialized state and wraps the dataset generator. It automatically fast-forwards through already completed steps of the active epoch without running execution graphs, maintaining total batch alignment.

---

## 🛠️ Local Setup Guide

### 1. Prerequisites
Ensure you have the following installed on your machine:
* Python 3.12+
* Node.js 18+
* Docker & Docker Compose
* NVIDIA GPU with CUDA Drivers (optional, falls back to CPU training if not detected)

### 2. Configure Database & Services
Spin up PostgreSQL, Redis, Prometheus, and Grafana using Docker Compose:
```bash
docker compose up -d db redis prometheus grafana
```

### 3. Initialize Python Environment
From the workspace root directory, create a virtual environment and install dependencies for the respective components:
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install main requirements
pip install -r backend/requirements.txt
pip install -r dataset_service/requirements.txt
pip install -r training/requirements.txt
```

### 4. Running Backend & Workers
Start the microservices locally in separate terminal tabs with the virtual environment active:
```bash
# Start API Gateway (Orchestrator)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Start Dataset Processing Service
uvicorn dataset_service.app.main:app --host 0.0.0.0 --port 8001 --reload

# Start Inference Service
uvicorn inference.app.main:app --host 0.0.0.0 --port 8002 --reload

# Start Celery Worker (Task Consumer)
celery -A workers.tasks.celery_app worker -Q dataset,training --pool=solo --loglevel=info
```

### 5. Running Frontend
Start the React development server:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser to access the dashboard.

---

## 🧪 Integration & Verification Testing
To run an automated integration verification of the Pause & Resume lifecycle:
```bash
python scratch/test_pause_resume_e2e.py
```
This script creates a tiny dataset, initiates training, triggers an automated pause at step 10, verifies state serialization, shuts down the process, resumes from the saved checkpoint, and completes execution successfully.
