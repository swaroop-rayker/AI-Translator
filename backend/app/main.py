import os
import sys

# Ensure root directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from backend.app.routes.orchestrator import router as jobs_router
from backend.app.routes.experiments import router as experiments_router
from backend.app.routes.registry import router as models_router
from backend.app.core.database import init_db

app = FastAPI(
    title="AI Translator - Backend Orchestrator",
    description="Central gateway API for workflow management, audit trails, and schedules",
    version="1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(jobs_router, prefix="/api")
app.include_router(experiments_router, prefix="/api")
app.include_router(models_router, prefix="/api")

# Setup Prometheus Instrumentation (must be at module level before startup)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Initialize and expose Prometheus metrics endpoint at startup
@app.on_event("startup")
def startup_event():
    # Create DB tables
    init_db()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "orchestrator"}
