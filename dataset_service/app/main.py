import os
import sys

# Ensure parent directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dataset_service.app.routes.datasets import router
from backend.app.core.database import init_db

app = FastAPI(
    title="AI Translator - Dataset Service",
    description="Dedicated microservice for dataset ingestion, validation, and cleaning",
    version="1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Dataset Routes
app.include_router(router)

@app.on_event("startup")
def on_startup():
    # Make sure DB tables exist
    init_db()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "dataset_service"}
