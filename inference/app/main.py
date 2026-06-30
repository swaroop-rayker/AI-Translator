from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from inference.app.translator import translator_engine

app = FastAPI(
    title="AI Translator - Inference Service",
    description="Dedicated stateless containerized translation hosting service",
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

class TranslationRequest(BaseModel):
    text: str
    src_lang: str  # 'en', 'kn', 'ml'
    tgt_lang: str  # 'en', 'kn', 'ml'

class ReloadRequest(BaseModel):
    model_path: str
    model_version: str
    engine: str = "pytorch"

@app.post("/translate")
def translate_text(request: TranslationRequest):
    """
    Translates input text using active model. Falls back to CPU if GPU is locked by training.
    """
    try:
        result = translator_engine.translate(
            text=request.text,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang
        )
        return result
    except Exception as e:
        # If detail is structured JSON error from translator, pass it through directly
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/reload")
def reload_model(request: ReloadRequest):
    """
    Forces the inference engine to load new model weights from shared disk space.
    """
    success = translator_engine.load_model(request.model_path, request.model_version, request.engine)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load weights from checkpoint: {request.model_path}"
        )
    return {
        "status": "success",
        "loaded_model_version": translator_engine.current_model_version,
        "loaded_model_path": translator_engine.current_model_path,
        "active_engine": translator_engine.current_engine
    }

@app.post("/unload")
def unload_model():
    """
    Releases loaded inference model/tokenizer weights from process RAM/VRAM.
    """
    translator_engine.unload_model()
    return {
        "status": "success",
        "model_loaded": False,
        "loaded_model_version": translator_engine.current_model_version,
        "loaded_model_path": translator_engine.current_model_path,
        "active_engine": translator_engine.current_engine
    }

@app.get("/status")
def get_inference_status():
    """
    Returns active model specs and execution device.
    """
    device = translator_engine.determine_device()
    return {
        "model_version": translator_engine.current_model_version,
        "model_path": translator_engine.current_model_path,
        "model_loaded": translator_engine.current_model_path is not None,
        "active_device": device,
        "active_engine": translator_engine.current_engine,
        "gpu_fallback_active": (device == "cpu" and translator_engine.default_device != "cpu")
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "inference"}

# Instrument Prometheus
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
