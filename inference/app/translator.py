import os
import time
import torch
import redis
import logging
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
if not os.path.exists("/.dockerenv") and "redis:6379" in REDIS_URL:
    REDIS_URL = REDIS_URL.replace("redis:6379", "localhost:6379")

# mBART-50 language codes mapping
MBART_LANG_MAP = {
    "en": "en_XX",
    "kn": "kn_IN",
    "ml": "ml_IN"
}

class TranslationEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        self.current_model_version = "None (Base mBART-50)"
        self.default_device = os.getenv("DEVICE", "cpu")
        
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis_client = None
            self.redis_available = False

        # Load default base mBART-50 model weights
        self.load_model("facebook/mbart-large-50-many-to-many-mmt")

    def determine_device(self) -> str:
        if self.default_device == "cpu":
            return "cpu"
            
        if self.redis_available and self.redis_client:
            try:
                is_gpu_locked = self.redis_client.exists("gpu_lock")
                if is_gpu_locked:
                    logger.info("GPU lock active by training worker. Falling back to CPU for inference.")
                    return "cpu"
            except Exception as e:
                logger.error(f"Error checking GPU lock in Redis: {e}")
                
        if not torch.cuda.is_available():
            return "cpu"
            
        return self.default_device

    def load_model(self, model_path: str, model_version: str = "Base"):
        logger.info(f"Loading translation model from {model_path}...")
        try:
            device = self.determine_device()
            
            # Load tokenizer and model for mBART-50
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
            loaded_model = MBartForConditionalGeneration.from_pretrained(model_path)
            loaded_model.to(device)
            
            self.model = loaded_model
            self.current_model_path = model_path
            self.current_model_version = model_version
            logger.info(f"Model loaded successfully on device: {device} (Version: {model_version})")
            return True
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            if model_path != "facebook/mbart-large-50-many-to-many-mmt":
                logger.info("Reverting to base model...")
                self.load_model("facebook/mbart-large-50-many-to-many-mmt")
            return False

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer is not loaded.")
            
        start_time = time.time()
        target_device = self.determine_device()
        
        if str(self.model.device) != target_device:
            logger.info(f"Moving model weights dynamically from {self.model.device} to {target_device}")
            self.model.to(target_device)
            
        # Get language codes for mBART-50
        src_lang_tag = MBART_LANG_MAP.get(src_lang, "en_XX")
        tgt_lang_tag = MBART_LANG_MAP.get(tgt_lang, "kn_IN")
        
        self.tokenizer.src_lang = src_lang_tag
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(target_device)
            
            # Get target language Bos token
            forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang_tag]
            
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id
                )
                
            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "translated_text": translated_text,
                "latency_ms": round(latency_ms, 2),
                "device": target_device,
                "model_version": self.current_model_version
            }
        except Exception as e:
            logger.error(f"Inference execution failed: {e}")
            raise RuntimeError(f"Translation execution failed: {str(e)}")

translator_engine = TranslationEngine()
