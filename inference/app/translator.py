import os
import time
import torch
import redis
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

# NLLB-200 language codes mapping
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "kn": "kan_Knda",
    "ml": "mal_Mlym"
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

        # Load default base NLLB-200 model weights
        self.load_model("facebook/nllb-200-distilled-600M")

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
            
            # Check if this is a PEFT (LoRA/QLoRA) adapter checkpoint
            is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
            if is_peft:
                logger.info("PEFT/LoRA adapter detected. Loading base model and overlaying adapter weights...")
                from peft import PeftModel
                base_model_name = "facebook/nllb-200-distilled-600M"
                try:
                    import json
                    with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                        adapter_cfg = json.load(f)
                        base_model_name = adapter_cfg.get("base_model_name_or_path", base_model_name)
                except Exception as e:
                    logger.warning(f"Could not read base model name from adapter config: {e}")
                
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                    
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                loaded_model = PeftModel.from_pretrained(base_model, model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                
            loaded_model.to(device)
            
            self.model = loaded_model
            self.current_model_path = model_path
            self.current_model_version = model_version
            logger.info(f"Model loaded successfully on device: {device} (Version: {model_version})")
            return True
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            if model_path != "facebook/nllb-200-distilled-600M":
                logger.info("Reverting to base model...")
                self.load_model("facebook/nllb-200-distilled-600M")
            return False

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer is not loaded.")
            
        start_time = time.time()
        target_device = self.determine_device()
        
        if str(self.model.device) != target_device:
            logger.info(f"Moving model weights dynamically from {self.model.device} to {target_device}")
            self.model.to(target_device)
            
        # Determine language tags based on model class
        is_nllb = "nllb" in self.current_model_path.lower()
        LANG_MAP = NLLB_LANG_MAP if is_nllb else MBART_LANG_MAP
        
        src_lang_tag = LANG_MAP.get(src_lang, "eng_Latn" if is_nllb else "en_XX")
        tgt_lang_tag = LANG_MAP.get(tgt_lang, "kan_Knda" if is_nllb else "ml_IN" if tgt_lang == "ml" else "kn_IN")
        
        self.tokenizer.src_lang = src_lang_tag
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(target_device)
            
            # Get target language Bos token
            if hasattr(self.tokenizer, "lang_code_to_id"):
                forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang_tag]
            else:
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_tag)
            
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
