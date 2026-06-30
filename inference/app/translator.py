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
        self.ct_translator = None
        self.current_engine = "pytorch"
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

    def load_model(self, model_path: str, model_version: str = "Base", engine: str = "pytorch"):
        # Convert path to Docker volume path (/data/...) if running inside Docker
        if "facebook" not in model_path.lower():
            cleaned_path = model_path.replace("\\", "/")
            if "data/" in cleaned_path:
                parts = cleaned_path.split("data/", 1)
                model_path = os.path.join("/data", parts[1].lstrip("/\\"))
        # Force pytorch engine for remote base models since they are not pre-converted to CTranslate2 format
        if model_path in ["facebook/nllb-200-distilled-600M", "facebook/mbart-large-50-many-to-many-mmt"]:
            engine = "pytorch"
            
        logger.info(f"Loading translation model from {model_path} using {engine}...")
        try:
            device = self.determine_device()
            
            # Clean up old models to release VRAM/RAM
            self.model = None
            self.ct_translator = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if engine == "ctranslate2":
                import ctranslate2
                ct_path = os.path.join(model_path, "ctranslate2")
                
                # Verify if ctranslate2 model folder exists, otherwise fall back to path itself
                if not os.path.exists(os.path.join(ct_path, "model.bin")):
                    ct_path = model_path
                    
                if not os.path.exists(os.path.join(ct_path, "model.bin")):
                    raise FileNotFoundError(f"CTranslate2 model files not found in {model_path} or {ct_path}")
                    
                # Load CTranslate2 translator
                self.ct_translator = ctranslate2.Translator(ct_path, device=device)
                
                # Tokenizer is still loaded via Transformers
                # Check for PEFT adapter to load correct tokenizer
                is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
                if is_peft:
                    base_model_name = "facebook/nllb-200-distilled-600M"
                    try:
                        import json
                        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                            cfg = json.load(f)
                            base_model_name = cfg.get("base_model_name_or_path", base_model_name)
                    except Exception:
                        pass
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    except Exception:
                        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                self.current_engine = "ctranslate2"
            else:
                # Standard PyTorch loading code
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
                self.current_engine = "pytorch"
                
            self.current_model_path = model_path
            self.current_model_version = model_version
            logger.info(f"Model loaded successfully using {self.current_engine} on device: {device} (Version: {model_version})")
            return True
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            if model_path != "facebook/nllb-200-distilled-600M":
                logger.info("Reverting to base model...")
                self.load_model("facebook/nllb-200-distilled-600M", "Base", "pytorch")
            return False

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        current_stage = "Resolving GPU locks"
        import json
        try:
            start_time = time.time()
            target_device = self.determine_device()
            
            # Determine language tags based on model class
            is_nllb = "nllb" in self.current_model_path.lower()
            if self.tokenizer:
                tokenizer_class = type(self.tokenizer).__name__.lower()
                if "nllb" in tokenizer_class:
                    is_nllb = True
                    
            LANG_MAP = NLLB_LANG_MAP if is_nllb else MBART_LANG_MAP
            
            src_lang_tag = LANG_MAP.get(src_lang, "eng_Latn" if is_nllb else "en_XX")
            tgt_lang_tag = LANG_MAP.get(tgt_lang, "kan_Knda" if is_nllb else "ml_IN" if tgt_lang == "ml" else "kn_IN")
            
            current_stage = "Tokenizing source text"
            self.tokenizer.src_lang = src_lang_tag
            
            if self.current_engine == "ctranslate2":
                if not self.ct_translator or not self.tokenizer:
                    raise RuntimeError("CTranslate2 model or tokenizer is not loaded.")
                
                # Tokenize source
                source_tokens = self.tokenizer.tokenize(text)
                if is_nllb:
                    source_tokens = [src_lang_tag] + source_tokens + ["</s>"]
                else:
                    source_tokens = source_tokens + [src_lang_tag] + ["</s>"]
                    
                current_stage = "Injecting language tags"
                target_prefix = [tgt_lang_tag]
                
                current_stage = "Running auto-regressive decoding"
                results = self.ct_translator.translate_batch([source_tokens], target_prefix=[target_prefix])
                
                current_stage = "Decoding output tokens"
                output_tokens = results[0].hypotheses[0]
                translated_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True)
            else:
                if not self.model or not self.tokenizer:
                    raise RuntimeError("PyTorch model or tokenizer is not loaded.")
                    
                if str(self.model.device) != target_device:
                    logger.info(f"Moving model weights dynamically from {self.model.device} to {target_device}")
                    self.model.to(target_device)
                    
                current_stage = "Injecting language tags"
                inputs = self.tokenizer(text, return_tensors="pt").to(target_device)
                
                # Get target language Bos token
                if hasattr(self.tokenizer, "lang_code_to_id"):
                    forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang_tag]
                else:
                    forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_tag)
                
                current_stage = "Running auto-regressive decoding"
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id
                    )
                    
                current_stage = "Decoding output tokens"
                translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "translated_text": translated_text,
                "latency_ms": round(latency_ms, 2),
                "device": target_device,
                "model_version": self.current_model_version,
                "engine": self.current_engine
            }
        except Exception as e:
            logger.error(f"Inference execution failed during stage '{current_stage}': {e}")
            raise RuntimeError(json.dumps({
                "stage": current_stage,
                "error": str(e)
            }))

translator_engine = TranslationEngine()
