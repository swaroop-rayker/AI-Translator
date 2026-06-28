import os
import sys
import json
import time
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import sacrebleu

# Ensure parent directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from backend.app.models.schemas import ModelRegistry, DatasetVersion

def get_db_session():
    # Detect if running inside Docker or on host
    db_host = "localhost" if not os.path.exists("/.dockerenv") else "db"
    db_url = f"postgresql://postgres:postgres@{db_host}:5432/ai_translator"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()

def list_registered_models(session):
    models = session.query(ModelRegistry).all()
    if not models:
        print("\n[!] No models registered in the database yet.")
        return []
    
    print("\nRegistered Models:")
    print("-" * 100)
    print(f"{'ID':<6} | {'Model Name':<28} | {'Version':<8} | {'Technique':<10} | {'Status':<10} | {'Current Metrics'}")
    print("-" * 100)
    for idx, m in enumerate(models):
        metrics_summary = ", ".join([f"{k}: {v}" for k, v in (m.metrics or {}).items() if k != "param_count"])
        tech = m.hyperparameters.get("training_technique", "full") if m.hyperparameters else "full"
        print(f"{idx:<6} | {m.model_name:<28} | {m.version:<8} | {tech:<10} | {m.approval_status:<10} | {metrics_summary}")
    print("-" * 100)
    return models

def evaluate_model(session, model_record, max_samples=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[+] Running evaluation on device: {device.upper()}")
    
    # 1. Fetch dataset validation split file path
    dataset_ver = session.query(DatasetVersion).filter(DatasetVersion.id == model_record.dataset_version_id).first()
    if not dataset_ver:
        print("[!] Error: Dataset version associated with this model was not found.")
        return
    
    # Resolve path (relative to D:\AI Translator)
    dataset_path = dataset_ver.storage_path
    if dataset_path.startswith("/data"):
        dataset_path = "." + dataset_path # Translate docker path to host path: ./data/...
        
    if not os.path.exists(dataset_path):
        print(f"[!] Error: Dataset file not found at: {dataset_path}")
        return
        
    print(f"[+] Loading validation dataset from: {dataset_path}")
    
    # 2. Read evaluation samples
    sources = []
    references = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sources.append(item["src"])
            references.append(item["tgt"])
            
    # Take validation subset (from the end of the lists as defined in datasets.py)
    val_split_size = int(len(sources) * 0.1) # 10% validation split
    if val_split_size == 0:
        val_split_size = min(len(sources), 100)
        
    val_sources = sources[-val_split_size:]
    val_references = references[-val_split_size:]
    
    if max_samples:
        val_sources = val_sources[:max_samples]
        val_references = val_references[:max_samples]
        
    print(f"[+] Found {len(val_sources)} validation sentence pairs.")
    
    # 3. Load model and tokenizer
    base_model_name = model_record.hyperparameters.get("model_name", "facebook/nllb-200-distilled-600M") if model_record.hyperparameters else "facebook/nllb-200-distilled-600M"
    print(f"[+] Loading base tokenizer and model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Configure NLLB tokenizer target language tag
    is_nllb = "nllb" in base_model_name.lower()
    src_lang = dataset_ver.src_lang
    tgt_lang = dataset_ver.tgt_lang
    
    NLLB_LANG_MAP = {"en": "eng_Latn", "kn": "kan_Knda", "ml": "mal_Mlym"}
    MBART_LANG_MAP = {"en": "en_XX", "kn": "kn_IN", "ml": "ml_IN"}
    
    LANG_MAP = NLLB_LANG_MAP if is_nllb else MBART_LANG_MAP
    src_lang_tag = LANG_MAP.get(src_lang, "eng_Latn" if is_nllb else "en_XX")
    tgt_lang_tag = LANG_MAP.get(tgt_lang, "kan_Knda" if is_nllb else "kn_IN")
    
    tokenizer.src_lang = src_lang_tag
    
    # Load base model
    if model_record.hyperparameters and model_record.hyperparameters.get("training_technique") == "qlora":
        # Load in 4-bit for memory savings if QLoRA
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)
        
    # Apply Adapter Weights if technique is LoRA or QLoRA
    tech = model_record.hyperparameters.get("training_technique", "full") if model_record.hyperparameters else "full"
    if tech in ["lora", "qlora"]:
        checkpoint_path = model_record.checkpoint_path
        if checkpoint_path.startswith("/data"):
            checkpoint_path = "." + checkpoint_path
            
        print(f"[+] Loading fine-tuned {tech.upper()} adapter weights from: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        model = base_model
        
    model.eval()
    
    # 4. Generate translations
    print("\n[+] Translating validation sentences...")
    translations = []
    start_time = time.time()
    
    # Get target language Bos token ID
    if hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang_tag]
    else:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_tag)
        
    for idx, src_text in enumerate(val_sources):
        inputs = tokenizer(src_text, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128
            )
        pred_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translations.append(pred_text)
        
        # Display progressive ticker
        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_sources):
            print(f"    Progress: {idx+1}/{len(val_sources)} translated...")
            
    latency = time.time() - start_time
    avg_latency = (latency / len(val_sources)) * 1000
    
    # 5. Compute BLEU and ChrF scores
    print("\n[+] Computing metrics...")
    
    # Sacrebleu expects list of reference lists
    bleu = sacrebleu.corpus_bleu(translations, [val_references])
    chrf = sacrebleu.corpus_chrf(translations, [val_references])
    
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model Version:         {model_record.version}")
    print(f"  Training Technique:    {tech.upper()}")
    print(f"  Total Sentences:       {len(val_sources)}")
    print(f"  Avg Latency/Sentence:  {avg_latency:.2f} ms")
    print("-" * 60)
    print(f"  BLEU Score:            {bleu.score:.2f}")
    print(f"  ChrF Score:            {chrf.score:.2f}")
    print("=" * 60)
    
    # 6. Save metrics back to Database
    save_choice = input("\n[?] Save these validation metrics to the database registry? (y/n): ").strip().lower()
    if save_choice == 'y':
        # Update model record metrics dict
        current_metrics = dict(model_record.metrics or {})
        current_metrics["bleu"] = round(bleu.score, 2)
        current_metrics["chrf"] = round(chrf.score, 2)
        current_metrics["eval_latency_ms"] = round(avg_latency, 2)
        current_metrics["eval_samples_count"] = len(val_sources)
        
        model_record.metrics = current_metrics
        # Also flag modified to force SQLAlchemy JSON write
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(model_record, "metrics")
        
        session.commit()
        print("[+] Metrics successfully written to the Model Registry database! They will now show up on the UI.")
    else:
        print("[+] Evaluation completed. Metrics not saved.")

if __name__ == "__main__":
    print("====================================================")
    print(" AI Translator - Model Evaluation Tool (sacreBLEU)")
    print("====================================================")
    
    session = get_db_session()
    try:
        models = list_registered_models(session)
        if not models:
            sys.exit(0)
            
        choice = input(f"\nEnter model index to evaluate (0-{len(models)-1}) or 'q' to quit: ").strip()
        if choice.lower() == 'q':
            sys.exit(0)
            
        try:
            model_idx = int(choice)
            if model_idx < 0 or model_idx >= len(models):
                raise ValueError
        except ValueError:
            print("[!] Invalid choice. Exiting.")
            sys.exit(1)
            
        selected_model = models[model_idx]
        
        # Option to limit validation samples for speed during testing
        samples_choice = input("[?] Enter max samples to evaluate (press Enter to run on entire validation set): ").strip()
        max_samples = int(samples_choice) if samples_choice.isdigit() else None
        
        evaluate_model(session, selected_model, max_samples)
        
    finally:
        session.close()
