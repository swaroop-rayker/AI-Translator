"""
Standalone evaluation subprocess.
Runs in its own process so the CUDA context is fully destroyed on exit.
Called by the Celery evaluate_model_task via subprocess.

Usage: python evaluator.py --model-id <uuid> --dataset-path <path> --checkpoint-path <path>
           --base-model <name> --technique <full|lora|qlora>
           --src-lang <code> --tgt-lang <code>

Writes JSON result to stdout on success.
"""
import os
import sys
import json
import argparse
import torch
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--technique", required=True, choices=["full", "lora", "qlora"])
    parser.add_argument("--src-lang", required=True)
    parser.add_argument("--tgt-lang", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Read dataset ──
    sources, references = [], []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sources.append(item["src"])
            references.append(item["tgt"])

    # 10% validation split (from the tail)
    val_size = max(int(len(sources) * 0.1), min(len(sources), 100))
    val_sources = sources[-val_size:]
    val_references = references[-val_size:]

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    is_nllb = "nllb" in args.base_model.lower()
    NLLB_MAP = {"en": "eng_Latn", "kn": "kan_Knda", "ml": "mal_Mlym"}
    MBART_MAP = {"en": "en_XX", "kn": "kn_IN", "ml": "ml_IN"}
    lang_map = NLLB_MAP if is_nllb else MBART_MAP

    src_tag = lang_map.get(args.src_lang, "eng_Latn" if is_nllb else "en_XX")
    tgt_tag = lang_map.get(args.tgt_lang, "kan_Knda" if is_nllb else "kn_IN")
    tokenizer.src_lang = src_tag

    # ── Load model ──
    if args.technique == "qlora":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model, quantization_config=bnb_config, device_map="auto"
        )
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model).to(device)

    if args.technique in ("lora", "qlora"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    else:
        model = base_model

    model.eval()

    # ── Translate validation set ──
    if hasattr(tokenizer, "lang_code_to_id"):
        forced_bos = tokenizer.lang_code_to_id[tgt_tag]
    else:
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_tag)

    translations = []
    for src_text in val_sources:
        inputs = tokenizer(src_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=128)
        translations.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0])

    # ── Score ──
    bleu = sacrebleu.corpus_bleu(translations, [val_references])
    chrf = sacrebleu.corpus_chrf(translations, [val_references])

    result = {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "samples": len(val_sources),
    }
    # Write JSON result to stdout for the parent process to read
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
