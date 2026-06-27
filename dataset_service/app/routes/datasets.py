import os
import shutil
import hashlib
import json
import csv
import random
import uuid
import redis
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from datetime import datetime

# Import DB and State models from backend directory
from backend.app.core.database import get_db
from backend.app.models.schemas import Dataset, DatasetVersion, StateTransition, ModelRegistry, ExperimentRun
from backend.app.core.state_manager import StateManager
from dataset_service.app.validation.validator import DatasetValidator
from dataset_service.app.cleaning.pipeline import DatasetCleaner

router = APIRouter(prefix="/datasets", tags=["datasets"])

DATA_DIR = os.getenv("DATA_DIR", "/data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MERGED_DIR = os.path.join(DATA_DIR, "merged")
BATCHED_DIR = os.path.join(DATA_DIR, "batched")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(BATCHED_DIR, exist_ok=True)

# Initialize Redis client for tracking merge progress
redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379/0"),
    decode_responses=True
)

def calculate_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

@router.post("/upload")
def upload_dataset(
    name: str = Form(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["txt", "csv", "tsv", "json", "jsonl"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_ext}"
        )
        
    new_dataset = Dataset(name=name)
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    
    raw_filename = f"{new_dataset.id}_v1_raw.{file_ext}"
    raw_path = os.path.join(RAW_DIR, raw_filename)
    
    try:
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        db.delete(new_dataset)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
        
    file_hash = calculate_sha256(raw_path)
    
    existing_version = db.query(DatasetVersion).filter(DatasetVersion.file_hash == file_hash).first()
    if existing_version:
        os.remove(raw_path)
        db.delete(new_dataset)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset with identical hash already exists (ID: {existing_version.dataset_id}, Version: {existing_version.version})"
        )
        
    new_version = DatasetVersion(
        dataset_id=new_dataset.id,
        version="v1_raw",
        status="Processing",
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        record_count=0,
        file_hash=file_hash,
        storage_path=raw_path,
        processing_history={"is_merged": False, "version_type": "raw"}
    )
    db.add(new_version)
    db.commit()
    db.refresh(new_version)
    
    transition = StateTransition(
        entity_type="dataset",
        entity_id=new_version.id,
        from_state="None",
        to_state="Processing",
        trigger_action="Upload Ingestion Start"
    )
    db.add(transition)
    db.commit()
    
    file_size = os.path.getsize(raw_path)
    redis_client.hset(f"merge_progress:{new_dataset.id}", mapping={
        "status": "processing",
        "processed_count": 0,
        "total_to_process": file_size,
        "error": "",
        "dataset_name": name,
        "version_id": new_version.id,
        "phase": "ingesting",
        "lines_merged": 0
    })
    redis_client.expire(f"merge_progress:{new_dataset.id}", 86400)
    
    background_tasks.add_task(
        bg_validate_single,
        dataset_id=new_dataset.id,
        version_id=new_version.id,
        file_path=raw_path,
        file_ext=file_ext,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        name=name
    )
    
    return {
        "dataset_id": new_dataset.id,
        "version_id": new_version.id,
        "name": new_dataset.name,
        "version": new_version.version,
        "status": "Processing",
        "is_merged": False
    }

def bg_validate_single(
    dataset_id: str,
    version_id: str,
    file_path: str,
    file_ext: str,
    src_lang: str,
    tgt_lang: str,
    name: str
):
    from backend.app.core.database import SessionLocal
    db = SessionLocal()
    
    try:
        file_size = os.path.getsize(file_path)
        
        redis_client.hset(f"merge_progress:{dataset_id}", mapping={
            "phase": "ingesting",
            "status": "processing",
            "processed_count": 0,
            "total_to_process": file_size,
            "error": ""
        })
        
        def validation_cb(processed, total):
            if redis_client.get(f"merge_cancel:{dataset_id}"):
                raise InterruptedError("Pipeline stopped by user.")
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "processed_count": processed,
                "total_to_process": total,
                "phase": "ingesting",
                "status": "processing"
            })

        validation_report = DatasetValidator.validate_file(
            file_path, 
            file_ext, 
            src_lang, 
            tgt_lang, 
            progress_callback=validation_cb
        )

        record_count = validation_report.get("record_count", 0)
        
        if record_count == 0:
            version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
            if version_record:
                version_record.status = "Failed"
                version_record.validation_report = validation_report
                db.commit()
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "failed",
                "error": "Ingestion failed: 0 valid records found during schema check."
            })
            return

        version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
        if version_record:
            version_record.record_count = record_count
            version_record.validation_report = validation_report
            version_record.status = "Validated"
            db.commit()

            transition = StateTransition(
                entity_type="dataset",
                entity_id=version_record.id,
                from_state="Processing",
                to_state="Validated",
                trigger_action="Single File Ingest Validation Complete"
            )
            db.add(transition)
            db.commit()

        redis_client.hset(f"merge_progress:{dataset_id}", mapping={
            "status": "completed",
            "processed_count": file_size,
            "total_to_process": file_size
        })

    except Exception as e:
        was_cancelled = bool(redis_client.get(f"merge_cancel:{dataset_id}"))
        
        # Roll back database changes: delete version
        version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
        if version_record:
            db.delete(version_record)
            db.commit()
            
        # Delete dataset if no other versions exist
        sibling_count = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == dataset_id).count()
        if sibling_count == 0:
            dataset_record = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset_record:
                db.delete(dataset_record)
                db.commit()

        # Delete partially uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
                
        if was_cancelled:
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "cancelled",
                "error": "Pipeline stopped by user. Partially uploaded files were deleted."
            })
            redis_client.delete(f"merge_cancel:{dataset_id}")
        else:
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "failed",
                "error": str(e)
            })
    finally:
        db.close()

class HashingTextWrapper:
    def __init__(self, file_obj, hash_obj):
        self.file_obj = file_obj
        self.hash_obj = hash_obj
        self.bytes_written = 0

    def write(self, s):
        self.file_obj.write(s)
        encoded = s.encode('utf-8', errors='ignore')
        self.hash_obj.update(encoded)
        self.bytes_written += len(encoded)

    def flush(self):
        self.file_obj.flush()

def bg_merge_moses(
    dataset_id: str,
    version_id: str,
    src_path: str,
    tgt_path: str,
    merged_path: str,
    src_lang: str,
    tgt_lang: str
):
    from backend.app.core.database import SessionLocal
    db = SessionLocal()
    
    try:
        # Phase 1: Merging separate Moses text files (Read-only on sources)
        src_size = os.path.getsize(src_path)
        redis_client.hset(f"merge_progress:{dataset_id}", mapping={
            "phase": "merging",
            "status": "processing",
            "processed_count": 0,
            "total_to_process": src_size,
            "lines_merged": 0,
            "error": ""
        })
        
        # Report progress dynamically (every 1MB or 1% of size)
        report_interval_bytes = max(1024 * 1024, src_size // 100)
        last_reported_bytes = 0
        bytes_processed = 0
        count = 0
        
        # Validation variables
        validation_issues = []
        empty_rows = 0
        extremely_short = 0
        extremely_long = 0
        lang_mismatches = 0
        max_val_rows = 20000
        
        import hashlib
        sha256 = hashlib.sha256()
        
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f_src, \
             open(tgt_path, "r", encoding="utf-8", errors="ignore") as f_tgt, \
             open(merged_path, "w", encoding="utf-8", newline="") as f_out:
             
            hash_wrapper = HashingTextWrapper(f_out, sha256)
            writer = csv.writer(hash_wrapper)
            writer.writerow(["src", "tgt"])
            
            for src_line, tgt_line in zip(f_src, f_tgt):
                # Throttle Redis cancel check to run only once every 100,000 lines
                if (count + empty_rows) % 100000 == 0:
                    if redis_client.get(f"merge_cancel:{dataset_id}"):
                        raise InterruptedError("Pipeline stopped by user.")
                
                src_text = src_line.strip()
                tgt_text = tgt_line.strip()
                
                # Approximate bytes processed from the source file to match total_to_process (src_size).
                # A multiplier of 3 is used for Kannada/Malayalam source script.
                bytes_processed += len(src_line) * (3 if src_lang in ["kn", "ml"] else 1) + 1
                
                if src_text and tgt_text:
                    writer.writerow([src_text, tgt_text])
                    count += 1
                    
                    # On-the-fly validation for the first max_val_rows
                    if count <= max_val_rows:
                        # Length checks
                        if len(src_text) < 2 or len(tgt_text) < 2:
                            extremely_short += 1
                        if len(src_text) > 500 or len(tgt_text) > 500:
                            extremely_long += 1
                            
                        # Language range checks
                        src_detected = DatasetValidator.detect_lang(src_text)
                        tgt_detected = DatasetValidator.detect_lang(tgt_text)
                        
                        if src_lang == "en" and src_detected not in ["en", "unknown"]:
                            lang_mismatches += 1
                            if len(validation_issues) < 100:
                                validation_issues.append(f"Item {count}: Source lang mismatch. Expected 'en', detected '{src_detected}'")
                        elif src_lang in ["kn", "ml"] and src_detected != src_lang and src_detected != "unknown":
                            lang_mismatches += 1
                            if len(validation_issues) < 100:
                                validation_issues.append(f"Item {count}: Source lang mismatch. Expected '{src_lang}', detected '{src_detected}'")
                                
                        if tgt_lang in ["kn", "ml"] and tgt_detected != tgt_lang and tgt_detected != "unknown":
                            lang_mismatches += 1
                            if len(validation_issues) < 100:
                                validation_issues.append(f"Item {count}: Target lang mismatch. Expected '{tgt_lang}', detected '{tgt_detected}'")
                else:
                    empty_rows += 1
                    
                if bytes_processed - last_reported_bytes >= report_interval_bytes:
                    redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                        "processed_count": bytes_processed,
                        "lines_merged": count
                    })
                    last_reported_bytes = bytes_processed

        if count == 0:
            if os.path.exists(merged_path):
                os.remove(merged_path)
            version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
            if version_record:
                version_record.status = "Failed"
                db.commit()
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "failed",
                "error": "Merging resulted in 0 records."
            })
            return

        merged_size = hash_wrapper.bytes_written
        file_hash = sha256.hexdigest()

        # Check for duplicates in model DB
        existing_version = db.query(DatasetVersion).filter(
            DatasetVersion.file_hash == file_hash,
            DatasetVersion.id != version_id
        ).first()
        
        if existing_version:
            os.remove(merged_path)
            version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
            if version_record:
                db.delete(version_record)
                db.commit()
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "failed",
                "error": f"Merged dataset already exists (ID: {existing_version.dataset_id})"
            })
            return

        # Phase 2: Report validation results directly from in-memory collection
        validation_report = {
            "is_valid": len(validation_issues) == 0 and count > 0,
            "record_count": count,
            "empty_rows": empty_rows,
            "missing_fields": 0,
            "malformed_rows": 0,
            "language_mismatches": lang_mismatches,
            "extremely_short_sequences": extremely_short,
            "extremely_long_sequences": extremely_long,
            "issues": validation_issues
        }

        # Simulate Phase 2 progress completion for frontend UI compatibility
        redis_client.hset(f"merge_progress:{dataset_id}", mapping={
            "phase": "ingesting",
            "status": "processing",
            "processed_count": merged_size,
            "total_to_process": merged_size,
            "lines_merged": count
        })

        version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
        if version_record:
            version_record.record_count = count
            version_record.file_hash = file_hash
            version_record.storage_path = merged_path
            version_record.validation_report = validation_report
            version_record.status = "Validated"
            db.commit()

            transition = StateTransition(
                entity_type="dataset",
                entity_id=version_record.id,
                from_state="None",
                to_state="Validated",
                trigger_action="Moses Merge & Ingest Validation"
            )
            db.add(transition)
            db.commit()

        redis_client.hset(f"merge_progress:{dataset_id}", mapping={
            "status": "completed",
            "processed_count": merged_size,
            "total_to_process": merged_size
        })

    except Exception as e:
        was_cancelled = bool(redis_client.get(f"merge_cancel:{dataset_id}"))
        
        # Roll back database changes: delete version
        version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
        if version_record:
            db.delete(version_record)
            db.commit()
            
        # Delete dataset if no other versions exist
        sibling_count = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == dataset_id).count()
        if sibling_count == 0:
            dataset_record = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset_record:
                db.delete(dataset_record)
                db.commit()

        # Delete partially merged CSV
        if os.path.exists(merged_path):
            try:
                os.remove(merged_path)
            except:
                pass
                
        if was_cancelled:
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "cancelled",
                "error": "Pipeline stopped by user. Partially merged files were deleted."
            })
            redis_client.delete(f"merge_cancel:{dataset_id}")
        else:
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "failed",
                "error": str(e)
            })
    finally:
        db.close()

@router.post("/merge-moses")
def merge_moses_files(
    name: str = Form(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    src_path_input: str = Form(...),
    tgt_path_input: str = Form(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Automated merging of separate Moses parallel corpus files into a single CSV.
    """
    def resolve_path(path_input: str) -> str:
        path = path_input.replace("\\", "/")
        if len(path) > 1 and path[1] == ":":
            path = path[2:].lstrip("/")

        if os.path.exists(path):
            return path
            
        parts = [p for p in path.split("/") if p]
        for i in range(len(parts)):
            subpath = "/".join(parts[i:])
            test_raw = os.path.join(RAW_DIR, subpath)
            if os.path.exists(test_raw):
                return test_raw
                
            test_data = os.path.join(DATA_DIR, subpath)
            if os.path.exists(test_data):
                return test_data
                
            if os.path.exists(subpath):
                return os.path.abspath(subpath)
                
        return path_input

    src_path = resolve_path(src_path_input)
    tgt_path = resolve_path(tgt_path_input)

    if not os.path.exists(src_path):
        raise HTTPException(status_code=400, detail=f"Source file not found: {src_path_input} (Resolved as: {src_path})")
    if not os.path.exists(tgt_path):
        raise HTTPException(status_code=400, detail=f"Target file not found: {tgt_path_input} (Resolved as: {tgt_path})")

    new_dataset = Dataset(name=name)
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)

    merged_filename = f"{new_dataset.id}_v1_merged.csv"
    merged_path = os.path.join(MERGED_DIR, merged_filename)

    new_version = DatasetVersion(
        dataset_id=new_dataset.id,
        version="v1_merged",
        status="Processing",
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        record_count=0,
        file_hash=f"placeholder-merge-{new_dataset.id}",
        storage_path=merged_path,
        processing_history={
            "is_merged": True,
            "version_type": "merged",
            "merged_from": {
                "src": src_path_input,
                "tgt": tgt_path_input
            }
        }
    )
    db.add(new_version)
    db.commit()
    db.refresh(new_version)

    redis_client.hset(f"merge_progress:{new_dataset.id}", mapping={
        "status": "processing",
        "processed_count": 0,
        "total_to_process": 0,
        "error": "",
        "dataset_name": name,
        "version_id": new_version.id,
        "phase": "merging"
    })
    redis_client.expire(f"merge_progress:{new_dataset.id}", 86400)

    background_tasks.add_task(
        bg_merge_moses,
        dataset_id=new_dataset.id,
        version_id=new_version.id,
        src_path=src_path,
        tgt_path=tgt_path,
        merged_path=merged_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    return {
        "dataset_id": new_dataset.id,
        "version_id": new_version.id,
        "name": new_dataset.name,
        "version": new_version.version,
        "status": "Processing",
        "is_merged": True
    }

@router.get("/merge-status/{dataset_id}")
def get_merge_status(dataset_id: str):
    data = redis_client.hgetall(f"merge_progress:{dataset_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Merge progress tracker not found.")
    return data

@router.post("/{version_id}/subset")
def subset_dataset(
    version_id: str,
    max_records: int,
    strategy: str = "first_n", # "first_n" or "random_sample"
    line_offset: int = -1,     # auto offset default
    db: Session = Depends(get_db)
):
    """
    Automated sentence batching. Extracts a specific number of sentence pairs (e.g. 50k, 100k)
    from a massive source dataset and saves it as a new immutable version.
    """
    parent_version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not parent_version:
        raise HTTPException(status_code=404, detail="Parent dataset version not found")

    if line_offset == -1:
        parent_history = parent_version.processing_history or {}
        line_offset = parent_history.get("next_subset_offset", 0)

    file_path = parent_version.storage_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Parent dataset file not found on disk")

    file_ext = file_path.split(".")[-1].lower()
    
    # Define subset output path with a unique UUID to prevent file clashes
    subset_filename = f"{parent_version.dataset_id}_{uuid.uuid4()}_subset_{max_records}.jsonl"
    subset_path = os.path.join(BATCHED_DIR, subset_filename)

    try:
        count = 0
        
        if strategy == "first_n":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as infile, \
                 open(subset_path, "w", encoding="utf-8") as outfile:
                 
                if file_ext == "jsonl" or "cleaned" in parent_version.version:
                    skipped = 0
                    for line in infile:
                        if line.strip():
                            if skipped < line_offset:
                                skipped += 1
                                continue
                            if count >= max_records:
                                break
                            outfile.write(line)
                            count += 1
                elif file_ext == "json":
                    data = json.load(infile)
                    records = data if isinstance(data, list) else [data]
                    for item in records[line_offset : line_offset + max_records]:
                        src = item.get("src") or item.get("source") or item.get("text")
                        tgt = item.get("tgt") or item.get("target") or item.get("translation")
                        if src and tgt:
                            outfile.write(json.dumps({"src": src, "tgt": tgt}, ensure_ascii=False) + "\n")
                            count += 1
                elif file_ext in ["csv", "tsv"]:
                    delimiter = "," if file_ext == "csv" else "\t"
                    reader = csv.reader(infile, delimiter=delimiter)
                    next(reader, None)  # Skip header
                    skipped = 0
                    for row in reader:
                        if len(row) >= 2:
                            if skipped < line_offset:
                                skipped += 1
                                continue
                            if count >= max_records:
                                break
                            outfile.write(json.dumps({"src": row[0], "tgt": row[1]}, ensure_ascii=False) + "\n")
                            count += 1
                elif file_ext == "txt":
                    skipped = 0
                    for line in infile:
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            if skipped < line_offset:
                                skipped += 1
                                continue
                            if count >= max_records:
                                break
                            outfile.write(json.dumps({"src": parts[0].strip(), "tgt": parts[1].strip()}, ensure_ascii=False) + "\n")
                            count += 1

        elif strategy == "random_sample":
            reservoir = []
            
            with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                def parsed_lines():
                    if file_ext == "jsonl" or "cleaned" in parent_version.version:
                        for line in infile:
                            if line.strip():
                                yield line.strip()
                    elif file_ext == "json":
                        data = json.load(infile)
                        records = data if isinstance(data, list) else [data]
                        for item in records:
                            src = item.get("src") or item.get("source") or item.get("text")
                            tgt = item.get("tgt") or item.get("target") or item.get("translation")
                            if src and tgt:
                                yield json.dumps({"src": src, "tgt": tgt}, ensure_ascii=False)
                    elif file_ext in ["csv", "tsv"]:
                        delimiter = "," if file_ext == "csv" else "\t"
                        reader = csv.reader(infile, delimiter=delimiter)
                        next(reader, None)
                        for row in reader:
                            if len(row) >= 2:
                                yield json.dumps({"src": row[0], "tgt": row[1]}, ensure_ascii=False)
                    elif file_ext == "txt":
                        for line in infile:
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                yield json.dumps({"src": parts[0].strip(), "tgt": parts[1].strip()}, ensure_ascii=False)
 
                for i, parsed_line in enumerate(parsed_lines()):
                    if len(reservoir) < max_records:
                        reservoir.append(parsed_line)
                    else:
                        r = random.randint(0, i)
                        if r < max_records:
                            reservoir[r] = parsed_line
                            
            count = len(reservoir)
            with open(subset_path, "w", encoding="utf-8") as outfile:
                for line in reservoir:
                    outfile.write(line + "\n")
                    
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported subsetting strategy: '{strategy}'")

        if count == 0:
            if os.path.exists(subset_path):
                os.remove(subset_path)
            raise HTTPException(status_code=400, detail="Subsetting resulted in 0 records. Check file format.")

        file_hash = calculate_sha256(subset_path)
        if line_offset > 0:
            new_version_name = f"v1_subset_{max_records}_offset_{line_offset}"
        else:
            new_version_name = f"v1_subset_{max_records}"
        
        existing_version = db.query(DatasetVersion).filter(
            DatasetVersion.file_hash == file_hash,
            DatasetVersion.dataset_id == parent_version.dataset_id
        ).first()
        
        if existing_version:
            os.remove(subset_path)
            return {
                "message": "Subset with identical content already exists.",
                "version_id": existing_version.id,
                "status": existing_version.status
            }

        new_version = DatasetVersion(
            dataset_id=parent_version.dataset_id,
            version=new_version_name,
            parent_version_id=parent_version.id,
            status="Uploaded",
            src_lang=parent_version.src_lang,
            tgt_lang=parent_version.tgt_lang,
            record_count=count,
            file_hash=file_hash,
            storage_path=subset_path,
            processing_history={
                "is_merged": parent_version.processing_history.get("is_merged") if parent_version.processing_history else False,
                "version_type": "batched"
            }
        )
        db.add(new_version)
        
        # Update parent version's processing_history to track next subset offset
        parent_history = dict(parent_version.processing_history or {})
        parent_history["next_subset_offset"] = line_offset + count
        parent_version.processing_history = parent_history
        flag_modified(parent_version, "processing_history")
        
        db.commit()
        db.refresh(new_version)
        
        transition = StateTransition(
            entity_type="dataset",
            entity_id=new_version.id,
            from_state="None",
            to_state="Uploaded",
            trigger_action=f"Automated Sentence Batching ({max_records} pairs)"
        )
        db.add(transition)
        db.commit()
        
        validate_dataset(new_version.id, db=db)
        db.refresh(new_version)
        
        return {
            "message": f"Successfully batched {count} sentences into new version '{new_version_name}'",
            "version_id": new_version.id,
            "version": new_version.version,
            "status": new_version.status,
            "record_count": new_version.record_count
        }

    except Exception as e:
        if os.path.exists(subset_path):
            os.remove(subset_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Subsetting failed: {str(e)}"
        )

@router.post("/{version_id}/validate")
def validate_dataset(version_id: str, db: Session = Depends(get_db)):
    version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version_record:
        raise HTTPException(status_code=404, detail="Dataset version not found")
        
    # If already in Validated or any later valid state, skip transition validation
    if version_record.status not in ["Validated", "Processing", "Processed", "TrainReady", "TrainingUsed"]:
        try:
            StateManager.validate_transition("dataset", version_record.status, "Validated")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        
    file_ext = version_record.storage_path.split(".")[-1].lower()
    
    # Perform validation
    report = DatasetValidator.validate_file(
        version_record.storage_path,
        file_ext,
        version_record.src_lang,
        version_record.tgt_lang
    )
    
    version_record.validation_report = report
    version_record.record_count = report["record_count"]
    
    target_state = "Validated" if report["is_valid"] or report["record_count"] > 0 else "Failed"
    
    # Only transition if status changes
    if version_record.status not in ["Validated", "Processing", "Processed", "TrainReady", "TrainingUsed"]:
        StateManager.transition_dataset(db, version_record, target_state, "Schema Validation Check")
    else:
        db.commit()
    
    return {
        "version_id": version_record.id,
        "status": version_record.status,
        "validation_report": report
    }

def bg_clean_dataset(
    dataset_id: str,
    version_id: str,
    min_length: int,
    max_length: int
):
    from backend.app.core.database import SessionLocal
    db = SessionLocal()
    
    parent_version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not parent_version:
        db.close()
        return

    temp_canonical_path = parent_version.storage_path + ".canonical.jsonl"
    clean_path = os.path.join(PROCESSED_DIR, f"{parent_version.dataset_id}_{parent_version.version}_cleaned.jsonl")
    
    try:
        file_ext = parent_version.storage_path.split(".")[-1].lower()
        
        # Update progress to "in progress"
        redis_client.hset(f"merge_progress:{dataset_id}", mapping={
            "phase": "in progress",
            "status": "processing"
        })
        
        # If it is a subset file generated by our subsetting router, it is already a clean JSONL
        if file_ext == "jsonl" or "subset" in parent_version.version:
            shutil.copyfile(parent_version.storage_path, temp_canonical_path)
        else:
            with open(parent_version.storage_path, "r", encoding="utf-8") as f, \
                 open(temp_canonical_path, "w", encoding="utf-8") as temp_out:
                if file_ext == "json":
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if redis_client.get(f"merge_cancel:{dataset_id}"):
                                raise InterruptedError("Pipeline stopped by user.")
                            src = item.get("src") or item.get("source") or item.get("text")
                            tgt = item.get("tgt") or item.get("target") or item.get("translation")
                            if src and tgt:
                                temp_out.write(json.dumps({"src": src, "tgt": tgt}, ensure_ascii=False) + "\n")
                elif file_ext in ["csv", "tsv"]:
                    delimiter = "," if file_ext == "csv" else "\t"
                    f.seek(0)
                    reader = csv.reader(f, delimiter=delimiter)
                    header = next(reader, None)
                    for row in reader:
                        if redis_client.get(f"merge_cancel:{dataset_id}"):
                            raise InterruptedError("Pipeline stopped by user.")
                        if len(row) >= 2:
                            temp_out.write(json.dumps({"src": row[0], "tgt": row[1]}, ensure_ascii=False) + "\n")
                elif file_ext == "txt":
                    for line in f:
                        if redis_client.get(f"merge_cancel:{dataset_id}"):
                            raise InterruptedError("Pipeline stopped by user.")
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            temp_out.write(json.dumps({"src": parts[0].strip(), "tgt": parts[1].strip()}, ensure_ascii=False) + "\n")

        # Now clean
        total_records = parent_version.record_count or 1
        
        def clean_progress_callback(count):
            if redis_client.get(f"merge_cancel:{dataset_id}"):
                raise InterruptedError("Pipeline stopped by user.")
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "processed_count": count,
                "total_to_process": total_records,
                "phase": "in progress",
                "status": "processing"
            })
            
        clean_metrics = DatasetCleaner.clean_dataset(
            temp_canonical_path,
            clean_path,
            {"min_length": min_length, "max_length": max_length},
            progress_callback=clean_progress_callback
        )
        
        if os.path.exists(temp_canonical_path):
            os.remove(temp_canonical_path)
            
        if redis_client.get(f"merge_cancel:{dataset_id}"):
            raise InterruptedError("Pipeline stopped by user.")
            
        file_hash = calculate_sha256(clean_path)
        
        existing_version = db.query(DatasetVersion).filter(
            DatasetVersion.file_hash == file_hash,
            DatasetVersion.dataset_id == parent_version.dataset_id
        ).first()
        
        if existing_version:
            os.remove(clean_path)
            parent_version.status = "Validated"
            db.commit()
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "ended",
                "phase": "ended",
                "processed_count": total_records,
                "total_to_process": total_records
            })
            return
            
        new_version = DatasetVersion(
            dataset_id=parent_version.dataset_id,
            version=f"{parent_version.version}_cleaned",
            parent_version_id=parent_version.id,
            status="Processed",
            src_lang=parent_version.src_lang,
            tgt_lang=parent_version.tgt_lang,
            record_count=clean_metrics["total_records_out"],
            file_hash=file_hash,
            storage_path=clean_path,
            processing_history={
                **clean_metrics, 
                "is_merged": parent_version.processing_history.get("is_merged") if parent_version.processing_history else False,
                "version_type": "cleaned"
            }
        )
        db.add(new_version)
        
        # Parent version status is transitioned to "Processed"
        parent_version.status = "Processed"
        db.commit()
        db.refresh(new_version)
        
        transition = StateTransition(
            entity_type="dataset",
            entity_id=new_version.id,
            from_state="None",
            to_state="Processed",
            trigger_action="Cleaning Pipeline Completed"
        )
        db.add(transition)
        db.commit()
        
        StateManager.transition_dataset(db, new_version, "TrainReady", "Auto Promote to TrainReady")
        
        # Set Redis status to "ended"
        redis_client.hset(f"merge_progress:{dataset_id}", mapping={
            "status": "ended",
            "phase": "ended",
            "processed_count": total_records,
            "total_to_process": total_records
        })
        
    except Exception as e:
        was_cancelled = bool(redis_client.get(f"merge_cancel:{dataset_id}"))
        
        # Roll back parent version back to Validated status
        parent_version.status = "Validated"
        db.commit()
        
        if os.path.exists(temp_canonical_path):
            try:
                os.remove(temp_canonical_path)
            except:
                pass
                
        if os.path.exists(clean_path):
            try:
                os.remove(clean_path)
            except:
                pass
                
        if was_cancelled:
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "cancelled",
                "phase": "cancelled",
                "error": "Pipeline stopped by user. Partially cleaned files were deleted."
            })
            redis_client.delete(f"merge_cancel:{dataset_id}")
        else:
            redis_client.hset(f"merge_progress:{dataset_id}", mapping={
                "status": "failed",
                "error": str(e)
            })
    finally:
        db.close()

@router.post("/{version_id}/process")
def process_dataset(
    version_id: str,
    min_length: int = 2,
    max_length: int = 300,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    parent_version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not parent_version:
        raise HTTPException(status_code=404, detail="Dataset version not found")
        
    if parent_version.status != "Validated":
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be in 'Validated' state to process (Current: {parent_version.status})"
        )
        
    StateManager.transition_dataset(db, parent_version, "Processing", "Start Clean Pipeline")
    
    # Initialize progress tracker in Redis
    redis_client.hset(f"merge_progress:{parent_version.dataset_id}", mapping={
        "status": "processing",
        "processed_count": 0,
        "total_to_process": parent_version.record_count,
        "error": "",
        "dataset_name": parent_version.version,
        "version_id": parent_version.id,
        "phase": "started",
        "is_cleaning": "1"
    })
    redis_client.expire(f"merge_progress:{parent_version.dataset_id}", 86400)
    
    if background_tasks:
        background_tasks.add_task(
            bg_clean_dataset,
            dataset_id=parent_version.dataset_id,
            version_id=parent_version.id,
            min_length=min_length,
            max_length=max_length
        )
    else:
        bg_clean_dataset(
            dataset_id=parent_version.dataset_id,
            version_id=parent_version.id,
            min_length=min_length,
            max_length=max_length
        )
    
    return {
        "dataset_id": parent_version.dataset_id,
        "version_id": parent_version.id,
        "status": "Processing"
    }

@router.get("/{version_id}/preview")
def preview_dataset(version_id: str, limit: int = 50, db: Session = Depends(get_db)):
    version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version_record:
        raise HTTPException(status_code=404, detail="Dataset version not found")
        
    file_path = version_record.storage_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")
        
    records = []
    file_ext = file_path.split(".")[-1].lower()
    
    try:
        if file_ext == "jsonl" or "cleaned" in version_record.version or "subset" in version_record.version:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(records) >= limit:
                        break
                    if line.strip():
                        records.append(json.loads(line))
        elif file_ext == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data[:limit] if isinstance(data, list) else [data]
        elif file_ext in ["csv", "tsv"]:
            delimiter = "," if file_ext == "csv" else "\t"
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                for row in reader:
                    if len(records) >= limit:
                        break
                    if len(row) >= 2:
                        records.append({"src": row[0], "tgt": row[1]})
        elif file_ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(records) >= limit:
                        break
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        records.append({"src": parts[0].strip(), "tgt": parts[1].strip()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")
        
    return {
        "version_id": version_record.id,
        "record_count": len(records),
        "records": records
    }

@router.get("")
def list_dataset_versions(db: Session = Depends(get_db)):
    """
    Lists all dataset versions registered in the system.
    """
    return db.query(DatasetVersion).order_by(DatasetVersion.created_at.asc()).all()

@router.post("/merge-cancel/{dataset_id}")
def cancel_merge(dataset_id: str):
    # Set cancel flag in Redis
    redis_client.set(f"merge_cancel:{dataset_id}", "1", ex=300)
    
    # Update progress status so the UI knows immediately
    redis_client.hset(f"merge_progress:{dataset_id}", mapping={
        "status": "cancelled",
        "error": "Cancellation requested. Rolling back changes..."
    })
    
    return {"message": "Cancellation request registered."}

@router.post("/{version_id}/archive")
def archive_dataset_version(version_id: str, db: Session = Depends(get_db)):
    version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version_record:
        raise HTTPException(status_code=404, detail="Dataset version not found")
        
    old_status = version_record.status
    version_record.status = "Archived"
    db.commit()
    
    transition = StateTransition(
        entity_type="dataset",
        entity_id=version_record.id,
        from_state=old_status,
        to_state="Archived",
        trigger_action="Developer manual archive"
    )
    db.add(transition)
    db.commit()
    
    return {"message": f"Dataset version {version_id} successfully archived."}

@router.delete("/{version_id}")
def delete_dataset_version(version_id: str, db: Session = Depends(get_db)):
    version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version_record:
        raise HTTPException(status_code=404, detail="Dataset version not found")
        
    dataset_id = version_record.dataset_id
    file_path = version_record.storage_path
    parent_version_id = version_record.parent_version_id
    
    # 1. Nullify parent_version_id references in child dataset versions
    db.query(DatasetVersion).filter(DatasetVersion.parent_version_id == version_id).update(
        {DatasetVersion.parent_version_id: None}
    )
    db.commit()
    
    # 2. Delete referencing ModelRegistry records
    db.query(ModelRegistry).filter(ModelRegistry.dataset_version_id == version_id).delete()
    db.commit()
    
    # 3. Delete referencing ExperimentRuns and their ModelRegistries
    runs = db.query(ExperimentRun).filter(ExperimentRun.dataset_version_id == version_id).all()
    run_ids = [r.id for r in runs]
    if run_ids:
        db.query(ModelRegistry).filter(ModelRegistry.experiment_run_id.in_(run_ids)).delete()
        db.query(ExperimentRun).filter(ExperimentRun.id.in_(run_ids)).delete()
        db.commit()
        
    # Delete DB record
    db.delete(version_record)
    db.commit()
    
    # Revert parent dataset status back to "Validated" if it was "Processed"
    if parent_version_id:
        parent_version = db.query(DatasetVersion).filter(DatasetVersion.id == parent_version_id).first()
        if parent_version and parent_version.status == "Processed":
            parent_version.status = "Validated"
            db.commit()
    
    # Delete parent dataset if no other versions exist
    sibling_count = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == dataset_id).count()
    if sibling_count == 0:
        dataset_record = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset_record:
            db.delete(dataset_record)
            db.commit()
            
    # Remove file from disk
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            pass
            
    return {"message": f"Dataset version {version_id} successfully deleted."}

class BulkDeleteRequest(BaseModel):
    version_ids: list[str]

@router.post("/bulk-delete")
def bulk_delete_dataset_versions(req: BulkDeleteRequest, db: Session = Depends(get_db)):
    deleted_ids = []
    for version_id in req.version_ids:
        version_record = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
        if not version_record:
            continue
            
        dataset_id = version_record.dataset_id
        file_path = version_record.storage_path
        parent_version_id = version_record.parent_version_id
        
        # 1. Nullify parent_version_id references in child dataset versions
        db.query(DatasetVersion).filter(DatasetVersion.parent_version_id == version_id).update(
            {DatasetVersion.parent_version_id: None}
        )
        db.commit()
        
        # 2. Delete referencing ModelRegistry records
        db.query(ModelRegistry).filter(ModelRegistry.dataset_version_id == version_id).delete()
        db.commit()
        
        # 3. Delete referencing ExperimentRuns and their ModelRegistries
        runs = db.query(ExperimentRun).filter(ExperimentRun.dataset_version_id == version_id).all()
        run_ids = [r.id for r in runs]
        if run_ids:
            db.query(ModelRegistry).filter(ModelRegistry.experiment_run_id.in_(run_ids)).delete()
            db.query(ExperimentRun).filter(ExperimentRun.id.in_(run_ids)).delete()
            db.commit()
            
        # Delete DB record
        db.delete(version_record)
        db.commit()
        
        # Revert parent dataset status back to "Validated" if it was "Processed"
        if parent_version_id:
            parent_version = db.query(DatasetVersion).filter(DatasetVersion.id == parent_version_id).first()
            if parent_version and parent_version.status == "Processed":
                parent_version.status = "Validated"
                db.commit()
        
        # Delete parent dataset if no other versions exist
        sibling_count = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == dataset_id).count()
        if sibling_count == 0:
            dataset_record = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset_record:
                db.delete(dataset_record)
                db.commit()
                
        # Remove file from disk
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        deleted_ids.append(version_id)
        
    return {"message": f"Successfully deleted {len(deleted_ids)} dataset versions.", "deleted_ids": deleted_ids}

@router.post("/purge")
def purge_all_datasets(db: Session = Depends(get_db)):
    from sqlalchemy import text
    try:
        db.execute(text("TRUNCATE datasets, dataset_versions, state_transitions, jobs, experiment_runs, model_registry CASCADE;"))
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database purge failed: {str(e)}")
        
    directories = ["merged", "batched", "processed"]
    for folder in directories:
        path = os.path.join(DATA_DIR, folder)
        if os.path.exists(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    pass
                    
    return {"message": "All database records and dataset files have been successfully purged."}
