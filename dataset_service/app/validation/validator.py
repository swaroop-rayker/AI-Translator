import re
import json
import csv
import os

class ByteCounterFileWrapper:
    def __init__(self, file_obj):
        self.file_obj = file_obj
        self.bytes_read = 0

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.file_obj)
        self.bytes_read += len(line.encode('utf-8', errors='ignore'))
        return line

class DatasetValidator:
    @staticmethod
    def detect_lang(text: str) -> str:
        """
        Detects if text contains characters from Kannada, Malayalam, or English/Latin.
        Returns 'kn', 'ml', 'en', or 'unknown'.
        """
        if not text:
            return "unknown"
        
        # Count character types
        kn_count = 0
        ml_count = 0
        en_count = 0
        
        for char in text:
            val = ord(char)
            if 0x0C80 <= val <= 0x0CFF:
                kn_count += 1
            elif 0x0D00 <= val <= 0x0D7F:
                ml_count += 1
            elif 32 <= val <= 126 or val in [10, 13, 9]: # Basic ASCII printable + formatting
                en_count += 1
                
        total = kn_count + ml_count + en_count
        if total == 0:
            return "unknown"
            
        # Determine language by majority character count
        max_val = max(kn_count, ml_count, en_count)
        if max_val == kn_count and kn_count > 0:
            return "kn"
        elif max_val == ml_count and ml_count > 0:
            return "ml"
        elif max_val == en_count and en_count > 0:
            return "en"
        return "unknown"

    @classmethod
    def validate_file(cls, file_path: str, format_type: str, src_lang: str, tgt_lang: str, progress_callback=None, max_val_rows: int = 20000) -> dict:
        """
        Validates the schema, content, and quality of the uploaded dataset.
        Returns a dict containing validation metrics and list of validation issues.
        """
        issues = []
        record_count = 0
        empty_rows = 0
        missing_fields = 0
        malformed_rows = 0
        lang_mismatches = 0
        extremely_short = 0
        extremely_long = 0
        
        def validate_item(idx, item):
            nonlocal record_count, empty_rows, missing_fields, lang_mismatches, extremely_short, extremely_long
            
            src = item.get("src") or item.get("source") or item.get("text")
            tgt = item.get("tgt") or item.get("target") or item.get("translation")
            
            if src is None or tgt is None:
                missing_fields += 1
                if len(issues) < 100:
                    issues.append(f"Item {idx}: Missing 'src' or 'tgt' field")
                return
            
            src = str(src).strip()
            tgt = str(tgt).strip()
            
            if not src or not tgt:
                empty_rows += 1
                return
                
            record_count += 1
                
            # Length checks
            if len(src) < 2 or len(tgt) < 2:
                extremely_short += 1
            if len(src) > 500 or len(tgt) > 500:
                extremely_long += 1
                
            # Language checks
            src_detected = cls.detect_lang(src)
            tgt_detected = cls.detect_lang(tgt)
            
            if src_lang == "en" and src_detected not in ["en", "unknown"]:
                lang_mismatches += 1
                if len(issues) < 100:
                    issues.append(f"Item {idx}: Source lang mismatch. Expected 'en', detected '{src_detected}'")
            elif src_lang in ["kn", "ml"] and src_detected != src_lang and src_detected != "unknown":
                lang_mismatches += 1
                if len(issues) < 100:
                    issues.append(f"Item {idx}: Source lang mismatch. Expected '{src_lang}', detected '{src_detected}'")
                
            if tgt_lang in ["kn", "ml"] and tgt_detected != tgt_lang and tgt_detected != "unknown":
                lang_mismatches += 1
                if len(issues) < 100:
                    issues.append(f"Item {idx}: Target lang mismatch. Expected '{tgt_lang}', detected '{tgt_detected}'")
 
        # Set up byte-based progress reporting
        try:
            file_size = os.path.getsize(file_path)
        except:
            file_size = 1
            
        report_interval_bytes = max(1024 * 1024, file_size // 100) # Report every 1% or 1MB
        last_reported_bytes = 0
        bytes_processed = 0

        # Read file contents and parse line-by-line to protect system memory (O(1) Memory footprint)
        try:
            if format_type == "jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, 1):
                        bytes_processed += len(line.encode('utf-8', errors='ignore'))
                        
                        if line_idx > max_val_rows:
                            record_count += 1
                            for _ in f:
                                record_count += 1
                            break
                            
                        if not line.strip():
                            empty_rows += 1
                            continue
                        try:
                            item = json.loads(line)
                            validate_item(line_idx, item)
                            if progress_callback and bytes_processed - last_reported_bytes >= report_interval_bytes:
                                progress_callback(bytes_processed, file_size)
                                last_reported_bytes = bytes_processed
                        except Exception as e:
                            malformed_rows += 1
                            if len(issues) < 100:
                                issues.append(f"Line {line_idx}: Malformed JSON - {str(e)}")
            elif format_type == "json":
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            total_items = len(data)
                            record_count = total_items
                            # Validate only a sample
                            for idx, item in enumerate(data[:max_val_rows], 1):
                                validate_item(idx, item)
                                if progress_callback and idx % max(1, min(total_items, max_val_rows) // 100) == 0:
                                    progress_callback(int((idx / total_items) * file_size), file_size)
                        elif isinstance(data, dict):
                            validate_item(1, data)
                            if progress_callback:
                                progress_callback(file_size, file_size)
                        else:
                            issues.append("Root JSON must be a list or object")
                    except Exception as e:
                        malformed_rows += 1
                        issues.append(f"Malformed JSON file: {str(e)}")
            elif format_type in ["csv", "tsv"]:
                delimiter = "," if format_type == "csv" else "\t"
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        sample = f.read(2048)
                        f.seek(0)
                        # Avoid calling has_header directly to avoid sniffer issues
                        has_header = any(h in sample.lower() for h in ["src", "tgt", "source", "target", "english", "kannada", "malayalam"])
                    except:
                        has_header = True
                        
                    wrapper = ByteCounterFileWrapper(f)
                    reader = csv.reader(wrapper, delimiter=delimiter)
                    header = next(reader, None) if has_header else ["src", "tgt"]
                    
                    # Normalize header fields
                    src_col_idx = -1
                    tgt_col_idx = -1
                    for idx, h in enumerate(header or []):
                        h_clean = h.strip().lower()
                        if h_clean in ["src", "source", "english", "text", "kannada", "malayalam"]:
                            if h_clean in ["src", "source", "english", "text"] and src_col_idx == -1:
                                src_col_idx = idx
                            else:
                                tgt_col_idx = idx
                    
                    if src_col_idx == -1 or tgt_col_idx == -1:
                        src_col_idx = 0
                        tgt_col_idx = min(1, len(header or []) - 1)
                        
                    for line_idx, row in enumerate(reader, 1):
                        bytes_processed = wrapper.bytes_read
                        
                        if line_idx > max_val_rows:
                            record_count += 1
                            for _ in reader:
                                record_count += 1
                            break
                            
                        if not row or not any(row):
                            empty_rows += 1
                            continue
                        if len(row) <= max(src_col_idx, tgt_col_idx):
                            malformed_rows += 1
                            if len(issues) < 100:
                                issues.append(f"Row {line_idx}: Missing target column index")
                            continue
                        
                        validate_item(line_idx, {"src": row[src_col_idx], "tgt": row[tgt_col_idx]})
                        if progress_callback and bytes_processed - last_reported_bytes >= report_interval_bytes:
                            progress_callback(bytes_processed, file_size)
                            last_reported_bytes = bytes_processed
            elif format_type == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, 1):
                        bytes_processed += len(line.encode('utf-8', errors='ignore'))
                        
                        if line_idx > max_val_rows:
                            record_count += 1
                            for _ in f:
                                record_count += 1
                            break
                            
                        if not line.strip():
                            empty_rows += 1
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            validate_item(line_idx, {"src": parts[0].strip(), "tgt": parts[1].strip()})
                            if progress_callback and bytes_processed - last_reported_bytes >= report_interval_bytes:
                                progress_callback(bytes_processed, file_size)
                                last_reported_bytes = bytes_processed
                        else:
                            malformed_rows += 1
                            if len(issues) < 100:
                                issues.append(f"Line {line_idx}: Missing tab separator for src/tgt")
            else:
                issues.append(f"Unsupported format type: '{format_type}'")
                
        except Exception as e:
            issues.append(f"Fatal parsing error: {str(e)}")
            
        # Perform one final progress report to force 100% when finished
        if progress_callback:
            try:
                progress_callback(file_size, file_size)
            except:
                pass

        is_valid = len(issues) == 0 and record_count > 0
        
        return {
            "is_valid": is_valid,
            "record_count": record_count,
            "empty_rows": empty_rows,
            "missing_fields": missing_fields,
            "malformed_rows": malformed_rows,
            "language_mismatches": lang_mismatches,
            "extremely_short_sequences": extremely_short,
            "extremely_long_sequences": extremely_long,
            "issues": issues[:100]  # Cap issues in report to first 100
        }

