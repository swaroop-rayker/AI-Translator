import unicodedata
import re
import json

class DatasetCleaner:
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Applies Unicode NFKC normalization and trims whitespace.
        """
        if not text:
            return ""
        # Unicode normalization (NFKC)
        normalized = unicodedata.normalize("NFKC", text)
        # Collapse multiple spaces, tabs, newlines
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @classmethod
    def clean_record(cls, src: str, tgt: str) -> tuple:
        """
        Cleans a single translation record.
        """
        return cls.normalize_text(src), cls.normalize_text(tgt)

    @classmethod
    def clean_dataset(cls, input_file_path: str, output_file_path: str, config: dict, progress_callback=None) -> dict:
        """
        Applies whitespace/unicode normalization, deduplication, and length filters.
        Writes clean records to output_file_path.
        Returns a dict of metrics (duplicates_removed, length_filtered, total_in, total_out).
        """
        min_len = config.get("min_length", 2)
        max_len = config.get("max_length", 300)
        
        seen_pairs = set()
        
        total_in = 0
        total_out = 0
        duplicates_removed = 0
        empty_removed = 0
        length_filtered = 0
        
        # Read raw/parsed records from input file (assumed to be canonical jsonl format)
        with open(input_file_path, "r", encoding="utf-8") as infile, \
             open(output_file_path, "w", encoding="utf-8") as outfile:
             
            for line_idx, line in enumerate(infile, 1):
                if not line.strip():
                    continue
                if progress_callback and line_idx % 1000 == 0:
                    progress_callback(line_idx)
                total_in += 1
                try:
                    record = json.loads(line)
                    # Normalize keys
                    src = record.get("src") or record.get("source") or record.get("text")
                    tgt = record.get("tgt") or record.get("target") or record.get("translation")
                    
                    if not src or not tgt:
                        empty_removed += 1
                        continue
                        
                    src_clean, tgt_clean = cls.clean_record(str(src), str(tgt))
                    
                    # Checks
                    if not src_clean or not tgt_clean:
                        empty_removed += 1
                        continue
                        
                    # Length check
                    if len(src_clean) < min_len or len(tgt_clean) < min_len or \
                       len(src_clean) > max_len or len(tgt_clean) > max_len:
                        length_filtered += 1
                        continue
                        
                    # Deduplication
                    pair_key = (src_clean, tgt_clean)
                    if pair_key in seen_pairs:
                        duplicates_removed += 1
                        continue
                        
                    seen_pairs.add(pair_key)
                    
                    # Write clean canonical format
                    outfile.write(json.dumps({"src": src_clean, "tgt": tgt_clean}, ensure_ascii=False) + "\n")
                    total_out += 1
                except Exception:
                    empty_removed += 1
                    
        return {
            "total_records_in": total_in,
            "total_records_out": total_out,
            "duplicates_removed": duplicates_removed,
            "empty_or_null_removed": empty_removed,
            "length_filtered": length_filtered
        }
