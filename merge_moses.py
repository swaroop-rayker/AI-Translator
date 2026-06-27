import csv
import argparse
import os

def merge_moses(src_file, tgt_file, output_csv):
    """
    Merges two separate Moses text files (source and target)
    into a single CSV file ready for platform upload.
    """
    if not os.path.exists(src_file):
        print(f"Error: Source file '{src_file}' not found.")
        return
    if not os.path.exists(tgt_file):
        print(f"Error: Target file '{tgt_file}' not found.")
        return

    print(f"Merging Moses files:")
    print(f"  Source: {src_file}")
    print(f"  Target: {tgt_file}")
    print(f"Writing to: {output_csv}...")

    with open(src_file, "r", encoding="utf-8", errors="ignore") as f_src, \
         open(tgt_file, "r", encoding="utf-8", errors="ignore") as f_tgt, \
         open(output_csv, "w", encoding="utf-8", newline="") as f_out:
         
        writer = csv.writer(f_out)
        # Write header
        writer.writerow(["src", "tgt"])
        
        count = 0
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_text = src_line.strip()
            tgt_text = tgt_line.strip()
            
            if src_text and tgt_text:
                writer.writerow([src_text, tgt_text])
                count += 1

    print(f"Successfully merged {count} parallel sentences into '{output_csv}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge separate Moses parallel corpus files into a single CSV.")
    parser.add_argument("--src", "-s", type=str, required=True, help="Path to source language file (e.g. corpus.en)")
    parser.add_argument("--tgt", "-t", type=str, required=True, help="Path to target language file (e.g. corpus.kn)")
    parser.add_argument("--output", "-o", type=str, default="moses_merged.csv", help="Output CSV path")
    
    args = parser.parse_args()
    merge_moses(args.src, args.tgt, args.output)
