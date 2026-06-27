import argparse
import os

def create_subset(input_path, output_path, max_sentences=50000):
    """
    Reads a large parallel corpus file (TSV/CSV/TXT) and creates a smaller
    subset of max_sentences for fast training and testing on laptop hardware.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    print(f"Reading from {input_path}...")
    print(f"Extracting first {max_sentences} sentences to {output_path}...")

    # Detect extension and delimiter
    ext = os.path.splitext(input_path)[1].lower()
    
    count = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
         
        # If CSV/TSV, write header
        if ext in [".csv", ".tsv"]:
            header = infile.readline()
            outfile.write(header)
            
        for line in infile:
            if count >= max_sentences:
                break
            if line.strip():
                outfile.write(line)
                count += 1

    print(f"Successfully wrote {count} sentences to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset massive parallel translation corpora.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the downloaded NLLB/OPUS file")
    parser.add_argument("--output", "-o", type=str, default="sample_subset.csv", help="Path to save the smaller subset")
    parser.add_argument("--size", "-s", type=int, default=50000, help="Number of sentences to extract")
    
    args = parser.parse_args()
    create_subset(args.input, args.output, args.size)
