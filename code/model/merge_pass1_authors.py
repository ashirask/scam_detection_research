"""
merge_pass1_authors.py
------------------------
Lightweight merge script for Pass 1 author lists.
Reads all pass1_authors_*.txt files from all months and file types,
takes the set union, and writes a single deduplicated global author list.
Supports optional capping via random sample for balanced bot:human ratio.

Usage:
  python merge_pass1_authors.py \
    --input-dir results/ \
    --output bot_authors_global.txt \
    --max-authors 5000 \
    --seed 42
"""

import argparse
import glob
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge Pass 1 author lists into global bot author list.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing pass1_authors_*.txt files"
    )
    parser.add_argument(
        "--output",
        default="bot_authors_global.txt",
        help="Output file for global author list (default: bot_authors_global.txt)"
    )
    parser.add_argument(
        "--max-authors",
        type=int,
        default=None,
        help="Cap total unique bot authors via random sample. "
                 "If not set, all qualifying authors are kept."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Collect all unique authors across all months and file types
    all_authors = set()
    pattern = str(input_dir / "pass1_authors_*.txt")
    files = sorted(glob.glob(pattern))
    
    print(f"Scanning for pass1_authors_*.txt files in {input_dir}...")
    print(f"  Found {len(files)} files")
    
    for filepath in files:
        # Skip summary files (they end with _summary.txt)
        if "_summary" in filepath:
            print(f"  Skipping {filepath} (summary file)")
            continue
        print(f"  Reading {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                username = line.strip()
                if username:
                    all_authors.add(username)
    
    print(f"\nTotal unique bot authors found: {len(all_authors)}")
    
    # Apply cap via simple random sample if requested
    random.seed(args.seed)
    if args.max_authors and len(all_authors) > args.max_authors:
        selected = set(random.sample(sorted(all_authors), args.max_authors))
        print(f"Capped to {len(selected)} authors (random sample, seed={args.seed})")
    else:
        selected = all_authors
        if args.max_authors:
            print(f"No cap applied — only {len(all_authors)} authors found (< {args.max_authors})")
        else:
            print(f"No cap applied — keeping all {len(selected)} authors")
    
    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        for author in sorted(selected):
            f.write(author + '\n')
    
    print(f"\nWritten to: {output_path}")


if __name__ == "__main__":
    main()
