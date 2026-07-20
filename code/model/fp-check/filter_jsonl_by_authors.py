#!/usr/bin/env python3
"""
filter_jsonl_by_authors.py
===========================
Filter JSONL files by keeping or removing specific authors.

This script is used to clean up merged JSONL files after bot detection rule
improvements. It can either keep only specified authors or remove specified
authors from the JSONL file.

USAGE:
# Remove eliminated authors
python filter_jsonl_by_authors.py \
    --input user_comments_bot.jsonl \
    --output user_comments_bot_filtered.jsonl \
    --remove-list eliminated_authors.txt

# Keep only specified authors  
python filter_jsonl_by_authors.py \
    --input user_comments_bot.jsonl \
    --output user_comments_bot_filtered.jsonl \
    --keep-list keep_authors.txt
"""

import argparse
import json
import re
from pathlib import Path


def extract_author_from_jsonl(line: str):
    """
    Extract the top-level 'author' field from a JSONL line without parsing the entire JSON.
    
    This is more efficient than json.loads() for large records with big arrays.
    It uses regex to find the first "author":"value" pattern (top-level field).
    
    Args:
        line: A JSONL line string
    
    Returns:
        str or None: The author value if found, None otherwise
    """
    # Look for author in the first 100 characters to avoid matching nested authors
    # Top-level author should be near the start of the JSON object
    line_start = line[:100]
    
    # Pattern to match "author":"value" handling escaped quotes
    pattern = r'"author"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
    match = re.search(pattern, line_start)
    if match:
        # Unescape the value (handle \", \\, etc.)
        author = match.group(1)
        # Decode JSON escape sequences
        try:
            return json.loads(f'"{author}"')
        except json.JSONDecodeError:
            return author
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file to filter"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file (filtered)"
    )
    parser.add_argument(
        "--remove-list",
        help="Text file with authors to remove (one per line)"
    )
    parser.add_argument(
        "--keep-list",
        help="Text file with authors to keep (one per line)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.remove_list and not args.keep_list:
        print("[ERROR] Must specify either --remove-list or --keep-list")
        return 1
    
    if args.remove_list and args.keep_list:
        print("[ERROR] Cannot specify both --remove-list and --keep-list")
        return 1
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1
    
    # Load author filter list
    if args.remove_list:
        filter_path = Path(args.remove_list)
        mode = "remove"
        print(f"Loading remove list from {filter_path}...")
    else:
        filter_path = Path(args.keep_list)
        mode = "keep"
        print(f"Loading keep list from {filter_path}...")
    
    with open(filter_path, 'r', encoding='utf-8') as f:
        filter_authors = set(line.strip() for line in f if line.strip())
    
    print(f"  Loaded {len(filter_authors)} authors to {mode}")
    
    # Process JSONL file
    print(f"\nProcessing {input_path}...")
    total_lines = 0
    kept_lines = 0
    removed_lines = 0
    parse_errors = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_lines += 1
            
            if total_lines % 10000 == 0:
                print(f"  Processed {total_lines:,} lines, kept {kept_lines:,}, removed {removed_lines:,}...")
            
            line = line.strip()
            if not line:
                continue
            
            try:
                # Extract author efficiently without parsing entire JSON
                author = extract_author_from_jsonl(line)
                
                if not author:
                    parse_errors += 1
                    continue
                
                # Apply filter
                if mode == "remove":
                    should_keep = author not in filter_authors
                else:  # keep
                    should_keep = author in filter_authors
                
                if should_keep:
                    outfile.write(line + '\n')
                    kept_lines += 1
                else:
                    removed_lines += 1
                    
            except Exception:
                parse_errors += 1
                continue
    
    print(f"\nDone! Results written to {output_path}")
    print(f"\nSummary:")
    print(f"  Total lines processed: {total_lines:,}")
    print(f"  Lines kept: {kept_lines:,}")
    print(f"  Lines removed: {removed_lines:,}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Filter mode: {mode}")
    print(f"  Filter list size: {len(filter_authors)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
