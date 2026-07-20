#!/usr/bin/env python3
"""
merge_parent_comments.py

Merges multiple parent comment JSONL files (from parallel array jobs) into a single file.
Deduplicates by comment ID since Reddit IDs are globally unique.

Usage:
  python merge_parent_comments.py \
    --input-dir output/parent_comments \
    --output parent_comments.jsonl
"""

import argparse
import orjson
import glob


def stream_jsonl(path):
    """
    Generator to read JSONL file line by line.
    
    Args:
        path: Path to JSONL file
    
    Yields:
        Parsed JSON objects (dicts) from each line
    """
    with open(path, "rb") as f:
        for line in f:
            yield orjson.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description="Merge parent comment JSONL files from parallel array jobs"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing parent comment JSONL files")
    parser.add_argument("--output", required=True, help="Output merged JSONL file path")
    parser.add_argument("--pattern", default="parent_comments_*.jsonl", help="File pattern to match (default: parent_comments_*.jsonl)")
    
    args = parser.parse_args()
    
    # Find all input files matching the pattern
    input_pattern = f"{args.input_dir}/{args.pattern}"
    input_files = sorted(glob.glob(input_pattern))
    
    if not input_files:
        print(f"[ERROR] No files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(input_files)} input files to merge:")
    for f in input_files:
        print(f"  - {f}")
    
    # Merge files, deduplicating by comment ID
    seen_ids = set()
    merged_count = 0
    
    print(f"\nMerging to {args.output}...")
    with open(args.output, "wb") as out_f:
        for input_file in input_files:
            print(f"  Processing {input_file}...")
            file_count = 0
            for record in stream_jsonl(input_file):
                comment_id = record.get("id")
                # Skip duplicates (shouldn't happen with Reddit IDs, but safe to check)
                if comment_id not in seen_ids:
                    seen_ids.add(comment_id)
                    out_f.write(orjson.dumps(record) + b"\n")
                    merged_count += 1
                    file_count += 1
            print(f"    Added {file_count} unique comments from this file")
    
    print(f"\n=== Summary ===")
    print(f"Total unique parent comments merged: {merged_count}")
    print(f"Output file: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
