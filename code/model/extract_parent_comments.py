#!/usr/bin/env python3
"""
extract_parent_comments.py

Extracts parent comment information from ZST files to improve temporal feature coverage.

Two modes of operation:

1. Single-file mode (for parallel array jobs):
   python extract_parent_comments.py \
     --parent-ids-file parent_ids.txt \
     --comments-zst RC_2024-01.zst \
     --output parent_comments_RC_2024-01.jsonl

2. Full mode (collect IDs + search all ZSTs sequentially):
   python extract_parent_comments.py \
     --comments-bot user_comments_bots.jsonl \
     --comments-human user_comments_humans.jsonl \
     --comments-zst RC_2024-01.zst RC_2024-02.zst ... \
     --output parent_comments.jsonl

This addresses the limitation where parent comments are not in the 10,000 user dataset,
which causes high NaN rates in temporal features (currently ~77% missing).
"""

import argparse
import orjson
import zstandard as zstd
import sys
import io
from collections import defaultdict


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


def collect_parent_ids(comments_bot_path, comments_human_path):
    """
    Collect all unique parent comment IDs from bot and human comment JSONL files.
    Only collects parent IDs that start with "t1_" (comment-to-comment replies).
    
    Args:
        comments_bot_path: Path to bot comments JSONL
        comments_human_path: Path to human comments JSONL
    
    Returns:
        Set of parent comment IDs (without "t1_" prefix)
    """
    parent_ids = set()
    
    print("Collecting parent IDs from bot comments...")
    for record in stream_jsonl(comments_bot_path):
        for comment in record.get("comments", []):
            parent_id = comment.get("parent_id")
            # Convert to string in case it's stored as integer
            parent_id = str(parent_id) if parent_id is not None else ""
            # Only collect parent IDs for comment-to-comment replies (t1_)
            if parent_id.startswith("t1_"):
                parent_comment_id = parent_id[3:]  # Strip "t1_" prefix
                parent_ids.add(parent_comment_id)
    
    print(f"  Found {len(parent_ids)} unique parent IDs from bot comments")
    
    print("Collecting parent IDs from human comments...")
    for record in stream_jsonl(comments_human_path):
        for comment in record.get("comments", []):
            parent_id = comment.get("parent_id")
            parent_id = str(parent_id) if parent_id is not None else ""
            if parent_id.startswith("t1_"):
                parent_comment_id = parent_id[3:]
                parent_ids.add(parent_comment_id)
    
    print(f"  Total unique parent IDs: {len(parent_ids)}")
    
    return parent_ids


def load_parent_ids_from_file(parent_ids_path):
    """
    Load parent comment IDs from a text file (one ID per line).
    Used for parallel array jobs where IDs are pre-collected.
    
    Args:
        parent_ids_path: Path to text file with parent IDs
    
    Returns:
        Set of parent comment IDs
    """
    parent_ids = set()
    with open(parent_ids_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parent_ids.add(line)
    print(f"Loaded {len(parent_ids)} parent IDs from {parent_ids_path}")
    return parent_ids


def save_parent_ids_to_file(parent_ids, output_path):
    """
    Save parent comment IDs to a text file (one ID per line).
    Used to pre-collect IDs for parallel array jobs.
    
    Args:
        parent_ids: Set of parent comment IDs
        output_path: Path to output text file
    """
    with open(output_path, "w") as f:
        for parent_id in parent_ids:
            f.write(parent_id + "\n")
    print(f"Saved {len(parent_ids)} parent IDs to {output_path}")


def decompress_zst(zst_path):
    """
    Decompress a ZST file and return a file-like object for reading.
    
    Args:
        zst_path: Path to .zst file
    
    Returns:
        File-like object for reading decompressed data
    """
    with open(zst_path, "rb") as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        reader = dctx.stream_reader(f)
        return reader


def search_zst_for_parents(zst_path, parent_ids, found_parents):
    """
    Search a single ZST file for parent comment IDs.
    Extracts minimal data (id, created_utc, author) for found parents.
    Handles corrupted files gracefully by continuing with partial results.
    
    Args:
        zst_path: Path to .zst file
        parent_ids: Set of parent comment IDs to search for
        found_parents: Dictionary to store found parent comments {id: data}
    
    Returns:
        Tuple of (new_count, corruption_detected)
        - new_count: Number of new parents found in this file
        - corruption_detected: Boolean indicating if file corruption was detected
    """
    new_count = 0
    corruption_detected = False
    stats = {
        "total_lines": 0,
        "parse_errors": 0,
    }
    
    print(f"  Searching {zst_path}...")
    
    try:
        # Decompress and read line by line with error handling
        with open(zst_path, "rb") as f:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                for line in text_stream:
                    stats["total_lines"] += 1
                    
                    try:
                        comment = orjson.loads(line)
                    except Exception:
                        stats["parse_errors"] += 1
                        continue
                    
                    comment_id = comment.get("id")
                    
                    # Check if this comment is one of the parents we're looking for
                    if comment_id in parent_ids and comment_id not in found_parents:
                        # Extract only the minimal data needed for temporal features
                        parent_data = {
                            "id": comment_id,
                            "created_utc": comment.get("created_utc"),
                            "author": comment.get("author"),
                        }
                        found_parents[comment_id] = parent_data
                        new_count += 1
                    
                    # Progress update every 1M lines
                    if stats["total_lines"] % 1_000_000 == 0:
                        print(f"    Processed {stats['total_lines']:,} lines, found {new_count} parents so far...")
    except zstd.ZstdError as e:
        corruption_detected = True
        print(f"    [ERROR] Zstd decompression error after {stats['total_lines']} lines: {e}", file=sys.stderr)
        print(f"    [ERROR] File may be corrupted or truncated: {zst_path}", file=sys.stderr)
    except Exception as e:
        corruption_detected = True
        print(f"    [ERROR] Unexpected error after {stats['total_lines']} lines: {e}", file=sys.stderr)
    
    print(f"    Found {new_count} new parents in this file")
    if stats["parse_errors"] > 0:
        print(f"    Parse errors: {stats['parse_errors']:,}")
    if corruption_detected:
        print(f"    [WARN] Corruption detected - partial results saved")
    
    return new_count, corruption_detected


def main():
    parser = argparse.ArgumentParser(
        description="Extract parent comment data from ZST files to improve temporal feature coverage"
    )
    
    # Mode 1: Collect parent IDs from JSONL files (for pre-collection step)
    parser.add_argument("--comments-bot", help="Path to bot comments JSONL")
    parser.add_argument("--comments-human", help="Path to human comments JSONL")
    
    # Mode 2: Load pre-collected parent IDs from file (for parallel jobs)
    parser.add_argument("--parent-ids-file", help="Path to text file with pre-collected parent IDs")
    parser.add_argument("--save-parent-ids", help="Save collected parent IDs to this file (for parallel jobs)")
    
    # Common arguments
    parser.add_argument("--comments-zst", nargs="+", help="One or more .zst comment files to search (optional for ID collection only)")
    parser.add_argument("--output", required=True, help="Output JSONL file path for parent comments")
    
    args = parser.parse_args()
    
    # Determine mode and get parent IDs
    parent_ids = None
    
    if args.parent_ids_file:
        # Mode: Load pre-collected IDs from file (for parallel array jobs)
        print("=== Loading pre-collected parent IDs ===")
        parent_ids = load_parent_ids_from_file(args.parent_ids_file)
    elif args.comments_bot and args.comments_human:
        # Mode: Collect IDs from JSONL files
        print("=== Phase 1: Collecting parent IDs ===")
        parent_ids = collect_parent_ids(args.comments_bot, args.comments_human)
        
        # Save to file if requested (for parallel jobs)
        if args.save_parent_ids:
            save_parent_ids_to_file(parent_ids, args.save_parent_ids)
    else:
        parser.error("Must provide either --parent-ids-file OR both --comments-bot and --comments-human")
    
    print(f"\nTotal parent IDs to search for: {len(parent_ids)}")
    
    # Phase 2: Search for these parent IDs in the ZST files (optional)
    # Skip if only collecting IDs for parallel jobs
    if not args.comments_zst:
        print("\nSkipping ZST search (--comments-zst not provided)")
        print("Parent IDs saved. Run with --comments-zst to search for parent comments.")
        return
    
    print("\n=== Phase 2: Searching ZST files for parent comments ===")
    found_parents = {}  # Dictionary to store found parent comments {id: data}
    any_corruption = False
    
    for zst_path in args.comments_zst:
        new_count, corruption = search_zst_for_parents(zst_path, parent_ids, found_parents)
        if corruption:
            any_corruption = True
    
    print(f"\n=== Summary ===")
    print(f"Parent IDs searched for: {len(parent_ids)}")
    print(f"Parents found: {len(found_parents)}")
    print(f"Coverage: {len(found_parents) / len(parent_ids) * 100:.1f}%")
    if any_corruption:
        print(f"Corruption detected in one or more files - partial results saved")
    
    # Write found parent comments to output JSONL file
    print(f"\nWriting {len(found_parents)} parent comments to {args.output}")
    with open(args.output, "wb") as f:
        for parent_id, parent_data in found_parents.items():
            f.write(orjson.dumps(parent_data) + b"\n")
    
    print("Done!")
    
    # Exit with error code if corruption was detected
    if any_corruption:
        sys.exit(1)


if __name__ == "__main__":
    main()
