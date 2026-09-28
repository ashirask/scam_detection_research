#!/usr/bin/env python3
"""
merge_offset_indexes.py

Merge offset index parquet files from multiple JSONL sources, using duplicate
resolution decisions from merge_parquet_files.py to maintain consistency.
This ensures that the index merge follows the same duplicate resolution
as the parquet data merge.

This script is HPC-cluster compatible:
- Handles index merging efficiently
- Uses duplicate_details.json for consistent resolution
- Includes comprehensive error handling and logging
- All imports are at top level for cluster compatibility

Usage:
  python merge_offset_indexes.py \
    --comments-index-main comments_main_index.parquet \
    --comments-index-manual comments_manual_index.parquet \
    --submissions-index-main submissions_main_index.parquet \
    --submissions-index-manual submissions_manual_index.parquet \
    --jsonl-main /path/to/main/ \
    --jsonl-manual /path/to/manual/ \
    --duplicate-details data/merged_authors.duplicate_details.json \
    --output-dir inference/indexes/ \
    --comments-file user_comments_human.jsonl \
    --submissions-file user_submissions_human.jsonl

Note: Expects JSONL files at:
  - /path/to/main/user_comments_human.jsonl
  - /path/to/manual/user_comments_human.jsonl
  - /path/to/main/user_submissions_human.jsonl
  - /path/to/manual/user_submissions_human.jsonl

HPC Cluster Example:
  sbatch merge_indexes.sbatch  # Or your cluster's job submission system
"""

import os
import sys
import argparse
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging setup for HPC cluster compatibility
# ---------------------------------------------------------------------------

def setup_logging():
    """
    Configure logging for HPC cluster output to .out and .err files.
    
    INFO and DEBUG messages go to stdout (.out file)
    WARNING and ERROR messages go to stderr (.err file)
    All messages include timestamps for debugging.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # stdout handler for INFO/DEBUG (goes to .out file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                          datefmt='%Y-%m-%d %H:%M:%S')
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)
    
    # stderr handler for WARNING/ERROR (goes to .err file)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                          datefmt='%Y-%m-%d %H:%M:%S')
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)
    
    return logger


# Initialize logging
logger = setup_logging()


# ---------------------------------------------------------------------------
# Key functions
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy/pandas types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_duplicate_resolution(duplicate_details_path: str) -> dict:
    """
    Load duplicate resolution decisions from merge_parquet_files.py output.
    
    Args:
        duplicate_details_path: Path to merged_authors.duplicate_details.json
    
    Returns:
        Dictionary mapping author -> "primary" or "secondary"
    """
    logger.info(f"Loading duplicate resolution from {duplicate_details_path}")
    
    if not os.path.exists(duplicate_details_path):
        raise FileNotFoundError(f"Duplicate details file not found: {duplicate_details_path}")
    
    with open(duplicate_details_path, "r") as f:
        duplicate_data = json.load(f)
    
    # Build resolution mapping from between_files section
    resolution_map = {}
    between_files = duplicate_data.get("between_files", [])
    
    for entry in between_files:
        author = entry["author"]
        kept_from = entry["kept_from"]
        resolution_map[author] = kept_from
    
    logger.info(f"Loaded {len(resolution_map)} duplicate resolution decisions")
    return resolution_map


def merge_indexes_with_resolution(
    index_main: pd.DataFrame,
    index_manual: pd.DataFrame,
    resolution_map: dict,
    jsonl_main_path: str,
    jsonl_manual_path: str,
    source_name: str
) -> pd.DataFrame:
    """
    Merge two offset indexes using duplicate resolution mapping.
    
    Args:
        index_main: Index from main JSONL file
        index_manual: Index from manual JSONL file
        resolution_map: Author -> "primary"/"secondary" mapping
        jsonl_main_path: Path to main JSONL file (for source_file column)
        jsonl_manual_path: Path to manual JSONL file (for source_file column)
        source_name: Name for logging (e.g., "comments" or "submissions")
    
    Returns:
        Merged index with source_file column
    """
    logger.info(f"Merging {source_name} indexes with duplicate resolution...")
    
    # Add source_file column to each index
    index_main = index_main.copy()
    index_main["source_file"] = jsonl_main_path
    
    index_manual = index_manual.copy()
    index_manual["source_file"] = jsonl_manual_path
    
    # Find authors in both indexes
    main_authors = set(index_main["author"].values)
    manual_authors = set(index_manual["author"].values)
    duplicate_authors = main_authors & manual_authors
    
    logger.info(f"  Found {len(duplicate_authors)} duplicate authors in {source_name} indexes")
    
    if len(duplicate_authors) == 0:
        # No duplicates, simple concatenation
        merged = pd.concat([index_main, index_manual], ignore_index=True)
        logger.info(f"  No duplicates - simple concatenation")
        return merged
    
    # Resolve duplicates using resolution_map
    kept_from_main = []
    kept_from_manual = []
    conflicts = []
    
    for author in duplicate_authors:
        if author in resolution_map:
            kept_from = resolution_map[author]
            if kept_from == "primary":
                kept_from_main.append(author)
            elif kept_from == "secondary":
                kept_from_manual.append(author)
            else:
                conflicts.append((author, kept_from))
        else:
            # Author not in resolution map - default to primary
            logger.warning(f"  Author {author} not in resolution map, defaulting to primary")
            kept_from_main.append(author)
    
    if conflicts:
        logger.warning(f"  Found {len(conflicts)} conflicts in resolution mapping")
        for author, kept_from in conflicts[:5]:  # Show first 5
            logger.warning(f"    {author}: kept_from={kept_from} (unrecognized)")
    
    logger.info(f"  Keeping {len(kept_from_main)} from main, {len(kept_from_manual)} from manual")
    
    # Build merged index
    # Start with main non-duplicates
    main_non_duplicates = index_main[~index_main["author"].isin(duplicate_authors)]
    
    # Add manual non-duplicates
    manual_non_duplicates = index_manual[~index_manual["author"].isin(duplicate_authors)]
    
    # Add resolved duplicates
    resolved_main = index_main[index_main["author"].isin(kept_from_main)]
    resolved_manual = index_manual[index_manual["author"].isin(kept_from_manual)]
    
    merged = pd.concat([
        main_non_duplicates,
        manual_non_duplicates,
        resolved_main,
        resolved_manual
    ], ignore_index=True)
    
    logger.info(f"  Merged {source_name} index: {len(merged)} unique authors")
    
    return merged


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for merge_offset_indexes.py.
    
    This function orchestrates the index merging process:
    1. Parse command-line arguments
    2. Load duplicate resolution from merge_parquet_files.py
    3. Load 4 index files (2 comments + 2 submissions)
    4. Merge comments indexes with resolution
    5. Merge submissions indexes with resolution
    6. Save merged indexes
    7. Output summary statistics
    
    The function ensures HPC compatibility by:
    - Using logger for structured output to .out/.err files
    - Proper error handling and progress logging
    - Path handling that works across different filesystems
    
    Returns:
        None (saves merged indexes to disk)
    
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If data validation fails
    """
    parser = argparse.ArgumentParser(
        description="Merge offset indexes using duplicate resolution from parquet merge"
    )
    parser.add_argument("--comments-index-main", required=True,
                        help="Path to comments index from main JSONL")
    parser.add_argument("--comments-index-manual", required=True,
                        help="Path to comments index from manual JSONL")
    parser.add_argument("--submissions-index-main", required=True,
                        help="Path to submissions index from main JSONL")
    parser.add_argument("--submissions-index-manual", required=True,
                        help="Path to submissions index from manual JSONL")
    parser.add_argument("--jsonl-main", required=True,
                        help="Base path to main JSONL files (e.g., /path/to/main/)")
    parser.add_argument("--jsonl-manual", required=True,
                        help="Base path to manual JSONL files (e.g., /path/to/manual/)")
    parser.add_argument("--comments-file", default="user_comments_human.jsonl",
                        help="Comments JSONL filename (default: user_comments_human.jsonl)")
    parser.add_argument("--submissions-file", default="user_submissions_human.jsonl",
                        help="Submissions JSONL filename (default: user_submissions_human.jsonl)")
    parser.add_argument("--duplicate-details", required=True,
                        help="Path to merged_authors.duplicate_details.json")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for merged indexes")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Merge Offset Indexes")
    logger.info("="*60)
    logger.info(f"Comments index main: {args.comments_index_main}")
    logger.info(f"Comments index manual: {args.comments_index_manual}")
    logger.info(f"Submissions index main: {args.submissions_index_main}")
    logger.info(f"Submissions index manual: {args.submissions_index_manual}")
    logger.info(f"JSONL main path: {args.jsonl_main}")
    logger.info(f"JSONL manual path: {args.jsonl_manual}")
    logger.info(f"Duplicate details: {args.duplicate_details}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Validate input files exist
    required_files = [
        args.comments_index_main, args.comments_index_manual,
        args.submissions_index_main, args.submissions_index_manual,
        args.duplicate_details
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Construct JSONL file paths
    jsonl_comments_main = os.path.join(args.jsonl_main, args.comments_file)
    jsonl_comments_manual = os.path.join(args.jsonl_manual, args.comments_file)
    jsonl_submissions_main = os.path.join(args.jsonl_main, args.submissions_file)
    jsonl_submissions_manual = os.path.join(args.jsonl_manual, args.submissions_file)
    
    # Validate JSONL files exist
    jsonl_files = [jsonl_comments_main, jsonl_comments_manual, 
                   jsonl_submissions_main, jsonl_submissions_manual]
    for file_path in jsonl_files:
        if not os.path.exists(file_path):
            logger.warning(f"JSONL file not found: {file_path}")
    
    # Load duplicate resolution
    logger.info("\n" + "="*60)
    logger.info("Loading duplicate resolution")
    logger.info("="*60)
    resolution_map = load_duplicate_resolution(args.duplicate_details)
    
    # Load indexes
    logger.info("\n" + "="*60)
    logger.info("Loading offset indexes")
    logger.info("="*60)
    
    logger.info(f"Loading comments index (main): {args.comments_index_main}")
    comments_index_main = pd.read_parquet(args.comments_index_main)
    logger.info(f"  {len(comments_index_main)} authors")
    
    logger.info(f"Loading comments index (manual): {args.comments_index_manual}")
    comments_index_manual = pd.read_parquet(args.comments_index_manual)
    logger.info(f"  {len(comments_index_manual)} authors")
    
    logger.info(f"Loading submissions index (main): {args.submissions_index_main}")
    submissions_index_main = pd.read_parquet(args.submissions_index_main)
    logger.info(f"  {len(submissions_index_main)} authors")
    
    logger.info(f"Loading submissions index (manual): {args.submissions_index_manual}")
    submissions_index_manual = pd.read_parquet(args.submissions_index_manual)
    logger.info(f"  {len(submissions_index_manual)} authors")
    
    # Merge indexes
    logger.info("\n" + "="*60)
    logger.info("Merging indexes")
    logger.info("="*60)
    
    merged_comments = merge_indexes_with_resolution(
        comments_index_main, comments_index_manual, resolution_map,
        jsonl_comments_main, jsonl_comments_manual, "comments"
    )
    
    merged_submissions = merge_indexes_with_resolution(
        submissions_index_main, submissions_index_manual, resolution_map,
        jsonl_submissions_main, jsonl_submissions_manual, "submissions"
    )
    
    # Save merged indexes
    logger.info("\n" + "="*60)
    logger.info("Saving merged indexes")
    logger.info("="*60)
    
    comments_output = os.path.join(args.output_dir, "comments_index.parquet")
    submissions_output = os.path.join(args.output_dir, "submissions_index.parquet")
    
    merged_comments.to_parquet(comments_output, index=False)
    logger.info(f"Saved merged comments index to {comments_output}")
    
    merged_submissions.to_parquet(submissions_output, index=False)
    logger.info(f"Saved merged submissions index to {submissions_output}")
    
    # Write manifest
    logger.info("\n" + "="*60)
    logger.info("Writing manifest")
    logger.info("="*60)
    
    manifest = {
        "comments_index_main": args.comments_index_main,
        "comments_index_manual": args.comments_index_manual,
        "submissions_index_main": args.submissions_index_main,
        "submissions_index_manual": args.submissions_index_manual,
        "jsonl_main_base": args.jsonl_main,
        "jsonl_manual_base": args.jsonl_manual,
        "comments_file": args.comments_file,
        "submissions_file": args.submissions_file,
        "jsonl_comments_main": jsonl_comments_main,
        "jsonl_comments_manual": jsonl_comments_manual,
        "jsonl_submissions_main": jsonl_submissions_main,
        "jsonl_submissions_manual": jsonl_submissions_manual,
        "duplicate_details": args.duplicate_details,
        "n_resolutions_applied": int(len(resolution_map)),
        "merged_comments_authors": int(len(merged_comments)),
        "merged_submissions_authors": int(len(merged_submissions)),
        "comments_output": comments_output,
        "submissions_output": submissions_output,
        "merge_timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    manifest_path = os.path.join(args.output_dir, "index_merge_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved manifest to {manifest_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Index merge complete")
    logger.info("="*60)
    logger.info(f"Merged comments index: {comments_output}")
    logger.info(f"Merged submissions index: {submissions_output}")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
