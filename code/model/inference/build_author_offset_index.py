#!/usr/bin/env python3
"""
build_author_offset_index.py

Phase C1 of the inference pipeline: Build a byte-offset index for efficient
random access into large JSONL corpus files. This is a one-time setup step
that enables Phase C2 to quickly retrieve specific author records without
scanning the entire corpus.

This script is HPC-cluster compatible:
- Handles large files efficiently with binary mode
- Uses orjson for fast JSON parsing
- Includes comprehensive error handling and logging
- All imports are at top level for cluster compatibility

Usage:
  python build_author_offset_index.py \
    --jsonl-path user_comments_human.jsonl \
    --index-output comments_index.parquet

Run once per corpus file:
  python build_author_offset_index.py --jsonl-path user_comments_human.jsonl --index-output comments_index.parquet
  python build_author_offset_index.py --jsonl-path user_submissions_human.jsonl --index-output submissions_index.parquet

Note: This script handles duplicates within a single JSONL file by keeping
the first occurrence for each author.

HPC Cluster Example:
  sbatch build_index.sbatch  # Or your cluster's job submission system
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

import orjson


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

def build_author_offset_index(jsonl_path: str, index_output_path: str):
    """
    Opens jsonl_path in binary mode ("rb"). For each line:
        offset = f.tell()          # position BEFORE reading the line
        line = f.readline()
        length = len(line)
        author = orjson.loads(line)["author"]
    Writes a parquet file with columns:
        author (str), byte_offset (int64), byte_length (int64)
    Also writes {index_output_path}.manifest.json:
        {"source_file": jsonl_path, "file_size_bytes": ..., "n_lines_indexed": ...,
         "built_at": "<ISO8601>"}
    The manifest lets a later run detect staleness by comparing
    file_size_bytes against the current file size before trusting the index.
    
    Args:
        jsonl_path: Path to the JSONL file to index
        index_output_path: Path where to write the index parquet file
    
    Returns:
        None (writes index parquet and manifest to disk)
    
    Raises:
        FileNotFoundError: If JSONL file doesn't exist
        ValueError: If JSONL file is malformed or missing author field
    """
    logger.info("="*60)
    logger.info("Building Author Offset Index")
    logger.info("="*60)
    logger.info(f"JSONL path: {jsonl_path}")
    logger.info(f"Index output: {index_output_path}")
    
    # Validate input file exists
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found at {jsonl_path}")
    
    file_size_bytes = jsonl_file.stat().st_size
    logger.info(f"File size: {file_size_bytes / 1024 / 1024 / 1024:.2f} GB")
    
    # Prepare output directory
    output_dir = os.path.dirname(index_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Build index
    logger.info("Scanning JSONL file and building index...")
    
    index_records = []
    n_lines = 0
    n_errors = 0
    last_log_time = datetime.now()
    log_interval_seconds = 30  # Log progress every 30 seconds
    
    try:
        with open(jsonl_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                
                # Check for EOF
                if not line:
                    break
                
                length = len(line)
                n_lines += 1
                
                try:
                    # Parse JSON and extract author
                    record = orjson.loads(line)
                    author = record.get("author")
                    
                    if author is None:
                        logger.warning(f"Line {n_lines}: Missing 'author' field, skipping")
                        n_errors += 1
                        continue
                    
                    index_records.append({
                        "author": author,
                        "byte_offset": offset,
                        "byte_length": length
                    })
                    
                except Exception as e:
                    logger.warning(f"Line {n_lines}: Failed to parse JSON: {e}")
                    n_errors += 1
                    continue
                
                # Periodic progress logging
                current_time = datetime.now()
                if (current_time - last_log_time).total_seconds() >= log_interval_seconds:
                    progress_pct = 100 * offset / file_size_bytes
                    logger.info(f"  Progress: {progress_pct:.1f}% ({n_lines:,} lines indexed, {n_errors} errors)")
                    last_log_time = current_time
    
    except Exception as e:
        logger.error(f"Error while scanning file: {e}")
        raise
    
    logger.info(f"Scan complete: {n_lines:,} lines indexed, {n_errors} errors")
    
    if n_errors > 0:
        logger.warning(f"Encountered {n_errors} errors during indexing")
    
    # Convert to DataFrame and save as parquet
    logger.info("Writing index to parquet...")
    index_df = pd.DataFrame(index_records)
    
    # Check for duplicate authors
    duplicate_authors = index_df[index_df["author"].duplicated(keep=False)]
    if not duplicate_authors.empty:
        n_duplicates = duplicate_authors["author"].nunique()
        logger.warning(f"Found {n_duplicates} duplicate authors in index")
        logger.warning("Keeping first occurrence for each duplicate author")
        index_df = index_df.drop_duplicates(subset=["author"], keep="first")
    
    index_df.to_parquet(index_output_path, index=False)
    logger.info(f"Saved index to {index_output_path}")
    logger.info(f"  Index size: {len(index_df):,} unique authors")
    
    # Write manifest
    manifest = {
        "source_file": jsonl_path,
        "file_size_bytes": int(file_size_bytes),
        "n_lines_indexed": int(n_lines),
        "n_unique_authors": int(len(index_df)),
        "n_errors": int(n_errors),
        "built_at": datetime.utcnow().isoformat() + "Z"
    }
    
    manifest_path = f"{index_output_path}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Saved manifest to {manifest_path}")
    
    logger.info("="*60)
    logger.info("Index building complete")
    logger.info("="*60)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for build_author_offset_index.py.
    
    This function orchestrates the index building process:
    1. Parse command-line arguments
    2. Validate input file exists
    3. Scan JSONL file and record byte offsets for each author
    4. Write index as parquet file
    5. Write manifest with metadata
    
    The function ensures HPC compatibility by:
    - Using logger for structured output to .out/.err files
    - Proper error handling and progress logging
    - Binary mode file reading for efficiency
    - Path handling that works across different filesystems
    
    Returns:
        None (saves index and manifest to disk)
    
    Raises:
        FileNotFoundError: If JSONL file doesn't exist
        ValueError: If file is malformed
    """
    parser = argparse.ArgumentParser(
        description="Build byte-offset index for efficient JSONL author lookup"
    )
    parser.add_argument("--jsonl-path", required=True,
                        help="Path to JSONL file (e.g., comments.jsonl or submissions.jsonl)")
    parser.add_argument("--index-output", required=True,
                        help="Output path for index parquet file (e.g., comments_index.parquet)")
    
    args = parser.parse_args()
    
    build_author_offset_index(args.jsonl_path, args.index_output)


if __name__ == "__main__":
    main()
