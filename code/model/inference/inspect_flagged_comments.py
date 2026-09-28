#!/usr/bin/env python3
"""
inspect_flagged_comments.py

Phase C2 of the inference pipeline: Use the offset indexes to retrieve
specific author records from the JSONL corpus files for flagged authors
from Phase B, and summarize them for qualitative review.

This script is HPC-cluster compatible:
- Uses efficient binary file seeking with offset indexes
- Handles missing authors gracefully with logging
- Includes comprehensive error handling and logging
- All imports are at top level for cluster compatibility

Usage:
  python inspect_flagged_comments.py \
    --flagged-authors inference/review/xgboost_none/threshold_0.5/flagged_authors.csv \
    --comments-jsonl-main /path/to/main/user_comments_human.jsonl \
    --comments-jsonl-manual /path/to/manual/user_comments_human.jsonl \
    --comments-index inference/indexes/comments_index.parquet \
    --submissions-jsonl-main /path/to/main/user_submissions_human.jsonl \
    --submissions-jsonl-manual /path/to/manual/user_submissions_human.jsonl \
    --submissions-index inference/indexes/submissions_index.parquet \
    --output-dir inference/review/xgboost_none/threshold_0.5/raw_posts/

HPC Cluster Example:
  sbatch inspect_comments.sbatch  # Or your cluster's job submission system
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

def verify_index_freshness(index_path: str, jsonl_path: str) -> bool:
    """
    Check if the index is still fresh by comparing the file_size_bytes
    in the manifest against the current file size.
    
    Note: This function is skipped for merged indexes since they have
    multiple source files.
    
    Args:
        index_path: Path to the index parquet file
        jsonl_path: Path to the JSONL corpus file
    
    Returns:
        True if index is fresh (sizes match), False otherwise
    
    Raises:
        FileNotFoundError: If index manifest or JSONL file doesn't exist
    """
    manifest_path = f"{index_path}.manifest.json"
    
    if not os.path.exists(manifest_path):
        logger.warning(f"Index manifest not found at {manifest_path}")
        return False
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found at {jsonl_path}")
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    # Check if this is a merged index manifest (from merge_offset_indexes.py)
    if "jsonl_comments_main" in manifest:
        logger.info("Detected merged index manifest - skipping freshness check")
        return True
    
    indexed_size = manifest["file_size_bytes"]
    current_size = Path(jsonl_path).stat().st_size
    
    if indexed_size != current_size:
        logger.warning(
            f"Index staleness detected for {jsonl_path}: "
            f"indexed size {indexed_size} != current size {current_size}"
        )
        return False
    
    logger.info(f"Index freshness verified for {jsonl_path}")
    return True


def fetch_author_record(
    author: str, index_df: pd.DataFrame, jsonl_main_path: str, jsonl_manual_path: str = None
):
    """
    Looks up author in index_df (columns: author, byte_offset, byte_length, source_file).
    Uses source_file column to determine which JSONL to read from.
    If not found, returns None (log a warning — means the corpus predates
    or postdates this author, or the index is stale).
    Otherwise: open the appropriate jsonl_path in "rb", f.seek(byte_offset),
    line = f.read(byte_length), return orjson.loads(line).
    
    Args:
        author: Author username to look up
        index_df: DataFrame with author, byte_offset, byte_length, source_file columns
        jsonl_main_path: Path to the main JSONL corpus file
        jsonl_manual_path: Path to the manual JSONL corpus file (optional)
    
    Returns:
        Dictionary containing the author's record, or None if not found
    """
    # Look up author in index
    author_row = index_df[index_df["author"] == author]
    
    if author_row.empty:
        logger.warning(f"Author '{author}' not found in index")
        return None
    
    # Get offset, length, and source file
    offset = author_row.iloc[0]["byte_offset"]
    length = author_row.iloc[0]["byte_length"]
    source_file = author_row.iloc[0]["source_file"]
    
    # Determine which JSONL to use
    if source_file == jsonl_main_path:
        jsonl_path = jsonl_main_path
    elif jsonl_manual_path and source_file == jsonl_manual_path:
        jsonl_path = jsonl_manual_path
    else:
        # Try to match by path if exact match fails
        if jsonl_manual_path and "manual" in source_file.lower():
            jsonl_path = jsonl_manual_path
        else:
            jsonl_path = jsonl_main_path
        logger.info(f"Using path matching for source: {source_file} -> {jsonl_path}")
    
    try:
        with open(jsonl_path, "rb") as f:
            f.seek(offset)
            line = f.read(length)
            record = orjson.loads(line)
            return record
    except Exception as e:
        logger.error(f"Error fetching record for author '{author}' from {jsonl_path}: {e}")
        return None


def summarize_for_review(
    author: str, comments_record, submissions_record
) -> dict:
    """
    Extracts a reviewer-friendly subset of fields from the raw Reddit-style
    records — e.g. for each comment/submission: body/selftext, title
    (submissions only), subreddit, created_utc, score. Returns:
        {"author": author, "n_comments": ..., "n_submissions": ...,
         "comments": [...trimmed...], "submissions": [...trimmed...]}
    
    Args:
        author: Author username
        comments_record: Dictionary from comments.jsonl, or None if not found
        submissions_record: Dictionary from submissions.jsonl, or None if not found
    
    Returns:
        Dictionary with summarized information for review
    """
    summary = {
        "author": author,
        "n_comments": 0,
        "n_submissions": 0,
        "comments": [],
        "submissions": []
    }
    
    # Process comments
    if comments_record is not None:
        comments = comments_record.get("comments", [])
        summary["n_comments"] = len(comments)
        
        # Summarize each comment (limit to first 20 for review)
        for comment in comments[:20]:
            comment_summary = {
                "body": comment.get("body", "")[:500],  # Truncate long bodies
                "subreddit": comment.get("subreddit", ""),
                "created_utc": comment.get("created_utc"),
                "score": comment.get("score"),
                "id": comment.get("id")
            }
            summary["comments"].append(comment_summary)
        
        if len(comments) > 20:
            summary["comments_truncated"] = len(comments) - 20
    
    # Process submissions
    if submissions_record is not None:
        submissions = submissions_record.get("submissions", [])
        summary["n_submissions"] = len(submissions)
        
        # Summarize each submission (limit to first 20 for review)
        for submission in submissions[:20]:
            submission_summary = {
                "title": submission.get("title", "")[:200],  # Truncate long titles
                "selftext": submission.get("selftext", "")[:500],  # Truncate long selftext
                "subreddit": submission.get("subreddit", ""),
                "created_utc": submission.get("created_utc"),
                "score": submission.get("score"),
                "id": submission.get("id")
            }
            summary["submissions"].append(submission_summary)
        
        if len(submissions) > 20:
            summary["submissions_truncated"] = len(submissions) - 20
    
    return summary


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for inspect_flagged_comments.py.
    
    This function orchestrates the comment inspection process:
    1. Parse command-line arguments
    2. Load flagged authors from Phase B
    3. Load offset indexes for comments and submissions
    4. Verify index freshness
    5. For each flagged author: fetch records from both JSONL files
    6. Summarize records for review
    7. Write output as JSONL bundle
    
    The function ensures HPC compatibility by:
    - Using logger for structured output to .out/.err files
    - Proper error handling and progress logging
    - Efficient binary file seeking
    - Path handling that works across different filesystems
    
    Returns:
        None (saves raw posts bundle to disk)
    
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If data validation fails
    """
    parser = argparse.ArgumentParser(
        description="Retrieve and summarize posts for flagged authors using offset indexes"
    )
    parser.add_argument("--flagged-authors", required=True,
                        help="Path to flagged_authors.csv from Phase B")
    parser.add_argument("--comments-jsonl-main", required=True,
                        help="Base path to main comments.jsonl directory OR full path to main comments.jsonl")
    parser.add_argument("--comments-jsonl-manual", required=True,
                        help="Base path to manual comments.jsonl directory OR full path to manual comments.jsonl")
    parser.add_argument("--comments-index", required=True,
                        help="Path to merged comments index parquet file")
    parser.add_argument("--submissions-jsonl-main", required=True,
                        help="Base path to main submissions.jsonl directory OR full path to main submissions.jsonl")
    parser.add_argument("--submissions-jsonl-manual", required=True,
                        help="Base path to manual submissions.jsonl directory OR full path to manual submissions.jsonl")
    parser.add_argument("--submissions-index", required=True,
                        help="Path to merged submissions index parquet file")
    parser.add_argument("--comments-file", default="user_comments_human.jsonl",
                        help="Comments JSONL filename (default: user_comments_human.jsonl)")
    parser.add_argument("--submissions-file", default="user_submissions_human.jsonl",
                        help="Submissions JSONL filename (default: user_submissions_human.jsonl)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for raw_posts_bundle.jsonl")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Phase C2: Inspect Flagged Comments")
    logger.info("="*60)
    logger.info(f"Flagged authors: {args.flagged_authors}")
    logger.info(f"Comments JSONL (main): {args.comments_jsonl_main}")
    logger.info(f"Comments JSONL (manual): {args.comments_jsonl_manual}")
    logger.info(f"Comments index: {args.comments_index}")
    logger.info(f"Submissions JSONL (main): {args.submissions_jsonl_main}")
    logger.info(f"Submissions JSONL (manual): {args.submissions_jsonl_manual}")
    logger.info(f"Submissions index: {args.submissions_index}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Handle JSONL paths - if directory, append filename; if file, use as-is
    def resolve_jsonl_path(base_path, filename):
        if os.path.isdir(base_path):
            return os.path.join(base_path, filename)
        else:
            return base_path
    
    args.comments_jsonl_main = resolve_jsonl_path(args.comments_jsonl_main, args.comments_file)
    args.comments_jsonl_manual = resolve_jsonl_path(args.comments_jsonl_manual, args.comments_file)
    args.submissions_jsonl_main = resolve_jsonl_path(args.submissions_jsonl_main, args.submissions_file)
    args.submissions_jsonl_manual = resolve_jsonl_path(args.submissions_jsonl_manual, args.submissions_file)
    
    logger.info(f"Resolved comments JSONL (main): {args.comments_jsonl_main}")
    logger.info(f"Resolved comments JSONL (manual): {args.comments_jsonl_manual}")
    logger.info(f"Resolved submissions JSONL (main): {args.submissions_jsonl_main}")
    logger.info(f"Resolved submissions JSONL (manual): {args.submissions_jsonl_manual}")
    
    # Step 1: Load flagged authors
    logger.info("\n" + "="*60)
    logger.info("Step 1: Load flagged authors")
    logger.info("="*60)
    
    if not os.path.exists(args.flagged_authors):
        raise FileNotFoundError(f"Flagged authors file not found at {args.flagged_authors}")
    
    flagged_df = pd.read_csv(args.flagged_authors)
    flagged_authors = flagged_df["author"].tolist()
    logger.info(f"Loaded {len(flagged_authors)} flagged authors")
    
    # Step 2: Load offset indexes
    logger.info("\n" + "="*60)
    logger.info("Step 2: Load offset indexes")
    logger.info("="*60)
    
    if not os.path.exists(args.comments_index):
        raise FileNotFoundError(f"Comments index not found at {args.comments_index}")
    if not os.path.exists(args.submissions_index):
        raise FileNotFoundError(f"Submissions index not found at {args.submissions_index}")
    
    # Validate JSONL files exist
    jsonl_files = [
        args.comments_jsonl_main, args.comments_jsonl_manual,
        args.submissions_jsonl_main, args.submissions_jsonl_manual
    ]
    for jsonl_path in jsonl_files:
        if not os.path.exists(jsonl_path):
            logger.warning(f"JSONL file not found: {jsonl_path}")
    
    comments_index = pd.read_parquet(args.comments_index)
    submissions_index = pd.read_parquet(args.submissions_index)
    
    logger.info(f"Loaded comments index: {len(comments_index)} authors")
    logger.info(f"Loaded submissions index: {len(submissions_index)} authors")
    
    # Step 3: Verify index freshness (skip for merged indexes)
    logger.info("\n" + "="*60)
    logger.info("Step 3: Verify index freshness")
    logger.info("="*60)
    
    logger.info("Skipping freshness check for merged indexes (multiple source files)")
    comments_fresh = True
    submissions_fresh = True
    
    if not comments_fresh or not submissions_fresh:
        logger.warning("One or more indexes may be stale. Consider rebuilding the index.")
    
    # Step 4: Fetch and summarize records for each flagged author
    logger.info("\n" + "="*60)
    logger.info("Step 4: Fetch and summarize records")
    logger.info("="*60)
    
    output_path = os.path.join(args.output_dir, "raw_posts_bundle.jsonl")
    n_success = 0
    n_missing_comments = 0
    n_missing_submissions = 0
    n_total_missing = 0
    
    with open(output_path, "wb") as f:
        for i, author in enumerate(flagged_authors):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(flagged_authors)} authors...")
            
            # Fetch records
            comments_record = fetch_author_record(author, comments_index, args.comments_jsonl_main, args.comments_jsonl_manual)
            submissions_record = fetch_author_record(author, submissions_index, args.submissions_jsonl_main, args.submissions_jsonl_manual)
            
            # Track missing records
            if comments_record is None:
                n_missing_comments += 1
            if submissions_record is None:
                n_missing_submissions += 1
            if comments_record is None and submissions_record is None:
                n_total_missing += 1
            
            # Summarize for review
            summary = summarize_for_review(author, comments_record, submissions_record)
            
            # Write to JSONL
            f.write(orjson.dumps(summary) + b"\n")
            n_success += 1
    
    logger.info(f"Processed {n_success} authors")
    logger.info(f"  Missing from comments index: {n_missing_comments}")
    logger.info(f"  Missing from submissions index: {n_missing_submissions}")
    logger.info(f"  Missing from both: {n_total_missing}")
    
    # Step 5: Write manifest
    logger.info("\n" + "="*60)
    logger.info("Step 5: Write manifest")
    logger.info("="*60)
    
    manifest = {
        "source_flagged_authors": args.flagged_authors,
        "comments_jsonl_main": args.comments_jsonl_main,
        "comments_jsonl_manual": args.comments_jsonl_manual,
        "comments_index": args.comments_index,
        "comments_index_fresh": comments_fresh,
        "submissions_jsonl_main": args.submissions_jsonl_main,
        "submissions_jsonl_manual": args.submissions_jsonl_manual,
        "submissions_index": args.submissions_index,
        "submissions_index_fresh": submissions_fresh,
        "n_flagged_authors": int(len(flagged_authors)),
        "n_authors_processed": int(n_success),
        "n_missing_comments": int(n_missing_comments),
        "n_missing_submissions": int(n_missing_submissions),
        "n_missing_both": int(n_total_missing),
        "output_path": output_path,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    
    manifest_path = os.path.join(args.output_dir, "raw_posts_bundle.manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Saved manifest to {manifest_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Phase C2 complete")
    logger.info("="*60)
    logger.info(f"Output bundle: {output_path}")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
