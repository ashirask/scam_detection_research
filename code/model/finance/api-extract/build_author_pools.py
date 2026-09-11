#!/usr/bin/env python3
"""
build_author_pools.py
======================
Builds deduplicated author pools per subreddit from the raw metadata extracted
by Stage 1. This script reads the posts.jsonl.gz and comments.jsonl.gz files for
each subreddit, combines them, deduplicates by author name, and computes per-author
statistics (post count, comment count, first/last seen timestamps).

This is Stage 2 of the 4-stage subreddit-to-sampled-author pipeline. It runs
locally on the extracted data and produces author pools ready for sampling.

WHERE TO RUN THIS
-----------------
Run this on your local machine or login node. This script does not touch the network
and processes only local files. It can be run repeatedly as new subreddits complete
their Stage 1 extraction.

OUTPUT LAYOUT
-------------
<input-dir>/
    raw_meta/
        <subreddit>/
            posts.jsonl.gz
            comments.jsonl.gz

<output-dir>/
    author_pools/
        <subreddit>_authors.csv  # Columns: author, n_posts, n_comments, first_seen_utc, last_seen_utc

Each CSV file contains one row per unique author for that subreddit, with statistics
computed from both posts and comments combined.

USAGE
-----
python build_author_pools.py \
    --input-dir /scratch/$USER/subreddit_extraction \
    --output-dir /scratch/$USER/subreddit_extraction/author_pools
"""

import argparse
import csv
import gzip
import json
import logging
from pathlib import Path


def setup_logging(log_file=None):
    """
    Configures logging for the script.

    Parameters:
    - log_file: optional path to a file where logs should be written

    Returns:
        None: sets up the logging configuration as a side effect
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def load_subreddits_from_raw_meta(raw_meta_dir):
    """
    Discovers which subreddits have completed Stage 1 extraction.

    This function scans the raw_meta directory to find all subdirectories that
    contain at least one of posts.jsonl.gz or comments.jsonl.gz, indicating that
    Stage 1 has processed that subreddit.

    Parameters:
    - raw_meta_dir: path to the raw_meta directory containing subreddit subdirectories

    Returns:
        list[str]: list of subreddit names that have at least one extracted file
    """
    raw_meta_path = Path(raw_meta_dir)
    if not raw_meta_path.exists():
        logging.error(f"raw_meta directory does not exist: {raw_meta_dir}")
        return []
    
    subreddits = []
    for subreddit_dir in raw_meta_path.iterdir():
        if not subreddit_dir.is_dir():
            continue  # Skip non-directory entries
        
        # Check if at least one of posts.jsonl.gz or comments.jsonl.gz exists
        posts_file = subreddit_dir / "posts.jsonl.gz"
        comments_file = subreddit_dir / "comments.jsonl.gz"
        
        if posts_file.exists() or comments_file.exists():
            subreddits.append(subreddit_dir.name)
    
    return subreddits


def stream_jsonl_gz(file_path):
    """
    Streams records from a gzipped JSONL file.

    This function reads a .jsonl.gz file line by line, decompressing on the fly,
    and yields each parsed JSON record. This is memory-efficient for large files.

    Parameters:
    - file_path: path to the .jsonl.gz file to read

    Yields:
        dict: one JSON record per line of the file

    Raises:
        FileNotFoundError: if the file does not exist
        json.JSONDecodeError: if a line is not valid JSON
    """
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            yield json.loads(line)


def build_author_pool_for_subreddit(raw_meta_dir, subreddit):
    """
    Builds a deduplicated author pool for a single subreddit.

    This function reads both posts.jsonl.gz and comments.jsonl.gz for the given
    subreddit, combines all records, and builds a dictionary keyed by author name
    with statistics about each author's activity in that subreddit.

    Parameters:
    - raw_meta_dir: path to the raw_meta directory
    - subreddit: subreddit name to process

    Returns:
        tuple: (author_stats, has_posts, has_comments) where:
              - author_stats is a dict mapping author names to their statistics:
                {author: {"n_posts": int, "n_comments": int, "first_seen_utc": int, "last_seen_utc": int}}
              - has_posts is True if posts.jsonl.gz exists
              - has_comments is True if comments.jsonl.gz exists
    """
    subreddit_dir = Path(raw_meta_dir) / subreddit
    posts_file = subreddit_dir / "posts.jsonl.gz"
    comments_file = subreddit_dir / "comments.jsonl.gz"
    
    # Dictionary to accumulate author statistics
    # Key: author name, Value: dict with n_posts, n_comments, first_seen_utc, last_seen_utc
    author_stats = {}
    
    # Process posts file if it exists
    if posts_file.exists():
        for rec in stream_jsonl_gz(posts_file):
            author = rec.get("author")
            # Drop None, empty string, or [deleted] authors
            if author is None or author == "" or author == "[deleted]":
                continue
            
            if author not in author_stats:
                # Initialize author entry
                author_stats[author] = {
                    "n_posts": 0,
                    "n_comments": 0,
                    "first_seen_utc": None,
                    "last_seen_utc": None,
                }
            
            # Update post count
            author_stats[author]["n_posts"] += 1
            
            # Update timestamp range
            created_utc = rec.get("created_utc")
            if created_utc is not None:
                created_utc = int(created_utc)
                if author_stats[author]["first_seen_utc"] is None or created_utc < author_stats[author]["first_seen_utc"]:
                    author_stats[author]["first_seen_utc"] = created_utc
                if author_stats[author]["last_seen_utc"] is None or created_utc > author_stats[author]["last_seen_utc"]:
                    author_stats[author]["last_seen_utc"] = created_utc
    
    # Process comments file if it exists
    if comments_file.exists():
        for rec in stream_jsonl_gz(comments_file):
            author = rec.get("author")
            # Drop None, empty string, or [deleted] authors
            if author is None or author == "" or author == "[deleted]":
                continue
            
            if author not in author_stats:
                # Initialize author entry
                author_stats[author] = {
                    "n_posts": 0,
                    "n_comments": 0,
                    "first_seen_utc": None,
                    "last_seen_utc": None,
                }
            
            # Update comment count
            author_stats[author]["n_comments"] += 1
            
            # Update timestamp range
            created_utc = rec.get("created_utc")
            if created_utc is not None:
                created_utc = int(created_utc)
                if author_stats[author]["first_seen_utc"] is None or created_utc < author_stats[author]["first_seen_utc"]:
                    author_stats[author]["first_seen_utc"] = created_utc
                if author_stats[author]["last_seen_utc"] is None or created_utc > author_stats[author]["last_seen_utc"]:
                    author_stats[author]["last_seen_utc"] = created_utc
    
    return author_stats, posts_file.exists(), comments_file.exists()


def write_author_pool_csv(author_stats, output_path, subreddit):
    """
    Writes the author pool statistics to a CSV file.

    This function writes the author statistics dictionary to a CSV file with
    the columns: author, n_posts, n_comments, first_seen_utc, last_seen_utc.

    Parameters:
    - author_stats: dictionary mapping author names to their statistics
    - output_path: path where the CSV file should be written
    - subreddit: subreddit name (used for logging)

    Returns:
        int: the number of authors written to the CSV file

    Side effect:
        Creates or overwrites the CSV file at output_path with the author statistics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV file
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["author", "n_posts", "n_comments", "first_seen_utc", "last_seen_utc"])
        
        # Write author rows
        for author, stats in author_stats.items():
            writer.writerow([
                author,
                stats["n_posts"],
                stats["n_comments"],
                stats["first_seen_utc"] if stats["first_seen_utc"] is not None else "",
                stats["last_seen_utc"] if stats["last_seen_utc"] is not None else "",
            ])
    
    return len(author_stats)


def process_subreddit(raw_meta_dir, output_dir, subreddit):
    """
    Processes a single subreddit: builds its author pool and writes to CSV.

    This function orchestrates the building of an author pool for one subreddit,
    from reading the raw metadata to writing the final CSV file.

    Parameters:
    - raw_meta_dir: path to the raw_meta directory
    - output_dir: path to the author_pools output directory
    - subreddit: subreddit name to process

    Returns:
        tuple: (subreddit, pool_size, success, has_posts, has_comments) where:
              - success is True if processing succeeded, False if it failed
              - has_posts is True if posts.jsonl.gz exists
              - has_comments is True if comments.jsonl.gz exists
    """
    try:
        logging.info(f"Processing subreddit: {subreddit}")
        
        # Build author pool from raw metadata
        author_stats, has_posts, has_comments = build_author_pool_for_subreddit(raw_meta_dir, subreddit)
        pool_size = len(author_stats)
        
        # Write to CSV
        output_path = Path(output_dir) / f"{subreddit}_authors.csv"
        write_author_pool_csv(author_stats, output_path, subreddit)
        
        logging.info(f"  {subreddit}: {pool_size} unique authors written to {output_path.name}")
        
        return subreddit, pool_size, True, has_posts, has_comments
    except Exception as e:
        logging.error(f"  {subreddit}: FAILED - {type(e).__name__}: {e}")
        return subreddit, 0, False, False, False


def main():
    """
    Main entry point for the script.

    Parses command-line arguments, sets up logging, discovers subreddits with
    completed Stage 1 extraction, and processes each one to build author pools.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="Directory containing raw_meta/ subdirectory with subreddit data")
    ap.add_argument("--output-dir", required=True, help="Directory where author_pools/ will be created")
    ap.add_argument("--log-file", default=None, help="Optional file to write log messages to")
    args = ap.parse_args()

    setup_logging(args.log_file)

    input_dir = Path(args.input_dir)
    raw_meta_dir = input_dir / "raw_meta"
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover which subreddits have completed Stage 1
    subreddits = load_subreddits_from_raw_meta(raw_meta_dir)
    logging.info(f"Found {len(subreddits)} subreddits with completed Stage 1 extraction")
    
    if not subreddits:
        logging.warning("No subreddits found to process. Check that --input-dir points to the correct location.")
        return
    
    # Process each subreddit
    total_authors = 0
    failed_subreddits = []
    incomplete_subreddits = []  # Track subreddits missing posts or comments
    
    for subreddit in subreddits:
        _, pool_size, success, has_posts, has_comments = process_subreddit(raw_meta_dir, output_dir, subreddit)
        if success:
            total_authors += pool_size
            # Track incomplete subreddits (missing posts or comments)
            if not (has_posts and has_comments):
                incomplete_subreddits.append((subreddit, has_posts, has_comments))
        else:
            failed_subreddits.append(subreddit)
    
    # Write incomplete subreddits report
    incomplete_path = output_dir / "incomplete_subreddits.txt"
    with open(incomplete_path, "w", encoding="utf-8") as f:
        for sub, has_posts, has_comments in incomplete_subreddits:
            f.write(f"{sub}\thas_posts={has_posts}\thas_comments={has_comments}\n")
    
    # Summary
    complete_count = len(subreddits) - len(incomplete_subreddits)
    logging.info(f"Done. Processed {len(subreddits)} subreddits.")
    logging.info(f"Complete subreddits (both posts + comments): {complete_count}")
    logging.info(f"Incomplete subreddits (missing posts or comments): {len(incomplete_subreddits)}")
    logging.info(f"Incomplete subreddits report written to {incomplete_path}")
    logging.info(f"Total unique authors across all subreddits: {total_authors}")
    if failed_subreddits:
        logging.warning(f"Failed to process {len(failed_subreddits)} subreddits: {', '.join(failed_subreddits)}")


if __name__ == "__main__":
    main()
