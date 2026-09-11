#!/usr/bin/env python3
"""
sample_authors.py
================
Samples a fixed number of authors from each subreddit's author pool using
deterministic random sampling. This script supports multiple batches per subreddit
with a registry mechanism to avoid re-sampling the same authors across batches.

This is Stage 3 of the 4-stage subreddit-to-sampled-author pipeline. It reads
the author pools built by Stage 2 and produces sampled author lists ready for
merging into a master list.

WHERE TO RUN THIS
-----------------
Run this on your local machine or login node. This script does not touch the network
and processes only local files. It can be run repeatedly to draw new batches from
the same author pools without re-running Stage 1.

OUTPUT LAYOUT
-------------
<pools-dir>/
    <subreddit>_authors.csv  # Input: author pool from Stage 2

<output-dir>/
    <subreddit>_batch{N}_sample1000.txt  # One username per line
    _registry/
        <subreddit>_sampled_so_far.txt  # Union of all sampled authors for this subreddit
    sampling_summary.csv  # Summary of sampling results per subreddit

The registry file tracks which authors have already been sampled across all batches,
enabling non-overlapping resampling without re-hitting the API.

USAGE
-----
python sample_authors.py \
    --pools-dir author_pools/ \
    --output-dir samples/ \
    --sample-size 1000 \
    --seed 42 \
    --batch-id 1
"""

import argparse
import csv
import hashlib
import logging
import random
from datetime import datetime
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


def hash_to_int(seed_string):
    """
    Converts a seed string to a deterministic integer for use as a random seed.

    This function uses SHA-256 hashing to convert an arbitrary string into a
    consistent integer value, which can then be used as a seed for Python's
    random number generator.

    Parameters:
    - seed_string: the string to hash (e.g., "42:wallstreetbets:1")

    Returns:
        int: a 32-bit integer derived from the hash of the seed string
    """
    hash_bytes = hashlib.sha256(seed_string.encode()).hexdigest()
    return int(hash_bytes, 16) % (2 ** 32)


def load_author_pool(pool_path):
    """
    Loads the author pool from a CSV file.

    This function reads the author pool CSV file and extracts the author names
    into a list. The CSV file is expected to have an 'author' column.

    Parameters:
    - pool_path: path to the author pool CSV file

    Returns:
        list[str]: list of author names from the pool

    Raises:
        FileNotFoundError: if the pool file does not exist
        KeyError: if the CSV file does not have an 'author' column
    """
    authors = []
    with open(pool_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            author = row.get("author")
            if author:
                authors.append(author)
    return authors


def load_registry(registry_path):
    """
    Loads the registry of previously sampled authors for a subreddit.

    This function reads the registry file (if it exists) and returns a set of
    author names that have already been sampled in previous batches.

    Parameters:
    - registry_path: path to the registry file for a subreddit

    Returns:
        set[str]: set of author names that have been previously sampled
    """
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return set()
    
    sampled_authors = set()
    with open(registry_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sampled_authors.add(line)
    
    return sampled_authors


def update_registry(registry_path, new_authors):
    """
    Updates the registry file with newly sampled authors.

    This function appends the newly sampled authors to the registry file,
    creating the file if it doesn't exist. The registry tracks the union of
    all sampled authors across all batches for a subreddit.

    Parameters:
    - registry_path: path to the registry file for a subreddit
    - new_authors: list of author names to append to the registry

    Returns:
        None: writes to the registry file as a side effect
    """
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(registry_path, "a", encoding="utf-8") as f:
        for author in new_authors:
            f.write(author + "\n")


def sample_from_pool(pool_authors, global_seed, subreddit, batch_id, sample_size, already_sampled=None):
    """
    Samples authors from a pool using deterministic random sampling.

    This function uses a seeded random number generator to sample a fixed number
    of authors from the pool. The seed is derived from the global seed, subreddit
    name, and batch ID, ensuring reproducibility and independence across subreddits
    and batches.

    Parameters:
    - pool_authors: list of all authors in the pool
    - global_seed: the global seed from the command line
    - subreddit: subreddit name (used in seed derivation)
    - batch_id: batch ID (used in seed derivation)
    - sample_size: number of authors to sample
    - already_sampled: set of authors already sampled in previous batches (optional)

    Returns:
        tuple: (sampled_authors, available_pool_size, insufficient_flag, seed_used)
               where sampled_authors is the list of sampled author names,
               available_pool_size is the size of the pool after excluding already-sampled authors,
               insufficient_flag is True if the pool was too small to sample the requested size,
               and seed_used is the integer seed used for sampling
    """
    # Compute the seed for this subreddit/batch combination
    seed_string = f"{global_seed}:{subreddit}:{batch_id}"
    seed_used = hash_to_int(seed_string)
    
    # Remove already-sampled authors from the pool
    if already_sampled is None:
        already_sampled = set()
    
    available_pool = [author for author in pool_authors if author not in already_sampled]
    available_pool_size = len(available_pool)
    
    # Sample from the available pool
    rng = random.Random(seed_used)
    actual_sample_size = min(sample_size, available_pool_size)
    sampled_authors = rng.sample(available_pool, actual_sample_size)
    
    # Flag if we couldn't sample the full requested size
    insufficient_flag = available_pool_size < sample_size
    
    return sampled_authors, available_pool_size, insufficient_flag, seed_used


def write_sample_file(sampled_authors, output_path):
    """
    Writes the sampled authors to a text file.

    This function writes the list of sampled authors to a text file, one author
    per line, in the order they were sampled.

    Parameters:
    - sampled_authors: list of sampled author names
    - output_path: path where the sample file should be written

    Returns:
        None: creates the sample file as a side effect
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for author in sampled_authors:
            f.write(author + "\n")


def process_subreddit(pools_dir, output_dir, subreddit, sample_size, global_seed, batch_id):
    """
    Processes a single subreddit: samples authors from its pool and writes output files.

    This function orchestrates the sampling process for one subreddit, from loading
    the author pool to writing the sample file and updating the registry.

    Parameters:
    - pools_dir: path to the directory containing author pool CSV files
    - output_dir: path to the output directory for samples
    - subreddit: subreddit name to process
    - sample_size: number of authors to sample
    - global_seed: global seed for reproducibility
    - batch_id: batch ID for this sampling run

    Returns:
        dict: summary dictionary with keys: subreddit, batch_id, pool_size, available_pool_size,
              n_sampled, seed_used, insufficient_flag, timestamp
    """
    pool_path = Path(pools_dir) / f"{subreddit}_authors.csv"
    registry_path = Path(output_dir) / "_registry" / f"{subreddit}_sampled_so_far.txt"
    sample_path = Path(output_dir) / f"{subreddit}_batch{batch_id}_sample{sample_size}.txt"
    
    # Load author pool
    if not pool_path.exists():
        logging.warning(f"  {subreddit}: pool file not found, skipping")
        return {
            "subreddit": subreddit,
            "batch_id": batch_id,
            "pool_size": 0,
            "available_pool_size": 0,
            "n_sampled": 0,
            "seed_used": 0,
            "insufficient_flag": True,
            "timestamp": datetime.utcnow().isoformat(),
            "error": "pool file not found",
        }
    
    pool_authors = load_author_pool(pool_path)
    pool_size = len(pool_authors)
    
    # Load registry of previously sampled authors
    already_sampled = load_registry(registry_path)
    
    # Sample from the pool
    sampled_authors, available_pool_size, insufficient_flag, seed_used = sample_from_pool(
        pool_authors, global_seed, subreddit, batch_id, sample_size, already_sampled
    )
    
    # Write sample file
    write_sample_file(sampled_authors, sample_path)
    
    # Update registry
    update_registry(registry_path, sampled_authors)
    
    # Log result
    if insufficient_flag:
        logging.warning(f"  {subreddit}: sampled {len(sampled_authors)} authors (pool insufficient, only {available_pool_size} available)")
    else:
        logging.info(f"  {subreddit}: sampled {len(sampled_authors)} authors from pool of {pool_size} (available: {available_pool_size})")
    
    return {
        "subreddit": subreddit,
        "batch_id": batch_id,
        "pool_size": pool_size,
        "available_pool_size": available_pool_size,
        "n_sampled": len(sampled_authors),
        "seed_used": seed_used,
        "insufficient_flag": insufficient_flag,
        "timestamp": datetime.utcnow().isoformat(),
    }


def write_sampling_summary(summary_rows, output_path):
    """
    Writes the sampling summary to a CSV file.

    This function writes the summary of sampling results for all processed
    subreddits to a CSV file, creating or overwriting it.

    Parameters:
    - summary_rows: list of summary dictionaries (one per subreddit)
    - output_path: path where the summary CSV should be written

    Returns:
        None: creates the summary CSV file as a side effect
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["subreddit", "batch_id", "pool_size", "available_pool_size", "n_sampled", "seed_used", "insufficient_flag", "timestamp"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            # Remove error key if present (not part of standard schema)
            row_to_write = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(row_to_write)


def discover_subreddits(pools_dir):
    """
    Discovers which subreddits have author pool files available.

    This function scans the pools directory to find all CSV files matching
    the pattern <subreddit>_authors.csv and extracts the subreddit names.

    Parameters:
    - pools_dir: path to the directory containing author pool CSV files

    Returns:
        list[str]: list of subreddit names that have pool files
    """
    pools_path = Path(pools_dir)
    if not pools_path.exists():
        logging.error(f"Pools directory does not exist: {pools_dir}")
        return []
    
    subreddits = []
    for pool_file in pools_path.glob("*_authors.csv"):
        # Extract subreddit name from filename
        subreddit = pool_file.stem.replace("_authors", "")
        subreddits.append(subreddit)
    
    return subreddits


def load_incomplete_subreddits(incomplete_path):
    """
    Loads the list of incomplete subreddits from Stage 2's report file.

    This function reads the incomplete_subreddits.txt file generated by
    build_author_pools.py and returns a set of subreddit names that are
    missing either posts or comments.

    Parameters:
    - incomplete_path: path to the incomplete_subreddits.txt file

    Returns:
        set[str]: set of subreddit names that are incomplete (missing posts or comments)
    """
    incomplete_path = Path(incomplete_path)
    if not incomplete_path.exists():
        logging.warning(f"Incomplete subreddits file not found: {incomplete_path}")
        return set()
    
    incomplete = set()
    with open(incomplete_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Format: subreddit\thas_posts=True/False\thas_comments=True/False
                parts = line.split("\t")
                if parts:
                    subreddit = parts[0]
                    incomplete.add(subreddit)
    
    return incomplete


def main():
    """
    Main entry point for the script.

    Parses command-line arguments, sets up logging, discovers subreddits with
    author pools, and samples authors from each one.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools-dir", required=True, help="Directory containing author pool CSV files")
    ap.add_argument("--output-dir", required=True, help="Directory where samples will be written")
    ap.add_argument("--sample-size", type=int, default=1000, help="Number of authors to sample per subreddit (default: 1000)")
    ap.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility (default: 42)")
    ap.add_argument("--batch-id", type=int, required=True, help="Batch ID for this sampling run (required for idempotency)")
    ap.add_argument("--log-file", default=None, help="Optional file to write log messages to")
    ap.add_argument("--incomplete-subreddits", default=None, help="Path to incomplete_subreddits.txt from Stage 2 to skip incomplete subreddits")
    args = ap.parse_args()

    setup_logging(args.log_file)

    pools_dir = Path(args.pools_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover which subreddits have author pools
    subreddits = discover_subreddits(pools_dir)
    logging.info(f"Found {len(subreddits)} subreddits with author pools")
    
    # Load incomplete subreddits if provided
    incomplete_set = set()
    if args.incomplete_subreddits:
        incomplete_set = load_incomplete_subreddits(args.incomplete_subreddits)
        logging.info(f"Loaded {len(incomplete_set)} incomplete subreddits to skip")
    
    # Filter out incomplete subreddits
    if incomplete_set:
        original_count = len(subreddits)
        subreddits = [s for s in subreddits if s not in incomplete_set]
        skipped_count = original_count - len(subreddits)
        logging.info(f"Skipping {skipped_count} incomplete subreddits, processing {len(subreddits)} complete subreddits")
    
    if not subreddits:
        logging.warning("No subreddits found to process after filtering. Check that --pools-dir points to the correct location.")
        return
    
    # Process each subreddit
    summary_rows = []
    total_sampled = 0
    insufficient_count = 0
    
    for subreddit in subreddits:
        summary = process_subreddit(
            pools_dir, output_dir, subreddit, args.sample_size, args.seed, args.batch_id
        )
        summary_rows.append(summary)
        
        if "error" not in summary:
            total_sampled += summary["n_sampled"]
            if summary["insufficient_flag"]:
                insufficient_count += 1
    
    # Write sampling summary
    summary_path = output_dir / "sampling_summary.csv"
    write_sampling_summary(summary_rows, summary_path)
    
    # Summary
    logging.info(f"Done. Processed {len(subreddits)} subreddits.")
    logging.info(f"Total authors sampled: {total_sampled}")
    if insufficient_count > 0:
        logging.warning(f"{insufficient_count} subreddits had insufficient pool size for the requested sample size")
    logging.info(f"Sampling summary written to {summary_path}")


if __name__ == "__main__":
    main()
