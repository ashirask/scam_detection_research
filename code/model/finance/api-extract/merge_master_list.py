#!/usr/bin/env python3
"""
merge_master_list.py
====================
Merges sampled author lists from all subreddits into a globally deduplicated master
list, preserving subreddit membership information. This script concatenates all
sample files, removes duplicate authors (since the same author may appear in multiple
subreddit samples), and produces two output files: a deduplicated author list ready
for the downstream author-history downloader, and a membership mapping showing which
subreddit(s) each author was sampled from.

This is Stage 4 of the 4-stage subreddit-to-sampled-author pipeline. It produces
the final input for download_arctic_shift_V2.3.py.

WHERE TO RUN THIS
-----------------
Run this on your local machine or login node. This script does not touch the network
and processes only local files. It can be run any time new batches are added to
regenerate the master list fresh from the current samples.

OUTPUT LAYOUT
-------------
<samples-dir>/
    <subreddit>_batch{N}_sample1000.txt  # Input: sampled author lists from Stage 3

<output-dir>/
    master/
        all_sampled_authors.txt  # Globally deduplicated author list, one per line
        author_subreddit_membership.csv  # author, subreddit, batch_id (long format)

The master list is ready to be passed to download_arctic_shift_V2.3.py as --humans-file.

USAGE
-----
python merge_master_list.py \
    --samples-dir samples/ \
    --output-dir master/ \
    --batches 1
"""

import argparse
import csv
import logging
import re
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


def parse_batch_filter(batches_arg):
    """
    Parses the --batches argument to determine which batch IDs to include.

    This function handles two cases:
    - "all": include all batches found in the samples directory
    - A comma-separated list of integers: include only those specific batch IDs

    Parameters:
    - batches_arg: the value of the --batches argument (either "all" or a comma-separated list)

    Returns:
        set[int] or str: a set of batch IDs to include, or the string "all" to include all batches
    """
    if batches_arg.lower() == "all":
        return "all"
    
    # Parse comma-separated list of integers
    try:
        batch_ids = set()
        for part in batches_arg.split(","):
            part = part.strip()
            if part:
                batch_ids.add(int(part))
        return batch_ids
    except ValueError as e:
        raise ValueError(f"Invalid --batches value: {batches_arg}. Expected 'all' or comma-separated integers.") from e


def discover_sample_files(samples_dir, batch_filter):
    """
    Discovers sample files in the samples directory, optionally filtering by batch ID.

    This function scans the samples directory for files matching the pattern
    <subreddit>_batch{N}_sample{M}.txt and returns information about each file,
    including the subreddit name and batch ID.

    Parameters:
    - samples_dir: path to the directory containing sample files
    - batch_filter: either "all" to include all batches, or a set of batch IDs to include

    Returns:
        list[dict]: list of dictionaries with keys: path, subreddit, batch_id, sample_size
    """
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        logging.error(f"Samples directory does not exist: {samples_dir}")
        return []
    
    # Pattern to match sample files: <subreddit>_batch{N}_sample{M}.txt
    pattern = re.compile(r"^(.+)_batch(\d+)_sample(\d+)\.txt$")
    
    sample_files = []
    for sample_file in samples_path.glob("*_batch*_sample*.txt"):
        match = pattern.match(sample_file.name)
        if not match:
            logging.warning(f"Skipping file with unexpected name: {sample_file.name}")
            continue
        
        subreddit = match.group(1)
        batch_id = int(match.group(2))
        sample_size = int(match.group(3))
        
        # Apply batch filter if not "all"
        if batch_filter != "all" and batch_id not in batch_filter:
            continue
        
        sample_files.append({
            "path": sample_file,
            "subreddit": subreddit,
            "batch_id": batch_id,
            "sample_size": sample_size,
        })
    
    return sample_files


def load_authors_from_sample_file(sample_file_info):
    """
    Loads author names from a single sample file.

    This function reads a sample file and returns the list of author names,
    one per line, along with the metadata (subreddit, batch_id) for each author.

    Parameters:
    - sample_file_info: dictionary with keys: path, subreddit, batch_id

    Returns:
        list[tuple]: list of (author, subreddit, batch_id) tuples
    """
    path = sample_file_info["path"]
    subreddit = sample_file_info["subreddit"]
    batch_id = sample_file_info["batch_id"]
    
    authors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                authors.append((line, subreddit, batch_id))
    
    return authors


def build_master_list(sample_files):
    """
    Builds a globally deduplicated master author list from all sample files.

    This function reads all sample files, collects all author-subreddit-batch
    tuples, and performs global deduplication by author name. It returns both
    the deduplicated author list and the membership mapping.

    Parameters:
    - sample_files: list of sample file info dictionaries (from discover_sample_files)

    Returns:
        tuple: (deduplicated_authors, membership_rows)
               where deduplicated_authors is a list of unique author names,
               and membership_rows is a list of (author, subreddit, batch_id) tuples
    """
    # Collect all author-subreddit-batch tuples from all sample files
    all_tuples = []
    for sample_file_info in sample_files:
        authors = load_authors_from_sample_file(sample_file_info)
        all_tuples.extend(authors)
    
    logging.info(f"Loaded {len(all_tuples)} author entries from {len(sample_files)} sample files")
    
    # Perform global deduplication by author name
    # Use a dict to track first occurrence of each author (preserves insertion order)
    author_to_memberships = {}
    for author, subreddit, batch_id in all_tuples:
        if author not in author_to_memberships:
            author_to_memberships[author] = []
        author_to_memberships[author].append((subreddit, batch_id))
    
    # Extract deduplicated author list (preserves order of first appearance)
    deduplicated_authors = list(author_to_memberships.keys())
    
    # Build membership rows (long format: one row per author-subreddit-batch combination)
    membership_rows = []
    for author, memberships in author_to_memberships.items():
        for subreddit, batch_id in memberships:
            membership_rows.append((author, subreddit, batch_id))
    
    logging.info(f"After deduplication: {len(deduplicated_authors)} unique authors")
    logging.info(f"Membership mapping: {len(membership_rows)} author-subreddit-batch combinations")
    
    return deduplicated_authors, membership_rows


def write_master_author_list(deduplicated_authors, output_path):
    """
    Writes the globally deduplicated author list to a text file.

    This function writes the list of unique author names to a text file,
    one author per line, in the order they first appeared in the sample files.

    Parameters:
    - deduplicated_authors: list of unique author names
    - output_path: path where the master author list should be written

    Returns:
        None: creates the master author list file as a side effect
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for author in deduplicated_authors:
            f.write(author + "\n")


def write_membership_csv(membership_rows, output_path):
    """
    Writes the author-subreddit membership mapping to a CSV file.

    This function writes the membership mapping in long format (one row per
    author-subreddit-batch combination) to a CSV file with columns:
    author, subreddit, batch_id.

    Parameters:
    - membership_rows: list of (author, subreddit, batch_id) tuples
    - output_path: path where the membership CSV should be written

    Returns:
        None: creates the membership CSV file as a side effect
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["author", "subreddit", "batch_id"])
        for author, subreddit, batch_id in membership_rows:
            writer.writerow([author, subreddit, batch_id])


def main():
    """
    Main entry point for the script.

    Parses command-line arguments, sets up logging, discovers sample files,
    builds the globally deduplicated master list, and writes the output files.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-dir", required=True, help="Directory containing sampled author text files")
    ap.add_argument("--output-dir", required=True, help="Directory where master/ will be created")
    ap.add_argument("--batches", default="1", help="Batch IDs to include (comma-separated integers, or 'all' for all batches, default: '1')")
    ap.add_argument("--log-file", default=None, help="Optional file to write log messages to")
    args = ap.parse_args()

    setup_logging(args.log_file)

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse batch filter
    batch_filter = parse_batch_filter(args.batches)
    if batch_filter == "all":
        logging.info("Including all batches")
    else:
        logging.info(f"Including batches: {', '.join(sorted(str(b) for b in batch_filter))}")
    
    # Discover sample files
    sample_files = discover_sample_files(samples_dir, batch_filter)
    logging.info(f"Found {len(sample_files)} sample files matching batch filter")
    
    if not sample_files:
        logging.warning("No sample files found to process. Check that --samples-dir points to the correct location and that batch filter is appropriate.")
        return
    
    # Build master list with global deduplication
    deduplicated_authors, membership_rows = build_master_list(sample_files)
    
    # Write output files
    master_list_path = output_dir / "master" / "all_sampled_authors.txt"
    membership_csv_path = output_dir / "master" / "author_subreddit_membership.csv"
    
    write_master_author_list(deduplicated_authors, master_list_path)
    write_membership_csv(membership_rows, membership_csv_path)
    
    # Summary
    logging.info(f"Done. Master list written to {master_list_path}")
    logging.info(f"Membership mapping written to {membership_csv_path}")
    logging.info(f"Total unique authors: {len(deduplicated_authors)}")


if __name__ == "__main__":
    main()
