#!/usr/bin/env python3
"""
download_arctic_shift.py
=========================
Downloads full comment/submission history for a list of bot and human authors
from the Arctic Shift API (https://arctic-shift.photon-reddit.com/api), one
file per author, resumable, rate-limit-aware.

WHERE TO RUN THIS
-----------------
Run this on your cluster's LOGIN NODE or a DATA-TRANSFER NODE, not inside a
compute job spread across many nodes. Most HPC clusters block outbound
internet from compute nodes; even where it's allowed, hammering a free public
API from many parallel nodes is both against the spirit of the service and
liable to get you rate-limited or IP-banned. Point --output-dir at your
scratch/project filesystem so the cluster's disk is what you're using, then
run the (offline, CPU-only) feature-building step as a normal batch job.

OUTPUT LAYOUT
-------------
<output-dir>/
    comments/
        bot/<author>.jsonl.gz
        human/<author>.jsonl.gz
    submissions/
        bot/<author>.jsonl.gz
        human/<author>.jsonl.gz
    _done/
        comments/bot/<author>.done
        ...
    failures.log

Each .jsonl.gz holds one raw record per line, exactly as returned by the API.
Re-running the script skips any author whose .done sentinel already exists,
so it is safe to resubmit after a walltime kill.

USAGE
-----
python download_arctic_shift.py \
    --bots-file   bot_authors.txt \
    --humans-file human_authors.txt \
    --output-dir  raw_arctic_shift/ \
    --after 2025-01-01 --before 2026-01-01 \
    --workers 6
"""

# Import standard library modules for argument parsing, file compression, JSON handling, logging, etc.
import argparse
import gzip
import json
import logging
import os
import threading
import time
# ThreadPoolExecutor allows us to run multiple API requests in parallel (concurrently)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# requests is the library for making HTTP requests to the API
import requests

# Base URL for the Arctic Shift API - this is the server we're fetching Reddit data from
BASE_URL = "https://arctic-shift.photon-reddit.com/api"
# Dictionary mapping the type of content we want to fetch to their specific API endpoints
# "comments" -> endpoint for fetching Reddit comments
# "submissions" -> endpoint for fetching Reddit posts (called "posts" in the API)
KIND_ENDPOINT = {
    "comments": f"{BASE_URL}/comments/search",
    "submissions": f"{BASE_URL}/posts/search",
}

# Global variables shared across all worker threads for rate limiting
# Arctic Shift is a free public API, so we need to be respectful and not overwhelm it
# _rate_lock: ensures only one thread can update the rate limit counter at a time (thread safety)
# _min_remaining_seen: tracks the minimum number of API calls we have left before being rate-limited
_rate_lock = threading.Lock()
_min_remaining_seen = [None]


def polite_get(session, url, params, min_sleep, max_retries=6):
    """
    Makes an HTTP GET request to the API with rate-limit awareness and automatic retries.
    
    This is the core function that talks to the Arctic Shift API. It handles:
    - Rate limiting (waiting when we're about to hit the API's limits)
    - Automatic retries on failures (network errors, server errors, timeouts)
    - Exponential backoff (waiting longer between retries)
    
    Parameters:
    - session: a requests.Session object (reuses connections for efficiency)
    - url: the API endpoint to call
    - params: dictionary of query parameters (e.g., author name, limit, sort order)
    - min_sleep: minimum time to wait between requests (politeness delay)
    - max_retries: how many times to retry before giving up
    
    Returns: the response object if successful
    Raises: RuntimeError if all retries are exhausted
    """
    backoff = 2.0  # Initial backoff time in seconds (doubles after each retry)
    for attempt in range(max_retries):
        try:
            # Make the actual HTTP GET request to the API
            # timeout=60 means wait at most 60 seconds for a response
            resp = session.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            # If we get a network error (no internet, DNS failure, connection reset, etc.)
            logging.warning(f"  network error ({e}); retrying in {backoff:.0f}s")
            time.sleep(backoff)  # Wait before retrying
            backoff = min(backoff * 2, 120)  # Double the backoff time, max 120 seconds
            continue  # Try again with the next attempt

        # Check rate limit headers sent by the API
        # X-RateLimit-Remaining: how many API calls we have left in the current time window
        # X-RateLimit-Reset: when the rate limit will reset (either Unix timestamp or seconds from now)
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining is not None:
            with _rate_lock:
                # Update the global rate limit counter (thread-safe)
                _min_remaining_seen[0] = int(remaining)
            if int(remaining) < 3:
                # We're running low on API calls - pause to avoid being rate-limited
                sleep_for = 5.0
                if reset is not None:
                    try:
                        # Parse the reset time to know exactly when we can make more requests
                        reset_val = float(reset)
                        # If reset_val is a Unix timestamp, convert to seconds-from-now
                        sleep_for = max(reset_val - time.time(), 1.0) if reset_val > time.time() else reset_val
                    except ValueError:
                        pass
                logging.info(f"  near rate limit (remaining={remaining}); sleeping {sleep_for:.1f}s")
                time.sleep(min(sleep_for, 120))

        if resp.status_code == 200:
            # Success! Wait a bit to be polite, then return the response
            time.sleep(min_sleep)  # baseline politeness delay between requests
            return resp
        if resp.status_code == 429:
            # HTTP 429 = Too Many Requests - we've hit the rate limit
            # The API tells us how long to wait in the Retry-After header
            wait = float(resp.headers.get("Retry-After", backoff))
            logging.warning(f"  429 rate limited; sleeping {wait:.0f}s")
            time.sleep(wait)
            backoff = min(backoff * 2, 120)
            continue
        if resp.status_code >= 500 or resp.status_code == 422 or "timed out" in resp.text.lower():
            # HTTP 5xx = server error (temporary, retryable)
            logging.warning(f"  server error/timeout ({resp.status_code}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
            continue

        # Unrecoverable error (4xx other than 429, like 400, 404, 422)
        # These are client errors - something wrong with our request, not the server
        # Raise an exception to surface this to the caller
        #resp.raise_for_status()
        if resp.status_code >= 400:
            logging.error(
                "Status=%s\nURL=%s\nBody=%s",
                resp.status_code,
                resp.url,
                resp.text,
            )
            resp.raise_for_status()

    raise RuntimeError(f"Exceeded retries for {url} params={params}")


def extract_records(payload):
    """
    Extracts the actual data records from the API response.
    
    The Arctic Shift API returns data in a specific format: {'data': [...]}
    This function extracts the list of records from that wrapper.
    It's defensive - it can handle if the API changes format.
    
    Parameters:
    - payload: the parsed JSON response from the API
    
    Returns: list of record dictionaries
    Raises: ValueError if the response format is unrecognized
    """
    if isinstance(payload, dict) and "data" in payload:
        # Standard format: {'data': [record1, record2, ...]}
        return payload["data"]
    if isinstance(payload, list):
        # Sometimes the API returns just a list directly
        return payload
    raise ValueError(f"Unrecognized response shape: {type(payload)} keys={getattr(payload, 'keys', lambda: None)()}")


def fetch_author_records(session, author, kind, after, before, limit, min_sleep):
    """
    Fetches ALL records (comments or submissions) for a single author.
    
    This function handles pagination - the API only returns a limited number of
    records per request (default 100), so we need to make multiple requests to get
    all of an author's history. We paginate by using the 'after' parameter which
    tells the API to give us records after a certain timestamp.
    
    Parameters:
    - session: requests.Session for making HTTP requests
    - author: Reddit username to fetch records for
    - kind: either 'comments' or 'submissions'
    - after: Unix timestamp to start fetching from (optional)
    - before: Unix timestamp to stop fetching at (optional)
    - limit: number of records per page (max 100 for this API)
    - min_sleep: politeness delay between requests
    
    Yields: one record dictionary at a time (generator function)
    """
    endpoint = KIND_ENDPOINT[kind]  # Get the correct API endpoint URL
    cursor = after  # cursor tracks where we are in pagination (timestamp)
    seen_ids = set()  # Track IDs we've already seen to avoid duplicates

    while True:
        # Build the query parameters for this API request
        params = {
            "author": author,  # The Reddit username
            #"limit": limit,  # How many records to return per page
            "limit": "auto",  # API requires "auto" instead of numeric limit
            "sort": "asc",  # Sort in ascending order by timestamp (oldest first)
            "meta-app": "download-tool",  # Required parameter for the API
        }
        if cursor is not None:
            # If we have a cursor, only fetch records after this timestamp
            params["after"] = cursor
        if before is not None:
            # If we have a before date, only fetch records before this timestamp
            params["before"] = before

        # Make the API request (polite_get handles rate limiting and retries)
        resp = polite_get(session, endpoint, params, min_sleep)
        # Parse the JSON response and extract the actual records
        records = extract_records(resp.json())

        if not records:
            # No records returned - we've reached the end
            break

        new_count = 0
        for rec in records:
            rid = rec.get("id")
            if rid in seen_ids:
                # Skip duplicates (can happen with pagination)
                continue
            seen_ids.add(rid)
            new_count += 1
            yield rec  # Yield one record at a time (generator pattern)

        if len(records) < limit:
            # If we got fewer records than we asked for, this is the last page
            break

        # Advance the cursor to fetch the next page
        # We use the last record's timestamp + 1 to avoid re-fetching the same record
        last_created = records[-1].get("created_utc")
        if last_created is None:
            break
        new_cursor = int(last_created) + 1
        if cursor is not None and new_cursor <= cursor and new_count == 0:
            # Safety check: if cursor isn't advancing and we got no new records,
            # we're stuck in a loop - stop to avoid infinite requests
            logging.warning(f"  cursor not advancing for {author}/{kind}; stopping early")
            break
        cursor = new_cursor


def process_author(author, label, output_dir, after, before, limit, min_sleep):
    """
    Downloads both comments and submissions for a single author.
    
    This is the main worker function that processes one author completely.
    It's designed to be resumable - if the script crashes, it can skip authors
    that were already processed by checking for the .done marker files.
    
    Parameters:
    - author: Reddit username
    - label: either 'bot' or 'human' (for organizing output)
    - output_dir: base directory where data will be saved
    - after, before: time filters (optional)
    - limit: page size for API requests
    - min_sleep: politeness delay
    
    Returns: tuple of (author, label, results_dict) where results_dict has
             'comments' and 'submissions' keys with either counts or error messages
    """
    # Create a session object for making HTTP requests (more efficient than individual requests)
    session = requests.Session()
    # Set a User-Agent header so the API knows who is making the request
    # You should replace the email with your actual email for the API provider to contact you
    session.headers["User-Agent"] = "bot-detection-research/1.0"

    results = {}
    # Process both comments and submissions for this author
    for kind in ("comments", "submissions"):
        # Path to the .done marker file - if this exists, we already processed this author
        done_marker = output_dir / "_done" / kind / label / f"{author}.done"
        # Path to the actual data file (compressed JSON lines)
        out_path = output_dir / kind / label / f"{author}.jsonl.gz"

        if done_marker.exists():
            # Already processed - skip it (resumable feature)
            results[kind] = "skipped"
            continue

        # Create the output directories if they don't exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done_marker.parent.mkdir(parents=True, exist_ok=True)

        # Use a .tmp file while writing - if we crash, we don't leave a partial file
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        count = 0
        try:
            # Open a compressed file for writing (gzip compression)
            # wt = write text mode, utf-8 encoding
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                # Fetch all records and write them one per line (JSONL format)
                for rec in fetch_author_records(session, author, kind, after, before, limit, min_sleep):
                    f.write(json.dumps(rec) + "\n")  # Write each record as a JSON string + newline
                    count += 1
            # If we got here without errors, rename .tmp to the final filename (atomic operation)
            tmp_path.replace(out_path)
            # Create the .done marker file with the count of records
            done_marker.write_text(str(count))
            results[kind] = count
        except Exception as e:
            # If anything went wrong, clean up the .tmp file
            if tmp_path.exists():
                tmp_path.unlink()
            logging.error(f"  FAILED {author}/{kind}: {e}")
            results[kind] = f"error: {e}"

    return author, label, results


def load_authors(path, label):
    """
    Loads author usernames from a text file.
    
    Parameters:
    - path: path to the text file containing one username per line
    - label: either 'bot' or 'human' - attached to each author for organization
    
    Returns: list of tuples [(username, label), ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        # Read each line, strip whitespace, filter empty lines, add the label
        return [(line.strip(), label) for line in f if line.strip()]


def main():
    """
    Main entry point for the script.
    
    Parses command-line arguments, sets up logging, loads author lists,
    and coordinates the parallel downloading of data.
    """
    # Set up command-line argument parsing
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bots-file", required=True, help="Text file with bot usernames, one per line")
    ap.add_argument("--humans-file", required=True, help="Text file with human usernames, one per line")
    ap.add_argument("--output-dir", required=True, help="Directory where downloaded data will be saved")
    ap.add_argument("--after", default=None, help="Only fetch posts/comments after this date (e.g. 2025-01-01)")
    ap.add_argument("--before", default=None, help="Only fetch posts/comments before this date (e.g. 2026-01-01)")
    ap.add_argument("--limit", type=int, default=100, help="Deprecated: API now uses 'auto' limit (ignored)")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent authors in flight (default: 6, keep modest)")
    ap.add_argument("--min-sleep", type=float, default=0.2, help="Baseline delay between requests, seconds (default: 0.2)")
    ap.add_argument("--log-file", default=None, help="Optional file to write log messages to")
    args = ap.parse_args()

    # Configure logging to show timestamp, log level, and message
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(args.log_file)] if args.log_file else []),
    )

    # Create the output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load both bot and human author lists
    authors = load_authors(args.bots_file, "bot") + load_authors(args.humans_file, "human")
    logging.info(f"Loaded {len(authors)} authors total ({sum(1 for _, l in authors if l=='bot')} bot, "
                 f"{sum(1 for _, l in authors if l=='human')} human)")

    # Path to the failures log file
    failures_path = output_dir / "failures.log"
    completed = 0
    last_hourly_log = time.time()  # Track when we last logged hourly stats
    
    # Use ThreadPoolExecutor to process multiple authors in parallel
    # This creates a pool of worker threads (specified by --workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(failures_path, "a", encoding="utf-8") as fail_f:
        # Submit all authors to the thread pool for processing
        # Each future represents one author being processed
        futures = {
            pool.submit(process_author, author, label, output_dir, args.after, args.before, args.limit, args.min_sleep): (author, label)
            for author, label in authors
        }
        # Process futures as they complete (not in submission order)
        for fut in as_completed(futures):
            author, label = futures[fut]
            try:
                # Get the result from the completed future
                _, _, results = fut.result()
            except Exception as e:
                # If the worker itself crashed (not an API error)
                logging.error(f"Unhandled error for {author} ({label}): {e}")
                fail_f.write(f"{author}\t{label}\tunhandled\t{e}\n")
                fail_f.flush()
                continue

            completed += 1
            # Check if any errors occurred during processing
            has_error = any(isinstance(v, str) and v.startswith("error") for v in results.values())
            if has_error:
                # Log the failure to the failures.log file
                fail_f.write(f"{author}\t{label}\t{results}\n")
                fail_f.flush()

            # Print progress every 100 authors
            if completed % 100 == 0:
                with _rate_lock:
                    remaining = _min_remaining_seen[0]
                logging.info(f"Progress: {completed}/{len(authors)} authors done (last seen rate-remaining={remaining})")
            
            # Print hourly rate limit stats
            if time.time() - last_hourly_log >= 3600:  # Every hour (3600 seconds)
                with _rate_lock:
                    remaining = _min_remaining_seen[0]
                hours_elapsed = (time.time() - last_hourly_log) / 3600
                authors_per_hour = 100 / hours_elapsed if completed >= 100 else completed / hours_elapsed
                logging.info(f"Hourly stats: {completed}/{len(authors)} authors done ({authors_per_hour:.1f} authors/hour), rate-remaining={remaining}")
                last_hourly_log = time.time()

    logging.info(f"Done. {completed}/{len(authors)} authors processed. See {failures_path} for any failures.")


# This is the standard Python idiom for running a script
# If this file is run directly (not imported), execute main()
if __name__ == "__main__":
    main()
