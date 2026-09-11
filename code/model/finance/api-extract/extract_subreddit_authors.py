#!/usr/bin/env python3
"""
extract_subreddit_authors.py
============================
Extracts minimal author-bearing records (author, created_utc, id) from Reddit
subreddits via the Arctic Shift API, one file per subreddit per kind (posts/comments),
resumable, rate-limit-aware.

This is Stage 1 of the 4-stage subreddit-to-sampled-author pipeline. It pulls the
raw metadata needed to build author pools and sample authors, without downloading
full post/comment bodies.

WHERE TO RUN THIS
-----------------
Run this on your cluster's LOGIN NODE or a DATA-TRANSFER NODE, not inside a
compute job spread across many nodes. Most HPC clusters block outbound
internet from compute nodes; even where it's allowed, hammering a free public
API from many parallel nodes is both against the spirit of the service and
liable to get you rate-limited or IP-banned. Point --output-dir at your
scratch/project filesystem so the cluster's disk is what you're using.

OUTPUT LAYOUT
-------------
<output-dir>/
    raw_meta/
        <subreddit>/
            posts.jsonl.gz          # {"author":..., "created_utc":..., "id":...}
            comments.jsonl.gz
    _done/
        <subreddit>/
            posts.done              # {"count": N, "last_cursor": X}
            comments.done
    logs/
        run.log
        heartbeat.log
    failures.log

Each .jsonl.gz holds one minimal record per line. The raw_meta data is permanent,
not scratch-temp-and-delete, so that Stage 3 can be re-run later to draw additional
non-overlapping batches of sampled authors without re-hitting the API.

USAGE
-----
python extract_subreddit_authors.py \
    --subreddits-file subreddits_200plus.json \
    --output-dir /scratch/$USER/subreddit_extraction \
    --workers 6 \
    --min-sleep 0.2 \
    --heartbeat-interval 300 \
    --log-file /scratch/$USER/subreddit_extraction/logs/run.log \
    --max-pages-safety 500000
"""

import argparse
import gzip
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = "https://arctic-shift.photon-reddit.com/api"
KIND_ENDPOINT = {
    "posts": f"{BASE_URL}/posts/search",
    "comments": f"{BASE_URL}/comments/search",
}

# Global variables shared across all worker threads for rate limiting
_rate_lock = threading.Lock()  # Lock to ensure thread-safe access to rate limit counter
_min_remaining_seen = [None]  # List (mutable) to track minimum API calls remaining seen

# Global tracking for heartbeat logging
_active_workers = {}  # Dictionary tracking each worker's current task: {worker_id: (subreddit, kind, pages_fetched, records_written)}
_active_workers_lock = threading.Lock()  # Lock for thread-safe access to active_workers dictionary
_worker_id_counter = 0  # Counter to assign unique IDs to each worker thread
_worker_id_counter_lock = threading.Lock()  # Lock for thread-safe increment of worker ID counter
_completed_subreddits = [0]  # Global counter for completed subreddits (list for mutability, shared with heartbeat logger)
_completed_subreddits_lock = threading.Lock()  # Lock for thread-safe access to completed counter


def get_worker_id():
    """
    Get a unique worker ID for logging purposes.

    Returns:
        int: A unique worker ID that increments each time this function is called.
    """
    global _worker_id_counter
    with _worker_id_counter_lock:  # Ensure only one thread increments at a time
        _worker_id_counter += 1  # Increment counter and return new ID
        return _worker_id_counter


def polite_get(session, url, params, min_sleep, max_retries=6):
    """
    Makes an HTTP GET request to the API with rate-limit awareness and automatic retries.

    This is the core function that talks to the Arctic Shift API. It handles:
    - Rate limiting (waiting when we're about to hit the API's limits)
    - Automatic retries on failures (network errors, server errors, timeouts)
    - Exponential backoff (waiting longer between retries)
    - Special handling for 422 errors (limited retries, detailed logging)

    Parameters:
    - session: a requests.Session object (reuses connections for efficiency)
    - url: the API endpoint to call
    - params: dictionary of query parameters (e.g., subreddit name, limit, sort order)
    - min_sleep: minimum time to wait between requests (politeness delay)
    - max_retries: how many times to retry before giving up (except 422 which uses fewer)

    Returns:
        requests.Response: the response object if successful

    Raises:
        RuntimeError: if all retries are exhausted
    """
    backoff = 2.0  # Initial backoff time: wait 2 seconds before first retry
    for attempt in range(max_retries):  # Try up to max_retries times before giving up
        try:
            resp = session.get(url, params=params, timeout=60)  # Make HTTP request with 60s timeout
        except requests.RequestException as e:
            # Network error occurred (no internet, DNS failure, connection reset, etc.)
            logging.warning(f"  network error ({e}); retrying in {backoff:.0f}s")
            time.sleep(backoff)  # Wait before retrying
            backoff = min(backoff * 2, 120)  # Double backoff time, max 120 seconds (exponential backoff)
            continue  # Try again with next attempt

        # Extract rate limit headers from API response
        remaining = resp.headers.get("X-RateLimit-Remaining")  # How many API calls we have left
        reset = resp.headers.get("X-RateLimit-Reset")  # When rate limit resets (timestamp or seconds)
        reset_at = resp.headers.get("X-RateLimit-Reset-At")  # Unix timestamp of next reset (newer header)
        if remaining is not None:
            with _rate_lock:  # Thread-safe update of global rate limit counter
                _min_remaining_seen[0] = int(remaining)
            if int(remaining) < 3:  # If we're running low on API calls (<3 remaining)
                sleep_for = 5.0  # Default sleep time
                header_used = "default"
                
                # Try X-RateLimit-Reset-At first (timestamp format - more precise)
                if reset_at is not None:
                    try:
                        reset_ts = float(reset_at)
                        sleep_for = max(reset_ts - time.time(), 1.0)
                        header_used = f"X-RateLimit-Reset-At (timestamp {reset_at})"
                    except ValueError:
                        logging.warning(f"Failed to parse X-RateLimit-Reset-At: {reset_at}")
                
                # Fall back to X-RateLimit-Reset (seconds-from-now or timestamp)
                elif reset is not None:
                    try:
                        reset_val = float(reset)
                        # If reset_val is a Unix timestamp, convert to seconds-from-now
                        if reset_val > time.time():
                            sleep_for = max(reset_val - time.time(), 1.0)
                            header_used = f"X-RateLimit-Reset (timestamp {reset_val})"
                        else:
                            sleep_for = max(reset_val, 1.0)
                            header_used = f"X-RateLimit-Reset (seconds {reset_val})"
                    except ValueError:
                        pass  # If parsing fails, use default sleep_for
                
                logging.info(f"  near rate limit (remaining={remaining}); sleeping {sleep_for:.1f}s using {header_used}")
                time.sleep(min(sleep_for, 120))  # Sleep until reset or max 120 seconds

        if resp.status_code == 200:
            # Success! Wait politeness delay and return response
            time.sleep(min_sleep)
            return resp
        if resp.status_code == 429:
            # HTTP 429 = Too Many Requests - we've hit the rate limit
            wait = float(resp.headers.get("Retry-After", backoff))  # API tells us how long to wait
            logging.warning(f"  429 rate limited; sleeping {wait:.0f}s")
            time.sleep(wait)  # Wait for the specified time
            backoff = min(backoff * 2, 120)  # Increase backoff for next retry
            continue
        if resp.status_code == 422:
            # HTTP 422 = Unprocessable Entity - client error (bad parameters)
            # Special handling: log details, retry only once, then give up
            logging.error(f"  422 Unprocessable Entity - URL: {resp.url}")
            logging.error(f"  422 Response body: {resp.text[:500]}")  # Log first 500 chars of response
            logging.error(f"  422 Params: {params}")  # Log the parameters we sent
            if attempt >= 1:  # Only retry once for 422 (attempt 0 and 1)
                logging.error(f"  422 failed after 1 retry, giving up")
                resp.raise_for_status()  # Raise exception to surface error to caller
            logging.warning(f"  422 retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
            continue
        if resp.status_code >= 500 or "timed out" in resp.text.lower():
            # HTTP 5xx = server error (temporary, should retry)
            logging.warning(f"  server error/timeout ({resp.status_code}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
            continue

        # Unrecoverable error (4xx other than 429, 422) - something wrong with our request
        if resp.status_code >= 400:
            logging.error(
                "Unrecoverable error - Status=%s\nURL=%s\nParams=%s\nBody=%s",
                resp.status_code,
                resp.url,
                params,
                resp.text[:500],  # First 500 chars of response body for debugging
            )
            resp.raise_for_status()  # Raise HTTPError with full details

    raise RuntimeError(f"Exceeded retries for {url} params={params}")  # All retries exhausted


def extract_records(payload):
    """
    Extracts the actual data records from the API response.

    The Arctic Shift API returns data in a specific format: {'data': [...]}
    This function extracts the list of records from that wrapper.
    It's defensive - it can handle if the API changes format.

    Parameters:
    - payload: the parsed JSON response from the API

    Returns:
        list: list of record dictionaries

    Raises:
        ValueError: if the response format is unrecognized
    """
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unrecognized response shape: {type(payload)} keys={getattr(payload, 'keys', lambda: None)()}")


def load_subreddits(path):
    """
    Loads subreddit names from a .txt or .json file.

    This function handles both text files (one subreddit per line) and JSON files
    (either a flat array of strings or an array of objects with common keys).

    Parameters:
    - path: path to the file containing subreddit names

    Returns:
        list[str]: a flat, deduped, order-preserving list of subreddit name strings

    Raises:
        ValueError: if the JSON file has an unrecognized shape
    """
    path = Path(path)
    if path.suffix == ".txt":
        # Text file: one subreddit per line, skip blank lines
        with open(path, "r", encoding="utf-8") as f:
            subreddits = [line.strip() for line in f if line.strip()]
        # Dedupe while preserving order
        seen = set()
        deduped = []
        for sub in subreddits:
            if sub not in seen:
                seen.add(sub)
                deduped.append(sub)
        return deduped
    elif path.suffix == ".json":
        # JSON file: try to parse as flat array of strings first
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if len(data) == 0:
                return []
            # Case 1: flat array of strings
            if all(isinstance(item, str) for item in data):
                # Dedupe while preserving order
                seen = set()
                deduped = []
                for sub in data:
                    if sub not in seen:
                        seen.add(sub)
                        deduped.append(sub)
                return deduped
            # Case 2: array of objects, try common keys
            if all(isinstance(item, dict) for item in data):
                # Try keys in order: subreddit, name, display_name
                for key in ["subreddit", "name", "display_name"]:
                    if key in data[0]:
                        subreddits = [item[key] for item in data if key in item]
                        # Dedupe while preserving order
                        seen = set()
                        deduped = []
                        for sub in subreddits:
                            if sub not in seen:
                                seen.add(sub)
                                deduped.append(sub)
                        return deduped
        
        # If we get here, the JSON shape is unrecognized
        raise ValueError(f"Unrecognized JSON shape in {path}: expected flat array of strings or array of objects with 'subreddit'/'name'/'display_name' keys")
    
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix} (expected .txt or .json)")


def fetch_subreddit_records(session, subreddit, kind, after, before, limit, min_sleep, worker_id, starting_cursor=None, last_cursor_container=None, max_pages_safety=None):
    """
    Fetches ALL records (posts or comments) for a single subreddit.

    This function handles pagination - the API only returns a limited number of
    records per request (default 100), so we need to make multiple requests to get
    all of a subreddit's history. We paginate by using the 'after' parameter which
    tells the API to give us records after a certain timestamp.

    Parameters:
    - session: requests.Session for making HTTP requests
    - subreddit: subreddit name to fetch records for
    - kind: either 'posts' or 'comments'
    - after: Unix timestamp to start fetching from (optional)
    - before: Unix timestamp to stop fetching at (optional)
    - limit: number of records per page (max 100 for this API)
    - min_sleep: politeness delay between requests
    - worker_id: unique worker ID for logging
    - starting_cursor: if resuming, the cursor position to start from (optional)
    - last_cursor_container: mutable container [cursor] to store last cursor position (for resumability)
    - max_pages_safety: maximum number of pages to fetch before stopping (safety guard)

    Yields:
        dict: one record dictionary at a time (generator function)
    """
    endpoint = KIND_ENDPOINT[kind]  # Get the correct API endpoint URL
    cursor = starting_cursor if starting_cursor is not None else after  # Start from resume point or beginning
    seen_ids = set()  # Track record IDs we've already seen to avoid duplicates
    pages_fetched = 0  # Counter for how many API pages we've fetched
    records_yielded = 0  # Counter for total records yielded to caller

    if starting_cursor is not None:
        logging.info(f"[Worker-{worker_id}] Resuming {subreddit}/{kind} from cursor {starting_cursor}")
    else:
        logging.info(f"[Worker-{worker_id}] Starting {subreddit}/{kind}")

    try:
        while True:
            # Check max-pages safety guard
            if max_pages_safety is not None and pages_fetched >= max_pages_safety:
                logging.warning(f"[Worker-{(worker_id)}] {subreddit}/{kind}: hit max-pages-safety {max_pages_safety}, stopping early")
                break

            # Build query parameters for this API request
            params = {
                "subreddit": subreddit,
                "fields": "author,created_utc,id",  # Field-limiting to reduce payload size
                "limit": limit,  # Numeric limit for programmatic pagination
                "sort": "asc",  # Sort ascending by timestamp (oldest first)
            }
            if cursor is not None:
                params["after"] = cursor  # Only fetch records after this timestamp
            if before is not None:
                params["before"] = before  # Only fetch records before this timestamp

            # Update active worker status for heartbeat logging
            with _active_workers_lock:
                _active_workers[worker_id] = (subreddit, kind, pages_fetched, records_yielded)

            resp = polite_get(session, endpoint, params, min_sleep)  # Make API request
            records = extract_records(resp.json())  # Extract records from response
            pages_fetched += 1  # Increment page counter

            if not records:
                # No records returned - we've reached the end
                logging.info(f"[Worker-{worker_id}] {subreddit}/{kind}: fetched {pages_fetched} pages, {records_yielded} records total (no more records)")
                break

            new_count = 0  # Count of new records in this page (not seen before)
            for rec in records:
                rid = rec.get("id")
                if rid in seen_ids:
                    continue  # Skip duplicate records
                seen_ids.add(rid)  # Mark this record as seen
                new_count += 1
                records_yielded += 1
                yield rec  # Yield record to caller (generator pattern)

                # Log progress every 1000 records for large subreddits
                if records_yielded % 1000 == 0:
                    logging.info(f"[Worker-{worker_id}] {subreddit}/{kind}: {records_yielded} records fetched so far (page {pages_fetched})")

            if len(records) < limit:
                # Got fewer records than requested - this is the last page
                logging.info(f"[Worker-{worker_id}] {subreddit}/{kind}: fetched {pages_fetched} pages, {records_yielded} records total (last page)")
                break

            # Advance cursor to fetch next page
            last_created = records[-1].get("created_utc")  # Get timestamp of last record
            if last_created is None:
                logging.warning(f"[Worker-{worker_id}] {subreddit}/{kind}: no created_utc in last record, stopping")
                break
            new_cursor = int(last_created) + 1  # Move cursor past last record
            if cursor is not None and new_cursor <= cursor and new_count == 0:
                # Safety check: cursor not advancing and no new records - avoid infinite loop
                logging.warning(f"[Worker-{worker_id}] {subreddit}/{kind}: cursor not advancing, stopping early")
                break
            cursor = new_cursor  # Update cursor for next iteration
            if last_cursor_container is not None:
                last_cursor_container[0] = cursor  # Store cursor in container for caller
    finally:
        # Clear active worker status when done (even if exception occurs)
        with _active_workers_lock:
            if worker_id in _active_workers:
                del _active_workers[worker_id]


def process_subreddit(subreddit, output_dir, after, before, limit, min_sleep, max_pages_safety):
    """
    Downloads both posts and comments for a single subreddit.

    This is the main worker function that processes one subreddit completely.
    It's designed to be resumable - if the script crashes, it can skip subreddits
    that were already processed by checking for the .done marker files.

    Parameters:
    - subreddit: subreddit name
    - output_dir: base directory where data will be saved
    - after, before: time filters (optional)
    - limit: page size for API requests
    - min_sleep: politeness delay
    - max_pages_safety: maximum pages to fetch per subreddit/kind (safety guard)

    Returns:
        tuple: (subreddit, results_dict) where results_dict has 'posts' and 'comments' keys
               with either counts or error messages
    """
    worker_id = get_worker_id()  # Get unique ID for this worker thread
    session = requests.Session()  # Create session for connection reuse
    session.headers["User-Agent"] = "bot-detection-research/2.0"  # Identify our client

    results = {}  # Dictionary to store results for posts and comments
    try:
        for kind in ("posts", "comments"):  # Process both types
            done_marker = output_dir / "_done" / subreddit / f"{kind}.done"  # Path to completion marker
            out_path = output_dir / "raw_meta" / subreddit / f"{kind}.jsonl.gz"  # Path to output file

            # Check for existing progress (resumability)
            starting_cursor = None
            existing_count = 0
            if done_marker.exists():
                try:
                    done_content = done_marker.read_text().strip()
                    # Try to parse as JSON (new format with cursor)
                    done_data = json.loads(done_content)
                    if isinstance(done_data, dict) and "count" in done_data:
                        existing_count = done_data["count"]
                        starting_cursor = done_data.get("last_cursor")
                        if starting_cursor is not None:
                            logging.info(f"[Worker-{worker_id}] {subreddit}/{kind}: resuming from cursor {starting_cursor} (already have {existing_count} records)")
                        else:
                            # Old format or completed without cursor - skip
                            results[kind] = "skipped"
                            logging.info(f"[Worker-{worker_id}] {subreddit}/{kind}: skipped (already done)")
                            continue
                    else:
                        # Old format (just count) - skip
                        results[kind] = "skipped"
                        logging.info(f"[Worker-{worker_id}] {subreddit}/{kind}: skipped (already done)")
                        continue
                except (json.JSONDecodeError, ValueError):
                    # Old format (just count as string) - skip
                    results[kind] = "skipped"
                    logging.info(f"[Worker-{worker_id}] {subreddit}/{kind}: skipped (already done)")
                    continue

            # Create output directories if they don't exist
            out_path.parent.mkdir(parents=True, exist_ok=True)
            done_marker.parent.mkdir(parents=True, exist_ok=True)

            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")  # Use .tmp while writing
            count = existing_count  # Start from existing count if resuming
            last_cursor_container = [None]  # Container to store last cursor position
            try:
                # If resuming, copy existing file to .tmp first
                if starting_cursor is not None and out_path.exists():
                    shutil.copy(out_path, tmp_path)
                
                # Open compressed file for writing (gzip compression)
                # If resuming, append to existing file; otherwise create new
                mode = "at" if starting_cursor is not None else "wt"
                with gzip.open(tmp_path, mode, encoding="utf-8") as f:
                    # Fetch all records and write them one per line (JSONL format)
                    for rec in fetch_subreddit_records(session, subreddit, kind, after, before, limit, min_sleep, worker_id, starting_cursor, last_cursor_container, max_pages_safety):
                        f.write(json.dumps(rec) + "\n")  # Write each record as JSON string + newline
                        count += 1
                # Atomic rename: .tmp -> final filename (prevents partial files on crash)
                tmp_path.replace(out_path)
                # Create .done marker with record count and last cursor (for resumability)
                last_cursor = last_cursor_container[0]
                done_data = {"count": count, "last_cursor": last_cursor}
                done_marker.write_text(json.dumps(done_data))
                results[kind] = count  # Store success result
                logging.info(f"[Worker-{worker_id}] {subreddit}/{kind}: completed ({count} records)")
            except Exception as e:
                # If anything went wrong, clean up the .tmp file
                if tmp_path.exists():
                    tmp_path.unlink()
                error_msg = f"error: {type(e).__name__}: {str(e)}"
                logging.error(f"[Worker-{worker_id}] {subreddit}/{kind}: FAILED - {error_msg}")
                results[kind] = error_msg  # Store error result
    finally:
        session.close()  # Ensure session is closed when done

    return subreddit, results  # Return results to main thread


def heartbeat_logger(total_subreddits, interval_seconds=300):
    """
    Background thread that logs periodic progress updates.

    Parameters:
    - total_subreddits: total number of subreddits to process
    - interval_seconds: how often to log (default 300 = 5 minutes)
    """
    while True:
        time.sleep(interval_seconds)  # Wait for the specified interval
        
        # Get current status of all active workers
        with _active_workers_lock:
            active_info = []
            for worker_id, (subreddit, kind, pages, records) in _active_workers.items():
                active_info.append(f"Worker-{worker_id}: {subreddit}/{kind} (page {pages}, {records} records)")
        
        # Get completed subreddits count
        with _completed_subreddits_lock:
            completed = _completed_subreddits[0]
        
        if active_info:
            logging.info(f"Heartbeat - {completed}/{total_subreddits} subreddits done, {len(active_info)} active workers")
            for info in active_info:
                logging.info(f"  {info}")  # Log each worker's current status
        else:
            logging.info(f"Heartbeat - {completed}/{total_subreddits} subreddits done, no active workers (all idle)")


def main():
    """
    Main entry point for the script.

    Parses command-line arguments, sets up logging, loads subreddit list,
    and coordinates the parallel downloading of data.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subreddits-file", required=True, help="Text or JSON file with subreddit names")
    ap.add_argument("--output-dir", required=True, help="Directory where downloaded data will be saved")
    ap.add_argument("--after", default=None, help="Only fetch posts/comments after this Unix timestamp (optional)")
    ap.add_argument("--before", default=None, help="Only fetch posts/comments before this Unix timestamp (optional)")
    ap.add_argument("--limit", type=int, default=100, help="Page size for API requests (default: 100)")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent subreddits in flight (default: 6, keep modest)")
    ap.add_argument("--min-sleep", type=float, default=0.2, help="Baseline delay between requests, seconds (default: 0.2)")
    ap.add_argument("--heartbeat-interval", type=int, default=300, help="Heartbeat logging interval in seconds (default: 300 = 5 minutes)")
    ap.add_argument("--log-file", default=None, help="Optional file to write log messages to")
    ap.add_argument("--max-pages-safety", type=int, default=500000, help="Maximum pages per subreddit/kind before stopping (safety guard, default: 500000)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(args.log_file)] if args.log_file else []),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load subreddits from file (handles both .txt and .json)
    subreddits = load_subreddits(args.subreddits_file)
    logging.info(f"Loaded {len(subreddits)} subreddits from {args.subreddits_file}")

    failures_path = output_dir / "failures.log"  # Path to failures log file
    completed = 0  # Counter for completed subreddits
    
    # Initialize global completed counter for heartbeat logger
    with _completed_subreddits_lock:
        _completed_subreddits[0] = completed
    
    # Start heartbeat logger thread (daemon thread will die when main thread exits)
    heartbeat_thread = threading.Thread(
        target=heartbeat_logger,
        args=(len(subreddits), args.heartbeat_interval),
        daemon=True  # Daemon thread: killed when main thread exits
    )
    heartbeat_thread.start()
    logging.info(f"Started heartbeat logger (interval: {args.heartbeat_interval}s)")
    
    # Use ThreadPoolExecutor to process multiple subreddits in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(failures_path, "a", encoding="utf-8") as fail_f:
        # Submit all subreddits to the thread pool for processing
        futures = {
            pool.submit(process_subreddit, subreddit, output_dir, args.after, args.before, args.limit, args.min_sleep, args.max_pages_safety): subreddit
            for subreddit in subreddits
        }
        # Process futures as they complete (not in submission order)
        for fut in as_completed(futures):
            subreddit = futures[fut]
            try:
                _, results = fut.result()  # Get result from completed future
            except Exception as e:
                # If the worker itself crashed (not an API error)
                logging.error(f"Unhandled error for {subreddit}: {type(e).__name__}: {e}")
                fail_f.write(f"{subreddit}\tunhandled\t{type(e).__name__}: {e}\n")
                fail_f.flush()
                continue

            completed += 1
            # Update global completed counter for heartbeat logger
            with _completed_subreddits_lock:
                _completed_subreddits[0] = completed
            
            # Check if any errors occurred during processing
            has_error = any(isinstance(v, str) and v.startswith("error") for v in results.values())
            if has_error:
                # Enhanced failure logging with context
                for kind, result in results.items():
                    if isinstance(result, str) and result.startswith("error"):
                        fail_f.write(f"{subreddit}\t{kind}\t{result}\n")
                fail_f.flush()
                logging.warning(f"Subreddit {subreddit} had errors: {results}")

            # Print progress every 10 subreddits
            if completed % 10 == 0:
                with _rate_lock:
                    remaining = _min_remaining_seen[0]
                logging.info(f"Progress: {completed}/{len(subreddits)} subreddits done (rate-remaining={remaining})")

    logging.info(f"Done. {completed}/{len(subreddits)} subreddits processed. See {failures_path} for any failures.")


if __name__ == "__main__":
    main()
