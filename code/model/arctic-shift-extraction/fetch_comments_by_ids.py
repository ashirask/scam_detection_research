#!/usr/bin/env python3
"""
fetch_comments_by_ids.py
=========================
Fetches comment metadata (author, created_utc) for a large list of comment IDs
from the Arctic Shift API (https://arctic-shift.photon-reddit.com/api), using the
ID Lookup Endpoint (/api/comments/ids). Designed for processing millions of IDs
efficiently with memory-conscious streaming and resumable execution.

KEY FEATURES:
- Batch processing: up to 500 IDs per API request (API maximum)
- Memory efficient: streams IDs from input file, doesn't load all into memory
- Parallel processing: multiple workers process batches concurrently
- Rate limiting: respects API rate limits with automatic retries
- Resumable: tracks progress to resume from interruptions
- Compressed output: writes to .jsonl.gz to save disk space

WHERE TO RUN THIS
-----------------
Run this on your cluster's LOGIN NODE or a DATA-TRANSFER NODE, not inside a
compute job spread across many nodes. Most HPC clusters block outbound
internet from compute nodes; even where it's allowed, hammering a free public
API from many parallel nodes is both against the spirit of the service and
liable to get you rate-limited or IP-banned.

OUTPUT FORMAT
-------------
<output-file>.jsonl.gz - One JSON record per line, compressed with gzip
Each record contains: {"id": "...", "author": "...", "created_utc": ...}

USAGE
-----
python fetch_comments_by_ids.py \
    --input-file comment_ids.txt \
    --output-file comment_metadata.jsonl.gz \
    --fields author,created_utc \
    --workers 6 \
    --batch-size 500
"""

import argparse
import gzip
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = "https://arctic-shift.photon-reddit.com/api"
COMMENTS_IDS_ENDPOINT = f"{BASE_URL}/comments/ids"

# Global variables shared across all worker threads for rate limiting
_rate_lock = threading.Lock()  # Lock to ensure thread-safe access to rate limit counter
_min_remaining_seen = [None]  # List (mutable) to track minimum API calls remaining seen

# Global tracking for heartbeat logging
_active_workers = {}  # Dictionary tracking each worker's current task: {worker_id: (batch_num, ids_processed)}
_active_workers_lock = threading.Lock()  # Lock for thread-safe access to active_workers dictionary
_worker_id_counter = 0  # Counter to assign unique IDs to each worker thread
_worker_id_counter_lock = threading.Lock()  # Lock for thread-safe increment of worker ID counter
_total_batches_processed = [0]  # Global counter for completed batches (list for mutability)
_total_batches_processed_lock = threading.Lock()  # Lock for thread-safe access to batch counter
_total_ids_processed = [0]  # Global counter for total IDs processed (list for mutability)
_total_ids_processed_lock = threading.Lock()  # Lock for thread-safe access to IDs counter


def get_worker_id():
    """
    Get a unique worker ID for logging purposes.
    
    This function is thread-safe and ensures each worker thread gets a unique
    identifier that can be used in log messages to track which worker is
    doing what.
    
    Returns: int - unique worker ID
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
    - params: dictionary of query parameters (e.g., list of IDs, fields to fetch)
    - min_sleep: minimum time to wait between requests (politeness delay)
    - max_retries: how many times to retry before giving up (except 422 which uses fewer)
    
    Returns: the response object if successful
    Raises: RuntimeError if all retries are exhausted
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
        if remaining is not None:
            with _rate_lock:  # Thread-safe update of global rate limit counter
                _min_remaining_seen[0] = int(remaining)
            if int(remaining) < 3:  # If we're running low on API calls (<3 remaining)
                sleep_for = 5.0  # Default sleep time
                if reset is not None:
                    try:
                        reset_val = float(reset)
                        # If reset_val is a Unix timestamp, convert to seconds-from-now
                        sleep_for = max(reset_val - time.time(), 1.0) if reset_val > time.time() else reset_val
                    except ValueError:
                        pass  # If parsing fails, use default sleep_for
                logging.info(f"  near rate limit (remaining={remaining}); sleeping {sleep_for:.1f}s")
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
    
    Returns: list of record dictionaries
    Raises: ValueError if the response format is unrecognized
    """
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unrecognized response shape: {type(payload)} keys={getattr(payload, 'keys', lambda: None)()}")


def fetch_batch(session, ids_batch, fields, min_sleep, worker_id, batch_num):
    """
    Fetches comment metadata for a single batch of comment IDs.
    
    This function processes one batch of IDs (up to 500) by making a single
    API call to the /api/comments/ids endpoint. It handles rate limiting,
    retries, and extracts the requested fields from the response.
    
    Parameters:
    - session: requests.Session for making HTTP requests
    - ids_batch: list of comment IDs to fetch (max 500)
    - fields: comma-separated string of fields to request (e.g., "author,created_utc")
    - min_sleep: politeness delay between requests
    - worker_id: unique worker ID for logging
    - batch_num: batch number for logging/progress tracking
    
    Returns: tuple of (batch_num, list_of_records) or (batch_num, error_message)
    """
    # Update active worker status for heartbeat logging
    with _active_workers_lock:
        _active_workers[worker_id] = (batch_num, len(ids_batch))
    
    try:
        # Build query parameters for the API request
        # Always include 'id' in fields - critical for downstream pipeline compatibility
        field_list = fields.split(",")
        if "id" not in field_list:
            field_list.append("id")
        params = {
            "ids": ",".join(ids_batch),  # Comma-separated list of IDs
            "fields": ",".join(field_list),  # Comma-separated list of fields to fetch (always includes id)
        }
        
        logging.info(f"[Worker-{worker_id}] Fetching batch {batch_num} ({len(ids_batch)} IDs)")
        
        # Make API request with rate limiting and retries
        resp = polite_get(session, COMMENTS_IDS_ENDPOINT, params, min_sleep)
        
        # Extract records from response
        records = extract_records(resp.json())
        
        # Filter to only include requested fields (API might return extra fields)
        field_list = fields.split(",")
        filtered_records = []
        for rec in records:
            # Create a new dict with only the requested fields
            filtered_rec = {f: rec.get(f) for f in field_list if f in rec}
            # Always include the ID - this is critical for the downstream pipeline
            # The API returns the ID as the primary identifier for each record
            comment_id = rec.get("id")
            if comment_id:
                filtered_rec["id"] = comment_id
            else:
                # If ID is missing, this is a data issue - log it but still include the record
                logging.warning(f"[Worker-{worker_id}] Record missing ID in batch {batch_num}")
                filtered_rec["id"] = None
            filtered_records.append(filtered_rec)
        
        logging.info(f"[Worker-{worker_id}] Batch {batch_num} complete: {len(filtered_records)} records")
        return batch_num, filtered_records
        
    except Exception as e:
        # Log error and return error message
        error_msg = f"error: {type(e).__name__}: {str(e)}"
        logging.error(f"[Worker-{worker_id}] Batch {batch_num} FAILED - {error_msg}")
        return batch_num, error_msg
    finally:
        # Clear active worker status when done (even if exception occurs)
        with _active_workers_lock:
            if worker_id in _active_workers:
                del _active_workers[worker_id]


def write_records_to_file(records, output_path):
    """
    Writes records to the output file in JSONL format with gzip compression.
    
    This function appends records to the output file. Each record is written
    as a single line of JSON. The file is compressed with gzip to save space.
    
    Parameters:
    - records: list of record dictionaries to write
    - output_path: Path object for the output file
    """
    # Open file in append mode ('at') to add to existing file
    with gzip.open(output_path, "at", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")  # Write each record as JSON string + newline


def load_ids_from_file(input_path, batch_size, progress_file=None):
    """
    Generator that yields batches of IDs from the input file.
    
    This function streams IDs from the input file without loading all of them
    into memory at once. It yields batches of IDs (up to batch_size) for processing.
    If a progress file exists, it skips IDs that have already been processed.
    
    Parameters:
    - input_path: Path to the text file containing one ID per line
    - batch_size: maximum number of IDs per batch
    - progress_file: optional Path to file tracking processed IDs (for resumability)
    
    Yields: tuple of (batch_num, list_of_ids)
    """
    # Load set of already processed IDs if progress file exists
    processed_ids = set()
    if progress_file and progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                processed_ids = set(line.strip() for line in f if line.strip())
            logging.info(f"Loaded {len(processed_ids)} already-processed IDs from progress file")
        except Exception as e:
            logging.warning(f"Could not load progress file: {e}")
    
    batch = []
    batch_num = 0
    skipped_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            comment_id = line.strip()
            if not comment_id:
                continue  # Skip empty lines
            
            # Skip if already processed (resumability)
            if comment_id in processed_ids:
                skipped_count += 1
                continue
            
            batch.append(comment_id)
            
            # Yield batch when it reaches the batch size
            if len(batch) >= batch_size:
                batch_num += 1
                yield batch_num, batch
                batch = []  # Reset batch for next iteration
        
        # Yield final partial batch if there are remaining IDs
        if batch:
            batch_num += 1
            yield batch_num, batch
    
    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} already-processed IDs")


def update_progress_file(progress_file, ids):
    """
    Updates the progress file with newly processed IDs.
    
    This function appends IDs to the progress file to track which IDs have
    been successfully processed. This enables resumability if the script
    is interrupted.
    
    Parameters:
    - progress_file: Path to the progress file
    - ids: list of IDs that were successfully processed
    """
    with open(progress_file, "a", encoding="utf-8") as f:
        for comment_id in ids:
            f.write(comment_id + "\n")


def heartbeat_logger(total_batches, interval_seconds=300):
    """
    Background thread that logs periodic progress updates.
    
    This function runs in a separate thread and logs progress information
    at regular intervals. It shows how many batches have been processed,
    how many IDs have been processed, and what each active worker is doing.
    
    Parameters:
    - total_batches: total number of batches to process (estimated)
    - interval_seconds: how often to log (default 300 = 5 minutes)
    """
    while True:
        time.sleep(interval_seconds)  # Wait for the specified interval
        
        # Get current status of all active workers
        with _active_workers_lock:
            active_info = []
            for worker_id, (batch_num, batch_size) in _active_workers.items():
                active_info.append(f"Worker-{worker_id}: batch {batch_num} ({batch_size} IDs)")
        
        # Get global counters
        with _total_batches_processed_lock:
            batches_done = _total_batches_processed[0]
        with _total_ids_processed_lock:
            ids_done = _total_ids_processed[0]
        
        if active_info:
            logging.info(f"Heartbeat - {batches_done}/{total_batches} batches done, {ids_done} IDs processed, {len(active_info)} active workers")
            for info in active_info:
                logging.info(f"  {info}")  # Log each worker's current status
        else:
            logging.info(f"Heartbeat - {batches_done}/{total_batches} batches done, {ids_done} IDs processed, no active workers (all idle)")


def main():
    """
    Main entry point for the script.
    
    Parses command-line arguments, sets up logging, streams IDs from the
    input file, and coordinates the parallel fetching of comment metadata.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-file", required=True, help="Text file with comment IDs, one per line")
    ap.add_argument("--output-file", required=True, help="Output file for comment metadata (will be .jsonl.gz)")
    ap.add_argument("--fields", default=" author,created_utc", help="Comma-separated fields to fetch (default: author,created_utc)")
    ap.add_argument("--batch-size", type=int, default=500, help="Number of IDs per API request (max: 500, default: 500)")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent batches in flight (default: 6, keep modest)")
    ap.add_argument("--min-sleep", type=float, default=0.2, help="Baseline delay between requests, seconds (default: 0.2)")
    ap.add_argument("--heartbeat-interval", type=int, default=500, help="Heartbeat logging interval in seconds (default: 500 = 8.3 minutes)")
    ap.add_argument("--log-file", default=None, help="Optional file to write log messages to")
    ap.add_argument("--progress-file", default=None, help="Optional file to track processed IDs for resumability")
    args = ap.parse_args()

    # Validate batch size (API limit is 500)
    if args.batch_size > 500:
        logging.warning(f"Batch size {args.batch_size} exceeds API maximum of 500, using 500 instead")
        args.batch_size = 500

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(args.log_file)] if args.log_file else []),
    )

    # Ensure output file has .jsonl.gz extension
    output_path = Path(args.output_file)
    if not output_path.suffixes or output_path.suffixes[-2:] != ['.jsonl', '.gz']:
        output_path = output_path.with_suffix('.jsonl.gz')
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup progress file if specified
    progress_path = Path(args.progress_file) if args.progress_file else None
    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)

    # Count total lines in input file for progress estimation
    logging.info(f"Counting total IDs in {args.input_file}...")
    total_ids = sum(1 for _ in open(args.input_file, "r", encoding="utf-8") if _.strip())
    total_batches = (total_ids + args.batch_size - 1) // args.batch_size  # Ceiling division
    logging.info(f"Total IDs to process: {total_ids:,}")
    logging.info(f"Estimated batches: {total_batches:,}")
    logging.info(f"Output file: {output_path}")
    logging.info(f"Fields to fetch: {args.fields}")

    # Initialize global counters
    with _total_batches_processed_lock:
        _total_batches_processed[0] = 0
    with _total_ids_processed_lock:
        _total_ids_processed[0] = 0

    # Start heartbeat logger thread (daemon thread will die when main thread exits)
    heartbeat_thread = threading.Thread(
        target=heartbeat_logger,
        args=(total_batches, args.heartbeat_interval),
        daemon=True  # Daemon thread: killed when main thread exits
    )
    heartbeat_thread.start()
    logging.info(f"Started heartbeat logger (interval: {args.heartbeat_interval}s)")

    # Create output file only if it doesn't exist (for resumability)
    # If file exists, we'll append to it later; don't truncate existing data
    if not output_path.exists():
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            pass  # Just create empty file
        logging.info(f"Created new output file: {output_path}")
    else:
        logging.info(f"Output file already exists, will append to it: {output_path}")

    # Use ThreadPoolExecutor to process multiple batches in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Dictionary to track futures: {future: batch_num}
        futures = {}
        
        # Submit batches to the thread pool as they are generated
        for batch_num, ids_batch in load_ids_from_file(args.input_file, args.batch_size, progress_path):
            # Submit this batch for processing
            session = requests.Session()  # Create session for this worker
            session.headers["User-Agent"] = "bot-detection-research/1.0"  # Identify our client
            
            future = executor.submit(
                fetch_batch,
                session,
                ids_batch,
                args.fields,
                args.min_sleep,
                get_worker_id(),
                batch_num
            )
            futures[future] = batch_num
        
        # Process futures as they complete (not in submission order)
        for future in as_completed(futures):
            batch_num = futures[future]
            try:
                result_batch_num, result = future.result()  # Get result from completed future
                
                if isinstance(result, str) and result.startswith("error"):
                    # Error occurred during batch processing
                    logging.error(f"Batch {result_batch_num} failed: {result}")
                    continue
                
                # Success - write records to output file
                if result:  # result is a list of records
                    write_records_to_file(result, output_path)
                    
                    # Update progress file if specified
                    if progress_path:
                        processed_ids = [rec.get("id") for rec in result if rec.get("id")]
                        update_progress_file(progress_path, processed_ids)
                    
                    # Update global counters
                    with _total_batches_processed_lock:
                        _total_batches_processed[0] += 1
                    with _total_ids_processed_lock:
                        _total_ids_processed[0] += len(result)
                    
                    # Log progress every 100 batches
                    with _total_batches_processed_lock:
                        batches_done = _total_batches_processed[0]
                    if batches_done % 100 == 0:
                        with _total_ids_processed_lock:
                            ids_done = _total_ids_processed[0]
                        with _rate_lock:
                            remaining = _min_remaining_seen[0]
                        logging.info(f"Progress: {batches_done}/{total_batches} batches done, {ids_done}/{total_ids} IDs processed (rate-remaining={remaining})")
                
            except Exception as e:
                # If the worker itself crashed (not an API error)
                logging.error(f"Unhandled error for batch {batch_num}: {type(e).__name__}: {e}")
                continue

    # Final summary
    with _total_batches_processed_lock:
        final_batches = _total_batches_processed[0]
    with _total_ids_processed_lock:
        final_ids = _total_ids_processed[0]
    
    logging.info(f"Done. Processed {final_batches:,} batches, {final_ids:,} IDs.")
    logging.info(f"Output saved to: {output_path}")
    if progress_path:
        logging.info(f"Progress saved to: {progress_path}")


if __name__ == "__main__":
    main()
