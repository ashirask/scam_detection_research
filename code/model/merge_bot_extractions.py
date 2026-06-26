# =============================================================================
# merge_bot_extractions.py
# =============================================================================
# Consolidates Pass 2 compressed output files into per-author aggregated records.
#
# This script reads all Pass 2 output files (comments_bots_*.jsonl.gz and
# submissions_bots_*.jsonl.gz), aggregates records by author, and writes
# per-author JSONL files that match the format of the human-side files.
#
# The output files are direct inputs to build_features.py alongside the
# equivalent human files. Label assignment (y=1 for bots) happens in
# build_features.py, not here.
#
# Usage:
#   python merge_bot_extractions.py \
#     --input-dir    results/ \
#     --authors-file bot_authors_global.txt \
#     --output-dir   merged/ \
#     --min-posts    3
# =============================================================================

import argparse
import glob
import gzip
import json
from pathlib import Path

try:
    import orjson
except ImportError:
    orjson = None

# =============================================================================
# CONSTANTS
# =============================================================================

# Hardcoded skip list - same as extraction script
# Belt-and-suspenders guard against edge cases
SKIP_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}


# =============================================================================
# FUNCTIONS
# =============================================================================

def stream_jsonl_gz(path):
    """
    Stream JSON records from a gzip-compressed JSONL file.
    
    Args:
        path (str or Path): Path to the .jsonl.gz file
    
    Yields:
        dict: Parsed JSON record for each line
    
    Note:
        - Uses orjson if available for faster parsing
        - Skips malformed lines silently
        - Handles empty lines gracefully
    """
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if orjson is not None:
                    yield orjson.loads(line)
                else:
                    yield json.loads(line)
            except Exception:
                continue  # Skip malformed lines


def main():
    """
    Main entry point for merging Pass 2 bot extractions.
    
    This function:
    1. Streams all compressed comment and submission files
    2. Aggregates records by author
    3. Applies minimum post threshold
    4. Writes per-author JSONL files
    5. Performs sanity check against global authors list
    """
    parser = argparse.ArgumentParser(
        description="Merge Pass 2 bot extractions into per-author JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing Pass 2 output files (comments_bots_*.jsonl.gz, submissions_bots_*.jsonl.gz)"
    )
    parser.add_argument(
        "--authors-file",
        required=True,
        help="Path to bot_authors_global.txt from merge_pass1_authors.py"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for merged files"
    )
    parser.add_argument(
        "--min-posts",
        type=int,
        default=3,
        help="Minimum posts threshold (default: 3)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    stats = {
        "comment_files": 0,
        "submission_files": 0,
        "total_comments": 0,
        "total_submissions": 0,
        "unique_authors": 0,
        "authors_dropped": 0,
        "authors_kept": 0,
    }
    
    print("=== Merging Pass 2 Bot Extractions ===")
    
    # -------------------------------------------------------------------------
    # Step 1: Stream and aggregate all comment files
    # -------------------------------------------------------------------------
    print("\nProcessing comment files...")
    user_comments = {}  # {author: [full_record, ...]}
    
    comment_pattern = str(input_dir / "comments_bots_*.jsonl.gz")
    comment_files = sorted(glob.glob(comment_pattern))
    
    for filepath in comment_files:
        # Skip summary files
        if "_summary" in filepath:
            print(f"  Skipping {filepath} (summary file)")
            continue
        
        print(f"  Reading {filepath}...")
        stats["comment_files"] += 1
        
        for record in stream_jsonl_gz(filepath):
            author = record.get("author", "")
            if author in SKIP_AUTHORS:
                continue
            
            if author not in user_comments:
                user_comments[author] = []
            user_comments[author].append(record)
            stats["total_comments"] += 1
    
    print(f"  Processed {stats['comment_files']} comment files, {stats['total_comments']:,} records")
    
    # -------------------------------------------------------------------------
    # Step 2: Stream and aggregate all submission files
    # -------------------------------------------------------------------------
    print("\nProcessing submission files...")
    user_submissions = {}  # {author: [full_record, ...]}
    
    submission_pattern = str(input_dir / "submissions_bots_*.jsonl.gz")
    submission_files = sorted(glob.glob(submission_pattern))
    
    for filepath in submission_files:
        # Skip summary files
        if "_summary" in filepath:
            print(f"  Skipping {filepath} (summary file)")
            continue
        
        print(f"  Reading {filepath}...")
        stats["submission_files"] += 1
        
        for record in stream_jsonl_gz(filepath):
            author = record.get("author", "")
            if author in SKIP_AUTHORS:
                continue
            
            if author not in user_submissions:
                user_submissions[author] = []
            user_submissions[author].append(record)
            stats["total_submissions"] += 1
    
    print(f"  Processed {stats['submission_files']} submission files, {stats['total_submissions']:,} records")
    
    # -------------------------------------------------------------------------
    # Step 3: Compute per-author post counts and apply minimum post filter
    # -------------------------------------------------------------------------
    print(f"\nApplying minimum post threshold ({args.min_posts})...")
    
    all_authors = set(user_comments.keys()) | set(user_submissions.keys())
    stats["unique_authors"] = len(all_authors)
    
    kept_authors = set()
    dropped_authors = set()
    
    for author in all_authors:
        n_comments = len(user_comments.get(author, []))
        n_submissions = len(user_submissions.get(author, []))
        total = n_comments + n_submissions
        
        if total >= args.min_posts:
            kept_authors.add(author)
        else:
            dropped_authors.add(author)
            # Free memory for dropped authors
            user_comments.pop(author, None)
            user_submissions.pop(author, None)
    
    stats["authors_dropped"] = len(dropped_authors)
    stats["authors_kept"] = len(kept_authors)
    
    print(f"  Unique authors found: {stats['unique_authors']:,}")
    print(f"  Dropped (< {args.min_posts} posts): {stats['authors_dropped']:,}")
    print(f"  Final bot authors kept: {stats['authors_kept']:,}")
    
    # -------------------------------------------------------------------------
    # Step 4: Write per-author JSONL files
    # -------------------------------------------------------------------------
    print("\nWriting output files...")
    
    comments_out = output_dir / "user_comments_bots.jsonl"
    submissions_out = output_dir / "user_submissions_bots.jsonl"
    
    # Write comments file
    with open(comments_out, 'w', encoding='utf-8') as f:
        for author in sorted(kept_authors):
            records = user_comments.get(author, [])
            output_record = {"author": author, "comments": records}
            f.write(json.dumps(output_record) + '\n')
    print(f"  Wrote {len(kept_authors):,} authors to {comments_out}")
    
    # Write submissions file
    with open(submissions_out, 'w', encoding='utf-8') as f:
        for author in sorted(kept_authors):
            records = user_submissions.get(author, [])
            output_record = {"author": author, "submissions": records}
            f.write(json.dumps(output_record) + '\n')
    print(f"  Wrote {len(kept_authors):,} authors to {submissions_out}")
    
    # -------------------------------------------------------------------------
    # Step 5: Sanity check against authors file
    # -------------------------------------------------------------------------
    print("\nSanity check against global authors list...")
    
    with open(args.authors_file, 'r', encoding='utf-8') as f:
        expected_authors = set(line.strip() for line in f if line.strip())
    
    unexpected = kept_authors - expected_authors
    missing = expected_authors - kept_authors
    
    if unexpected:
        print(f"  [WARN] {len(unexpected)} authors in output not in global list")
    if missing:
        print(f"  [INFO] {len(missing)} expected authors have no records in Pass 2 output")
        print(f"         (likely posted only in corrupted months or below min-posts threshold)")
    
    # -------------------------------------------------------------------------
    # Step 6: Write summary
    # -------------------------------------------------------------------------
    summary_lines = [
        "=== merge_bot_extractions Summary ===",
        "",
        "Input",
        f"  Comment files processed  : {stats['comment_files']}",
        f"  Submission files processed: {stats['submission_files']}",
        f"  Total comment records    : {stats['total_comments']:,}",
        f"  Total submission records : {stats['total_submissions']:,}",
        "",
        "Authors",
        f"  Unique authors found     : {stats['unique_authors']:,}",
        f"  Dropped (< {args.min_posts} posts)      : {stats['authors_dropped']:,}",
        f"  Final bot authors kept   : {stats['authors_kept']:,}",
        "",
        "Output",
        f"  user_comments_bots.jsonl    : {stats['authors_kept']:,} authors",
        f"  user_submissions_bots.jsonl : {stats['authors_kept']:,} authors",
        "",
        "Note: label assignment (y=1) happens in build_features.py, not here.",
    ]
    
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    
    summary_file = output_dir / "merge_bot_extractions_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')
    print(f"\nSummary written to: {summary_file}")


if __name__ == "__main__":
    main()
