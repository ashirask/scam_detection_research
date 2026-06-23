"""
extract_bots_from_zst.py
------------------------
Two-pass bot extraction from Reddit .zst dumps:
- Pass 1: Author discovery (fast, author-only parse)
- Pass 2: Full record extraction (complete JSON for confirmed authors)

Usage:
  # Pass 1
  python extract_bots_from_zst.py \
    --pass-num 1 \
    --zst-file /data/reddit/RC_2024-01.zst \
    --file-type comments \
    --botrank botrank_top500.csv \
    --botrank-top-n 500 \
    --min-posts 3 \
    --output-dir results/

  # Pass 2
  python extract_bots_from_zst.py \
    --pass-num 2 \
    --zst-file /data/reddit/RC_2024-01.zst \
    --file-type comments \
    --authors-file bot_authors_global.txt \
    --output-dir results/
"""

import argparse
import gzip
import random
import re
import sys
import time
import io
import json
from pathlib import Path

import zstandard as zstd
import pandas as pd

try:
    import orjson
except ImportError:
    orjson = None

# Hardcoded skip list - applies to both passes
SKIP_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}

BOT_FALSE_POSITIVES = {
    "bottle", "bottom", "botox", "both", "bother", "botanical",
    "botanic", "botswana", "bought", "boots", "booth"
}


def rule_a_match(author: str):
    """
    Check if author matches bot username patterns.
    Returns (bool, pattern_name_or_None) where pattern_name is one of:
    "exact", "bot", "Auto", "auto", "mod", "_bot"
    """
    if not author:
        return False, None
    
    username_lower = author.lower()
    
    # False positive guard
    if username_lower in BOT_FALSE_POSITIVES:
        return False, None
    
    # Exact "bot"
    if username_lower == "bot":
        return True, "exact"
    
    # Whole-word "bot"
    if re.search(r'\bbot\b', username_lower):
        return True, "bot"
    
    # Starts with "Auto" (case-sensitive)
    if author.startswith("Auto"):
        return True, "Auto"
    
    # Whole-word "auto"
    if re.search(r'\bauto\b', username_lower):
        return True, "auto"
    
    # Whole-word "mod"
    if re.search(r'\bmod\b', username_lower):
        return True, "mod"
    
    # Underscore-bounded "_bot"
    if re.search(r'(^bot_|_bot$|_bot_)', username_lower):
        return True, "_bot"
    
    return False, None


def parse_json(line: str):
    """Parse JSON using orjson if available, fallback to standard json."""
    if orjson is not None:
        try:
            return orjson.loads(line)
        except Exception:
            pass
    
    try:
        return json.loads(line)
    except Exception:
        return None


def format_runtime(seconds: float) -> str:
    """Format runtime as Xm Ys."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def run_pass1(args, zst_path, zst_name):
    """Pass 1: Author discovery - author-only parse, write qualifying authors to text file."""
    print(f"=== Pass 1: Author Discovery ===")
    
    # Load BotRank set
    print(f"Loading BotRank from {args.botrank}...")
    botrank_df = pd.read_csv(args.botrank)
    botrank_top = botrank_df.head(args.botrank_top_n)
    botrank_set = set(botrank_top["bot_name"].str.lower())
    print(f"  Loaded {len(botrank_set)} unique bot names from top {args.botrank_top_n}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract period from filename (e.g., RC_2024-01 -> 2024-01)
    period = zst_name.replace("RC_", "").replace("RS_", "")
    output_authors = output_dir / f"pass1_authors_{args.file_type}_{period}.txt"
    output_summary = output_dir / f"pass1_authors_{args.file_type}_{period}_summary.txt"
    
    author_post_count = {}  # {author: count}
    stats = {
        "total_lines": 0,
        "parse_errors": 0,
        "skipped": 0,
        "candidate_authors": 0,
        "below_min_posts": 0,
    }
    
    corruption_detected = False
    start_time = time.time()
    
    print(f"Processing {args.zst_file}...")
    
    try:
        with open(args.zst_file, 'rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                for line in text_stream:
                    stats["total_lines"] += 1
                    
                    try:
                        record = parse_json(line)
                    except Exception:
                        stats["parse_errors"] += 1
                        continue
                    
                    if record is None:
                        stats["parse_errors"] += 1
                        continue
                    
                    author = record.get("author", "")
                    if author in SKIP_AUTHORS:
                        stats["skipped"] += 1
                        continue
                    
                    rule_hit, matched_pattern = rule_a_match(author)
                    botrank_hit = author.lower() in botrank_set
                    
                    if rule_hit or botrank_hit:
                        author_post_count[author] = author_post_count.get(author, 0) + 1
                        stats["candidate_authors"] = len(author_post_count)
                    
                    # Progress update every 1M lines
                    if stats["total_lines"] % 1_000_000 == 0:
                        print(f"  Processed {stats['total_lines']:,} lines, {stats['candidate_authors']:,} candidate authors...")
    except zstd.ZstdError as e:
        corruption_detected = True
        print(f"[ERROR] Zstd decompression error after {stats['total_lines']} lines: {e}", file=sys.stderr)
        print(f"[ERROR] File may be corrupted or truncated: {args.zst_file}", file=sys.stderr)
    except Exception as e:
        corruption_detected = True
        print(f"[ERROR] Unexpected error after {stats['total_lines']} lines: {e}", file=sys.stderr)
    
    runtime = time.time() - start_time
    
    # Apply minimum post threshold
    qualifying = {
        author for author, count in author_post_count.items()
        if count >= args.min_posts
    }
    stats["below_min_posts"] = len(author_post_count) - len(qualifying)
    
    # Write qualifying authors
    with open(output_authors, 'w', encoding='utf-8') as f:
        for author in sorted(qualifying):
            f.write(author + '\n')
    
    # Write summary
    summary_lines = [
        f"=== Pass 1 Summary: {args.file_type} {zst_name} ===",
        f"Source file              : {args.zst_file}",
        f"Total lines scanned      : {stats['total_lines']:,}",
        f"Parse errors             : {stats['parse_errors']:,}",
        f"Skipped (skip list)      : {stats['skipped']:,}",
        f"Candidate authors found  : {stats['candidate_authors']:,}",
        f"Below min-posts (< {args.min_posts})    : {stats['below_min_posts']:,}",
        f"Qualifying authors output: {len(qualifying):,}",
        f"Output file              : {output_authors}",
        f"Corrupted                : {corruption_detected}",
        f"Runtime                  : {format_runtime(runtime)}",
    ]
    
    summary_text = "\n".join(summary_lines)
    
    with open(output_summary, 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')
    
    print("\n" + summary_text)
    
    if corruption_detected:
        print(f"[WARN] Exiting with partial results: {len(qualifying)} authors found before failure.")
        sys.exit(1)
    else:
        print(f"\nDone! Results written to {output_authors}")
        sys.exit(0)


def run_pass2(args, zst_path, zst_name):
    """Pass 2: Full record extraction - load full JSON for authors in global list."""
    print(f"=== Pass 2: Full Record Extraction ===")
    
    # Load author set
    print(f"Loading author list from {args.authors_file}...")
    with open(args.authors_file, 'r', encoding='utf-8') as f:
        bot_authors = set(line.strip() for line in f if line.strip())
    print(f"  Loaded {len(bot_authors)} authors")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output: compressed jsonl
    period = zst_path.stem.replace("RC_", "").replace("RS_", "")
    output_jsonl = output_dir / f"{args.file_type}_bots_{zst_path.stem}.jsonl.gz"
    output_summary = output_dir / f"{args.file_type}_bots_{zst_path.stem}_summary.txt"
    
    stats = {
        "total_lines": 0,
        "parse_errors": 0,
        "skipped": 0,
        "matched": 0,
    }
    unique_authors_matched = set()
    
    corruption_detected = False
    start_time = time.time()
    
    print(f"Processing {args.zst_file}...")
    
    out_f = None
    
    try:
        out_f = gzip.open(output_jsonl, 'wt', encoding='utf-8', compresslevel=6)
        
        with open(args.zst_file, 'rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                for line in text_stream:
                    stats["total_lines"] += 1
                    
                    try:
                        record = parse_json(line)
                    except Exception:
                        stats["parse_errors"] += 1
                        continue
                    
                    if record is None:
                        stats["parse_errors"] += 1
                        continue
                    
                    author = record.get("author", "")
                    if author in SKIP_AUTHORS:
                        stats["skipped"] += 1
                        continue
                    
                    if author not in bot_authors:
                        continue
                    
                    # Write full record
                    if orjson is not None:
                        out_f.write(orjson.dumps(record).decode('utf-8') + '\n')
                    else:
                        out_f.write(json.dumps(record) + '\n')
                    
                    stats["matched"] += 1
                    unique_authors_matched.add(author)
                    
                    # Progress update every 1M lines
                    if stats["total_lines"] % 1_000_000 == 0:
                        print(f"  Processed {stats['total_lines']:,} lines, {stats['matched']:,} matched so far...")
    except zstd.ZstdError as e:
        corruption_detected = True
        print(f"[ERROR] Zstd decompression error after {stats['total_lines']} lines: {e}", file=sys.stderr)
        print(f"[ERROR] File may be corrupted or truncated: {args.zst_file}", file=sys.stderr)
    except Exception as e:
        corruption_detected = True
        print(f"[ERROR] Unexpected error after {stats['total_lines']} lines: {e}", file=sys.stderr)
    finally:
        if out_f:
            out_f.close()
    
    runtime = time.time() - start_time
    
    # Write summary
    summary_lines = [
        f"=== Pass 2 Summary: {args.file_type} {zst_path.stem} ===",
        f"Source file              : {args.zst_file}",
        f"Authors in global list   : {len(bot_authors):,}",
        f"Total lines scanned      : {stats['total_lines']:,}",
        f"Parse errors             : {stats['parse_errors']:,}",
        f"Skipped (skip list)      : {stats['skipped']:,}",
        f"Matched and written      : {stats['matched']:,}",
        f"Unique authors matched   : {len(unique_authors_matched):,}",
        f"Output file              : {output_jsonl}",
        f"Corrupted                : {corruption_detected}",
        f"Runtime                  : {format_runtime(runtime)}",
    ]
    
    summary_text = "\n".join(summary_lines)
    
    with open(output_summary, 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')
    
    print("\n" + summary_text)
    
    if corruption_detected:
        print(f"[WARN] Exiting with partial results: {stats['matched']} matches found before failure.")
        sys.exit(1)
    else:
        print(f"\nDone! Results written to {output_jsonl}")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Two-pass bot extraction from Reddit .zst dumps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pass-num",
        type=int,
        required=True,
        choices=[1, 2],
        help="Pass number: 1 = author discovery, 2 = full record extraction",
    )
    parser.add_argument(
        "--zst-file",
        required=True,
        help="Path to .zst file (RC_YYYY-MM.zst or RS_YYYY-MM.zst)",
    )
    parser.add_argument(
        "--file-type",
        required=True,
        choices=["comments", "submissions"],
        help="Type of Reddit data in the zst file",
    )
    parser.add_argument(
        "--botrank",
        help="Path to BotRank CSV file (Pass 1 only)",
    )
    parser.add_argument(
        "--botrank-top-n",
        type=int,
        default=500,
        help="Number of top bots from BotRank to use (default: 500, Pass 1 only)",
    )
    parser.add_argument(
        "--min-posts",
        type=int,
        default=3,
        help="Minimum posts threshold for Pass 1 (default: 3)",
    )
    parser.add_argument(
        "--authors-file",
        help="Path to global author list file (Pass 2 only)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for results (default: current directory)",
    )
    
    args = parser.parse_args()
    
    zst_path = Path(args.zst_file)
    zst_name = zst_path.stem  # e.g., RC_2024-01 or RS_2024-01
    
    # Validate arguments per pass
    if args.pass_num == 1:
        if not args.botrank:
            print("[ERROR] --botrank is required for Pass 1", file=sys.stderr)
            sys.exit(1)
        run_pass1(args, zst_path, zst_name)
    else:  # pass_num == 2
        if not args.authors_file:
            print("[ERROR] --authors-file is required for Pass 2", file=sys.stderr)
            sys.exit(1)
        run_pass2(args, zst_path, zst_name)


if __name__ == "__main__":
    main()
