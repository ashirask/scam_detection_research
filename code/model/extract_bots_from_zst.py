"""
extract_bots_from_zst.py
------------------------
Streams a single month's Reddit .zst dump (comments or submissions) line by line,
and extracts every record whose author matches a bot username pattern or appears
in a BotRank known-bot list.

Usage:
  python extract_bots_from_zst.py \
    --zst-file /data/reddit/RC_2024-01.zst \
    --file-type comments \
    --botrank botrank_top500.csv \
    --botrank-top-n 500 \
    --output-dir results/
"""

import argparse
import re
import time
import io
import json
import sys
from pathlib import Path

import zstandard as zstd
import pandas as pd

try:
    import orjson
except ImportError:
    orjson = None

BOT_FALSE_POSITIVES = {
    "bottle", "bottom", "botox", "both", "bother", "botanical",
    "botanic", "botswana", "bought", "boots", "booth"
}


def rule_a_match(author: str):
    """
    Check if author matches bot username patterns.
    Returns (bool, pattern_name_or_None) where pattern_name is one of:
    "exact", "bot", "auto", "mod", "_bot"
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
        return True, "auto"
    
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


def process_line(line: str, botrank_set: set, stats: dict, unique_authors: set):
    """Process a single line from the zst file."""
    try:
        record = parse_json(line)
    except Exception:
        stats["parse_errors"] += 1
        return None
    
    if record is None:
        stats["parse_errors"] += 1
        return None
    
    author = record.get("author", "")
    if author == "[deleted]" or author == "":
        stats["skipped_deleted"] += 1
        return None
    
    rule_hit, matched_pattern = rule_a_match(author)
    botrank_hit = author.lower() in botrank_set
    
    if rule_hit or botrank_hit:
        reason = "rule" if rule_hit else "botrank"
        record["_match_reason"] = reason
        record["_rule_matched"] = rule_hit
        record["_matched_pattern"] = matched_pattern
        record["_botrank_matched"] = botrank_hit
        
        unique_authors.add(author.lower())
        
        if rule_hit and botrank_hit:
            stats["matched_both"] += 1
        elif rule_hit:
            stats["matched_rule"] += 1
        else:
            stats["matched_botrank"] += 1
        
        stats["matched"] += 1
        return record
    
    return None


def format_runtime(seconds: float) -> str:
    """Format runtime as Xm Ys."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def main():
    parser = argparse.ArgumentParser(
        description="Extract bot records from Reddit .zst dumps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        required=True,
        help="Path to BotRank CSV file",
    )
    parser.add_argument(
        "--botrank-top-n",
        type=int,
        default=500,
        help="Number of top bots from BotRank to use (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for results (default: current directory)",
    )
    
    args = parser.parse_args()
    
    # Load BotRank set
    print(f"Loading BotRank from {args.botrank}...")
    botrank_df = pd.read_csv(args.botrank)
    botrank_top = botrank_df.head(args.botrank_top_n)
    botrank_set = set(botrank_top["bot_name"].str.lower())
    print(f"  Loaded {len(botrank_set)} unique bot names from top {args.botrank_top_n}")
    
    # Determine output filenames
    zst_path = Path(args.zst_file)
    zst_name = zst_path.stem  # e.g., RC_2024-01 or RS_2024-01
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_jsonl = output_dir / f"{args.file_type}_bots_{zst_name}.jsonl"
    output_summary = output_dir / f"{args.file_type}_bots_{zst_name}_summary.txt"
    
    # Stats tracking
    stats = {
        "total_lines": 0,
        "parse_errors": 0,
        "skipped_deleted": 0,
        "matched": 0,
        "matched_rule": 0,
        "matched_botrank": 0,
        "matched_both": 0,
    }
    unique_authors = set()
    
    start_time = time.time()
    
    print(f"Processing {args.zst_file}...")
    
    corruption_detected = False
    out_f = None
    
    try:
        # Stream the zst file
        with open(args.zst_file, 'rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                
                out_f = open(output_jsonl, 'w', encoding='utf-8')
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    
                    stats["total_lines"] += 1
                    
                    matched_record = process_line(line, botrank_set, stats, unique_authors)
                    
                    if matched_record is not None:
                        out_f.write(json.dumps(matched_record) + '\n')
                    
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
    
    # Write summary regardless of whether corruption occurred
    summary_lines = [
        f"=== Bot Extraction Summary: {args.file_type}_bots_{zst_name} ===",
        f"Source file          : {args.zst_file}",
        f"Total lines processed: {stats['total_lines']:,}",
        f"Parse errors          : {stats['parse_errors']:,}",
        f"Skipped (deleted)     : {stats['skipped_deleted']:,}",
        f"Matched (total)       : {stats['matched']:,}",
        f"  via rule            : {stats['matched_rule']:,}",
        f"  via botrank         : {stats['matched_botrank']:,}",
        f"  via both             : {stats['matched_both']:,}",
        f"Unique authors matched: {len(unique_authors):,}",
        f"Output                : {output_jsonl}",
        f"Runtime               : {format_runtime(runtime)}",
        f"Corrupted             : {corruption_detected}",
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


if __name__ == "__main__":
    main()
