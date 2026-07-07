# =============================================================================
# extract_bots_from_zst.py
# =============================================================================
# Two-pass bot extraction from Reddit .zst dumps:
#
# PASS 1 (Author Discovery):
#   - Streams a single month's .zst file (comments or submissions)
#   - Extracts only the 'author' and 'text' fields from each line (fast, minimal memory)
#   - Applies bot username pattern matching (Rule A) and BotRank lookup (Rule B) and "i am bot" pattern matching (Rule C)
#   - Counts posts per author and filters by --min-posts threshold
#   - Writes qualifying bot usernames to a text file
#
# PASS 2 (Full Record Extraction):
#   - Loads the global bot author list (produced by merge_pass1_authors.py)
#   - Re-streams the same .zst file
#   - Extracts complete JSON records only for authors in the global list
#   - Writes compressed .jsonl.gz output (all JSON fields preserved)
#
# This two-pass design enables:
#   1. Fast author discovery across all months
#   2. Optional capping of total bot authors via random sampling
#   3. Efficient full record extraction only for selected authors
#
# Usage:
#   # Pass 1
#   python extract_bots_from_zst.py \
#     --pass-num 1 \
#     --zst-file /data/reddit/RC_2024-01.zst \
#     --file-type comments \
#     --botrank botrank_top500.csv \
#     --botrank-top-n 500 \
#     --min-posts 3 \
#     --output-dir results/
#
#   # Pass 2
#   python extract_bots_from_zst.py \
#     --pass-num 2 \
#     --zst-file /data/reddit/RC_2024-01.zst \
#     --file-type comments \
#     --authors-file bot_authors_global.txt \
#     --output-dir results/
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import gzip
import random
import re
import sys
import time
import io
import json
from pathlib import Path

import zstandard as zstd  # Streaming zst decompression
import pandas as pd  # BotRank CSV loading

# Try to import orjson for faster JSON parsing, fallback to stdlib json
try:
    import orjson
except ImportError:
    orjson = None

# =============================================================================
# CONSTANTS
# =============================================================================

# Hardcoded skip list - applies to both passes
# These authors are unconditionally skipped before any matching logic
# Rationale:
#   - [deleted]/[removed]: No username means no features can be computed
#   - AutoModerator: Posts in nearly every subreddit, adds no discriminative signal
#   - "": Malformed records
SKIP_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}

# False-positive guard for the \bbot\b pattern
# These words contain "bot" as a substring but are not actual bot usernames
# They are excluded to avoid false matches like "bottle", "bottom", etc.
BOT_FALSE_POSITIVES = {
    "bottle", "bottom", "botox", "both", "bother", "botanical",
    "botanic", "botswana", "bought", "boots", "booth"
}

# Bot phrases for text-based detection (Rule C)
# These phrases commonly appear in bot comments/submissions
BOT_PHRASES = [
    "i am a bot",
    "beep beep",
    "beep boop",
    "i'm a bot",
    "im a bot",
    "this is a bot",
    "automated message",
    "auto moderator",
    "automated response",
    "bot message",
]


# =============================================================================
# FUNCTIONS
# =============================================================================

def rule_c_match(text: str):
    """
    Check if text content contains bot identification phrases (Rule C).
    
    This function searches for common bot phrases in comment body or
    submission selftext/title. It uses case-insensitive substring matching.
    
    Args:
        text (str): The text content to check (can be empty string)
    
    Returns:
        bool: True if the text contains any bot phrase, False otherwise
    
    Note:
        - Matching is case-insensitive
        - Empty strings return False
        - Phrases are defined in BOT_PHRASES constant
    """
    if not text:
        return False
    
    text_lower = text.lower()
    for phrase in BOT_PHRASES:
        if phrase in text_lower:
            return True
    return False


def rule_a_match(author: str):
    """
    Check if a Reddit username matches bot identification patterns (Rule A).
    
    This function implements regex-based pattern matching to identify potential
    bot accounts based on username characteristics. It checks multiple patterns
    in order and returns the first matching pattern name.
    
    Args:
        author (str): The Reddit username to check. Can be empty string.
    
    Returns:
        tuple: (bool, str or None)
            - bool: True if the author matches any bot pattern, False otherwise
            - str or None: The pattern name that matched (e.g., "exact", "bot", "mod", "_bot"), or None if no match

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
    
    # Whole-word "mod"
    if re.search(r'\bmod\b', username_lower):
        return True, "mod"
    
    # Underscore-bounded "_bot"
    if re.search(r'(^bot_|_bot$|_bot_)', username_lower):
        return True, "_bot"
    
    return False, None


def parse_json(line: str):
    """
    Parse a JSON string using orjson if available, with fallback to stdlib json.
    
    This function provides a unified interface for JSON parsing with preference
    for the faster orjson library. If orjson is not installed or fails, it
    gracefully falls back to the standard library json module.
    
    Args:
        line (str): A JSON-formatted string (typically a single line from a .zst file)
    
    Returns:
        dict or None: The parsed JSON object as a Python dictionary, or None if
                      parsing fails
    """
    if orjson is not None:
        try:
            return orjson.loads(line)
        except Exception:
            pass  # Fall through to stdlib json
    
    try:
        return json.loads(line)
    except Exception:
        return None


def format_runtime(seconds: float) -> str:
    """
    Convert a runtime in seconds to a human-readable string format.
    
    Args:
        seconds (float): Runtime duration in seconds (can be float for sub-second precision)
    
    Returns:
        str: Formatted string in "Xm Ys" format (e.g., "6m 14s")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def run_pass1(args, zst_path, zst_name):
    """
    Execute Pass 1: Author discovery phase.
    
    This function streams a single month's Reddit .zst file, extracts only the
    'author' field from each record, and identifies bot accounts using pattern
    matching (Rule A) and BotRank lookup (Rule B). It counts posts per author,
    applies a minimum post threshold, and writes qualifying bot usernames to a
    text file.
    
    Args:
        args: Parsed command-line arguments containing:
            - zst_file: Path to the .zst file
            - file_type: "comments" or "submissions"
            - botrank: Path to BotRank CSV file
            - botrank_top_n: Number of top bots to use from BotRank
            - min_posts: Minimum posts threshold for qualifying authors
            - output_dir: Output directory path
        zst_path (Path): Path object for the .zst file
        zst_name (str): Stem of the .zst filename (e.g., "RC_2024-01")
    
    Outputs:
        - pass1_authors_{file_type}_{period}.txt: Text file with one qualifying
          bot username per line (sorted alphabetically)
        - pass1_authors_{file_type}_{period}_summary.txt: Summary statistics
    
    Exit codes:
        - 0: Success
        - 1: Corruption or error detected (partial results may be saved)
    """
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
                    except (json.JSONDecodeError, ValueError) as e:
                        # Line-level JSON error - can skip and continue
                        stats["parse_errors"] += 1
                        if stats["parse_errors"] <= 10:  # Only log first 10 to avoid spam
                            print(f"[WARN] JSON parse error at line {stats['total_lines']}: {e}", file=sys.stderr)
                        continue
                    
                    if record is None:
                        stats["parse_errors"] += 1
                        continue
                    
                    author = record.get("author", "")
                    if author in SKIP_AUTHORS:
                        stats["skipped"] += 1
                        continue
                    
                    # Rule A: Username pattern matching
                    rule_hit, matched_pattern = rule_a_match(author)
                    
                    # Rule B: BotRank lookup
                    botrank_hit = author.lower() in botrank_set
                    
                    # Rule C: Text-based phrase matching
                    # Extract text based on file type (comments use 'body', submissions use 'selftext')
                    if args.file_type == "comments":
                        text_content = record.get("body", "")
                    else:  # submissions
                        # Check both selftext and title for submissions
                        selftext = record.get("selftext", "")
                        title = record.get("title", "")
                        text_content = f"{selftext} {title}"
                    
                    rule_c_hit = rule_c_match(text_content)
                    
                    # Determine if author matches bot detection rules
                    is_bot = rule_hit or botrank_hit or rule_c_hit
                    
                    # Match based on mode: bot mode extracts bots, human mode extracts non-bots
                    if (args.mode == "bot" and is_bot) or (args.mode == "human" and not is_bot):
                        author_post_count[author] = author_post_count.get(author, 0) + 1
                        stats["candidate_authors"] = len(author_post_count)
                    
                    # Progress update every 1M lines
                    if stats["total_lines"] % 1_000_000 == 0:
                        print(f"  Processed {stats['total_lines']:,} lines, {stats['candidate_authors']:,} candidate authors...")
    except zstd.ZstdError as e:
        # File-level corruption - cannot continue past this point
        corruption_detected = True
        print(f"[ERROR] Zstd decompression error at line {stats['total_lines']}: {e}", file=sys.stderr)
        print(f"[ERROR] File may be corrupted or truncated: {args.zst_file}", file=sys.stderr)
        print(f"[INFO] Partial results saved: {len(author_post_count)} authors found before error", file=sys.stderr)
    except Exception as e:
        # Unexpected error - treat as corruption
        corruption_detected = True
        print(f"[ERROR] Unexpected error at line {stats['total_lines']}: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"[INFO] Partial results saved: {len(author_post_count)} authors found before error", file=sys.stderr)
    
    runtime = time.time() - start_time
    
    # Apply minimum post threshold
    qualifying = {
        author for author, count in author_post_count.items()
        if count >= args.min_posts
    }
    stats["below_min_posts"] = len(author_post_count) - len(qualifying)
    
    # Apply max-authors cap via random sample if specified
    if args.max_authors and len(qualifying) > args.max_authors:
        import random
        random.seed(args.seed)
        qualifying = set(random.sample(sorted(qualifying), args.max_authors))
        stats["capped"] = True
        stats["cap_reason"] = f"random sample (seed={args.seed})"
    else:
        stats["capped"] = False
        stats["cap_reason"] = "no cap applied"
    
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
        f"Capped                   : {stats['capped']}",
        f"Cap reason               : {stats['cap_reason']}",
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
    """
    Execute Pass 2: Full record extraction phase.
    
    This function re-streams a single month's Reddit .zst file and extracts
    complete JSON records only for authors present in the global bot author list
    (produced by merge_pass1_authors.py). The output is compressed to save disk
    space and contains all JSON fields from the original records.
    
    Args:
        args: Parsed command-line arguments containing:
            - zst_file: Path to the .zst file
            - file_type: "comments" or "submissions"
            - authors_file: Path to global bot author list (from Pass 1 merge)
            - output_dir: Output directory path
        zst_path (Path): Path object for the .zst file
        zst_name (str): Stem of the .zst filename (e.g., "RC_2024-01")
    
    Outputs:
        - {file_type}_bots_{zst_stem}.jsonl.gz: Gzip-compressed JSONL file
          containing full Reddit records for matched authors
        - {file_type}_bots_{zst_stem}_summary.txt: Summary statistics
    
    Exit codes:
        - 0: Success
        - 1: Corruption or error detected (partial results may be saved)
    
    Note:
        The global author list is produced by merge_pass1_authors.py which takes
        the union of all Pass 1 outputs across months and optionally caps the
        total via random sampling.
    """
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
                    except (json.JSONDecodeError, ValueError) as e:
                        # Line-level JSON error - can skip and continue
                        stats["parse_errors"] += 1
                        if stats["parse_errors"] <= 10:  # Only log first 10 to avoid spam
                            print(f"[WARN] JSON parse error at line {stats['total_lines']}: {e}", file=sys.stderr)
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
        # File-level corruption - cannot continue past this point
        corruption_detected = True
        print(f"[ERROR] Zstd decompression error at line {stats['total_lines']}: {e}", file=sys.stderr)
        print(f"[ERROR] File may be corrupted or truncated: {args.zst_file}", file=sys.stderr)
        print(f"[INFO] Partial results saved: {stats['matched']} matches found before error", file=sys.stderr)
    except Exception as e:
        # Unexpected error - treat as corruption
        corruption_detected = True
        print(f"[ERROR] Unexpected error at line {stats['total_lines']}: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"[INFO] Partial results saved: {stats['matched']} matches found before error", file=sys.stderr)
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
    """
    Main entry point for the bot extraction script.
    
    This function parses command-line arguments, validates them based on the
    selected pass number, and dispatches to the appropriate pass function
    (run_pass1 or run_pass2).
    
    Command-line arguments:
        --pass-num (required): Pass number (1 or 2)
        --zst-file (required): Path to .zst file
        --file-type (required): "comments" or "submissions"
        --botrank (Pass 1 only): Path to BotRank CSV file
        --botrank-top-n (Pass 1 only): Number of top bots from BotRank (default: 500)
        --min-posts (Pass 1 only): Minimum posts threshold (default: 3)
        --authors-file (Pass 2 only): Path to global author list
        --output-dir: Output directory (default: current directory)
    
    Validation:
        - Pass 1 requires --botrank to be specified
        - Pass 2 requires --authors-file to be specified
        - Invalid arguments result in error message and exit code 1
    
    The function does not return; it calls sys.exit() with appropriate exit codes.
    """
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
        "--mode",
        choices=["bot", "human"],
        default="bot",
        help="Extraction mode: 'bot' (extract accounts matching bot rules) or 'human' (extract accounts NOT matching bot rules) (default: bot, Pass 1 only)",
    )
    parser.add_argument(
        "--max-authors",
        type=int,
        default=None,
        help="Cap total authors extracted via random sample (Pass 1 only, useful for human mode to avoid millions of results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when using --max-authors (default: 42)",
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
