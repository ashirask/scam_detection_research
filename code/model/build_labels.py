"""
build_labels.py
---------------
Reads raw Reddit JSONL files (comments and submissions) plus optional sampled-authors file
and BotRank CSV, aggregates activity per author, and produces a labeled dataset.

Output: labels.parquet with binary labels (bot/human) and per-user statistics.
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# Try to import orjson for fast JSON parsing, fallback to stdlib json
try:
    import orjson as json_lib
except ImportError:
    import json as json_lib

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# False-positive guard for bot pattern matching
# These words contain "bot" as a substring but are not bot-related
BOT_FALSE_POSITIVES = {
    "bottle", "bottom", "botox", "both", "bother", "botanical",
    "botanic", "botswana", "bought", "boots", "booth"
}


def load_authors_filter(authors_file: Optional[str]) -> Optional[Set[str]]:
    """
    Load sampled usernames from a text file (one per line).
    
    Args:
        authors_file: Path to text file with one username per line
        
    Returns:
        Set of usernames if file provided, None otherwise
    """
    if not authors_file:
        return None
    
    logger.info(f"Loading authors filter from: {authors_file}")
    authors = set()
    with open(authors_file, 'r', encoding='utf-8') as f:
        for line in f:
            username = line.strip()
            if username:  # Skip empty lines
                authors.add(username)
    
    logger.info(f"  Loaded {len(authors)} authors from filter")
    return authors


def parse_jsonl_line(line: str) -> Optional[dict]:
    """
    Parse a single JSONL line using orjson or stdlib json.
    
    Args:
        line: JSONL line string
        
    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        # orjson returns bytes, need to decode
        if hasattr(json_lib, 'loads'):
            data = json_lib.loads(line)
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            return data
        else:
            return json_lib.loads(line)
    except Exception as e:
        logger.debug(f"Failed to parse line: {e}")
        return None


def stream_jsonl(
    filepath: str,
    authors_filter: Optional[Set[str]],
    record_type: str
) -> Dict[str, List[dict]]:
    """
    Stream a JSONL file and aggregate records by author.
    
    Args:
        filepath: Path to JSONL file
        authors_filter: Optional set of usernames to filter by
        record_type: 'comment' or 'submission' for logging
        
    Returns:
        Dict mapping author -> list of full record dicts
    """
    logger.info(f"Streaming {record_type}s from: {filepath}")
    
    user_records = defaultdict(list)
    skipped_deleted = 0
    skipped_filtered = 0
    total_lines = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            
            record = parse_jsonl_line(line)
            if not record:
                continue
            
            author = record.get('author')
            
            # Skip deleted/empty authors
            if not author or author in ['[deleted]', 'AutoModerator', '']:
                skipped_deleted += 1
                continue
            
            # Apply authors filter if provided
            if authors_filter and author not in authors_filter:
                skipped_filtered += 1
                continue
            
            # Store full record
            user_records[author].append(record)
    
    logger.info(f"  Processed {total_lines} lines")
    logger.info(f"  Found {len(user_records)} unique authors")
    logger.info(f"  Skipped {skipped_deleted} deleted/empty authors")
    if authors_filter:
        logger.info(f"  Skipped {skipped_filtered} authors not in filter")
    
    return user_records


def compute_author_stats(
    user_comments: Dict[str, List[dict]],
    user_submissions: Dict[str, List[dict]]
) -> Dict[str, dict]:
    """
    Compute per-author summary statistics.
    
    Args:
        user_comments: Dict mapping author -> list of comment records
        user_submissions: Dict mapping author -> list of submission records
        
    Returns:
        Dict mapping author -> summary stats dict
    """
    logger.info("Computing per-author statistics")
    
    # Get all unique authors across both datasets
    all_authors = set(user_comments.keys()) | set(user_submissions.keys())
    
    author_stats = {}
    
    for author in all_authors:
        comments = user_comments.get(author, [])
        submissions = user_submissions.get(author, [])
        
        # Get first author_fullname seen (from either comments or submissions)
        author_fullname = None
        if comments and comments[0].get('author_fullname'):
            author_fullname = comments[0]['author_fullname']
        elif submissions and submissions[0].get('author_fullname'):
            author_fullname = submissions[0]['author_fullname']
        
        # Collect all timestamps
        all_timestamps = []
        for record in comments:
            ts = record.get('created_utc')
            if ts is not None:
                all_timestamps.append(float(ts))
        for record in submissions:
            ts = record.get('created_utc')
            if ts is not None:
                all_timestamps.append(float(ts))
        
        # Compute time span
        if all_timestamps:
            first_seen = min(all_timestamps)
            last_seen = max(all_timestamps)
            span_days = (last_seen - first_seen) / 86400.0
        else:
            first_seen = None
            last_seen = None
            span_days = 0.0
        
        author_stats[author] = {
            'author': author,
            'author_fullname': author_fullname,
            'num_comments': len(comments),
            'num_submissions': len(submissions),
            'total_posts': len(comments) + len(submissions),
            'first_seen_utc': first_seen,
            'last_seen_utc': last_seen,
            'account_span_days': span_days,
        }
    
    logger.info(f"  Computed stats for {len(author_stats)} authors")
    return author_stats


def apply_bot_rules(username: str) -> tuple:
    """
    Apply username-based bot detection rules.
    
    Args:
        username: Reddit username
        
    Returns:
        Tuple of (label, label_source, label_confidence, reason)
        label is 'bot' or 'human'
    """
    username_lower = username.lower()
    
    # Pattern 1: Exact match "bot"
    if username_lower == "bot":
        return "bot", "rule", 1.0, "exact match 'bot'"
    
    # Pattern 2: Whole-word "bot"
    if re.search(r'\bbot\b', username_lower):
        # False-positive guard
        if username_lower not in BOT_FALSE_POSITIVES:
            return "bot", "rule", 1.0, r"whole-word \bbot\b match"
    
    # Pattern 3: Starts with "Auto" (case-sensitive)
    if username.startswith("Auto"):
        return "bot", "rule", 1.0, "startswith('Auto')"
    
    # Pattern 4: Whole-word "auto"
    if re.search(r'\bauto\b', username_lower):
        return "bot", "rule", 1.0, r"whole-word \bauto\b match"
    
    # Pattern 5: Whole-word "mod"
    if re.search(r'\bmod\b', username_lower):
        return "bot", "rule", 1.0, r"whole-word \bmod\b match"
    
    # Pattern 6: Underscore-bounded _bot or bot_
    if re.search(r'(^bot_|_bot$|_bot_)', username_lower):
        return "bot", "rule", 1.0, "underscore-bounded bot match"
    
    # Default: human
    return "human", "rule", 1.0, "no bot pattern"


def load_botrank(
    botrank_file: str,
    top_n: int
) -> Dict[str, float]:
    """
    Load BotRank CSV and create lookup dict for top-N users.
    Respects the CSV order from fetch_botrank.py (does not re-sort).
    
    Args:
        botrank_file: Path to BotRank CSV
        top_n: Number of top users to include (from CSV order)
        
    Returns:
        Dict mapping username.lower() -> score
    """
    logger.info(f"Loading BotRank from: {botrank_file}")
    
    df = pd.read_csv(botrank_file)
    
    # The CSV has columns: rank,bot_name,score,good_votes,bad_votes,comment_karma,link_karma
    # Respect the CSV order from fetch_botrank.py (do not re-sort)
    df_top = df.head(top_n)
    
    # Build lookup dict: username.lower() -> score
    botrank_lookup = {}
    for _, row in df_top.iterrows():
        username = row['bot_name']
        score = row['score']
        if pd.notna(username):
            botrank_lookup[username.lower()] = float(score)
    
    logger.info(f"  Loaded top {len(botrank_lookup)} users (respecting CSV order)")
    if botrank_lookup:
        logger.info(f"  Score range: {min(botrank_lookup.values()):.4f} - {max(botrank_lookup.values()):.4f}")
    
    return botrank_lookup


def apply_labels(
    author_stats: Dict[str, dict],
    botrank_lookup: Optional[Dict[str, float]],
    botrank_threshold: float
) -> Dict[str, dict]:
    """
    Apply rule-based and BotRank labels to all authors.
    
    Args:
        author_stats: Dict mapping author -> stats dict
        botrank_lookup: Optional BotRank lookup dict
        botrank_threshold: Minimum score to override with BotRank
        
    Returns:
        Updated author_stats with label fields added
    """
    logger.info("Applying labels")
    
    bot_rule_count = 0
    botrank_override_count = 0
    human_count = 0
    
    for author, stats in author_stats.items():
        # Step 1: Apply rule-based labeling
        label, label_source, confidence, reason = apply_bot_rules(author)
        
        # Log at debug level for auditability
        logger.debug(f"{author} → {label} [rule: {reason}]")
        
        # Step 2: Apply BotRank augmentation if provided
        botrank_score = np.nan
        if botrank_lookup:
            author_lower = author.lower()
            if author_lower in botrank_lookup:
                botrank_score = botrank_lookup[author_lower]
                # Override if score meets threshold
                if botrank_score >= botrank_threshold:
                    label = "bot"
                    label_source = "botrank"
                    confidence = botrank_score
                    logger.debug(f"{author} → bot [botrank: score={botrank_score:.4f}, top-{len(botrank_lookup)}]")
                    botrank_override_count += 1
        
        # Count labels
        if label == "bot":
            if label_source == "rule":
                bot_rule_count += 1
        else:
            human_count += 1
        
        # Add label fields to stats
        stats['label'] = label
        stats['label_source'] = label_source
        stats['label_confidence'] = confidence
        stats['botrank_score'] = botrank_score
        
        # Step 3: Compute numeric Y
        stats['y'] = 1 if label == "bot" else 0
    
    logger.info(f"  Rule-based bots: {bot_rule_count}")
    logger.info(f"  BotRank overrides: {botrank_override_count}")
    logger.info(f"  Humans: {human_count}")
    
    return author_stats


def filter_sparse_accounts(
    author_stats: Dict[str, dict],
    min_posts: int
) -> Dict[str, dict]:
    """
    Filter out accounts with insufficient posts.
    
    Args:
        author_stats: Dict mapping author -> stats dict
        min_posts: Minimum number of posts required
        
    Returns:
        Filtered author_stats dict
    """
    logger.info(f"Filtering accounts with < {min_posts} posts")
    
    filtered = {
        author: stats
        for author, stats in author_stats.items()
        if stats['total_posts'] >= min_posts
    }
    
    dropped = len(author_stats) - len(filtered)
    logger.info(f"  Dropped {dropped} sparse accounts")
    logger.info(f"  Retained {len(filtered)} accounts")
    
    return filtered


def write_user_records(
    user_records: Dict[str, List[dict]],
    output_file: str,
    record_type: str
):
    """
    Write per-user records to JSONL file for downstream use.
    
    Args:
        user_records: Dict mapping author -> list of records
        output_file: Output JSONL file path
        record_type: 'comment' or 'submission' for logging
    """
    logger.info(f"Writing user {record_type}s to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for author, records in user_records.items():
            # Write one JSON object per line with author as key
            output_record = {
                'author': author,
                'records': records
            }
            # Handle orjson (returns bytes) vs json (returns str)
            dumped = json_lib.dumps(output_record)
            if isinstance(dumped, bytes):
                dumped = dumped.decode('utf-8')
            f.write(dumped + '\n')
    
    logger.info(f"  Wrote {len(user_records)} authors")


def write_labels_parquet(
    author_stats: Dict[str, dict],
    output_file: str
):
    """
    Write labeled dataset to parquet file.
    
    Args:
        author_stats: Dict mapping author -> stats dict with labels
        output_file: Output parquet file path
    """
    logger.info(f"Writing labels to: {output_file}")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(author_stats, orient='index')
    
    # Ensure correct column order
    column_order = [
        'author', 'author_fullname', 'label', 'label_source',
        'label_confidence', 'y', 'num_comments', 'num_submissions',
        'total_posts', 'first_seen_utc', 'last_seen_utc',
        'account_span_days', 'botrank_score'
    ]
    
    # Select only columns that exist
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    # Write to parquet
    df.to_parquet(output_file, index=False)
    logger.info(f"  Wrote {len(df)} rows to {output_file}")


def print_summary(
    author_stats: Dict[str, dict],
    botrank_lookup: Optional[Dict[str, float]],
    output_file: str,
    summary_file: Optional[str] = None
):
    """
    Print and optionally write label summary.
    
    Args:
        author_stats: Dict mapping author -> stats dict with labels
        botrank_lookup: BotRank lookup dict (for top-N count)
        output_file: Output parquet file path
        summary_file: Optional file to write summary to
    """
    # Count labels
    bot_rule = sum(1 for s in author_stats.values() if s['label'] == 'bot' and s['label_source'] == 'rule')
    bot_botrank = sum(1 for s in author_stats.values() if s['label'] == 'bot' and s['label_source'] == 'botrank')
    human = sum(1 for s in author_stats.values() if s['label'] == 'human')
    
    # Count BotRank matches in sample
    botrank_matches = sum(1 for s in author_stats.values() if pd.notna(s['botrank_score']))
    
    summary = f"""=== Label Summary ===
Total labeled dataset          : {len(author_stats)}
  bot  (rule)                  : {bot_rule}
  bot  (botrank override)      : {bot_botrank}
  human                        : {human}

"""
    
    if botrank_lookup:
        summary += f"""BotRank top-N used             : {len(botrank_lookup)}
  BotRank matches in sample    : {botrank_matches}
"""
    
    summary += f"Output written to              : {output_file}\n"
    
    # Print to stdout
    print(summary)
    
    # Write to file if specified
    if summary_file:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Summary written to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Build labeled dataset from Reddit JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        '--comments',
        required=True,
        help='Path to comments JSONL file'
    )
    parser.add_argument(
        '--submissions',
        required=True,
        help='Path to submissions JSONL file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--authors-file',
        help='Path to sampled authors text file (one username per line)'
    )
    parser.add_argument(
        '--botrank',
        help='Path to BotRank CSV file'
    )
    parser.add_argument(
        '--botrank-top-n',
        type=int,
        default=500,
        help='Number of top BotRank users to use (default: 500)'
    )
    parser.add_argument(
        '--botrank-threshold',
        type=float,
        default=0.8,
        help='Minimum BotRank score to override label (default: 0.8)'
    )
    parser.add_argument(
        '--min-posts',
        type=int,
        default=3,
        help='Minimum posts per author to include (default: 3)'
    )
    parser.add_argument(
        '--output',
        default='labels.parquet',
        help='Output parquet file (default: labels.parquet)'
    )
    parser.add_argument(
        '--summary',
        help='Optional file to write label summary to'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Step 1: Load authors filter
    authors_filter = load_authors_filter(args.authors_file)
    
    # Step 2: Stream and aggregate comments
    user_comments = stream_jsonl(args.comments, authors_filter, 'comment')
    
    # Step 3: Stream and aggregate submissions
    user_submissions = stream_jsonl(args.submissions, authors_filter, 'submission')
    
    # Step 4: Compute per-author stats
    author_stats = compute_author_stats(user_comments, user_submissions)
    
    # Step 5: Load BotRank if provided
    botrank_lookup = None
    if args.botrank:
        botrank_lookup = load_botrank(args.botrank, args.botrank_top_n)
    
    # Step 6: Apply labels (rules + BotRank)
    author_stats = apply_labels(author_stats, botrank_lookup, args.botrank_threshold)
    
    # Step 7: Filter sparse accounts
    author_stats = filter_sparse_accounts(author_stats, args.min_posts)
    
    # Step 8: Write user records for downstream use
    comments_output = args.output.replace('.parquet', '_user_comments.jsonl')
    submissions_output = args.output.replace('.parquet', '_user_submissions.jsonl')
    write_user_records(user_comments, comments_output, 'comment')
    write_user_records(user_submissions, submissions_output, 'submission')
    
    # Step 9: Write labels parquet
    write_labels_parquet(author_stats, args.output)
    
    # Step 10: Print summary
    print_summary(author_stats, botrank_lookup, args.output, args.summary)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
