#!/usr/bin/env python3
"""
diagnose_bot_false_positives.py
================================
Diagnostic script to identify false positive bot accounts by re-applying
bot detection rules (A, B, C) to the actual downloaded content.

This addresses the source mismatch issue where authors were identified as bots
from .zst dumps (Pass 1) but their API-fetched content may not match the rules.

LOGIC:
- Rule A: Username pattern matching (skip if match)
- Rule B: BotRank lookup (skip if match)  
- Rule C: Text-based phrase matching (only checked if A and B don't match)

OPTIMIZATION:
- Only deep-dive into content if Rules A and B both fail
- This avoids expensive content scanning for obvious bots

IMPROVEMENTS:
- Word-boundary regex for Rule C to avoid false positives like "i'm a bottom"
- Added "bottom" to false positives list
- Outputs ranked list of suspicious authors with Rule C match counts and samples

USAGE:
python diagnose_bot_false_positives.py \
    --merged-dir merged/ \
    --botrank botrank_top500.csv \
    --output false_positives_report.txt
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging to output to both stdout and stderr for SLURM compatibility."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # stdout handler (for .out file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # stderr handler (for .err file)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    
    return logger


# =============================================================================
# CONSTANTS (from extract_bots_from_zst.py)
# =============================================================================

SKIP_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}

BOT_FALSE_POSITIVES = {
    "bottle", "bottom", "botox", "both", "bother", "botanical",
    "botanic", "botswana", "bought", "boots", "booth"
}

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

def rule_c_match_improved(text: str):
    """
    Check if text content contains bot identification phrases using word-boundary matching.
    
    This improved version uses regex with word boundaries to avoid false positives
    like "i'm a bottom" matching "i'm a bot".
    
    Args:
        text (str): The text content to check (can be empty string)
    
    Returns:
        bool: True if the text contains any bot phrase with word boundaries, False otherwise
    """
    if not text:
        return False
    
    text_lower = text.lower()
    for phrase in BOT_PHRASES:
        # Convert phrase to regex with word boundaries
        # Escape special regex characters in the phrase
        phrase_escaped = re.escape(phrase)
        # Add word boundaries at start and end
        pattern = rf'\b{phrase_escaped}\b'
        if re.search(pattern, text_lower):
            return True
    return False


def rule_a_match(author: str):
    """
    Check if a Reddit username matches bot identification patterns (Rule A).
    
    Args:
        author (str): The Reddit username to check. Can be empty string.
    
    Returns:
        tuple: (bool, str or None)
            - bool: True if the author matches any bot pattern, False otherwise
            - str or None: The pattern name that matched, or None if no match
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




def extract_text_from_record(record, record_type: str):
    """
    Extract text content from a JSON record (string or dict).
    
    Args:
        record: JSON string or dict representing a Reddit record
        record_type: Either "comments" or "submissions"
    
    Returns:
        str: The text content (body for comments, selftext+title for submissions)
    """
    try:
        # Handle both string and dict input
        if isinstance(record, str):
            obj = json.loads(record)
        else:
            obj = record
        
        if record_type == "comments":
            return obj.get("body", "")
        else:  # submissions
            selftext = obj.get("selftext", "")
            title = obj.get("title", "")
            return f"{selftext} {title}"
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


def scan_author_content(author_data: dict, label: str, sample_size: int, content_length: int, max_records: int = 100):
    """
    Scan an author's content for Rule C matches and collect samples.
    
    Args:
        author_data: Dictionary with 'comments' and 'submissions' keys
        label: Either 'bot' or 'human'
        sample_size: Number of samples to collect
        content_length: Maximum length of each sample in characters
        max_records: Maximum number of records to scan per author (default: 100)
    
    Returns:
        dict: {
            'rule_c_matches': int,
            'total_records': int,
            'rule_c_samples': list of str (Rule C matches),
            'general_samples': list of str (random samples regardless of Rule C)
        }
    """
    rule_c_matches = 0
    total_records = 0
    rule_c_samples = []
    general_samples = []
    
    # Parse comments and submissions
    comments = author_data.get("comments", [])
    if isinstance(comments, str):
        comments = json.loads(comments)
    
    submissions = author_data.get("submissions", [])
    if isinstance(submissions, str):
        submissions = json.loads(submissions)
    
    # Limit records to scan (for memory efficiency with bot accounts)
    if max_records is not None:
        comments = comments[:max_records]
        submissions = submissions[:max_records]
    
    # Calculate balanced sample allocation
    # Try to get roughly equal samples from comments and submissions
    num_comments = len(comments)
    num_submissions = len(submissions)
    
    if num_comments > 0 and num_submissions > 0:
        # Both types available - split evenly
        comment_samples_needed = (sample_size + 1) // 2  # ceil(sample_size/2)
        submission_samples_needed = sample_size - comment_samples_needed
    elif num_comments > 0:
        # Only comments available
        comment_samples_needed = sample_size
        submission_samples_needed = 0
    elif num_submissions > 0:
        # Only submissions available
        comment_samples_needed = 0
        submission_samples_needed = sample_size
    else:
        # No records
        comment_samples_needed = 0
        submission_samples_needed = 0
    
    # Scan comments
    for comment in comments:
        total_records += 1
        text = extract_text_from_record(comment, "comments")
        if rule_c_match_improved(text):
            rule_c_matches += 1
            if len(rule_c_samples) < sample_size:
                rule_c_samples.append(f"[COMMENT] {text[:content_length]}")
        
        # Collect general samples (up to allocated amount)
        if len(general_samples) < comment_samples_needed:
            general_samples.append(f"[COMMENT] {text[:content_length]}")
    
    # Scan submissions
    for submission in submissions:
        total_records += 1
        text = extract_text_from_record(submission, "submissions")
        if rule_c_match_improved(text):
            rule_c_matches += 1
            if len(rule_c_samples) < sample_size:
                rule_c_samples.append(f"[SUBMISSION] {text[:content_length]}")
        
        # Collect general samples (up to allocated amount)
        if len(general_samples) < submission_samples_needed:
            general_samples.append(f"[SUBMISSION] {text[:content_length]}")
    
    return {
        'rule_c_matches': rule_c_matches,
        'total_records': total_records,
        'rule_c_samples': rule_c_samples,
        'general_samples': general_samples
    }


def main():
    # Setup logging for SLURM compatibility
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--merged-dir",
        required=True,
        help="Directory containing merged files from merge_arctic_per_author.py"
    )
    parser.add_argument(
        "--botrank",
        required=True,
        help="Path to BotRank CSV file (for Rule B)"
    )
    parser.add_argument(
        "--output",
        default="false_positives_report.txt",
        help="Output file for the diagnostic report"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of sample records to show per author (default: 3)"
    )
    parser.add_argument(
        "--content-length",
        type=int,
        default=500,
        help="Maximum length of content samples in characters (default: 500)"
    )
    parser.add_argument(
        "--show-fp-samples",
        action="store_true",
        help="Show content samples for false positives (authors with no rule matches)"
    )
    parser.add_argument(
        "--sample-authors",
        type=int,
        default=0,
        help="Number of authors to randomly sample for diagnosis (0 = process all authors)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--author-list-output",
        default=None,
        help="Output file for list of all author names (one per line)"
    )
    parser.add_argument(
        "--max-records-per-author",
        type=int,
        default=None,
        help="Maximum number of records to scan per author for Rule C (default: no limit)"
    )
    
    args = parser.parse_args()
    
    merged_dir = Path(args.merged_dir)
    output_path = Path(args.output)
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")
    
    # Load BotRank for Rule B
    logger.info(f"Loading BotRank from {args.botrank}...")
    botrank_df = pd.read_csv(args.botrank)
    botrank_set = set(botrank_df["bot_name"].str.lower())
    logger.info(f"  Loaded {len(botrank_set)} unique bot names")
    
    # Load merged bot data (streaming approach)
    comments_bot_path = merged_dir / "user_comments_bot.jsonl"
    submissions_bot_path = merged_dir / "user_submissions_bot.jsonl"
    
    logger.info(f"Loading merged bot data from {merged_dir} (streaming)...")
    
    # Initialize results storage
    results = {
        'rule_a_match': [],      # Skipped - legitimate by username
        'rule_b_match': [],      # Skipped - legitimate by BotRank
        'rule_c_match': [],      # Has Rule C matches in content
        'no_match': [],           # False positive - no rules match
    }
    
    # Error counters
    parse_errors = 0
    
    # First pass: collect all author names (lightweight)
    logger.info("  First pass: collecting author names...")
    all_authors = set()
    comments_authors = set()
    submissions_authors = set()
    
    if comments_bot_path.exists():
        logger.info("    Scanning comments file...")
        line_count = 0
        with open(comments_bot_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % 100000 == 0:
                    logger.info(f"      Processed {line_count:,} lines, {len(comments_authors):,} authors found...")
                try:
                    obj = json.loads(line)
                    author = obj.get("author")
                    if author and author not in SKIP_AUTHORS:
                        comments_authors.add(author)
                        all_authors.add(author)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
        logger.info(f"    Found {len(comments_authors)} authors in comments ({line_count:,} total lines)")
    
    if submissions_bot_path.exists():
        logger.info("    Scanning submissions file...")
        line_count = 0
        with open(submissions_bot_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % 100000 == 0:
                    logger.info(f"      Processed {line_count:,} lines, {len(submissions_authors):,} authors found...")
                try:
                    obj = json.loads(line)
                    author = obj.get("author")
                    if author and author not in SKIP_AUTHORS:
                        submissions_authors.add(author)
                        all_authors.add(author)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
        logger.info(f"    Found {len(submissions_authors)} authors in submissions ({line_count:,} total lines)")
    
    # Verify overlap
    overlap = comments_authors & submissions_authors
    comments_only = comments_authors - submissions_authors
    submissions_only = submissions_authors - comments_authors
    logger.info("    Author overlap verification:")
    logger.info(f"      In both files: {len(overlap)}")
    logger.info(f"      Comments only: {len(comments_only)}")
    logger.info(f"      Submissions only: {len(submissions_only)}")
    logger.info(f"    Total unique authors: {len(all_authors)}")
    
    # Write author list if requested
    if args.author_list_output:
        author_list_path = Path(args.author_list_output)
        logger.info(f"  Writing author list to {author_list_path}...")
        with open(author_list_path, 'w', encoding='utf-8') as f:
            for author in sorted(all_authors):
                f.write(author + '\n')
        logger.info(f"    Wrote {len(all_authors)} authors")
    
    # Sample authors if requested
    if args.sample_authors > 0:
        sample_size = min(args.sample_authors, len(all_authors))
        sampled_authors = random.sample(list(all_authors), sample_size)
        logger.info(f"  Sampling {sample_size} authors from {len(all_authors)} total authors ({sample_size/len(all_authors)*100:.1f}%)")
        all_authors = set(sampled_authors)
    else:
        logger.info(f"  Processing all {len(all_authors)} authors")
    
    # Dictionary to accumulate author data from both files
    # Format: {author: {'comments': [...], 'submissions': [...]}}
    author_data_accumulator = {}
    
    # Stream comments file and accumulate (only for sampled authors)
    if comments_bot_path.exists():
        logger.info("  Processing comments file (for sampled authors)...")
        line_count = 0
        with open(comments_bot_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % 100000 == 0:
                    logger.info(f"    Processed {line_count:,} lines, {len(author_data_accumulator):,} authors loaded...")
                try:
                    obj = json.loads(line)
                    author = obj.get("author")
                    if author and author not in SKIP_AUTHORS and author in all_authors:
                        if author not in author_data_accumulator:
                            author_data_accumulator[author] = {'comments': [], 'submissions': []}
                        author_data_accumulator[author]['comments'] = obj.get('comments', [])
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
        logger.info(f"  Loaded {len(author_data_accumulator)} authors from comments ({line_count:,} total lines, {parse_errors} parse errors)")
    
    # Stream submissions file and diagnose incrementally
    authors_processed = 0
    if submissions_bot_path.exists():
        logger.info("  Processing submissions file and diagnosing incrementally...")
        line_count = 0
        with open(submissions_bot_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % 100000 == 0:
                    logger.info(f"    Processed {line_count:,} lines, {authors_processed} authors diagnosed...")
                try:
                    obj = json.loads(line)
                    author = obj.get("author")
                    if author and author not in SKIP_AUTHORS and author in all_authors:
                        if author not in author_data_accumulator:
                            author_data_accumulator[author] = {'comments': [], 'submissions': []}
                        author_data_accumulator[author]['submissions'] = obj.get('submissions', [])
                        
                        # Diagnose this author immediately now that we have both comments and submissions
                        author_data = author_data_accumulator[author]
                        
                        # Rule A: Username pattern matching
                        rule_hit, matched_pattern = rule_a_match(author)
                        if rule_hit:
                            results['rule_a_match'].append(author)
                        else:
                            # Rule B: BotRank lookup
                            botrank_hit = author.lower() in botrank_set
                            if botrank_hit:
                                results['rule_b_match'].append(author)
                            else:
                                # Rules A and B both failed - need to check Rule C in content
                                content_scan = scan_author_content(author_data, 'bot', args.sample_size, args.content_length, args.max_records_per_author)
                                
                                if content_scan['rule_c_matches'] > 0:
                                    results['rule_c_match'].append({
                                        'author': author,
                                        'matches': content_scan['rule_c_matches'],
                                        'total': content_scan['total_records'],
                                        'samples': content_scan['rule_c_samples']
                                    })
                                else:
                                    results['no_match'].append({
                                        'author': author,
                                        'total': content_scan['total_records'],
                                        'samples': content_scan['general_samples'] if args.show_fp_samples else []
                                    })
                        
                        # Clean up data immediately to save memory
                        del author_data_accumulator[author]
                        authors_processed += 1
                        
                        if authors_processed % 100 == 0:
                            logger.info(f"    Diagnosed {authors_processed} authors so far...")
                        if authors_processed % 500 == 0:
                            logger.info(f"      Rule A: {len(results['rule_a_match'])}, Rule B: {len(results['rule_b_match'])}, "
                                  f"Rule C: {len(results['rule_c_match'])}, No match: {len(results['no_match'])}")
                        
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
        logger.info(f"  Processed submissions ({line_count:,} total lines, {parse_errors} total parse errors)")
    
    # Process any authors that only had comments (no submissions)
    remaining_authors = list(author_data_accumulator.keys())
    if remaining_authors:
        logger.info(f"  Processing {len(remaining_authors)} authors with comments only (no submissions)...")
        for author in remaining_authors:
            author_data = author_data_accumulator[author]
            
            # Rule A: Username pattern matching
            rule_hit, matched_pattern = rule_a_match(author)
            if rule_hit:
                results['rule_a_match'].append(author)
            else:
                # Rule B: BotRank lookup
                botrank_hit = author.lower() in botrank_set
                if botrank_hit:
                    results['rule_b_match'].append(author)
                else:
                    # Rules A and B both failed - need to check Rule C in content
                    content_scan = scan_author_content(author_data, 'bot', args.sample_size, args.content_length, args.max_records_per_author)
                    
                    if content_scan['rule_c_matches'] > 0:
                        results['rule_c_match'].append({
                            'author': author,
                            'matches': content_scan['rule_c_matches'],
                            'total': content_scan['total_records'],
                            'samples': content_scan['rule_c_samples']
                        })
                    else:
                        results['no_match'].append({
                            'author': author,
                            'total': content_scan['total_records'],
                            'samples': content_scan['general_samples'] if args.show_fp_samples else []
                        })
            
            del author_data_accumulator[author]
            authors_processed += 1
    
    total_authors = len(results['rule_a_match']) + len(results['rule_b_match']) + len(results['rule_c_match']) + len(results['no_match'])
    logger.info(f"  Total unique bot authors diagnosed: {total_authors}")
    
    # Calculate bot rate (authors that match at least one rule)
    confirmed_bots = len(results['rule_a_match']) + len(results['rule_b_match']) + len(results['rule_c_match'])
    bot_rate = (confirmed_bots / total_authors * 100) if total_authors > 0 else 0
    false_positives = len(results['no_match'])
    fp_rate = (false_positives / total_authors * 100) if total_authors > 0 else 0
    
    # Calculate confidence interval (95% CI using Wilson score interval)
    if total_authors > 0 and args.sample_authors > 0:
        import math
        z = 1.96  # 95% confidence
        p = confirmed_bots / total_authors
        n = total_authors
        denominator = 1 + (z**2 / n)
        center = (p + (z**2 / (2*n))) / denominator
        margin = z * math.sqrt((p*(1-p) + (z**2)/(4*n)) / n) / denominator
        ci_lower = max(0, center - margin) * 100
        ci_upper = min(100, center + margin) * 100
    else:
        ci_lower = 0
        ci_upper = 0
    
    # Sort Rule C matches by frequency (descending)
    results['rule_c_match'].sort(key=lambda x: x['matches'], reverse=True)
    
    # Write report
    logger.info("\nWriting report to {}...".format(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BOT FALSE POSITIVE DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        if args.sample_authors > 0:
            f.write(f"SAMPLING MODE: Analyzed {total_authors} randomly sampled authors\n")
            f.write(f"Sample size: {args.sample_authors} authors\n")
            f.write(f"Bot rate (confirmed bots): {bot_rate:.2f}% ({confirmed_bots}/{total_authors})\n")
            f.write(f"False positive rate: {fp_rate:.2f}% ({false_positives}/{total_authors})\n")
            if ci_lower > 0 or ci_upper > 0:
                f.write(f"95% confidence interval for bot rate: [{ci_lower:.2f}%, {ci_upper:.2f}%]\n")
            f.write("\n")
        
        f.write(f"Total authors analyzed: {total_authors}\n")
        f.write(f"Rule A matches (username patterns): {len(results['rule_a_match'])}\n")
        f.write(f"Rule B matches (BotRank): {len(results['rule_b_match'])}\n")
        f.write(f"Rule C matches (content phrases): {len(results['rule_c_match'])}\n")
        f.write(f"No rule matches (FALSE POSITIVES): {len(results['no_match'])}\n")
        f.write(f"JSON parse errors: {parse_errors}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("FALSE POSITIVES (No rules match in downloaded content)\n")
        f.write("=" * 80 + "\n")
        f.write("These authors were flagged as bots in Pass 1 but don't match any bot\n")
        f.write("detection rules in their actual downloaded content.\n\n")
        
        for item in results['no_match']:
            f.write(f"Author: {item['author']}\n")
            f.write(f"  Total records: {item['total']}\n")
            f.write(f"  Rule A: NO\n")
            f.write(f"  Rule B: NO\n")
            f.write(f"  Rule C: NO (0 matches)\n")
            if item['samples']:
                f.write(f"  Sample content:\n")
                for sample in item['samples']:
                    f.write(f"    - {sample}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("RULE C MATCHES (Content-based bot detection)\n")
        f.write("=" * 80 + "\n")
        f.write("These authors don't match Rules A or B but have bot-like phrases in\n")
        f.write("their content. Review samples to determine if they are legitimate bots.\n\n")
        
        for item in results['rule_c_match']:
            f.write(f"Author: {item['author']}\n")
            f.write(f"  Total records: {item['total']}\n")
            f.write(f"  Rule C matches: {item['matches']}\n")
            f.write(f"  Sample matches:\n")
            for sample in item['samples']:
                f.write(f"    - {sample}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("SKIPPED (Legitimate bots by Rules A or B)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Rule A (username patterns): {len(results['rule_a_match'])} authors\n")
        f.write(f"Rule B (BotRank): {len(results['rule_b_match'])} authors\n")
        f.write("\n")
        f.write("These authors were not scanned for Rule C because they already match\n")
        f.write("stronger bot detection rules (username patterns or known bot lists).\n")
    
    logger.info("\nDone! Report written to {}".format(output_path))
    logger.info("\nSummary:")
    logger.info("  Total authors: {}".format(total_authors))
    logger.info("  Rule A matches (skipped): {}".format(len(results['rule_a_match'])))
    logger.info("  Rule B matches (skipped): {}".format(len(results['rule_b_match'])))
    logger.info("  Rule C matches (review needed): {}".format(len(results['rule_c_match'])))
    logger.info("  False positives (no matches): {}".format(len(results['no_match'])))


if __name__ == "__main__":
    main()
