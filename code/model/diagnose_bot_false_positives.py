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
import gzip
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


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


def load_jsonl_gz(path: Path):
    """
    Load records from a gzipped JSONL file.
    
    Args:
        path: Path to the .jsonl.gz file
    
    Returns:
        list: List of parsed JSON objects
    """
    if not path.exists():
        return []
    
    records = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


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


def scan_author_content(author_data: dict, label: str, sample_size: int, content_length: int):
    """
    Scan an author's content for Rule C matches and collect samples.
    
    Args:
        author_data: Dictionary with 'comments' and 'submissions' keys
        label: Either 'bot' or 'human'
        sample_size: Number of samples to collect
        content_length: Maximum length of each sample in characters
    
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
        if len(general_samples) < sample_size:
            general_samples.append(f"[SUBMISSION] {text[:content_length]}")
    
    return {
        'rule_c_matches': rule_c_matches,
        'total_records': total_records,
        'rule_c_samples': rule_c_samples,
        'general_samples': general_samples
    }


def main():
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
    
    args = parser.parse_args()
    
    merged_dir = Path(args.merged_dir)
    output_path = Path(args.output)
    
    # Load BotRank for Rule B
    print(f"Loading BotRank from {args.botrank}...")
    botrank_df = pd.read_csv(args.botrank)
    botrank_set = set(botrank_df["bot_name"].str.lower())
    print(f"  Loaded {len(botrank_set)} unique bot names")
    
    # Load merged bot data
    comments_bot_path = merged_dir / "user_comments_bot.jsonl"
    submissions_bot_path = merged_dir / "user_submissions_bot.jsonl"
    
    print(f"Loading merged bot data from {merged_dir}...")
    
    # Load comments
    comments_by_author = {}
    if comments_bot_path.exists():
        with open(comments_bot_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    author = obj.get("author")
                    if author:
                        comments_by_author[author] = obj
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {len(comments_by_author)} authors from comments")
    
    # Load submissions and merge
    submissions_by_author = {}
    if submissions_bot_path.exists():
        with open(submissions_bot_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    author = obj.get("author")
                    if author:
                        submissions_by_author[author] = obj
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {len(submissions_by_author)} authors from submissions")
    
    # Combine data
    all_authors = set(comments_by_author.keys()) | set(submissions_by_author.keys())
    print(f"  Total unique bot authors: {len(all_authors)}")
    
    # Diagnose each author
    results = {
        'rule_a_match': [],      # Skipped - legitimate by username
        'rule_b_match': [],      # Skipped - legitimate by BotRank
        'rule_c_match': [],      # Has Rule C matches in content
        'no_match': []           # False positive - no rules match
    }
    
    print("\nDiagnosing authors...")
    for i, author in enumerate(all_authors, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(all_authors)} authors...")
        
        # Skip deleted/removed authors
        if author in SKIP_AUTHORS:
            continue
        
        # Build author data dict
        author_data = {}
        if author in comments_by_author:
            author_data['comments'] = comments_by_author[author].get('comments', [])
        if author in submissions_by_author:
            author_data['submissions'] = submissions_by_author[author].get('submissions', [])
        
        # Rule A: Username pattern matching
        rule_hit, matched_pattern = rule_a_match(author)
        if rule_hit:
            results['rule_a_match'].append(author)
            continue
        
        # Rule B: BotRank lookup
        botrank_hit = author.lower() in botrank_set
        if botrank_hit:
            results['rule_b_match'].append(author)
            continue
        
        # Rules A and B both failed - need to check Rule C in content
        content_scan = scan_author_content(author_data, 'bot', args.sample_size, args.content_length)
        
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
    
    # Sort Rule C matches by frequency (descending)
    results['rule_c_match'].sort(key=lambda x: x['matches'], reverse=True)
    
    # Write report
    print(f"\nWriting report to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BOT FALSE POSITIVE DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total authors analyzed: {len(all_authors)}\n")
        f.write(f"Rule A matches (username patterns): {len(results['rule_a_match'])}\n")
        f.write(f"Rule B matches (BotRank): {len(results['rule_b_match'])}\n")
        f.write(f"Rule C matches (content phrases): {len(results['rule_c_match'])}\n")
        f.write(f"No rule matches (FALSE POSITIVES): {len(results['no_match'])}\n")
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
    
    print(f"\nDone! Report written to {output_path}")
    print(f"\nSummary:")
    print(f"  Total authors: {len(all_authors)}")
    print(f"  Rule A matches (skipped): {len(results['rule_a_match'])}")
    print(f"  Rule B matches (skipped): {len(results['rule_b_match'])}")
    print(f"  Rule C matches (review needed): {len(results['rule_c_match'])}")
    print(f"  False positives (no matches): {len(results['no_match'])}")


if __name__ == "__main__":
    main()
