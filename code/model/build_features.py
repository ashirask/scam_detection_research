#!/usr/bin/env python3
"""
build_features.py

Computes per-author features (X) and assigns labels (y) for all authors in both the bot
and human populations. Outputs a single dataset.parquet file — one row per author, all
features plus label — ready for model training.

Labels are assigned based on which input files an author came from:
- Authors from bot input files → y = 1
- Authors from human input files → y = 0

Usage (without temporal features):
  python build_features.py \
    --comments-bot      user_comments_bots.jsonl \
    --comments-human    user_comments_humans.jsonl \
    --submissions-bot   user_submissions_bots.jsonl \
    --submissions-human user_submissions_humans.jsonl \
    --domain-whitelist  domain_whitelist.txt \
    --output            dataset.parquet

Usage (with temporal features):
  python build_features.py \
    --comments-bot      user_comments_bots.jsonl \
    --comments-human    user_comments_humans.jsonl \
    --submissions-bot   user_submissions_bots.jsonl \
    --submissions-human user_submissions_humans.jsonl \
    --domain-whitelist  domain_whitelist.txt \
    --temporal \
    --output            dataset.parquet
"""

import re
import math
import argparse
from collections import Counter
from urllib.parse import urlparse
import statistics
import orjson
import pandas as pd


# Regex pattern to match HTTP/HTTPS URLs in text
URL_PATTERN = re.compile(r'https?://[^\s\)\]\"\']+')


def stream_jsonl(path):
    """
    Generator to read JSONL file line by line.
    
    Args:
        path: Path to JSONL file
    
    Yields:
        Parsed JSON objects (dicts) from each line
    """
    with open(path, "rb") as f:
        for line in f:
            yield orjson.loads(line)


def extract_domain(url):
    """
    Extract domain from a URL, stripping 'www.' prefix.
    
    Args:
        url: URL string to parse
    
    Returns:
        Domain name in lowercase without 'www.' prefix, or None if parsing fails
    """
    try:
        domain = urlparse(url).netloc.lower()
        return domain.lstrip("www.")
    except Exception:
        return None


def load_domain_whitelist(path):
    """
    Load domain whitelist from text file into a set for fast lookup.
    
    Args:
        path: Path to domain whitelist text file (one domain per line)
    
    Returns:
        Set of domain strings
    """
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def build_comment_lookup(comments_bot_path, comments_human_path):
    """
    Build a lookup dictionary mapping comment IDs to their creation timestamps.
    This is required for temporal features to compute reply times.
    
    Args:
        comments_bot_path: Path to bot comments JSONL
        comments_human_path: Path to human comments JSONL
    
    Returns:
        Dictionary {comment_id: created_utc} for all comments in both files
    """
    lookup = {}
    
    # Process bot comments
    for record in stream_jsonl(comments_bot_path):
        for comment in record.get("comments", []):
            cid = comment.get("id")
            ts = comment.get("created_utc")
            if cid and ts:
                lookup[cid] = ts
    
    # Process human comments
    for record in stream_jsonl(comments_human_path):
        for comment in record.get("comments", []):
            cid = comment.get("id")
            ts = comment.get("created_utc")
            if cid and ts:
                lookup[cid] = ts
    
    return lookup


# ============================================================================
# Feature Group 1: Activity / Volume
# ============================================================================

def activity_features(comments, submissions):
    """
    Compute activity-based features: post counts and temporal rates.
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
    
    Returns:
        Dictionary of activity features
    """
    n_c = len(comments)
    n_s = len(submissions)
    
    # Collect all timestamps from both comments and submissions
    all_ts = [r["created_utc"] for r in comments + submissions if r.get("created_utc")]
    
    # Calculate account age in days (span between first and last post)
    if len(all_ts) >= 2:
        span_days = (max(all_ts) - min(all_ts)) / 86400  # 86400 seconds per day
    else:
        span_days = 0.0
    
    # Avoid division by zero for per-day calculations
    span_safe = max(span_days, 1.0)
    
    return {
        "num_comments": n_c,
        "num_submissions": n_s,
        "total_posts": n_c + n_s,
        "comments_per_day": n_c / span_safe,
        "submissions_per_day": n_s / span_safe,
        "account_age_days": span_days,
    }


# ============================================================================
# Feature Group 2: Temporal / Reply Time (optional)
# ============================================================================

def temporal_features(comments, comment_lookup):
    """
    Compute reply-time features for comment-to-comment replies only.
    Excludes replies to submissions (parent_id starting with "t3_").
    
    Args:
        comments: List of comment dictionaries
        comment_lookup: Dictionary {comment_id: created_utc} for all comments
    
    Returns:
        Dictionary of temporal features (NaN if no reply times available)
    """
    reply_times = []
    
    for c in comments:
        parent_id = c.get("parent_id", "")
        # Only process replies to other comments (parent_id starts with "t1_")
        if not parent_id.startswith("t1_"):
            continue
        # Strip "t1_" prefix to get the parent comment ID
        parent_comment_id = parent_id[3:]
        parent_ts = comment_lookup.get(parent_comment_id)
        if parent_ts is None:
            continue  # Parent comment not in our dataset
        reply_time = c["created_utc"] - parent_ts
        # Guard against timestamp anomalies (negative reply times)
        if reply_time >= 0:
            reply_times.append(reply_time)
    
    if not reply_times:
        return {
            "mean_reply_time_seconds": float("nan"),
            "min_reply_time_seconds": float("nan"),
            "std_reply_time_seconds": float("nan"),
            "reply_time_coverage": 0.0,  # Diagnostic only, not a model feature
        }
    
    # Count how many comments are replies to other comments
    t1_replies = sum(1 for c in comments if c.get("parent_id", "").startswith("t1_"))
    
    return {
        "mean_reply_time_seconds": sum(reply_times) / len(reply_times),
        "min_reply_time_seconds": min(reply_times),
        "std_reply_time_seconds": statistics.pstdev(reply_times) if len(reply_times) > 1 else 0.0,
        "reply_time_coverage": len(reply_times) / max(t1_replies, 1),
    }


# ============================================================================
# Feature Group 3: Engagement / Karma
# ============================================================================

def engagement_features(comments, submissions):
    """
    Compute engagement features based on scores, upvote ratios, and comments received.
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
    
    Returns:
        Dictionary of engagement features (NaN if no data available)
    """
    # Extract scores from comments and submissions
    c_scores = [c["score"] for c in comments if "score" in c]
    s_scores = [s["score"] for s in submissions if "score" in s]
    
    # Extract upvote ratios from submissions (can be None)
    upvote_ratios = [s["upvote_ratio"] for s in submissions if s.get("upvote_ratio") is not None]
    
    # Extract number of comments received on submissions
    comments_received = [s["num_comments"] for s in submissions if "num_comments" in s]
    
    # Calculate mean scores
    mean_c_score = sum(c_scores) / len(c_scores) if c_scores else float("nan")
    mean_s_score = sum(s_scores) / len(s_scores) if s_scores else float("nan")
    
    # Calculate ratio of comment score to submission score
    # Add 1 to denominator to avoid division by zero
    if mean_c_score is not float("nan") and mean_s_score is not float("nan"):
        ratio = mean_c_score / (abs(mean_s_score) + 1)
    else:
        ratio = float("nan")
    
    return {
        "mean_comment_score": mean_c_score,
        "mean_submission_score": mean_s_score,
        "comment_to_post_score_ratio": ratio,
        "mean_upvote_ratio": sum(upvote_ratios) / len(upvote_ratios) if upvote_ratios else float("nan"),
        "mean_comments_received_per_submission": sum(comments_received) / len(comments_received) if comments_received else float("nan"),
    }


# ============================================================================
# Feature Group 4: Moderation Signals
# ============================================================================

def moderation_features(comments, submissions):
    """
    Compute moderation-related features: bans, removals, and controversiality.
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
    
    Returns:
        Dictionary of moderation features
    """
    all_records = comments + submissions
    total = max(len(all_records), 1)
    
    # Count posts where the author was banned at the time of posting
    banned_count = sum(1 for r in all_records if r.get("banned_at_utc") is not None)
    
    # Count submissions removed by moderators
    mod_removed = sum(1 for s in submissions if s.get("removed_by_category") == "moderator")
    
    # Extract controversiality values from comments (0 or 1)
    controversiality_vals = [c.get("controversiality", 0) for c in comments]
    
    return {
        "banned_post_ratio": banned_count / total,
        "submission_mod_removed_ratio": mod_removed / max(len(submissions), 1),
        "controversiality_mean": sum(controversiality_vals) / len(controversiality_vals) if controversiality_vals else float("nan"),
    }


# ============================================================================
# Feature Group 5: Subreddit Activity
# ============================================================================

def subreddit_features(comments, submissions):
    """
    Compute subreddit diversity and concentration features.
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
    
    Returns:
        Dictionary of subreddit features (NaN if no subreddit data)
    """
    # Collect all subreddit names from comments and submissions
    all_subreddits = (
        [c["subreddit"] for c in comments if c.get("subreddit")] +
        [s["subreddit"] for s in submissions if s.get("subreddit")]
    )
    
    if not all_subreddits:
        return {
            "num_unique_subreddits": float("nan"),
            "top_subreddit_concentration": float("nan"),
            "subreddit_entropy": float("nan"),
            "nsfw_subreddit_ratio": float("nan"),
        }
    
    # Count occurrences of each subreddit
    counts = Counter(all_subreddits)
    total = sum(counts.values())
    
    # Calculate entropy of subreddit distribution (measure of diversity)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )
    
    # Fraction of posts in the single most-used subreddit
    top_concentration = counts.most_common(1)[0][1] / total
    
    # Fraction of submissions marked as NSFW (over_18 field)
    nsfw_count = sum(1 for s in submissions if s.get("over_18") is True)
    nsfw_ratio = nsfw_count / max(len(submissions), 1)
    
    return {
        "num_unique_subreddits": len(counts),
        "top_subreddit_concentration": top_concentration,
        "subreddit_entropy": entropy,
        "nsfw_subreddit_ratio": nsfw_ratio,
    }


# ============================================================================
# Feature Group 6: Text Stylometrics
# ============================================================================

def get_texts(comments, submissions):
    """
    Extract valid text content from comments and submissions.
    Skips [deleted] and [removed] placeholders.
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
    
    Returns:
        List of text strings
    """
    texts = []
    
    # Extract text from comment bodies
    for c in comments:
        body = c.get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            texts.append(body)
    
    # Extract text from submission titles and selftext
    for s in submissions:
        title = s.get("title", "")
        selftext = s.get("selftext", "")
        # Skip placeholder text
        if selftext in ("[deleted]", "[removed]", ""):
            selftext = ""
        combined = (title + " " + selftext).strip()
        if combined:
            texts.append(combined)
    
    return texts


def stylometric_features(comments, submissions):
    """
    Compute text stylometric features: length, vocabulary diversity, repetition,
    capitalization, special characters, and whitespace patterns.
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
    
    Returns:
        Dictionary of stylometric features (NaN if no text available)
    """
    texts = get_texts(comments, submissions)
    
    if not texts:
        return {k: float("nan") for k in [
            "mean_text_length", "type_token_ratio", "repetition_ratio",
            "uppercase_ratio", "url_density", "caret_count_mean",
            "asterisk_count_mean", "max_consecutive_carets",
            "max_consecutive_asterisks", "whitespace_entropy"
        ]}
    
    # Mean text length per post
    mean_len = sum(len(t) for t in texts) / len(texts)
    
    # Type-token ratio: unique words / total words (vocabulary diversity)
    all_text = " ".join(texts).lower()
    tokens = re.findall(r'\b[a-z]+\b', all_text)
    ttr = len(set(tokens)) / max(len(tokens), 1)
    
    # Repetition ratio: fraction of posts that are exact duplicates
    normalized = [t.strip().lower() for t in texts]
    counts = Counter(normalized)
    dup_count = sum(c - 1 for c in counts.values() if c > 1)
    rep_ratio = dup_count / max(len(texts), 1)
    
    # Uppercase ratio: fraction of alphabetic characters that are uppercase
    alpha_chars = [ch for ch in all_text if ch.isalpha()]
    upper_count = sum(1 for ch in "".join(texts) if ch.isupper())
    upper_ratio = upper_count / max(len(alpha_chars), 1)
    
    # URL density: average number of URLs per post
    total_urls = sum(len(URL_PATTERN.findall(t)) for t in texts)
    url_density = total_urls / len(texts)
    
    # Caret (^) and asterisk (*) features
    caret_counts = [t.count("^") for t in texts]
    asterisk_counts = [t.count("*") for t in texts]
    
    def max_consecutive(text, char):
        """
        Find the maximum number of consecutive occurrences of a character in text.
        
        Args:
            text: String to search
            char: Character to count
        
        Returns:
            Maximum consecutive run length
        """
        max_run = 0
        current = 0
        for ch in text:
            if ch == char:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run
    
    max_carets = max((max_consecutive(t, "^") for t in texts), default=0)
    max_asterisks = max((max_consecutive(t, "*") for t in texts), default=0)
    
    # Whitespace entropy: entropy of whitespace token distribution per post
    def ws_entropy(text):
        """
        Calculate entropy of whitespace character distribution in a text.
        
        Args:
            text: String to analyze
        
        Returns:
            Entropy value (0.0 if no whitespace)
        """
        tokens = re.findall(r'\s+', text)
        if not tokens:
            return 0.0
        c = Counter(tokens)
        total = sum(c.values())
        return -sum((v / total) * math.log2(v / total) for v in c.values())
    
    mean_ws_entropy = sum(ws_entropy(t) for t in texts) / len(texts)
    
    return {
        "mean_text_length": mean_len,
        "type_token_ratio": ttr,
        "repetition_ratio": rep_ratio,
        "uppercase_ratio": upper_ratio,
        "url_density": url_density,
        "caret_count_mean": sum(caret_counts) / len(caret_counts),
        "asterisk_count_mean": sum(asterisk_counts) / len(asterisk_counts),
        "max_consecutive_carets": max_carets,
        "max_consecutive_asterisks": max_asterisks,
        "whitespace_entropy": mean_ws_entropy,
    }


# ============================================================================
# Feature Group 7: URL Features
# ============================================================================

def url_features(comments, submissions, domain_whitelist):
    """
    Compute URL-related features: suspicious domain ratio and domain entropy.
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
        domain_whitelist: Set of trusted domain names
    
    Returns:
        Dictionary of URL features (NaN if no URLs found)
    """
    all_urls = []
    
    # Extract URLs from comment bodies
    for c in comments:
        all_urls.extend(URL_PATTERN.findall(c.get("body", "") or ""))
    
    # Extract URLs from submission selftext and the submission URL field
    for s in submissions:
        all_urls.extend(URL_PATTERN.findall(s.get("selftext", "") or ""))
        link_url = s.get("url", "")
        if link_url and link_url.startswith("http"):
            all_urls.append(link_url)
    
    if not all_urls:
        return {
            "suspicious_url_ratio": float("nan"),
            "url_domain_entropy": float("nan"),
        }
    
    # Extract domains from all URLs
    domains = [extract_domain(u) for u in all_urls]
    domains = [d for d in domains if d]  # Filter out None values
    
    # Count domains not in whitelist (suspicious)
    suspicious = sum(1 for d in domains if d not in domain_whitelist)
    susp_ratio = suspicious / len(domains)
    
    # Calculate domain entropy (diversity of domains used)
    domain_counts = Counter(domains)
    total = sum(domain_counts.values())
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in domain_counts.values()
    )
    
    return {
        "suspicious_url_ratio": susp_ratio,
        "url_domain_entropy": entropy,
    }


# ============================================================================
# Feature Group 8: Username Features
# ============================================================================

def username_features(author):
    """
    Compute username-based features: length, digit ratio, entropy, and patterns.
    
    Args:
        author: Username string
    
    Returns:
        Dictionary of username features
    """
    length = len(author)
    digits = sum(1 for ch in author if ch.isdigit())
    alpha = sum(1 for ch in author if ch.isalpha())
    
    # Character entropy: measure of character diversity in username
    counts = Counter(author.lower())
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    
    return {
        "username_length": length,
        "username_digit_ratio": digits / max(length, 1),
        "username_entropy": entropy,
        "username_has_digits_at_end": int(bool(author and author[-1].isdigit())),
        "username_has_underscore": int("_" in author),
    }


# ============================================================================
# Main Processing Loop
# ============================================================================

def process_population(comments_path, submissions_path, label,
                        domain_whitelist, comment_lookup, args):
    """
    Process all authors in a single population (bots or humans).
    
    Args:
        comments_path: Path to comments JSONL for this population
        submissions_path: Path to submissions JSONL for this population
        label: Label to assign (1 for bots, 0 for humans)
        domain_whitelist: Set of trusted domain names
        comment_lookup: Dictionary for temporal features (or None if not using temporal)
        args: Command-line arguments
    
    Returns:
        List of feature dictionaries, one per author
    """
    results = []
    
    # Load submissions into a dictionary keyed by author for O(1) lookup
    # This allows us to efficiently match submissions with comments by author
    sub_by_author = {}
    for record in stream_jsonl(submissions_path):
        sub_by_author[record["author"]] = record.get("submissions", [])
    
    # Stream through comments file, processing one author at a time
    for record in stream_jsonl(comments_path):
        author = record["author"]
        comments = record.get("comments", [])
        submissions = sub_by_author.get(author, [])
        
        # Start with author identifier and label
        feats = {"author": author, "y": label}
        
        # Compute all feature groups
        feats.update(activity_features(comments, submissions))
        feats.update(engagement_features(comments, submissions))
        feats.update(moderation_features(comments, submissions))
        feats.update(subreddit_features(comments, submissions))
        feats.update(stylometric_features(comments, submissions))
        feats.update(url_features(comments, submissions, domain_whitelist))
        feats.update(username_features(author))
        
        # Add temporal features if flag is set
        if args.temporal and comment_lookup is not None:
            feats.update(temporal_features(comments, comment_lookup))
        
        results.append(feats)
    
    return results


def print_summary(df, args):
    """
    Print a summary of the feature building process to stdout.
    
    Args:
        df: DataFrame with all features
        args: Command-line arguments
    """
    print("\n=== build_features Summary ===")
    
    # Author counts
    bot_count = (df["y"] == 1).sum()
    human_count = (df["y"] == 0).sum()
    total = len(df)
    
    print(f"\nAuthors processed")
    print(f"  Bot authors    : {bot_count:,}")
    print(f"  Human authors  : {human_count:,}")
    print(f"  Total          : {total:,}")
    
    # Feature count
    feature_cols = [c for c in df.columns if c not in ["author", "y"]]
    print(f"\nFeatures computed : {len(feature_cols)}")
    print(f"  Temporal        : {'yes' if args.temporal else 'no'} (--temporal flag)")
    
    # NaN rates (top 5 by frequency)
    nan_rates = (df[feature_cols].isna().sum() / len(df)).sort_values(ascending=False)
    print(f"  NaN rates (top 5 by frequency):")
    for col, rate in nan_rates.head(5).items():
        print(f"    {col:35s} : {rate*100:5.1f}%")
    
    # Class balance
    print(f"\nClass balance")
    print(f"  y=1 (bot)   : {bot_count:,}  ({bot_count/total*100:.1f}%)")
    print(f"  y=0 (human) : {human_count:,}  ({human_count/total*100:.1f}%)")
    
    print(f"\nOutput : {args.output}  ({len(df)} rows × {len(df.columns)} columns)")


def main():
    parser = argparse.ArgumentParser(
        description="Build features for bot detection model training"
    )
    parser.add_argument("--comments-bot", required=True, help="Path to bot comments JSONL")
    parser.add_argument("--comments-human", required=True, help="Path to human comments JSONL")
    parser.add_argument("--submissions-bot", required=True, help="Path to bot submissions JSONL")
    parser.add_argument("--submissions-human", required=True, help="Path to human submissions JSONL")
    parser.add_argument("--domain-whitelist", required=True, help="Path to domain whitelist text file")
    parser.add_argument("--temporal", action="store_true", help="Include temporal reply-time features")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    
    args = parser.parse_args()
    
    # Step 1: Load domain whitelist
    print(f"Loading domain whitelist from {args.domain_whitelist}")
    domain_whitelist = load_domain_whitelist(args.domain_whitelist)
    print(f"  Loaded {len(domain_whitelist)} domains")
    
    # Step 2: Build comment lookup if temporal features are enabled
    comment_lookup = None
    if args.temporal:
        print("Building comment lookup for temporal features...")
        comment_lookup = build_comment_lookup(args.comments_bot, args.comments_human)
        print(f"  Built lookup with {len(comment_lookup)} comments")
    
    # Step 3: Process bot population (label = 1)
    print(f"\nProcessing bot population...")
    bot_results = process_population(
        args.comments_bot,
        args.submissions_bot,
        label=1,
        domain_whitelist=domain_whitelist,
        comment_lookup=comment_lookup,
        args=args
    )
    print(f"  Processed {len(bot_results)} bot authors")
    
    # Step 4: Process human population (label = 0)
    print(f"\nProcessing human population...")
    human_results = process_population(
        args.comments_human,
        args.submissions_human,
        label=0,
        domain_whitelist=domain_whitelist,
        comment_lookup=comment_lookup,
        args=args
    )
    print(f"  Processed {len(human_results)} human authors")
    
    # Step 5: Combine results into DataFrame
    print(f"\nCombining results...")
    all_results = bot_results + human_results
    df = pd.DataFrame(all_results)
    
    # Step 6: Write to parquet
    print(f"Writing to {args.output}")
    df.to_parquet(args.output, index=False)
    
    # Step 7: Print summary
    print_summary(df, args)


if __name__ == "__main__":
    main()
