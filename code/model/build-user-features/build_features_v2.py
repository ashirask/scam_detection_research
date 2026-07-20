#!/usr/bin/env python3
"""
build_features_v2.py

Memory-efficient version of build_features.py with enhanced temporal features.
Designed for large-scale datasets (100GB+ input files, 21M+ parent comments).

Key improvements:
- SQLite-based comment lookup (handles 21M+ IDs without memory issues)
- Enhanced temporal features: burstiness, predictability, reply-time predictability
- Progress tracking and resumability
- Memory-efficient streaming with chunking

Usage:
  python build_features_v2.py \
    --comments-bot      user_comments_bots.jsonl \
    --comments-human    user_comments_humans.jsonl \
    --submissions-bot   user_submissions_bots.jsonl \
    --submissions-human user_submissions_humans.jsonl \
    --domain-whitelist  domain_whitelist.txt \
    --parent-comments   parent_comments.jsonl.gz \
    --temporal \
    --output            dataset.parquet
"""

import re
import math
import argparse
import sqlite3
import tempfile
import logging
import sys
import warnings
from collections import Counter
from urllib.parse import urlparse
import statistics
from pathlib import Path
import gzip
import orjson
import pandas as pd
from scipy import stats

# Regex pattern to match HTTP/HTTPS URLs in text
URL_PATTERN = re.compile(r'https?://[^\s\)\]\"\']+')


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging to output to both stdout and stderr for SLURM compatibility."""
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


def stream_jsonl(path):
    """
    Generator to read JSONL file line by line.
    Handles both regular .jsonl and .jsonl.gz files.
    
    Args:
        path: Path to JSONL file (can be .jsonl or .jsonl.gz)
    
    Yields:
        Parsed JSON objects (dicts) from each line
    """
    path = Path(path)
    
    if path.suffix == '.gz':
        # Handle gzipped JSONL
        open_func = gzip.open
        mode = 'rt'
    else:
        # Handle regular JSONL
        open_func = open
        mode = 'rb'
    
    with open_func(path, mode) as f:
        for line in f:
            if isinstance(line, bytes):
                yield orjson.loads(line)
            else:
                yield orjson.loads(line.encode('utf-8'))


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


def build_comment_lookup_sqlite(comments_bot_path, comments_human_path, 
                                parent_comments_path=None, db_path=None):
    """
    Build a SQLite database for comment ID to timestamp lookup.
    Memory-efficient alternative to in-memory dict for large datasets.
    
    Args:
        comments_bot_path: Path to bot comments JSONL
        comments_human_path: Path to human comments JSONL
        parent_comments_path: Optional path to parent comments JSONL from ZST extraction
        db_path: Optional path for SQLite database (default: temp file)
    
    Returns:
        Path to SQLite database file
    """
    if db_path is None:
        # Create temporary database
        db_path = tempfile.mktemp(suffix='.db')
    
    logger = logging.getLogger(__name__)
    
    # Check if database already exists and skip rebuilding
    if Path(db_path).exists():
        logger.info(f"Using existing SQLite comment lookup at {db_path}")
        # Verify database has data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM comments')
        count = cursor.fetchone()[0]
        conn.close()
        logger.info(f"  Existing database has {count:,} comments")
        return db_path
    
    logger.info(f"Building SQLite comment lookup at {db_path}")
    
    # Create SQLite database with optimized schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table with appropriate indexes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            created_utc REAL
        )
    ''')
    
    # Use WAL mode for better concurrent performance
    cursor.execute('PRAGMA journal_mode=WAL')
    cursor.execute('PRAGMA synchronous=NORMAL')
    
    # Function to insert comments from a file
    def insert_comments_from_file(path, source_name):
        count = 0
        batch = []
        batch_size = 10000
        
        for record in stream_jsonl(path):
            # Handle different file formats
            if "comments" in record:
                # Format: {"author": "...", "comments": [...]}
                comments = record.get("comments", [])
            else:
                # Format: single comment records
                comments = [record]
            
            for comment in comments:
                cid = comment.get("id")
                ts = comment.get("created_utc")
                if cid and ts is not None:
                    batch.append((str(cid), float(ts)))
                    count += 1
                    
                    if len(batch) >= batch_size:
                        cursor.executemany(
                            'INSERT OR REPLACE INTO comments (id, created_utc) VALUES (?, ?)',
                            batch
                        )
                        conn.commit()
                        batch = []
        
        # Insert remaining batch
        if batch:
            cursor.executemany(
                'INSERT OR REPLACE INTO comments (id, created_utc) VALUES (?, ?)',
                batch
            )
            conn.commit()
        
        logger.info(f"  Inserted {count:,} comments from {source_name}")
        return count
    
    # Insert comments from all sources
    total = 0
    total += insert_comments_from_file(comments_bot_path, "bot comments")
    total += insert_comments_from_file(comments_human_path, "human comments")
    
    if parent_comments_path:
        total += insert_comments_from_file(parent_comments_path, "parent comments")
    
    # Create index for faster lookups
    logger.info("  Creating index...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_utc ON comments(created_utc)')
    conn.commit()
    
    # Optimize database
    cursor.execute('PRAGMA optimize')
    conn.close()
    
    logger.info(f"  Total comments in lookup: {total:,}")
    logger.info(f"  Database size: {Path(db_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    return db_path


class SQLiteConnectionPool:
    """
    Manages a persistent SQLite connection for efficient temporal feature queries.
    
    Instead of opening/closing a connection for each query (which is very slow),
    we keep one connection open and reuse it for all queries.
    """
    
    def __init__(self, db_path):
        """
        Initialize the connection pool with a persistent SQLite connection.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def get_timestamp(self, comment_id):
        """
        Get timestamp for a comment ID using the persistent connection.
        
        Args:
            comment_id: Comment ID to look up
        
        Returns:
            Timestamp as float, or None if not found
        """
        self.cursor.execute('SELECT created_utc FROM comments WHERE id = ?', (str(comment_id),))
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def get_timestamps_batch(self, comment_ids):
        """
        Get timestamps for multiple comment IDs using chunked batch queries.
        
        SQLite has a limit on the number of parameters per query (typically 999).
        This method chunks large requests into smaller batches to avoid this limit.
        
        Args:
            comment_ids: List of comment IDs to look up
        
        Returns:
            Dictionary mapping comment_id -> timestamp
        """
        if not comment_ids:
            return {}
        
        # SQLite parameter limit is typically 999, use 500 for safety
        chunk_size = 500
        all_timestamps = {}
        
        for i in range(0, len(comment_ids), chunk_size):
            chunk = comment_ids[i:i + chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            query = f'SELECT id, created_utc FROM comments WHERE id IN ({placeholders})'
            
            self.cursor.execute(query, chunk)
            chunk_timestamps = {row[0]: row[1] for row in self.cursor.fetchall()}
            all_timestamps.update(chunk_timestamps)
        
        return all_timestamps
    
    def close(self):
        """Close the persistent connection."""
        self.conn.close()


def get_comment_timestamp_sqlite(db_path, comment_id):
    """
    Get timestamp for a comment ID from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        comment_id: Comment ID to look up
    
    Returns:
        Timestamp as float, or None if not found
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT created_utc FROM comments WHERE id = ?', (str(comment_id),))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


# ============================================================================
# Feature Group 1: Activity / Volume (Enhanced with per-day rates)
# ============================================================================

def activity_features(comments, submissions):
    """
    Compute activity-based features with per-day rates as primary features.
    
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
# Feature Group 2: Temporal / Burstiness / Predictability
# ============================================================================

def compute_inter_event_times(timestamps):
    """
    Compute inter-event times from sorted timestamps.
    
    Args:
        timestamps: List of timestamps (created_utc)
    
    Returns:
        List of inter-event times in seconds
    """
    if len(timestamps) < 2:
        return []
    
    sorted_ts = sorted(timestamps)
    return [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]


def burstiness_features(comments, submissions):
    """
    Compute burstiness features: inter-event time statistics for comments and posts.
    Measures how quickly users post - bots tend to have regular, short intervals.
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
    
    Returns:
        Dictionary of burstiness features
    """
    # Extract timestamps
    comment_ts = [c["created_utc"] for c in comments if c.get("created_utc")]
    submission_ts = [s["created_utc"] for s in submissions if s.get("created_utc")]
    
    # Compute inter-event times
    comment_intervals = compute_inter_event_times(comment_ts)
    submission_intervals = compute_inter_event_times(submission_ts)
    
    # Comment burstiness
    if comment_intervals:
        comment_mean_interval = statistics.mean(comment_intervals)
        comment_min_interval = min(comment_intervals)
        comment_std_interval = statistics.pstdev(comment_intervals) if len(comment_intervals) > 1 else 0.0
    else:
        comment_mean_interval = float("nan")
        comment_min_interval = float("nan")
        comment_std_interval = float("nan")
    
    # Submission burstiness
    if submission_intervals:
        submission_mean_interval = statistics.mean(submission_intervals)
        submission_min_interval = min(submission_intervals)
        submission_std_interval = statistics.pstdev(submission_intervals) if len(submission_intervals) > 1 else 0.0
    else:
        submission_mean_interval = float("nan")
        submission_min_interval = float("nan")
        submission_std_interval = float("nan")
    
    # Combined post burstiness (comments + submissions)
    all_ts = comment_ts + submission_ts
    all_intervals = compute_inter_event_times(all_ts)
    
    if all_intervals:
        all_mean_interval = statistics.mean(all_intervals)
        all_min_interval = min(all_intervals)
        all_std_interval = statistics.pstdev(all_intervals) if len(all_intervals) > 1 else 0.0
    else:
        all_mean_interval = float("nan")
        all_min_interval = float("nan")
        all_std_interval = float("nan")
    
    return {
        "comment_mean_interval_seconds": comment_mean_interval,
        "comment_min_interval_seconds": comment_min_interval,
        "comment_std_interval_seconds": comment_std_interval,
        "submission_mean_interval_seconds": submission_mean_interval,
        "submission_min_interval_seconds": submission_min_interval,
        "submission_std_interval_seconds": submission_std_interval,
        "all_posts_mean_interval_seconds": all_mean_interval,
        "all_posts_min_interval_seconds": all_min_interval,
        "all_posts_std_interval_seconds": all_std_interval,
    }


def predictability_features(comments, submissions):
    """
    Compute predictability features using differential entropy of inter-event times.
    Bots: Gaussian (regular) or exponential (Poisson) distributions
    Humans: Heavy-tailed / power-law distributions
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
    
    Returns:
        Dictionary of predictability features
    """
    # Extract timestamps
    comment_ts = [c["created_utc"] for c in comments if c.get("created_utc")]
    submission_ts = [s["created_utc"] for s in submissions if s.get("created_utc")]
    
    # Compute inter-event times
    comment_intervals = compute_inter_event_times(comment_ts)
    submission_intervals = compute_inter_event_times(submission_ts)
    all_ts = comment_ts + submission_ts
    all_intervals = compute_inter_event_times(all_ts)
    
    # Compute differential entropy (measures unpredictability)
    # Higher entropy = more unpredictable = more human-like
    # Lower entropy = more predictable = more bot-like
    
    def safe_differential_entropy(intervals):
        if len(intervals) < 2:
            return float("nan")
        try:
            # Use natural logarithm for differential entropy
            return stats.differential_entropy(intervals)
        except:
            return float("nan")
    
    # Also compute variance as a simpler measure
    def safe_variance(intervals):
        if len(intervals) < 2:
            return float("nan")
        return statistics.variance(intervals)
    
    return {
        "comment_interval_entropy": safe_differential_entropy(comment_intervals),
        "comment_interval_variance": safe_variance(comment_intervals),
        "submission_interval_entropy": safe_differential_entropy(submission_intervals),
        "submission_interval_variance": safe_variance(submission_intervals),
        "all_posts_interval_entropy": safe_differential_entropy(all_intervals),
        "all_posts_interval_variance": safe_variance(all_intervals),
    }


def temporal_features(comments, connection_pool):
    """
    Compute reply-time features with predictability (entropy of reply times).
    Bots: short, regular reply times (low entropy)
    Humans: long, irregular reply times (high entropy)
    
    This version uses batch lookups for much better performance.
    
    Args:
        comments: List of comment dictionaries
        connection_pool: SQLiteConnectionPool object for efficient lookups
    
    Returns:
        Dictionary of temporal features
    """
    # Collect all parent IDs for this author (batch approach)
    parent_ids = []
    comment_times = {}
    
    for c in comments:
        parent_id = c.get("parent_id", "")
        parent_id = str(parent_id) if parent_id is not None else ""
        
        # Only process replies to other comments (parent_id starts with "t1_")
        if not parent_id.startswith("t1_"):
            continue
        
        parent_comment_id = parent_id[3:]
        parent_ids.append(parent_comment_id)
        comment_times[parent_comment_id] = c["created_utc"]
    
    if not parent_ids:
        return {
            "mean_reply_time_seconds": float("nan"),
            "min_reply_time_seconds": float("nan"),
            "std_reply_time_seconds": float("nan"),
            "reply_time_entropy": float("nan"),
            "reply_time_variance": float("nan"),
            "reply_time_coverage": 0.0,
        }
    
    # Single batch query for all parent timestamps (much faster than individual queries)
    parent_timestamps = connection_pool.get_timestamps_batch(parent_ids)
    
    # Compute reply times from batch results
    reply_times = []
    for parent_id, parent_ts in parent_timestamps.items():
        if parent_ts is not None:
            reply_time = comment_times[parent_id] - parent_ts
            if reply_time >= 0:
                reply_times.append(reply_time)
    
    if not reply_times:
        return {
            "mean_reply_time_seconds": float("nan"),
            "min_reply_time_seconds": float("nan"),
            "std_reply_time_seconds": float("nan"),
            "reply_time_entropy": float("nan"),
            "reply_time_variance": float("nan"),
            "reply_time_coverage": 0.0,
        }
    
    # Count how many comments are replies to other comments
    t1_replies = sum(1 for c in comments if str(c.get("parent_id", "")).startswith("t1_"))
    
    # Compute statistics
    mean_reply = statistics.mean(reply_times)
    min_reply = min(reply_times)
    std_reply = statistics.pstdev(reply_times) if len(reply_times) > 1 else 0.0
    
    # Compute differential entropy of reply times
    try:
        reply_entropy = stats.differential_entropy(reply_times)
    except:
        reply_entropy = float("nan")
    
    # Compute variance
    reply_variance = statistics.variance(reply_times) if len(reply_times) > 1 else float("nan")
    
    return {
        "mean_reply_time_seconds": mean_reply,
        "min_reply_time_seconds": min_reply,
        "std_reply_time_seconds": std_reply,
        "reply_time_entropy": reply_entropy,
        "reply_time_variance": reply_variance,
        "reply_time_coverage": len(reply_times) / max(t1_replies, 1),
    }


# ============================================================================
# Feature Group 3: Engagement / Karma (unchanged)
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
# Feature Group 4: Moderation Signals (unchanged)
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
# Feature Group 5: Subreddit Activity (unchanged)
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
# Feature Group 6: Text Stylometrics (unchanged)
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
# Feature Group 7: URL Features (unchanged)
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
# Feature Group 8: Username Features (unchanged)
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
# Main Processing Loop with Progress Tracking
# ============================================================================

def process_population(comments_path, submissions_path, label,
                        domain_whitelist, comment_lookup_db, args,
                        progress_file=None):
    """
    Process all authors in a single population (bots or humans).
    Enhanced with progress tracking for large files.
    
    Args:
        comments_path: Path to comments JSONL for this population
        submissions_path: Path to submissions JSONL for this population
        label: Label to assign (1 for bots, 0 for humans)
        domain_whitelist: Set of trusted domain names
        comment_lookup_db: Path to SQLite database for temporal features (or None)
        args: Command-line arguments
        progress_file: Optional path to track progress
    
    Returns:
        List of feature dictionaries, one per author
    """
    # Create connection pool if temporal features are enabled
    connection_pool = None
    if args.temporal and comment_lookup_db is not None:
        connection_pool = SQLiteConnectionPool(comment_lookup_db)
    results = []
    processed_authors = set()
    
    # Load progress if resuming
    if progress_file and Path(progress_file).exists():
        with open(progress_file, 'r') as f:
            processed_authors = set(line.strip() for line in f if line.strip())
        logger = logging.getLogger(__name__)
        logger.info(f"  Resuming from {len(processed_authors)} already-processed authors")
    
    # Load submissions into a dictionary keyed by author for O(1) lookup
    # For very large files, this could be memory-intensive
    # Consider chunking if memory becomes an issue
    logger = logging.getLogger(__name__)
    logger.info(f"  Loading submissions from {submissions_path}")
    sub_by_author = {}
    for record in stream_jsonl(submissions_path):
        author = record.get("author")
        if author:
            sub_by_author[author] = record.get("submissions", [])
    logger.info(f"  Loaded {len(sub_by_author)} authors with submissions")
    
    # Stream through comments file, processing one author at a time
    author_count = 0
    skipped_count = 0
    
    logger.info(f"  Processing comments from {comments_path}")
    for record in stream_jsonl(comments_path):
        author = record.get("author")
        if not author:
            continue
        
        # Skip if already processed (resumability)
        if author in processed_authors:
            skipped_count += 1
            continue
        
        comments = record.get("comments", [])
        submissions = sub_by_author.get(author, [])
        
        # Start with author identifier and label
        feats = {"author": author, "y": label}
        
        # Compute all feature groups
        feats.update(activity_features(comments, submissions))
        feats.update(burstiness_features(comments, submissions))
        feats.update(predictability_features(comments, submissions))
        feats.update(engagement_features(comments, submissions))
        feats.update(moderation_features(comments, submissions))
        feats.update(subreddit_features(comments, submissions))
        feats.update(stylometric_features(comments, submissions))
        feats.update(url_features(comments, submissions, domain_whitelist))
        feats.update(username_features(author))
        
        # Add temporal features if flag is set
        if args.temporal and connection_pool is not None:
            feats.update(temporal_features(comments, connection_pool))
        
        results.append(feats)
        processed_authors.add(author)
        author_count += 1
        
        # Log progress every 100 authors (changed from 1000 for faster feedback)
        if author_count % 100 == 0:
            logger.info(f"    Processed {author_count} authors...")
            
            # Save progress checkpoint
            if progress_file:
                with open(progress_file, 'a') as f:
                    f.write(author + '\n')
    
    # Final progress save
    if progress_file:
        with open(progress_file, 'a') as f:
            for author in processed_authors:
                f.write(author + '\n')
    
    logger.info(f"  Processed {author_count} authors (skipped {skipped_count} already processed)")
    
    # Close connection pool if it was created
    if connection_pool is not None:
        connection_pool.close()
    
    return results


def print_summary(df, args):
    """
    Print a summary of the feature building process to stdout.
    
    Args:
        df: DataFrame with all features
        args: Command-line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info("\n=== build_features_v2 Summary ===")
    
    # Author counts
    bot_count = (df["y"] == 1).sum()
    human_count = (df["y"] == 0).sum()
    total = len(df)
    
    logger.info(f"\nAuthors processed")
    logger.info(f"  Bot authors    : {bot_count:,}")
    logger.info(f"  Human authors  : {human_count:,}")
    logger.info(f"  Total          : {total:,}")
    
    # Feature count
    feature_cols = [c for c in df.columns if c not in ["author", "y"]]
    logger.info(f"\nFeatures computed : {len(feature_cols)}")
    logger.info(f"  Temporal        : {'yes' if args.temporal else 'no'} (--temporal flag)")
    
    # NaN rates (top 5 by frequency)
    nan_rates = (df[feature_cols].isna().sum() / len(df)).sort_values(ascending=False)
    logger.info(f"  NaN rates (top 5 by frequency):")
    for col, rate in nan_rates.head(5).items():
        logger.info(f"    {col:40s} : {rate*100:5.1f}%")
    
    # Class balance
    logger.info(f"\nClass balance")
    logger.info(f"  y=1 (bot)   : {bot_count:,}  ({bot_count/total*100:.1f}%)")
    logger.info(f"  y=0 (human) : {human_count:,}  ({human_count/total*100:.1f}%)")
    
    logger.info(f"\nOutput : {args.output}  ({len(df)} rows × {len(df.columns)} columns)")


def main():
    # Setup logging for SLURM compatibility
    logger = setup_logging()
    
    # Suppress scipy warnings for small samples (expected for authors with few posts)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    parser = argparse.ArgumentParser(
        description="Build features for bot detection model training (memory-efficient v2)"
    )
    parser.add_argument("--comments-bot", required=True, help="Path to bot comments JSONL")
    parser.add_argument("--comments-human", required=True, help="Path to human comments JSONL")
    parser.add_argument("--submissions-bot", required=True, help="Path to bot submissions JSONL")
    parser.add_argument("--submissions-human", required=True, help="Path to human submissions JSONL")
    parser.add_argument("--domain-whitelist", required=True, help="Path to domain whitelist text file")
    parser.add_argument("--temporal", action="store_true", help="Include temporal reply-time features")
    parser.add_argument("--parent-comments", help="Path to parent comments JSONL from ZST extraction (improves temporal coverage)")
    parser.add_argument("--comment-db", help="Path for SQLite comment lookup database (default: temp file)")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--progress-dir", help="Directory for progress tracking files (enables resumability)")
    
    args = parser.parse_args()
    
    # Setup progress tracking
    progress_dir = Path(args.progress_dir) if args.progress_dir else None
    if progress_dir:
        progress_dir.mkdir(parents=True, exist_ok=True)
        bot_progress = progress_dir / "bot_authors_progress.txt"
        human_progress = progress_dir / "human_authors_progress.txt"
    else:
        bot_progress = None
        human_progress = None
    
    # Step 1: Load domain whitelist
    logger.info(f"Loading domain whitelist from {args.domain_whitelist}")
    domain_whitelist = load_domain_whitelist(args.domain_whitelist)
    logger.info(f"  Loaded {len(domain_whitelist)} domains")
    
    # Step 2: Build SQLite comment lookup if temporal features are enabled
    comment_lookup_db = None
    if args.temporal:
        logger.info("Building SQLite comment lookup for temporal features...")
        comment_lookup_db = build_comment_lookup_sqlite(
            args.comments_bot, 
            args.comments_human,
            parent_comments_path=args.parent_comments,
            db_path=args.comment_db
        )
        logger.info(f"  Built SQLite lookup at {comment_lookup_db}")
    
    # Step 3: Process bot population (label = 1)
    logger.info(f"\nProcessing bot population...")
    bot_results = process_population(
        args.comments_bot,
        args.submissions_bot,
        label=1,
        domain_whitelist=domain_whitelist,
        comment_lookup_db=comment_lookup_db,
        args=args,
        progress_file=bot_progress
    )
    logger.info(f"  Processed {len(bot_results)} bot authors")
    
    # Step 4: Process human population (label = 0)
    logger.info(f"\nProcessing human population...")
    human_results = process_population(
        args.comments_human,
        args.submissions_human,
        label=0,
        domain_whitelist=domain_whitelist,
        comment_lookup_db=comment_lookup_db,
        args=args,
        progress_file=human_progress
    )
    logger.info(f"  Processed {len(human_results)} human authors")
    
    # Step 5: Combine results into DataFrame
    logger.info(f"\nCombining results...")
    all_results = bot_results + human_results
    df = pd.DataFrame(all_results)
    
    # Step 6: Write to parquet
    logger.info(f"Writing to {args.output}")
    df.to_parquet(args.output, index=False)
    
    # Step 7: Print summary
    print_summary(df, args)
    
    # Step 8: Cleanup temporary database if it was auto-generated
    if args.temporal and args.comment_db is None:
        logger.info(f"\nCleaning up temporary SQLite database...")
        Path(comment_lookup_db).unlink(missing_ok=True)
        # Also cleanup WAL file
        wal_path = Path(str(comment_lookup_db) + "-wal")
        wal_path.unlink(missing_ok=True)
        logger.info(f"  Cleaned up {comment_lookup_db}")


if __name__ == "__main__":
    main()
