"""
Fast reply detection using Reddit comment timing coordination.

This script identifies suspicious accounts that participate in coordinated fast replies—
multiple users responding to the same piece of content (post or comment) within a short
time window (e.g., 10 seconds), repeatedly across many threads.

Why this design:
- Detects coordinated behavior by timing, not content similarity
- Uses Reddit's natural reply structure: comments link to parents via link_id and parent_id
- Measures intra-content coordination: how often do the same author pairs fast-reply to shared content?
- Complements text-similarity detection (bot_detection_faiss_no_lsh.py) with timing signals

Pipeline:
1. Load sampled posts (JSONL) as content roots
2. Load sampled comments (JSONL) and resolve parent-child relationships
3. For each piece of content, collect direct replies sorted by timestamp
4. Find all reply pairs within time threshold (default 10 seconds)
5. Aggregate by author pair: count instances + compute reply time statistics
6. Filter and export suspicious author pairs + full events

Reddit vs Twitter:
- Twitter's fast_retweet: User A retweets from User B within 10 seconds of B's tweet
  Result: directed edge (A→B), bipartite graph
- Reddit's fast_reply: Users A and B both reply to content C within 10 seconds of each other
  Result: undirected edge (A↔B), ties to shared content

Questions:
i would like to go back and make sure the methodlogy is correct. as you said:
If A is a reply to the post at 4:10:10 and B is a reply to the same post at 4:10:13 
→ they are flagged (both share the post parent). -- this shouldn't be flagged as both the actions are independent and both the authors are seperately responsing to the post?
If A is a reply to the post at 4:10:10 and B is a reply to A at 4:10:13 (so B’s parent is A) 
→ A and B do not get paired by this detection (they do not share the same parent). The code does not record a directed parent→child edge; it only records co-replies to the same content root. This matches your understanding. -- well if the script doesnt include it, dont you think this is a better way to process and detect the scams? could we have a new script with better implementation?
what you think?

"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# Default paths
DEFAULT_POSTS_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample_posts_2024.jsonl")
DEFAULT_COMMENTS_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample_comments_2024.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")

# Reddit artifacts to skip
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}
SKIP_TEXT = {"[removed]", "[deleted]"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fast-reply detection.

    Example CLI:
        python fast_reddit.py \\
            --posts-file sampled_data/sample_posts_2024.jsonl \\
            --comments-file sampled_data/sample_comments_2024.jsonl \\
            --output-dir code/src_3/output \\
            --time-threshold 10 \\
            --min-fast-reply-instances 5

    Returns:
        argparse.Namespace with fields:
        - posts_file: path to posts JSONL
        - comments_file: path to comments JSONL
        - output_dir: directory for outputs (suspicious_fast_reply_pairs.csv, etc.)
        - time_threshold: max seconds between replies to count as "fast" (default 10)
        - min_fast_reply_instances: minimum coordinated fast-reply events to flag an author pair
        - seed: RNG seed for reproducibility
    """
    parser = argparse.ArgumentParser(
        description=(
            "Detect suspicious Reddit accounts from coordinated fast replies "
            "to the same content using comment timing analysis."
        )
    )
    parser.add_argument(
        "--posts-file",
        default=DEFAULT_POSTS_FILE,
        help="Path to sampled posts JSONL file (content roots).",
    )
    parser.add_argument(
        "--comments-file",
        default=DEFAULT_COMMENTS_FILE,
        help="Path to sampled comments JSONL file (reply events).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where detection outputs are written.",
    )
    parser.add_argument(
        "--time-threshold",
        type=int,
        default=10,
        help="Maximum seconds between two replies to the same content to count as a 'fast reply' pair (default 10).",
    )
    parser.add_argument(
        "--min-fast-reply-instances",
        type=int,
        default=5,
        help="Minimum number of coordinated fast-reply instances to flag an author pair as suspicious.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_posts(path: str) -> Dict[str, Dict]:
    """Load posts from JSONL and index by post ID.

    Input:
        path = "sampled_data/sample_posts_2024.jsonl"

    File format (one JSON per line):
        {"id": "18vki7n", "author": "user1", "created_utc": 1704067312.0, ...}
        {"id": "18vkiif", "author": "user2", "created_utc": 1704067337.0, ...}

    Computation:
        1. Parse each JSON line
        2. Extract key fields: id, author, created_utc, subreddit, title, selftext
        3. Index by post id for fast parent lookup
        4. Skip malformed or malicious records

    Output:
        Dictionary indexed by post id:
        {
            "18vki7n": {
                "id": "18vki7n",
                "author": "user1",
                "created_utc": 1704067312.0,
                "subreddit": "AskReddit",
                "title": "What is your story?",
                "selftext": "Tell me...",
                "raw": {...full JSON...}
            },
            ...
        }
    """
    posts: Dict[str, Dict] = {}

    with open(path, encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            post_id = record.get("id") or record.get("name")
            if not post_id:
                continue

            author = (record.get("author") or "").strip()
            if not author or author in SKIP_AUTHORS:
                continue

            created = record.get("created_utc")
            if created is None:
                continue

            posts[post_id] = {
                "id": post_id,
                "author": author,
                "created_utc": float(created),
                "subreddit": record.get("subreddit", ""),
                "title": (record.get("title") or "").strip()[:100],
                "selftext": (record.get("selftext") or "").strip()[:100],
                "raw": record,
            }

    return posts


def load_comments(path: str) -> pd.DataFrame:
    """Load comments from JSONL and build a DataFrame with parent linkage resolved.

    Input:
        path = "sampled_data/sample_comments_2024.jsonl"

    File format (one JSON per line):
        {"id": "abc123", "author": "user1", "body": "Great post!", "created_utc": 1704067320.0, 
         "link_id": "t3_18vki7n", "parent_id": "t1_xyz789", ...}

    Computation:
        1. Parse each JSON line
        2. Extract: id, author, body, created_utc, link_id, parent_id, subreddit
        3. link_id format: "t3_<post_id>" (prefix t3_ for posts)
        4. parent_id format: "t1_<comment_id>" (prefix t1_ for comments) or "t3_<post_id>"
        5. Strip prefixes to get usable IDs
        6. Skip deleted/removed content and malformed records

    Output:
        DataFrame with columns:
        - comment_id: unique comment identifier
        - author: comment author username
        - body: normalized comment text
        - created_utc: UNIX timestamp
        - link_id: post ID (without t3_ prefix)
        - parent_id: parent comment or post ID (without prefix)
        - parent_type: "comment" if parent_id is a comment, "post" if parent is a post
        - subreddit: community name
        - raw: original JSON record

        Example row:
        {
            "comment_id": "abc123",
            "author": "user1",
            "body": "Great post!",
            "created_utc": 1704067320.0,
            "link_id": "18vki7n",
            "parent_id": "18vki7n",  # or "xyz789" if replying to another comment
            "parent_type": "post",
            "subreddit": "AskReddit"
        }
    """
    rows: List[Dict] = []

    with open(path, encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            author = (record.get("author") or "").strip()
            if not author or author in SKIP_AUTHORS:
                continue

            body = (record.get("body") or "").strip()
            if not body or body in SKIP_TEXT:
                continue

            comment_id = record.get("id")
            link_id = record.get("link_id")
            parent_id = record.get("parent_id")
            created = record.get("created_utc")

            if not (comment_id and link_id and parent_id and created is not None):
                continue

            # Strip Reddit fullname prefixes: t3_<id> for posts, t1_<id> for comments
            # Format: "t3_abcd1234" -> "abcd1234", "t1_xyz5678" -> "xyz5678"
            if link_id.startswith("t3_"):
                link_id = link_id[3:]
            if parent_id.startswith("t1_"):
                parent_id_clean = parent_id[3:]
                parent_type = "comment"
            elif parent_id.startswith("t3_"):
                parent_id_clean = parent_id[3:]
                parent_type = "post"
            else:
                # Malformed parent_id, skip
                continue

            rows.append(
                {
                    "comment_id": comment_id,
                    "author": author,
                    "body": body,
                    "created_utc": float(created),
                    "link_id": link_id,
                    "parent_id": parent_id_clean,
                    "parent_type": parent_type,
                    "subreddit": record.get("subreddit", ""),
                    "raw": record,
                }
            )

    return pd.DataFrame(rows)


def find_fast_reply_pairs(
    comments: pd.DataFrame,
    posts: Dict[str, Dict],
    time_threshold: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Find all reply pairs within time threshold for each piece of content.

    Core algorithm:
    For each parent (post or comment as content root):
        1. Collect all direct replies (parent_id == content_id)
        2. Sort replies by created_utc
        3. For every pair (i, j) where i < j:
           - Compute delta_seconds = reply[j].created_utc - reply[i].created_utc
           - If delta_seconds <= time_threshold:
             * Record the pair: (author_i, author_j, content_id, content_type, delta_seconds)

    Input:
        comments = DataFrame with columns: comment_id, author, created_utc, link_id, parent_id, parent_type
                   shape: (100000,)  # typical comment count
        posts = Dict indexed by post_id
        time_threshold = 10 (seconds)

    Output:
        Tuple:
        - List of dicts, each representing one fast-reply pair:
          [
              {
                  "author_a": "user1",
                  "author_b": "user2",
                  "content_id": "post_abc123",
                  "content_type": "post",
                  "delta_seconds": 7,
                  "comment_a_id": "comment_1",
                  "comment_b_id": "comment_2",
                  "timestamp_a": 1704067320.0,
                  "timestamp_b": 1704067327.0,
                  "subreddit": "AskReddit",
              },
              ...
          ]

        - Dict with diagnostics:
          {
              "total_posts": 1000,
              "total_content_roots": 1500,  # posts + comments that had replies
              "total_comments": 100000,
              "fast_reply_events": 45123,
          }
    """
    fast_pairs: List[Dict] = []
    diagnostics = {
        "total_posts": len(posts),
        "total_content_roots": 0,  # posts + comments that are parents to other comments
        "total_comments": len(comments),
        "fast_reply_events": 0,
    }

    # Group comments by their parent
    comments_by_parent = comments.groupby("parent_id")

    # Track which post each content root belongs to (for subreddit lookup)
    post_id_for_subreddit = {}
    for parent_id in comments_by_parent.groups:
        # Check if parent is a post
        if parent_id in posts:
            post_id_for_subreddit[parent_id] = parent_id
        else:
            # Parent is a comment; find its root post via link_id
            # Get one comment with this parent to extract link_id
            sample_comment = comments_by_parent.get_group(parent_id).iloc[0]
            post_id_for_subreddit[parent_id] = sample_comment.get("link_id")

    # Process each content root (post or comment) that has replies
    for content_id, group in comments_by_parent:
        # Get all replies to this content
        replies = group.copy()

        if len(replies) < 2:
            continue

        diagnostics["total_content_roots"] += 1

        # Determine content type: post or comment
        if content_id in posts:
            content_type = "post"
            subreddit = posts[content_id].get("subreddit", "")
        else:
            content_type = "comment"
            # Get subreddit from the root post
            root_post_id = post_id_for_subreddit.get(content_id)
            subreddit = posts.get(root_post_id, {}).get("subreddit", "")

        # Sort by timestamp
        replies = replies.sort_values("created_utc").reset_index(drop=True)

        # Find all pairs within time threshold
        for i in range(len(replies)):
            for j in range(i + 1, len(replies)):
                row_i = replies.iloc[i]
                row_j = replies.iloc[j]

                delta_seconds = int(row_j["created_utc"] - row_i["created_utc"])

                # Only keep pairs within threshold
                if delta_seconds > time_threshold:
                    continue

                # Skip same-author pairs
                author_i = row_i["author"]
                author_j = row_j["author"]
                if author_i == author_j:
                    continue

                fast_pairs.append(
                    {
                        "author_a": author_i,
                        "author_b": author_j,
                        "content_id": content_id,
                        "content_type": content_type,
                        "delta_seconds": delta_seconds,
                        "comment_a_id": row_i["comment_id"],
                        "comment_b_id": row_j["comment_id"],
                        "timestamp_a": float(row_i["created_utc"]),
                        "timestamp_b": float(row_j["created_utc"]),
                        "subreddit": subreddit,
                    }
                )

    diagnostics["fast_reply_events"] = len(fast_pairs)
    return fast_pairs, diagnostics


def aggregate_suspicious_pairs(
    fast_pairs: List[Dict],
    min_instances: int,
) -> pd.DataFrame:
    """Aggregate fast-reply pairs by author-pair and compute summary statistics.

    Input:
        fast_pairs = List of dicts from find_fast_reply_pairs()
                     shape: [45123] events
        min_instances = 5 (minimum coordinated events to flag)

    Computation (per author pair):
        1. Collect all events where (author_a, author_b) appear together in fast reply
        2. Separate by content_type (post vs comment)
        3. Count events; calculate max_delta, avg_delta, min_delta
        4. Track example content and subreddits
        5. Compute normalized_frequency = instances / max_instances (0.0-1.0)
        6. Filter: keep only pairs with >= min_instances

    Output:
        DataFrame with columns:
        - author_a, author_b: usernames (normalized to alphabetical order)
        - content_type: "post" or "comment"
        - fast_reply_instances: count of coordinated events
        - normalized_frequency: instances / global max (0.0-1.0)
        - max_delta: longest reply gap in seconds
        - avg_delta: mean reply gap
        - min_delta: shortest reply gap
        - example_content_ids: sample content IDs where they coordinated
        - example_subreddits: communities where they coordinated
    """
    if not fast_pairs:
        return pd.DataFrame(
            columns=[
                "author_a",
                "author_b",
                "content_type",
                "fast_reply_instances",
                "normalized_frequency",
                "max_delta",
                "avg_delta",
                "min_delta",
                "example_content_ids",
                "example_subreddits",
            ]
        )

    # Normalize pairs: always use alphabetical order (a <= b)
    for pair in fast_pairs:
        if pair["author_a"] > pair["author_b"]:
            pair["author_a"], pair["author_b"] = pair["author_b"], pair["author_a"]

    # Group by normalized author pair and content_type
    grouped = {}
    for pair in fast_pairs:
        key = (pair["author_a"], pair["author_b"], pair["content_type"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(pair)

    # Find global max instances (across all author pairs and content types)
    max_instances = max((len(events) for events in grouped.values()), default=0)
    if max_instances == 0:
        max_instances = 1  # Prevent division by zero

    # Compute aggregate stats
    out_rows = []
    for (author_a, author_b, content_type), events in grouped.items():
        if len(events) < min_instances:
            continue

        deltas = [e["delta_seconds"] for e in events]
        content_ids = [e["content_id"] for e in events]
        subreddits = [e["subreddit"] for e in events]
        instances = len(events)

        # Unique examples (up to 5)
        unique_content_ids = list(dict.fromkeys(content_ids))[:5]
        unique_subreddits = list(dict.fromkeys(subreddits))[:5]

        out_rows.append(
            {
                "author_a": author_a,
                "author_b": author_b,
                "content_type": content_type,
                "fast_reply_instances": instances,
                "normalized_frequency": round(instances / max_instances, 3),
                "max_delta": int(np.max(deltas)),
                "avg_delta": round(float(np.mean(deltas)), 2),
                "min_delta": int(np.min(deltas)),
                "example_content_ids": ", ".join(unique_content_ids),
                "example_subreddits": ", ".join(unique_subreddits),
            }
        )

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "author_a",
                "author_b",
                "content_type",
                "fast_reply_instances",
                "normalized_frequency",
                "max_delta",
                "avg_delta",
                "min_delta",
                "example_content_ids",
                "example_subreddits",
            ]
        )

    return pd.DataFrame(out_rows).sort_values("fast_reply_instances", ascending=False).reset_index(drop=True)


def main() -> None:
    """Orchestrate the full fast-reply detection pipeline.

    High-level flow:
    1. Parse CLI arguments
    2. Load posts and comments from JSONL files
    3. Find all fast-reply pairs using comment timing
    4. Aggregate by author pair
    5. Export results for downstream analysis

    Outputs written to output_dir:
    - fast_reply_events.csv: all individual fast-reply events
    - suspicious_fast_reply_pairs.csv: aggregated author pairs above threshold
    - fast_reply_summary.json: diagnostic statistics
    """
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ========== STEP 1: Load posts ==========
    print(f"Loading posts from: {args.posts_file}")
    posts = load_posts(args.posts_file)
    if not posts:
        raise SystemExit("No usable posts found.")
    print(f"Loaded {len(posts)} posts")

    # ========== STEP 2: Load comments ==========
    print(f"Loading comments from: {args.comments_file}")
    comments = load_comments(args.comments_file)
    if comments.empty:
        raise SystemExit("No usable comments found.")
    print(f"Loaded {len(comments)} comments")

    # ========== DEBUG: Check data quality ==========
    print("\n--- Data Quality Checks ---")
    
    # Check if there's any overlap between post IDs and comment parent_ids
    post_ids = set(posts.keys())
    comment_parents = set(comments["parent_id"].unique())
    overlap = post_ids & comment_parents
    print(f"Posts with ID present in comment parent_ids: {len(overlap)}")
    
    if not overlap:
        print("  WARNING: No overlap! Posts and comments may be from different datasets.")
        if len(post_ids) > 0 and len(comment_parents) > 0:
            print(f"  Sample post IDs: {list(post_ids)[:3]}")
            print(f"  Sample comment parents: {list(comment_parents)[:3]}")
    
    # Check comment parent types
    print(f"Comments with parent_type='post': {(comments['parent_type'] == 'post').sum()}")
    print(f"Comments with parent_type='comment': {(comments['parent_type'] == 'comment').sum()}")
    
    # Check if any posts have multiple different-author comments
    top_level_comments = comments[comments["parent_type"] == "post"]
    if not top_level_comments.empty:
        comments_per_post = top_level_comments.groupby("parent_id").size()
        posts_with_2plus = (comments_per_post >= 2).sum()
        print(f"Top-level comments on posts: {len(top_level_comments)}")
        print(f"Posts with ≥2 top-level comments: {posts_with_2plus}")
    print(
        f"Finding fast-reply pairs (threshold={args.time_threshold} seconds)..."
    )
    fast_pairs, diagnostics = find_fast_reply_pairs(
        comments=comments,
        posts=posts,
        time_threshold=args.time_threshold,
    )

    # Debug: print diagnostics even if no pairs found
    print("\n--- Diagnostic Details ---")
    print(f"Total posts loaded: {diagnostics['total_posts']}")
    print(f"Content roots (posts + comments) with ≥2 replies: {diagnostics['total_content_roots']}")
    print(f"Total comments loaded: {diagnostics['total_comments']}")
    print(f"Fast-reply events found: {diagnostics['fast_reply_events']}")

    if not fast_pairs:
        print("\nNo fast-reply pairs found. Exiting.")
        print("\nPossible reasons:")
        print("1. No posts have multiple comments with different authors")
        print("2. All comments on each post are from the same author")
        print("3. All reply gaps exceed the time threshold")
        print("4. Posts and comments don't overlap (different threads/date ranges)")
        return

    print(f"\nFound {len(fast_pairs)} fast-reply events")

    # ========== STEP 4: Save raw fast-reply events ==========
    events_df = pd.DataFrame(fast_pairs)
    events_path = os.path.join(args.output_dir, "fast_reply_events.csv")
    events_df.to_csv(events_path, index=False)
    print(f"Saved fast-reply events: {events_path}")

    # ========== STEP 5: Aggregate by author pair ==========
    print(
        f"Aggregating by author pair (min_instances={args.min_fast_reply_instances})..."
    )
    suspicious_pairs = aggregate_suspicious_pairs(
        fast_pairs=fast_pairs,
        min_instances=args.min_fast_reply_instances,
    )

    if suspicious_pairs.empty:
        print("No suspicious pairs found after aggregation. Exiting.")
        return

    print(f"Flagged {len(suspicious_pairs)} suspicious author pairs")

    # ========== STEP 6: Save aggregated suspicious pairs ==========
    pairs_path = os.path.join(args.output_dir, "suspicious_fast_reply_pairs.csv")
    suspicious_pairs.to_csv(pairs_path, index=False)
    print(f"Saved suspicious pairs: {pairs_path}")

    # ========== STEP 7: Save diagnostic summary ==========
    summary = {
        "input_posts_file": args.posts_file,
        "input_comments_file": args.comments_file,
        "time_threshold_seconds": args.time_threshold,
        "min_fast_reply_instances": args.min_fast_reply_instances,
        "diagnostics": diagnostics,
        "suspicious_pairs_found": int(len(suspicious_pairs)),
    }

    summary_path = os.path.join(args.output_dir, "fast_reply_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\n" + "=" * 60)
    print("Fast-reply detection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
