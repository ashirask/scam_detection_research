"""
Detect fast reply coordination between comments that reply to the same parent comment.

This script is the comment-to-comment version of the fast reply detector.
It only looks at direct replies to a parent comment and asks whether two different
accounts replied to that same parent comment within a short time window.

Why this version exists:
- It matches a stricter methodology: compare sibling replies to the same parent comment
- It avoids treating two independent replies to a post as a coordination signal
- It works directly from the extracted comments JSONL produced by extract_comments_for_posts.py

Input requirements:
- A JSONL file of extracted comments
- No posts JSON is required for the core detection logic

Why posts JSON is not required:
- Each extracted comment already contains link_id, parent_id, parent_type, and subreddit
- root_post_id can be derived from link_id
- comment-to-comment comparisons only need parent_id groups and timestamps

Outputs:
- comment_fast_reply_events.csv
- suspicious_comment_fast_reply_pairs.csv
- comment_fast_reply_summary.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


DEFAULT_COMMENTS_FILE = str(
    Path(__file__).resolve().parent.parent.parent / "sampled_data" / "fast_100_10" / "extracted_comments_for_100_posts.jsonl"
)
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")

SKIP_AUTHORS = {"[deleted]", "AutoModerator"}
SKIP_TEXT = {"[removed]", "[deleted]"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect suspicious Reddit accounts that reply to the same parent comment "
            "within a short time window."
        )
    )
    parser.add_argument(
        "--comments-file",
        default=DEFAULT_COMMENTS_FILE,
        help="Path to extracted comments JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--time-threshold",
        type=int,
        default=10,
        help="Maximum seconds between two replies to the same parent comment to count as a fast-reply pair.",
    )
    parser.add_argument(
        "--min-fast-reply-instances",
        type=int,
        default=5,
        help="Minimum number of coordinated fast-reply events to flag an author pair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def normalize_fullname(value: Optional[str]) -> Optional[str]:
    """Strip Reddit fullname prefixes when present."""
    if not value:
        return None
    value = str(value).strip()
    if value.startswith(("t1_", "t3_", "t2_")):
        return value[3:]
    return value


def load_comments(path: str) -> pd.DataFrame:
    """Load extracted comments into a DataFrame."""
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

            comment_id = normalize_fullname(record.get("id") or record.get("name"))
            link_id = normalize_fullname(record.get("link_id"))
            parent_id = normalize_fullname(record.get("parent_id"))
            created = record.get("created_utc")
            subreddit = record.get("subreddit", "")

            if not (comment_id and link_id and parent_id and created is not None):
                continue

            parent_type = "comment" if str(record.get("parent_id", "")).startswith("t1_") else "post"

            rows.append(
                {
                    "comment_id": comment_id,
                    "author": author,
                    "body": body,
                    "created_utc": float(created),
                    "link_id": link_id,
                    "parent_id": parent_id,
                    "parent_type": parent_type,
                    "subreddit": subreddit,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["root_post_id"] = df["link_id"]
    return df


def find_comment_reply_pairs(
    comments: pd.DataFrame,
    time_threshold: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Find fast-reply pairs among comments that share the same parent comment.

    This only considers parent_type == 'comment'.
    """
    fast_pairs: List[Dict] = []
    diagnostics = {
        "total_comments": int(len(comments)),
        "total_parent_comments": 0,
        "total_event_pairs": 0,
        "fast_reply_events": 0,
    }

    parent_comments = comments[comments["parent_type"] == "comment"].copy()
    if parent_comments.empty:
        return fast_pairs, diagnostics

    grouped = parent_comments.groupby("parent_id")

    for parent_comment_id, replies in grouped:
        if len(replies) < 2:
            continue

        diagnostics["total_parent_comments"] += 1

        replies = replies.sort_values("created_utc").reset_index(drop=True)
        for i in range(len(replies)):
            for j in range(i + 1, len(replies)):
                row_i = replies.iloc[i]
                row_j = replies.iloc[j]

                delta_seconds = int(row_j["created_utc"] - row_i["created_utc"])
                diagnostics["total_event_pairs"] += 1

                if delta_seconds > time_threshold:
                    continue

                author_i = row_i["author"]
                author_j = row_j["author"]
                if author_i == author_j:
                    continue

                fast_pairs.append(
                    {
                        "author_a": author_i,
                        "author_b": author_j,
                        "parent_comment_id": parent_comment_id,
                        "root_post_id": row_i["root_post_id"],
                        "subreddit": row_i["subreddit"],
                        "delta_seconds": delta_seconds,
                        "reply_a_id": row_i["comment_id"],
                        "reply_b_id": row_j["comment_id"],
                        "timestamp_a": float(row_i["created_utc"]),
                        "timestamp_b": float(row_j["created_utc"]),
                    }
                )

    diagnostics["fast_reply_events"] = len(fast_pairs)
    return fast_pairs, diagnostics


def aggregate_suspicious_pairs(
    fast_pairs: List[Dict],
    min_instances: int,
) -> pd.DataFrame:
    """Aggregate comment-reply fast pairs by author pair."""
    if not fast_pairs:
        return pd.DataFrame(
            columns=[
                "author_a",
                "author_b",
                "fast_reply_instances",
                "normalized_frequency",
                "max_delta",
                "avg_delta",
                "min_delta",
                "example_parent_comment_ids",
                "example_root_post_ids",
                "example_subreddits",
            ]
        )

    for pair in fast_pairs:
        if pair["author_a"] > pair["author_b"]:
            pair["author_a"], pair["author_b"] = pair["author_b"], pair["author_a"]

    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for pair in fast_pairs:
        key = (pair["author_a"], pair["author_b"])
        grouped.setdefault(key, []).append(pair)

    max_instances = max((len(events) for events in grouped.values()), default=1)

    out_rows = []
    for (author_a, author_b), events in grouped.items():
        if len(events) < min_instances:
            continue

        deltas = [event["delta_seconds"] for event in events]
        parent_comment_ids = [event["parent_comment_id"] for event in events]
        root_post_ids = [event["root_post_id"] for event in events]
        subreddits = [event["subreddit"] for event in events]

        out_rows.append(
            {
                "author_a": author_a,
                "author_b": author_b,
                "fast_reply_instances": len(events),
                "normalized_frequency": round(len(events) / max_instances, 3),
                "max_delta": int(np.max(deltas)),
                "avg_delta": round(float(np.mean(deltas)), 2),
                "min_delta": int(np.min(deltas)),
                "example_parent_comment_ids": ", ".join(list(dict.fromkeys(parent_comment_ids))[:5]),
                "example_root_post_ids": ", ".join(list(dict.fromkeys(root_post_ids))[:5]),
                "example_subreddits": ", ".join(list(dict.fromkeys(subreddits))[:5]),
            }
        )

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "author_a",
                "author_b",
                "fast_reply_instances",
                "normalized_frequency",
                "max_delta",
                "avg_delta",
                "min_delta",
                "example_parent_comment_ids",
                "example_root_post_ids",
                "example_subreddits",
            ]
        )

    return pd.DataFrame(out_rows).sort_values("fast_reply_instances", ascending=False).reset_index(drop=True)


def main() -> None:
    """Run the comment-to-comment fast reply detection pipeline."""
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading comments from: {args.comments_file}")
    comments = load_comments(args.comments_file)
    if comments.empty:
        raise SystemExit("No usable comments found.")

    print(f"Loaded {len(comments)} comments")
    print(f"Parent comments available: {(comments['parent_type'] == 'comment').sum()}")

    fast_pairs, diagnostics = find_comment_reply_pairs(
        comments=comments,
        time_threshold=args.time_threshold,
    )

    print("\n--- Diagnostic Details ---")
    print(f"Total comments loaded: {diagnostics['total_comments']}")
    print(f"Parent comments with 2+ replies: {diagnostics['total_parent_comments']}")
    print(f"Candidate reply pairs checked: {diagnostics['total_event_pairs']}")
    print(f"Fast-reply events found: {diagnostics['fast_reply_events']}")

    if not fast_pairs:
        print("No fast-reply pairs found. Exiting.")
        return

    events_df = pd.DataFrame(fast_pairs)
    events_path = os.path.join(args.output_dir, "comment_fast_reply_events.csv")
    events_df.to_csv(events_path, index=False)
    print(f"Saved fast-reply events: {events_path}")

    suspicious_pairs = aggregate_suspicious_pairs(
        fast_pairs=fast_pairs,
        min_instances=args.min_fast_reply_instances,
    )

    pairs_path = os.path.join(args.output_dir, "suspicious_comment_fast_reply_pairs.csv")
    suspicious_pairs.to_csv(pairs_path, index=False)
    print(f"Saved suspicious pairs: {pairs_path}")

    summary = {
        "input_comments_file": args.comments_file,
        "time_threshold_seconds": args.time_threshold,
        "min_fast_reply_instances": args.min_fast_reply_instances,
        "diagnostics": diagnostics,
        "suspicious_pairs_found": int(len(suspicious_pairs)),
    }

    summary_path = os.path.join(args.output_dir, "comment_fast_reply_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()