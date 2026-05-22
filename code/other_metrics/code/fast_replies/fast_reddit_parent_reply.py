"""

**SELECTED DRAFT**

Detect fast replies where one comment directly replies to another comment.

This script follows the newer methodology:
- A user writes a parent comment
- Another user replies directly to that comment within a small time window
- That fast child reply is the event we keep

This is different from sibling-pair logic.
We do not compare two child replies to each other.
We only compare each child comment to its direct parent comment.

Why this matches the desired behavior:
- It captures direct comment-to-comment replies
- It ignores independent replies to the post
- It produces a clearer signal: who replies quickly to comments, how often, and where

Input:
- Extracted comments JSONL from extract_comments_for_posts.py

Posts JSON:
- Not required for the core logic
- The extracted comments file already contains parent_id, parent_type, link_id, and subreddit

Outputs:
- parent_reply_events.csv
- suspicious_parent_reply_accounts.csv
- parent_reply_summary.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_COMMENTS_FILE = str(
    Path(__file__).resolve().parent.parent.parent
    / "sampled_data"
    / "fast_100_10"
    / "extracted_comments_for_100_posts.jsonl"
)
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")

SKIP_AUTHORS = {"[deleted]", "AutoModerator"}
SKIP_TEXT = {"[removed]", "[deleted]"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect accounts that reply quickly to a parent comment using extracted Reddit comments."
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
        default=3,
        help="Maximum seconds between a parent comment and a child reply to count as fast.",
    )
    parser.add_argument(
        "--min-fast-reply-instances",
        type=int,
        default=3,
        help="Minimum number of fast child replies needed to flag an account.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def normalize_fullname(value: Optional[str]) -> Optional[str]:
    """Strip Reddit fullname prefixes like t1_ or t3_."""
    if not value:
        return None
    value = str(value).strip()
    if value.startswith(("t1_", "t2_", "t3_")):
        return value[3:]
    return value


def load_comments(path: str) -> pd.DataFrame:
    """Load the extracted comments JSONL into a clean DataFrame.

    The extracted file already contains the fields we need:
    - comment_id
    - author
    - created_utc
    - parent_id
    - parent_type
    - link_id
    - subreddit
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

            comment_id = normalize_fullname(record.get("id") or record.get("name"))
            link_id = normalize_fullname(record.get("link_id"))
            parent_id_raw = record.get("parent_id")
            parent_id = normalize_fullname(parent_id_raw)
            created = record.get("created_utc")

            if not (comment_id and link_id and parent_id and created is not None):
                continue

            parent_type = "comment" if str(parent_id_raw).startswith("t1_") else "post"

            rows.append(
                {
                    "comment_id": comment_id,
                    "author": author,
                    "body": body,
                    "created_utc": float(created),
                    "link_id": link_id,
                    "parent_id": parent_id,
                    "parent_type": parent_type,
                    "subreddit": record.get("subreddit", ""),
                }
            )

    return pd.DataFrame(rows)


def find_fast_parent_replies(comments: pd.DataFrame, time_threshold: int) -> Tuple[List[Dict], Dict[str, int]]:
    """Find direct comment replies that happen within the time threshold.

    We only flag replies where:
    - the child comment replies to another comment (parent_type == 'comment')
    - the child and parent are different authors
    - the child arrives within time_threshold seconds after the parent comment
    """
    fast_events: List[Dict] = []
    diagnostics = {
        "total_comments": int(len(comments)),
        "total_comment_replies": 0,
        "fast_reply_events": 0,
        "missing_parent_comments": 0,
    }

    if comments.empty:
        return fast_events, diagnostics

    # Build a quick lookup from comment_id -> row data.
    comment_lookup: Dict[str, Dict] = {}
    for row in comments.to_dict(orient="records"):
        comment_lookup[row["comment_id"]] = row

    # Only inspect comment-to-comment replies.
    child_replies = comments[comments["parent_type"] == "comment"].copy()
    diagnostics["total_comment_replies"] = int(len(child_replies))

    for _, child in child_replies.iterrows():
        parent_id = child["parent_id"]
        parent = comment_lookup.get(parent_id)

        # If the parent comment is not in the extracted sample, skip it.
        if parent is None:
            diagnostics["missing_parent_comments"] += 1
            continue

        parent_author = parent["author"]
        child_author = child["author"]

        # Skip same-author replies.
        if parent_author == child_author:
            continue

        delta_seconds = int(child["created_utc"] - parent["created_utc"])
        if delta_seconds < 0:
            continue

        if delta_seconds > time_threshold:
            continue

        fast_events.append(
            {
                "child_author": child_author,
                "parent_author": parent_author,
                "parent_comment_id": parent_id,
                "child_comment_id": child["comment_id"],
                "root_post_id": child["link_id"],
                "subreddit": child["subreddit"],
                "delta_seconds": delta_seconds,
                "parent_created_utc": float(parent["created_utc"]),
                "child_created_utc": float(child["created_utc"]),
            }
        )

    diagnostics["fast_reply_events"] = len(fast_events)
    return fast_events, diagnostics


def aggregate_suspicious_accounts(fast_events: List[Dict], min_instances: int) -> pd.DataFrame:
    """Aggregate fast replies by the child account that did the replying.

    This keeps the output focused on accounts that repeatedly reply quickly to comments.
    """
    if not fast_events:
        return pd.DataFrame(
            columns=[
                "child_author",
                "fast_reply_instances",
                "normalized_frequency",
                "max_delta",
                "avg_delta",
                "min_delta",
                "example_parent_comment_ids",
                "example_child_comment_ids",
                "example_parent_authors",
                "example_root_post_ids",
                "example_subreddits",
            ]
        )

    grouped: Dict[str, List[Dict]] = {}
    for event in fast_events:
        grouped.setdefault(event["child_author"], []).append(event)

    max_instances = max((len(events) for events in grouped.values()), default=1)

    out_rows = []
    for child_author, events in grouped.items():
        if len(events) < min_instances:
            continue

        deltas = [event["delta_seconds"] for event in events]
        parent_comment_ids = [event["parent_comment_id"] for event in events]
        child_comment_ids = [event["child_comment_id"] for event in events]
        parent_authors = [event["parent_author"] for event in events]
        root_post_ids = [event["root_post_id"] for event in events]
        subreddits = [event["subreddit"] for event in events]

        out_rows.append(
            {
                "child_author": child_author,
                "fast_reply_instances": len(events),
                "normalized_frequency": round(len(events) / max_instances, 3),
                "max_delta": int(np.max(deltas)),
                "avg_delta": round(float(np.mean(deltas)), 2),
                "min_delta": int(np.min(deltas)),
                "example_parent_comment_ids": ", ".join(list(dict.fromkeys(parent_comment_ids))[:5]),
                "example_child_comment_ids": ", ".join(list(dict.fromkeys(child_comment_ids))[:5]),
                "example_parent_authors": ", ".join(list(dict.fromkeys(parent_authors))[:5]),
                "example_root_post_ids": ", ".join(list(dict.fromkeys(root_post_ids))[:5]),
                "example_subreddits": ", ".join(list(dict.fromkeys(subreddits))[:5]),
            }
        )

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "child_author",
                "fast_reply_instances",
                "normalized_frequency",
                "max_delta",
                "avg_delta",
                "min_delta",
                "example_parent_comment_ids",
                "example_child_comment_ids",
                "example_parent_authors",
                "example_root_post_ids",
                "example_subreddits",
            ]
        )

    return pd.DataFrame(out_rows).sort_values("fast_reply_instances", ascending=False).reset_index(drop=True)


def main() -> None:
    """Run the direct parent-reply fast detection pipeline."""
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading comments from: {args.comments_file}")
    comments = load_comments(args.comments_file)
    if comments.empty:
        raise SystemExit("No usable comments found.")

    print(f"Loaded {len(comments)} comments")
    print(f"Comment-to-comment replies available: {(comments['parent_type'] == 'comment').sum()}")

    fast_events, diagnostics = find_fast_parent_replies(
        comments=comments,
        time_threshold=args.time_threshold,
    )

    print("\n--- Diagnostic Details ---")
    print(f"Total comments loaded: {diagnostics['total_comments']}")
    print(f"Direct comment replies inspected: {diagnostics['total_comment_replies']}")
    print(f"Missing parent comments skipped: {diagnostics['missing_parent_comments']}")
    print(f"Fast-reply events found: {diagnostics['fast_reply_events']}")

    if not fast_events:
        print("No fast-reply events found. Exiting.")
        return

    events_df = pd.DataFrame(fast_events)
    events_path = os.path.join(args.output_dir, "parent_reply_events.csv")
    events_df.to_csv(events_path, index=False)
    print(f"Saved fast-reply events: {events_path}")

    suspicious_accounts = aggregate_suspicious_accounts(
        fast_events=fast_events,
        min_instances=args.min_fast_reply_instances,
    )

    accounts_path = os.path.join(args.output_dir, "suspicious_parent_reply_accounts.csv")
    suspicious_accounts.to_csv(accounts_path, index=False)
    print(f"Saved suspicious accounts: {accounts_path}")

    summary = {
        "input_comments_file": args.comments_file,
        "time_threshold_seconds": args.time_threshold,
        "min_fast_reply_instances": args.min_fast_reply_instances,
        "diagnostics": diagnostics,
        "suspicious_accounts_found": int(len(suspicious_accounts)),
    }

    summary_path = os.path.join(args.output_dir, "parent_reply_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()