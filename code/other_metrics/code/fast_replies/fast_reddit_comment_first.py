"""
Comment-first fast-reply detection for Reddit.

Goal:
- Focus on account-level fast response behavior without pair aggregation.
- Identify accounts that repeatedly respond within a short time window
  in the same content thread (same root post via `link_id`).

Design choice:
- Comments are the source of truth for timing.
- We do not require a matching posts file for parent lookups.
- This avoids sparse overlap issues when posts/comments are sampled differently.

Signal definition used here:
- Sort comments by timestamp within each thread (`link_id`).
- If a comment arrives within `time_threshold` seconds of the immediately
  previous comment by a different author, mark it as a fast-response event.
- Aggregate these events by responding account.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_COMMENTS_FILE = str(
    Path(__file__).resolve().parents[2] / "sampled_data" / "sample_comments_2024.jsonl"
)
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")

SKIP_AUTHORS = {"[deleted]", "AutoModerator"}
SKIP_TEXT = {"[removed]", "[deleted]"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for comment-first fast-response detection.

    Example:
        python fast_reddit_comment_first.py \
            --comments-file sampled_data/sample_comments_2024.jsonl \
            --output-dir code/other_metrics/code/output \
            --time-threshold 30 \
            --min-fast-events 3 \
            --min-unique-threads 2
    """
    parser = argparse.ArgumentParser(
        description=(
            "Detect Reddit accounts that repeatedly respond quickly within the same thread "
            "using comment-first timing analysis."
        )
    )
    parser.add_argument(
        "--comments-file",
        default=DEFAULT_COMMENTS_FILE,
        help="Path to comments JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--time-threshold",
        type=int,
        default=10,
        help="Maximum seconds between adjacent comments in same thread to mark as fast response.",
    )
    parser.add_argument(
        "--min-fast-events",
        type=int,
        default=3,
        help="Minimum fast-response events for an account to be flagged.",
    )
    parser.add_argument(
        "--min-unique-threads",
        type=int,
        default=2,
        help="Minimum number of distinct threads where account shows fast-response behavior.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_comments(path: str) -> pd.DataFrame:
    """Load and clean Reddit comments from JSONL.

    Output columns:
    - comment_id
    - author
    - created_utc
    - link_id (root post id without `t3_` prefix)
    - parent_id (without fullname prefix)
    - parent_type ("post" | "comment")
    - subreddit
    - body
    - raw
    """
    rows: List[Dict] = []

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            author = (record.get("author") or "").strip()
            body = (record.get("body") or "").strip()
            comment_id = record.get("id")
            created = record.get("created_utc")
            link_id = record.get("link_id")
            parent_id = record.get("parent_id")

            if not author or author in SKIP_AUTHORS:
                continue
            if not body or body in SKIP_TEXT:
                continue
            if not comment_id or created is None or not link_id or not parent_id:
                continue

            # Normalize Reddit fullnames for stable joins/grouping.
            if isinstance(link_id, str) and link_id.startswith("t3_"):
                link_id = link_id[3:]

            if isinstance(parent_id, str) and parent_id.startswith("t1_"):
                parent_id_clean = parent_id[3:]
                parent_type = "comment"
            elif isinstance(parent_id, str) and parent_id.startswith("t3_"):
                parent_id_clean = parent_id[3:]
                parent_type = "post"
            else:
                continue

            rows.append(
                {
                    "comment_id": str(comment_id),
                    "author": author,
                    "created_utc": float(created),
                    "link_id": str(link_id),
                    "parent_id": str(parent_id_clean),
                    "parent_type": parent_type,
                    "subreddit": record.get("subreddit", ""),
                    "body": body,
                    "raw": record,
                }
            )

    return pd.DataFrame(rows)


def find_fast_response_events(
    comments: pd.DataFrame,
    time_threshold: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Create event-level fast-response rows from adjacent comments in each thread.

    Thread model:
    - `link_id` identifies the root post thread.
    - We sort all comments in that thread by `created_utc`.

    Event rule:
    - For adjacent comments i-1 -> i in same thread:
      - delta = t(i) - t(i-1)
      - keep event when delta <= threshold and authors differ.

    Why adjacent-only:
    - Captures direct rapid reaction dynamics.
    - Avoids quadratic all-pairs explosion for large threads.
    """
    if comments.empty:
        return pd.DataFrame(), {
            "total_comments": 0,
            "total_threads": 0,
            "threads_with_2plus_comments": 0,
            "fast_response_events": 0,
        }

    events: List[Dict] = []

    grouped = comments.groupby("link_id", sort=False)
    total_threads = int(grouped.ngroups)
    threads_with_2plus = 0

    for link_id, g in grouped:
        if len(g) < 2:
            continue
        threads_with_2plus += 1

        g = g.sort_values("created_utc").reset_index(drop=True)

        # Walk adjacent comments only; this keeps runtime linear per thread.
        for i in range(1, len(g)):
            prev_row = g.iloc[i - 1]
            cur_row = g.iloc[i]

            if prev_row["author"] == cur_row["author"]:
                continue

            delta_seconds = int(cur_row["created_utc"] - prev_row["created_utc"])
            if delta_seconds < 0:
                continue
            if delta_seconds > time_threshold:
                continue

            events.append(
                {
                    "thread_id": link_id,
                    "subreddit": cur_row["subreddit"],
                    "responder_author": cur_row["author"],
                    "previous_author": prev_row["author"],
                    "responder_comment_id": cur_row["comment_id"],
                    "previous_comment_id": prev_row["comment_id"],
                    "responder_time": float(cur_row["created_utc"]),
                    "previous_time": float(prev_row["created_utc"]),
                    "delta_seconds": delta_seconds,
                    "responder_parent_type": cur_row["parent_type"],
                }
            )

    events_df = pd.DataFrame(events)
    diagnostics = {
        "total_comments": int(len(comments)),
        "total_threads": total_threads,
        "threads_with_2plus_comments": int(threads_with_2plus),
        "fast_response_events": int(len(events_df)),
    }
    return events_df, diagnostics


def aggregate_fast_responder_accounts(
    events_df: pd.DataFrame,
    min_fast_events: int,
    min_unique_threads: int,
) -> pd.DataFrame:
    """Aggregate event rows to account-level fast-response statistics.

    For each `responder_author`, compute:
    - fast_response_events: number of qualifying fast responses
    - unique_threads: number of distinct threads where fast responses occurred
    - unique_previous_authors: number of distinct users responded to quickly
    - avg/min/max delta_seconds
    - sample subreddits

    Then filter by minimum counts to reduce noise.
    """
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "author",
                "fast_response_events",
                "unique_threads",
                "unique_previous_authors",
                "avg_delta_seconds",
                "min_delta_seconds",
                "max_delta_seconds",
                "sample_subreddits",
            ]
        )

    grouped = events_df.groupby("responder_author", sort=False)
    rows: List[Dict] = []

    for author, g in grouped:
        event_count = int(len(g))
        unique_threads = int(g["thread_id"].nunique())
        unique_prev_authors = int(g["previous_author"].nunique())

        if event_count < min_fast_events:
            continue
        if unique_threads < min_unique_threads:
            continue

        deltas = g["delta_seconds"].to_numpy(dtype=np.int32)
        subreddits = list(dict.fromkeys(g["subreddit"].tolist()))[:5]

        rows.append(
            {
                "author": author,
                "fast_response_events": event_count,
                "unique_threads": unique_threads,
                "unique_previous_authors": unique_prev_authors,
                "avg_delta_seconds": round(float(np.mean(deltas)), 2),
                "min_delta_seconds": int(np.min(deltas)),
                "max_delta_seconds": int(np.max(deltas)),
                "sample_subreddits": ", ".join(subreddits),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "author",
                "fast_response_events",
                "unique_threads",
                "unique_previous_authors",
                "avg_delta_seconds",
                "min_delta_seconds",
                "max_delta_seconds",
                "sample_subreddits",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["fast_response_events", "unique_threads", "avg_delta_seconds"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def main() -> None:
    """Run the comment-first fast-response pipeline and write outputs."""
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading comments from: {args.comments_file}")
    comments = load_comments(args.comments_file)
    if comments.empty:
        raise SystemExit("No usable comments found after filtering.")
    print(f"Usable comments: {len(comments)}")
    print(f"Unique authors: {comments['author'].nunique()}")
    print(f"Unique threads (link_id): {comments['link_id'].nunique()}")

    print(f"\nFinding fast-response events (threshold={args.time_threshold}s)...")
    events_df, diagnostics = find_fast_response_events(
        comments=comments,
        time_threshold=args.time_threshold,
    )

    print("\n--- Diagnostics ---")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    events_path = os.path.join(args.output_dir, "fast_response_events_comment_first.csv")
    events_df.to_csv(events_path, index=False)
    print(f"Saved event-level output: {events_path} ({len(events_df)} rows)")

    print(
        f"\nAggregating account-level fast responders "
        f"(min_fast_events={args.min_fast_events}, min_unique_threads={args.min_unique_threads})..."
    )
    accounts_df = aggregate_fast_responder_accounts(
        events_df=events_df,
        min_fast_events=args.min_fast_events,
        min_unique_threads=args.min_unique_threads,
    )

    accounts_path = os.path.join(args.output_dir, "fast_responder_accounts.csv")
    accounts_df.to_csv(accounts_path, index=False)
    print(f"Saved account-level output: {accounts_path} ({len(accounts_df)} rows)")

    summary = {
        "input_comments_file": args.comments_file,
        "time_threshold_seconds": int(args.time_threshold),
        "min_fast_events": int(args.min_fast_events),
        "min_unique_threads": int(args.min_unique_threads),
        "diagnostics": diagnostics,
        "accounts_flagged": int(len(accounts_df)),
    }

    summary_path = os.path.join(args.output_dir, "fast_response_comment_first_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\nComment-first fast-response detection complete.")


if __name__ == "__main__":
    main()
