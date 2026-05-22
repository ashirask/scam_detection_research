"""
Extract recent comments for randomly selected non-bot suspicious accounts
and compute parent-child delta seconds.

This reproduces the workflow used to generate:
- selected_authors_recent_comments.csv
- selected_authors_recent_comments_with_delta.csv

Inputs:
- suspicious_parent_reply_accounts_with_text.csv
- extracted_comments_for_30k_posts.jsonl (or any extracted comments JSONL)
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample non-bot child authors from suspicious account CSV, "
            "extract their recent comments from extracted comments JSONL, "
            "and compute delta_seconds to parent comments when available."
        )
    )
    parser.add_argument(
        "--suspicious-csv",
        required=True,
        help="Path to suspicious_parent_reply_accounts_with_text.csv.",
    )
    parser.add_argument(
        "--comments-jsonl",
        required=True,
        help="Path to extracted comments JSONL (for author comment history and parent timestamps).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output files.",
    )
    parser.add_argument(
        "--num-authors",
        type=int,
        default=10,
        help="How many non-bot authors to sample (default: 10).",
    )
    parser.add_argument(
        "--max-comments-per-author",
        type=int,
        default=100,
        help="Keep at most this many most recent comments per author (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible author sampling.",
    )
    parser.add_argument(
        "--bot-substring",
        default="bot",
        help="Substring used to exclude bot-like usernames (case-insensitive). Default: bot",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def build_comment_lookups(comments_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    created_lookup: Dict[str, float] = {}
    author_lookup: Dict[str, str] = {}
    body_lookup: Dict[str, str] = {}

    total = 0
    with comments_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1

            comment_id = record.get("id")
            comment_name = record.get("name")
            created = record.get("created_utc")
            author = (record.get("author") or "").strip()
            body = normalize_text(record.get("body") or "")

            keys: List[str] = []
            if comment_id:
                keys.append(str(comment_id))
                keys.append(f"t1_{comment_id}")
            if comment_name:
                keys.append(str(comment_name))

            for key in keys:
                if created is not None:
                    created_lookup[key] = float(created)
                author_lookup[key] = author
                body_lookup[key] = body

    return {
        "created_lookup": created_lookup,
        "author_lookup": author_lookup,
        "body_lookup": body_lookup,
        "total_comments": total,
    }


def select_authors(suspicious_csv: Path, num_authors: int, seed: int, bot_substring: str) -> List[str]:
    df = pd.read_csv(suspicious_csv)
    if "child_author" not in df.columns:
        raise ValueError("Input CSV must contain a 'child_author' column")

    candidates = df[~df["child_author"].astype(str).str.contains(bot_substring, case=False, na=False)]
    authors = candidates["child_author"].dropna().astype(str).tolist()
    if not authors:
        raise ValueError("No non-bot child authors found with the provided bot substring filter")

    random.seed(seed)
    sample_size = min(num_authors, len(authors))
    return random.sample(authors, k=sample_size)


def extract_recent_comments(
    comments_jsonl: Path,
    selected_authors: List[str],
    max_comments_per_author: int,
    lookups: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    selected_set = set(selected_authors)

    with comments_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            author = (record.get("author") or "").strip()
            if author not in selected_set:
                continue

            created_utc = record.get("created_utc")
            comment_id = record.get("id")
            comment_name = record.get("name")
            parent_id = record.get("parent_id")

            rows.append(
                {
                    "author": author,
                    "created_utc": float(created_utc) if created_utc is not None else None,
                    "comment_id": str(comment_id) if comment_id else "",
                    "comment_name": str(comment_name) if comment_name else "",
                    "subreddit": str(record.get("subreddit") or ""),
                    "link_id": str(record.get("link_id") or ""),
                    "parent_id": str(parent_id) if parent_id else "",
                    "body": normalize_text(record.get("body") or ""),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "author",
                "created_utc",
                "comment_id",
                "comment_name",
                "subreddit",
                "link_id",
                "parent_id",
                "body",
                "rank_most_recent",
                "parent_created_utc",
                "parent_author",
                "parent_body",
                "delta_seconds",
            ]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["author", "created_utc", "comment_id"], ascending=[True, False, False])
    df["rank_most_recent"] = df.groupby("author").cumcount() + 1
    df = df[df["rank_most_recent"] <= max_comments_per_author].copy()

    created_lookup = lookups["created_lookup"]
    author_lookup = lookups["author_lookup"]
    body_lookup = lookups["body_lookup"]

    def lookup_parent_created(parent_id: Any) -> Optional[float]:
        if pd.isna(parent_id):
            return None
        key = str(parent_id).strip()
        return created_lookup.get(key)

    def lookup_parent_author(parent_id: Any) -> str:
        if pd.isna(parent_id):
            return ""
        key = str(parent_id).strip()
        return author_lookup.get(key, "")

    def lookup_parent_body(parent_id: Any) -> str:
        if pd.isna(parent_id):
            return ""
        key = str(parent_id).strip()
        return body_lookup.get(key, "")

    df["parent_created_utc"] = df["parent_id"].apply(lookup_parent_created)
    df["parent_author"] = df["parent_id"].apply(lookup_parent_author)
    df["parent_body"] = df["parent_id"].apply(lookup_parent_body)
    df["delta_seconds"] = pd.to_numeric(df["created_utc"], errors="coerce") - pd.to_numeric(
        df["parent_created_utc"], errors="coerce"
    )

    df = df.sort_values(["author", "created_utc", "comment_id"], ascending=[True, False, False]).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()

    suspicious_csv = Path(args.suspicious_csv)
    comments_jsonl = Path(args.comments_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_authors = select_authors(
        suspicious_csv=suspicious_csv,
        num_authors=args.num_authors,
        seed=args.seed,
        bot_substring=args.bot_substring,
    )

    lookups = build_comment_lookups(comments_jsonl)
    result_df = extract_recent_comments(
        comments_jsonl=comments_jsonl,
        selected_authors=selected_authors,
        max_comments_per_author=args.max_comments_per_author,
        lookups=lookups,
    )

    recent_csv = output_dir / "selected_authors_recent_comments.csv"
    with_delta_csv = output_dir / "selected_authors_recent_comments_with_delta.csv"
    summary_json = output_dir / "selected_authors_summary.json"

    base_columns = [
        "author",
        "created_utc",
        "comment_id",
        "comment_name",
        "subreddit",
        "link_id",
        "parent_id",
        "body",
        "rank_most_recent",
    ]
    result_df[base_columns].to_csv(recent_csv, index=False)
    result_df.to_csv(with_delta_csv, index=False)

    summary = {
        "seed": args.seed,
        "bot_substring": args.bot_substring,
        "selected_authors": selected_authors,
        "selected_author_count": len(selected_authors),
        "rows_written": int(len(result_df)),
        "rows_by_author": (
            result_df.groupby("author").size().sort_values(ascending=False).to_dict() if not result_df.empty else {}
        ),
        "total_comments_in_input_jsonl": int(lookups["total_comments"]),
        "rows_with_parent_comment_timestamp": int(result_df["parent_created_utc"].notna().sum())
        if "parent_created_utc" in result_df
        else 0,
        "rows_with_delta_seconds": int(result_df["delta_seconds"].notna().sum()) if "delta_seconds" in result_df else 0,
        "output_recent_csv": str(recent_csv),
        "output_with_delta_csv": str(with_delta_csv),
    }

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
