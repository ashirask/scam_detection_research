"""
Fast reply detection using Reddit comment timing coordination.

This script identifies suspicious accounts that participate in coordinated fast replies,
meaning multiple users responding to the same piece of content (post or comment) within
a short time window, repeatedly across many threads.

Why this design:
- Detects coordinated behavior by timing, not content similarity
- Uses Reddit's natural reply structure: comments link to parents via link_id and parent_id
- Measures intra-content coordination: how often do the same author pairs fast-reply to shared content?
- Complements text-similarity detection with timing signals

Pipeline:
1. Load sampled posts (JSONL) as content roots
2. Load sampled comments (JSONL) and resolve parent-child relationships
3. For each piece of content, collect direct replies sorted by timestamp
4. Find all reply pairs within a time threshold
5. Aggregate by author pair: count instances + compute reply time statistics
6. Filter and export suspicious author pairs + full events
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_POSTS_FILE = str(Path(__file__).resolve().parents[2] / "sampled_data" / "sample_posts_2024.jsonl")
DEFAULT_COMMENTS_FILE = str(Path(__file__).resolve().parents[2] / "sampled_data" / "sample_comments_2024.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")

SKIP_AUTHORS = {"[deleted]", "AutoModerator"}
SKIP_TEXT = {"[removed]", "[deleted]"}


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for fast-reply detection."""
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
		help="Directory where detection outputs are written."
	)
	parser.add_argument(
		"--time-threshold",
		type=int,
		default=10,
		help="Maximum seconds between two replies to the same content to count as fast.",
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


def load_posts(path: str) -> Dict[str, Dict]:
	"""Load posts from JSONL and index by post ID.

	Each JSON line is expected to contain keys such as `id`, `author`, `created_utc`,
	`subreddit`, `title`, and `selftext`.
	"""
	posts: Dict[str, Dict] = {}

	with open(path, encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue

			try:
				record = json.loads(line)
			except json.JSONDecodeError:
				continue

			post_id = record.get("id") or record.get("name")
			author = (record.get("author") or "").strip()
			created = record.get("created_utc")

			if not post_id or not author or author in SKIP_AUTHORS or created is None:
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

	The `link_id` field points to the root post and `parent_id` points to either a
	post (`t3_...`) or another comment (`t1_...`).
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
			link_id = record.get("link_id")
			parent_id = record.get("parent_id")
			created = record.get("created_utc")

			if not author or author in SKIP_AUTHORS or not body or body in SKIP_TEXT:
				continue
			if not comment_id or not link_id or not parent_id or created is None:
				continue

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
	"""Find all reply pairs within the time threshold for each post.

	The core signal here is coordination: two different authors replying to the same
	post within a very short window.
	"""
	fast_pairs: List[Dict] = []
	diagnostics = {
		"total_posts": len(posts),
		"posts_with_comments": 0,
		"total_comments": len(comments),
		"fast_reply_events": 0,
	}

	comments_by_parent = comments.groupby("parent_id") if not comments.empty else None

	for post_id, post_record in posts.items():
		if comments_by_parent is None or post_id not in comments_by_parent.groups:
			continue

		post_comments = comments_by_parent.get_group(post_id).copy()
		post_comments = post_comments[post_comments["parent_type"] == "post"]
		if len(post_comments) < 2:
			continue

		diagnostics["posts_with_comments"] += 1
		post_comments = post_comments.sort_values("created_utc").reset_index(drop=True)

		for i in range(len(post_comments)):
			for j in range(i + 1, len(post_comments)):
				row_i = post_comments.iloc[i]
				row_j = post_comments.iloc[j]

				delta_seconds = int(row_j["created_utc"] - row_i["created_utc"])
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
						"content_id": post_id,
						"content_type": "post",
						"delta_seconds": delta_seconds,
						"comment_a_id": row_i["comment_id"],
						"comment_b_id": row_j["comment_id"],
						"timestamp_a": float(row_i["created_utc"]),
						"timestamp_b": float(row_j["created_utc"]),
						"subreddit": post_record.get("subreddit", ""),
					}
				)

	diagnostics["fast_reply_events"] = len(fast_pairs)
	return fast_pairs, diagnostics


def aggregate_suspicious_pairs(fast_pairs: List[Dict], min_instances: int) -> pd.DataFrame:
	"""Aggregate fast-reply events by author pair and compute summary statistics."""
	if not fast_pairs:
		return pd.DataFrame(
			columns=[
				"author_a",
				"author_b",
				"fast_reply_instances",
				"max_delta",
				"avg_delta",
				"min_delta",
				"example_content_ids",
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

	out_rows = []
	for (author_a, author_b), events in grouped.items():
		if len(events) < min_instances:
			continue

		deltas = [e["delta_seconds"] for e in events]
		content_ids = [e["content_id"] for e in events]
		subreddits = [e["subreddit"] for e in events]

		out_rows.append(
			{
				"author_a": author_a,
				"author_b": author_b,
				"fast_reply_instances": len(events),
				"max_delta": int(np.max(deltas)),
				"avg_delta": round(float(np.mean(deltas)), 2),
				"min_delta": int(np.min(deltas)),
				"example_content_ids": ", ".join(list(dict.fromkeys(content_ids))[:5]),
				"example_subreddits": ", ".join(list(dict.fromkeys(subreddits))[:5]),
			}
		)

	if not out_rows:
		return pd.DataFrame(
			columns=[
				"author_a",
				"author_b",
				"fast_reply_instances",
				"max_delta",
				"avg_delta",
				"min_delta",
				"example_content_ids",
				"example_subreddits",
			]
		)

	return pd.DataFrame(out_rows).sort_values("fast_reply_instances", ascending=False).reset_index(drop=True)


def main() -> None:
	"""Run the fast-reply detection pipeline and export outputs."""
	args = parse_args()
	np.random.seed(args.seed)

	os.makedirs(args.output_dir, exist_ok=True)

	print(f"Loading posts from: {args.posts_file}")
	posts = load_posts(args.posts_file)
	if not posts:
		raise SystemExit("No usable posts found.")

	print(f"Loading comments from: {args.comments_file}")
	comments = load_comments(args.comments_file)
	if comments.empty:
		raise SystemExit("No usable comments found.")

	print(f"Finding fast-reply pairs (threshold={args.time_threshold} seconds)...")
	fast_pairs, diagnostics = find_fast_reply_pairs(
		comments=comments,
		posts=posts,
		time_threshold=args.time_threshold,
	)

	if not fast_pairs:
		print("No fast-reply pairs found. Exiting.")
		return

	events_df = pd.DataFrame(fast_pairs)
	events_path = os.path.join(args.output_dir, "fast_reply_events.csv")
	events_df.to_csv(events_path, index=False)

	print(f"Aggregating by author pair (min_instances={args.min_fast_reply_instances})...")
	suspicious_pairs = aggregate_suspicious_pairs(
		fast_pairs=fast_pairs,
		min_instances=args.min_fast_reply_instances,
	)

	if suspicious_pairs.empty:
		print("No suspicious pairs found after aggregation. Exiting.")
		return

	pairs_path = os.path.join(args.output_dir, "suspicious_fast_reply_pairs.csv")
	suspicious_pairs.to_csv(pairs_path, index=False)

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

	print("Fast-reply detection complete!")


if __name__ == "__main__":
	main()
