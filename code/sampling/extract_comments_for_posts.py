"""
Extract comments from ZST files for randomly selected Reddit posts.

Pipeline:
1. Load a parsed posts JSONL file (output from sample_reddit_multi_v2.py)
2. Randomly sample N post IDs from it
3. Stream through comments ZST files
4. Extract all comments where link_id matches one of the sampled post IDs
5. Save comments to output JSONL and measure fast-reply timing

Usage:
    python extract_comments_for_posts.py \
        --posts-jsonl sampled_data/sample_posts_2024.jsonl \
        --comments-zst RC_2024-01.zst RC_2024-02.zst RC_2024-03.zst \
        --output-dir sampled_data \
        --output-file extracted_comments_for_sample_posts.jsonl \
        --num-posts 100 \
        --seed 42
"""

import argparse
import glob
import io
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import zstandard as zstd # type: ignore


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract comments from ZST files for a random sample of posts. "
            "Useful for analyzing timing coordination within specific threads."
        )
    )
    parser.add_argument(
        "--posts-jsonl",
        required=True,
        help="Path to posts JSONL file (parsed from sample_reddit_multi_v2.py).",
    )
    parser.add_argument(
        "--comments-zst",
        nargs="+",
        required=True,
        help="One or more .zst comment files to search.",
    )
    parser.add_argument(
        "--num-posts",
        type=int,
        default=100,
        help="Number of posts to randomly sample (default 100).",
    )
    parser.add_argument(
        "--output-dir",
        default="sampled_data",
        help="Directory for output JSONL.",
    )
    parser.add_argument(
        "--output-file",
        default="extracted_comments_for_sample_posts.jsonl",
        help="Output JSONL filename.",
    )
    parser.add_argument(
        "--time-threshold",
        type=int,
        default=10,
        help="Measure fast replies within this many seconds (default 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-window-size",
        type=int,
        default=2147483648,
        help="Max zstd decode window size.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print progress every N seen comments.",
    )
    return parser.parse_args()


def load_post_ids(posts_jsonl_path: str, num_posts: int, seed: int) -> Set[str]:
    """Load post IDs from parsed JSONL and randomly sample N of them.

    Input:
        posts_jsonl_path: path to JSONL with {"id": "...", ...} records
        num_posts: how many to randomly sample
        seed: RNG seed

    Output:
        Set of post IDs (strings) selected for extraction
    """
    random.seed(seed)
    all_post_ids: List[str] = []

    with open(posts_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            post_id = record.get("id")
            if post_id:
                all_post_ids.append(str(post_id))

    print(f"Loaded {len(all_post_ids)} post IDs from {posts_jsonl_path}")

    if len(all_post_ids) < num_posts:
        print(f"  Warning: only {len(all_post_ids)} posts available; using all")
        sampled = all_post_ids
    else:
        sampled = random.sample(all_post_ids, k=num_posts)

    print(f"Randomly sampled {len(sampled)} post IDs")
    return set(sampled)


def normalize_link_id(link_id: Optional[str]) -> Optional[str]:
    """Strip Reddit fullname prefix from link_id.

    t3_abc123 -> abc123
    abc123 -> abc123
    """
    if not link_id:
        return None
    link_id = str(link_id).strip()
    if link_id.startswith("t3_"):
        return link_id[3:]
    return link_id


def extract_comments_from_zst_files(
    zst_paths: List[str],
    target_post_ids: Set[str],
    output_jsonl_path: str,
    max_window_size: int,
    progress_every: int,
) -> Tuple[int, int]:
    """Stream through ZST comment files and extract comments for target posts.

    Input:
        zst_paths: list of .zst file paths
        target_post_ids: set of post IDs to extract comments for
        output_jsonl_path: where to write extracted comments
        max_window_size: zstd window size limit
        progress_every: print progress every N comments

    Output:
        Tuple (total_comments_seen, comments_matched)
    """
    total_seen = 0
    total_matched = 0
    bad_json = 0

    with open(output_jsonl_path, "w", encoding="utf-8") as out:
        for zst_path in zst_paths:
            if not os.path.isfile(zst_path):
                print(f"Warning: {zst_path} not found, skipping")
                continue

            print(f"\nProcessing: {zst_path}")
            file_seen = 0
            file_matched = 0
            file_bad = 0

            with open(zst_path, "rb") as f:
                dctx = zstd.ZstdDecompressor(max_window_size=max_window_size)
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

                    for line in text_stream:
                        total_seen += 1
                        file_seen += 1

                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            bad_json += 1
                            file_bad += 1
                            continue

                        # Extract and normalize link_id (points to root post)
                        link_id = normalize_link_id(record.get("link_id"))
                        if not link_id:
                            continue

                        # Check if this comment is for one of our target posts
                        if link_id in target_post_ids:
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_matched += 1
                            file_matched += 1

                        if progress_every > 0 and file_seen % progress_every == 0:
                            print(
                                f"  File progress: seen={file_seen}, matched={file_matched}, "
                                f"bad_json={file_bad}"
                            )

            print(
                f"Completed file: seen={file_seen}, matched={file_matched}, "
                f"bad_json={file_bad}"
            )

    return total_seen, total_matched


def measure_fast_replies(
    comments_jsonl_path: str,
    time_threshold: int,
) -> Dict[str, int]:
    """Analyze extracted comments and count fast-reply events.

    Fast reply: comment from user B arrives within time_threshold seconds
    of a previous comment from a different user A in the same thread.
    """
    comments_by_post: Dict[str, List[Dict]] = {}

    # Load comments and group by post
    with open(comments_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            link_id = normalize_link_id(record.get("link_id"))
            if not link_id:
                continue

            if link_id not in comments_by_post:
                comments_by_post[link_id] = []
            comments_by_post[link_id].append(record)

    # Measure fast replies within each post's comments
    total_posts = len(comments_by_post)
    posts_with_fast_replies = 0
    total_fast_reply_pairs = 0

    for post_id, comments in comments_by_post.items():
        if len(comments) < 2:
            continue

        # Sort by timestamp
        comments.sort(key=lambda x: x.get("created_utc", 0))

        # Check adjacent pairs for fast replies
        for i in range(1, len(comments)):
            prev_comment = comments[i - 1]
            cur_comment = comments[i]

            prev_author = (prev_comment.get("author") or "").strip()
            cur_author = (cur_comment.get("author") or "").strip()

            # Skip same-author or deleted comments
            if prev_author == cur_author or not prev_author or not cur_author:
                continue

            prev_time = prev_comment.get("created_utc")
            cur_time = cur_comment.get("created_utc")

            if prev_time is None or cur_time is None:
                continue

            delta = int(cur_time - prev_time)
            if delta < 0:
                continue

            if delta <= time_threshold:
                total_fast_reply_pairs += 1

        if len(comments) > 1:
            posts_with_fast_replies += 1

    return {
        "total_extracted_posts": total_posts,
        "posts_with_2plus_comments": posts_with_fast_replies,
        "adjacent_fast_reply_pairs": total_fast_reply_pairs,
        "time_threshold_seconds": time_threshold,
    }


def main() -> None:
    """Run the extraction and analysis pipeline."""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    # ========== STEP 1: Load and sample post IDs ==========
    print("=" * 60)
    print("STEP 1: Load and sample post IDs")
    print("=" * 60)
    target_post_ids = load_post_ids(
        args.posts_jsonl,
        num_posts=args.num_posts,
        seed=args.seed,
    )

    # ========== STEP 2: Extract comments for sampled posts ==========
    print("\n" + "=" * 60)
    print("STEP 2: Extract comments for sampled posts")
    print("=" * 60)
    total_comments_seen, comments_matched = extract_comments_from_zst_files(
        zst_paths=args.comments_zst,
        target_post_ids=target_post_ids,
        output_jsonl_path=output_path,
        max_window_size=args.max_window_size,
        progress_every=args.progress_every,
    )

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)
    print(f"Total comments seen in ZST files: {total_comments_seen}")
    print(f"Comments extracted (for target posts): {comments_matched}")
    print(f"Output saved to: {output_path}")

    # ========== STEP 3: Measure fast replies ==========
    print("\n" + "=" * 60)
    print(f"STEP 3: Measure fast replies (threshold={args.time_threshold}s)")
    print("=" * 60)
    timing_stats = measure_fast_replies(
        comments_jsonl_path=output_path,
        time_threshold=args.time_threshold,
    )

    print("\nTiming Analysis Results:")
    for key, value in timing_stats.items():
        print(f"  {key}: {value}")

    # Save summary
    summary = {
        "sampled_post_ids_count": len(target_post_ids),
        "total_comments_seen_in_zst": total_comments_seen,
        "comments_extracted_for_posts": comments_matched,
        "extraction_coverage": round(comments_matched / max(total_comments_seen, 1) * 100, 2),
        "timing_analysis": timing_stats,
    }

    summary_path = os.path.join(
        args.output_dir, "extraction_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
