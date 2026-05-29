# fast_reddit_parent_reply.py — README

Purpose
- A direct parent-reply detector that flags accounts that repeatedly reply quickly to an existing comment.
- The script keeps only child comments that reply to a parent comment within a configurable time window, which makes the signal easier to interpret than sibling-reply logic.

Why this method
- It captures direct comment-to-comment behavior for a single account.
- It avoids false positives from multiple accounts independently replying to the same post.
- It produces an account-level signal that is easy to aggregate, rank, and inspect manually.

Inputs
- Extracted Reddit comments JSONL from `extract_comments_for_posts.py`.
- The extracted file already contains the fields needed for this detector, including `comment_id`, `author`, `created_utc`, `parent_id`, `parent_type`, `link_id`, and `subreddit`.

Quick usage
```bash
python code/other_metrics/code/fast_replies/fast_reddit_parent_reply.py \
  --comments-file sampled_data/fast_100_10/extracted_comments_for_100_posts.jsonl \
  --output-dir code/other_metrics/code/fast_replies/output \
  --time-threshold 3 \
  --min-fast-reply-instances 3
```

Main CLI options
- `--comments-file` PATH : Path to the extracted comments JSONL file.
- `--output-dir` DIR : Directory where outputs are written.
- `--time-threshold` INT : Maximum seconds between a parent comment and a child reply to count as fast.
- `--min-fast-reply-instances` INT : Minimum number of fast replies needed to flag an account.
- `--seed` INT : Random seed for reproducibility.

Filtering pipeline
- Invalid JSON rows are skipped.
- Rows with missing or skipped authors such as `[deleted]` and `AutoModerator` are removed.
- Rows with empty or removed text are skipped.
- Only comment-to-comment replies are inspected.
- Replies from the same author as the parent are skipped.
- Replies with negative time deltas or deltas above the time threshold are skipped.

Outputs
- `parent_reply_events.csv` - event-level rows for each fast child reply.
- `suspicious_parent_reply_accounts.csv` - per-account summary of repeated fast reply behavior.
- `parent_reply_summary.json` - run metadata and diagnostics.

Output fields
- Event CSV includes child author, parent author, comment IDs, root post ID, subreddit, and delta seconds.
- Account summary includes instance counts, normalized frequency, delta statistics, and example parent/child IDs.

Notes
- This script is intentionally focused on direct parent-reply behavior rather than broad pairwise similarity.
- If you want a wider or narrower detector, the most useful knobs are `--time-threshold` and `--min-fast-reply-instances`.
- For further analysis, the output csv can be plugged into `extract_selected_authors_with_delta.py` to inspect accounts that don't appear to be a bot.