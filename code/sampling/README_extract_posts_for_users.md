# extract_posts_for_users.py

This script samples a random set of Reddit users from one or more JSONL/ZST source files, then extracts every post written by those users from one or more target JSONL/ZST files.

It is the post-level companion to `extract_comments_for_posts.py`.

## When to use it

Use this when you already have a sampled 3-month post file, comment file, or mixed JSONL file and want to:

1. sample a reproducible set of users from that source, and
2. collect all of their posts from the same file or a larger target dump.

The script works in a streaming way, so it does not need to load the full input into memory.

## Inputs

- `--source-jsonl` or `--source-dir`: file list or directory used to sample users.
- `--target-jsonl` or `--target-dir`: file list or directory to scan for matching posts. If omitted, the source files are also used as the target.
- Each record should contain an `author` field, and post records should also contain an `id` field for duplicate suppression.

## Outputs

The script writes these files to `--output-dir`:

- `extracted_posts_for_sample_users.jsonl` by default
- `sampled_users.txt`
- `extraction_summary.json`

Each extracted post is written as-is, with a `source_file` field added for provenance.

## Example

```bash
python code/sampling/extract_posts_for_users.py \
  --source-jsonl sampled_data/sample_posts_2024.jsonl \
  --target-jsonl sampled_data/sample_posts_2022-2025.jsonl \
  --num-users 100 \
  --seed 42 \
  --output-dir sampled_data/user_posts
```

Directory-based discovery works too:

```bash
python code/sampling/extract_posts_for_users.py \
  --source-dir sampled_data \
  --source-glob "sample_posts_2024*.jsonl" \
  --target-dir sampled_data \
  --target-recursive \
  --num-users 100 \
  --seed 42 \
  --output-dir sampled_data/user_posts
```

To extract posts from the same file used for user sampling, omit `--target-jsonl`.

## Useful flags

- `--num-users`: number of unique users to sample.
- `--min-source-posts`: only sample users who appear at least this many times in the source data.
- `--source-recursive` / `--target-recursive`: search subdirectories when discovering files from folders.
- `--exclude-author-tokens`: exclude usernames containing tokens such as `bot`.
- `--keep-deleted`: keep `[deleted]` users instead of filtering them out.
- `--allow-duplicates`: keep duplicate post IDs if the same record appears in multiple target files.
- `--verbose`: print per-file loading and extraction diagnostics.

## Recommendations

- If you are sampling users from a narrow time window, keep the source file separate from the larger extraction target so the same users can be expanded against a broader post history.
- Keep `--exclude-author-tokens bot` and the default deleted-user filter on unless you specifically need bot-like or placeholder accounts.
- For larger experiments, write outputs into a dedicated subfolder so the sampled user list and summary stay tied to the exact run.