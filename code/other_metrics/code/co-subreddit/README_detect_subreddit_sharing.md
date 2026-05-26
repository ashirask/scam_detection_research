# detect_subreddit_sharing.py — README

Purpose
- A direct co-subreddit detector that mirrors the TF-IDF workflow used for co-URL
  detection, but uses subreddit names as the token for each post/comment.
- For each user, the script collects the set or frequency distribution of subreddits
  they appear in, builds a TF-IDF document, computes pairwise cosine similarity, and
  uses the 99th percentile of observed similarities as the baseline threshold.

What the data field is
- The subreddit value is read from the plain `subreddit` field in the JSON objects.
- In the sample files, this field appears alongside `subreddit_id` and `subreddit_type`.

Inputs
- JSONL files containing Reddit posts or comments.
- The script accepts multiple files as long as `--input` and `--type` have the same length.

Quick usage
```bash
python code/other_metrics/code/co-subreddit/detect_subreddit_sharing.py \
  --input sampled_data/sample_posts_100k_2024.jsonl \
  --type posts \
  --output code/other_metrics/output/co-subreddit/demo \
  --min-posts 5 \
  --count-mode total \
  --min-df 2 --max-df 0.9 \
  --null-method observed \
  --percentile 99
```

Main CLI options
- `--input` PATH... : One or more JSONL files.
- `--type` posts|comments : One data type per input file.
- `--output` DIR : Output folder for CSV, Markdown report, and plot.
- `--min-posts` INT : Minimum total subreddit tokens a user must have to stay in the analysis.
- `--count-mode` unique|total : Count unique subreddit tokens or total occurrences.
- `--min-df` INT|FLOAT : Minimum TF-IDF document frequency for subreddit tokens.
- `--max-df` INT|FLOAT : Maximum TF-IDF document frequency for subreddit tokens.
- `--null-method` observed|sampled_pairs : Use all observed user pairs or a sampled subset to estimate the percentile threshold.
- `--sample-size` INT : Number of observed pairs to sample when using `sampled_pairs`.
- `--seed` INT : Random seed for sampled pair selection.
- `--percentile` INT : Percentile baseline used for flagging suspicious pairs.
- `--verbose` : Print step-by-step loader and vectorization traces.

Filtering pipeline
- Invalid JSON rows are skipped.
- Rows with missing or `[deleted]` authors are skipped.
- Rows with missing subreddit names are skipped.
- Users with fewer than `--min-posts` subreddit tokens are removed.
- TF-IDF pruning removes tokens outside the `--min-df` / `--max-df` range.
- Suspicious pairs are those with cosine similarity above the 99th-percentile baseline.

Outputs
- CSV: `co_subreddit_suspicious_pairs_{timestamp}.csv`
  - Pairwise summary rows with user IDs, cosine similarity, percentile rank, shared subreddit count, and previews.
- Markdown: `co_subreddit_user_context_{timestamp}.md`
  - One section per user showing their subreddit context and example text.
- PNG: `co_subreddit_similarity_distribution_{timestamp}.png`
  - Histogram and KDE of the observed pairwise similarity distribution.

Notes
- The report is for context only. The pairwise results themselves are already in the CSV.
