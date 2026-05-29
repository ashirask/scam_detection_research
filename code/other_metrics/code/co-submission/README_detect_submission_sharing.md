# detect_submission_sharing.py — README

Purpose
- A direct co-submission detector that mirrors the TF-IDF workflow used for co-URL and co-subreddit detection, but uses Reddit submissions as the token for each comment.
- For each user, the script collects the submissions they commented on, builds a TF-IDF document, computes pairwise cosine similarity, and uses the observed percentile of similarities as the baseline threshold.

Why this method
- It tests whether two users repeatedly comment on the same submissions more often than we would expect by chance.
- It down-weights popular submissions that attract many comments, which helps reduce false positives from high-traffic threads such as r/all.
- It keeps the logic aligned with the existing TF-IDF detectors in this folder.

What the token is
- The token is the comment's `link_id`, normalized to remove the Reddit fullname prefix.
- In practice, this is the submission identifier for the post the comment belongs to.
- The script treats repeated comments on the same submission as repeated TF counts unless `--count-mode unique` is selected.

Inputs
- JSONL files containing Reddit comments.
- The script is compatible with the extracted comments JSON produced by `extract_comments_for_posts.py`, because that output already includes `link_id`, `author`, `body`, `created_utc`, `subreddit`, and `permalink`.
- No new extraction step is required if you already have a comments JSONL file in that format.

Quick usage
```bash
python code/other_metrics/code/co-submission/detect_submission_sharing.py \
  --input sampled_data/extract_30k_for-fast/extracted_comments_for_30k_posts.jsonl \
  --output code/other_metrics/output/co-submission/demo \
  --min-comments 5 \
  --count-mode total \
  --min-df 2 --max-df 0.9 \
  --null-method observed \
  --percentile 99
```

Main CLI options
- `--input` PATH... : One or more comments JSONL files.
- `--output` DIR : Output folder for CSV, Markdown report, and plot.
- `--min-comments` INT : Minimum total submission-comment tokens a user must have to stay in the analysis.
- `--count-mode` unique|total : Count unique submissions or total comment occurrences.
- `--min-df` INT|FLOAT : Minimum TF-IDF document frequency for submission tokens.
- `--max-df` INT|FLOAT : Maximum TF-IDF document frequency for submission tokens.
- `--null-method` observed|sampled_pairs : Use all observed user pairs or a sampled subset to estimate the percentile threshold.
- `--sample-size` INT : Number of observed pairs to sample when using `sampled_pairs`.
- `--seed` INT : Random seed for sampled pair selection.
- `--percentile` INT : Percentile baseline used for flagging suspicious pairs.
- `--verbose` : Print step-by-step loader and vectorization traces.
- `--preview-limit` INT : Maximum number of verbose debug events per loader.
- `--run-id` STRING : Optional label added to output filenames.

Filtering pipeline
- Invalid JSON rows are skipped.
- Rows with missing or `[deleted]` authors are skipped.
- Rows with missing `link_id` are skipped.
- Rows from `AutoModerator` are skipped.
- Users with fewer than `--min-comments` submission tokens are removed.
- TF-IDF pruning removes tokens outside the `--min-df` / `--max-df` range.
- Suspicious pairs are those with cosine similarity above the selected percentile threshold.

TF-IDF post-row pruning (new)
- After TF-IDF vocabulary pruning, the script removes authors who have fewer than 2 non-zero TF-IDF features. This prevents the "one-token collapse" case where many users have a single shared submission token (or a single high-weight feature) that artificially inflates observed similarities and can lead to thresholds of 1.0.
- The default behavior uses a minimum of 2 non-zero features (hardcoded as `min_features=2` in the script). If you want to change this, modify the `filter_authors_by_tfidf_features()` call in `detect_submission_sharing.py`.

Outputs
- CSV: `co_submission__*.csv`
  - Pairwise summary rows with user IDs, cosine similarity, percentile rank, shared submission count, and previews.
- Markdown: `co_submission__*.md`
  - One section per user showing their submission context and example comment text.
- PNG: `co_submission__*.png`
  - Histogram and KDE of the observed pairwise similarity distribution.

Notes
- This detector is comments-only. It does not need post metadata beyond the submission identifier stored in `link_id`.
- The current input file produced by `extract_comments_for_posts.py` is sufficient for testing this metric.
- If you want a wider or narrower signal, the most useful knobs are `--min-comments`, `--min-df`, `--max-df`, and `--count-mode`.
