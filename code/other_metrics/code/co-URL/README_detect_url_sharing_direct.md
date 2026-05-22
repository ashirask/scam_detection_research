# detect_url_sharing_direct.py — README

Purpose
- A lightweight, direct co-URL detection script that builds per-author TF‑IDF "documents"
  from normalized URLs (from posts or comment bodies) and flags author pairs whose
  cosine similarity exceeds an observed percentile threshold.

Key behaviors
- Threshold is computed from the observed round-robin (upper-triangle) author-author
  similarity distribution (default: 99th percentile). Optionally a sampled subset can
  be used for faster estimation.
- Tokenization: authors are represented by lists of URL tokens. For posts the script
  can use either the `domain` or the normalized `full_url` (controlled by `--post-mode`).
- URL normalization: host is lowercased, `www.` removed, path preserved, query/fragment
  are stripped in the current implementation (see notes below about YouTube collapsing).
- TF‑IDF pruning is supported via `--min-df` and `--max-df` to remove rare/common tokens.

Inputs
- JSONL files containing Reddit posts (`--type posts`) or comments (`--type comments`).
  The script expects the JSON objects to contain `author` and (for posts) `url` or
  `url_overridden_by_dest`, and optionally `title`/`selftext`.

Quick usage
```bash
python code/other_metrics/code/co-URL/detect_url_sharing_direct.py \
  --input sampled_data/sample_posts_100k_2024.jsonl \
  --type posts \
  --output code/other_metrics/output/co-url/direct_results \
  --post-mode full_url \
  --min-urls 2 \
  --count-mode total \
  --min-df 2 --max-df 0.9 \
  --null-method sampled_pairs --sample-size 500 --percentile 99
```

Main CLI options (summary)
- `--input` PATH... : One or more JSONL input files.
- `--type` posts|comments : Type for each input file (must match number of `--input`).
- `--output` DIR : Output directory (created if missing).
- `--post-mode` domain|full_url : Use post `domain` or `url/url_overridden_by_dest` as main token.
- `--min-urls` INT : Minimum tokens required per author (default 2).
- `--count-mode` unique|total : Count tokens uniquely or by total occurrences.
- `--min-df` INT|FLOAT : TF‑IDF `min_df` (e.g., 2 or 0.01).
- `--max-df` INT|FLOAT : TF‑IDF `max_df` (e.g., 0.9).
- `--null-method` observed|sampled_pairs : How to derive threshold from observed pairs (sample or complete data).
- `--sample-size` INT : When using `sampled_pairs` the number of observed pairs to sample.
- `--seed` INT : RNG seed for sampling.
- `--percentile` INT : Percentile for threshold (default 99).

Outputs
- CSV: `co_url_suspicious_pairs_direct_{timestamp}.csv` — summary rows for each suspicious pair
  (author_1, author_2, cosine_similarity, percentile_rank, shared_features, previews).
- Markdown: `co_url_pair_report_direct_{timestamp}.md` — author content reference listing each
  author once with their captured posts/comments (the CSV already contains pair details).
- PNG: similarity distribution histogram saved to the output directory.

Which URL is printed in the report?
- The report shows the `url` field from `author_sources` entries. For posts this is assigned
  from `url_overridden_by_dest` or, if missing, `url` (stored as `source_url` when loading posts).
- URLs that were only found inside `title` or `selftext` are normalized and added as features
  (they participate in TF‑IDF and similarity) but are NOT added to the `author_sources` 'url'
  attribute by default — so they will not appear as the printed `URL` unless the post provided one.

Notes and caveats
- URL normalization currently strips query parameters and fragments. That makes many
  YouTube links collapse to `https://youtube.com/watch` — if you want to preserve
  video identity, change `normalize_url()` to keep the `v=` parameter or use `--post-mode domain`.
- Using aggressive `min_df`/`max_df` can collapse author vectors to very few features and
  artificially inflate cosine similarities; if you want the threshold to reflect the
  unpruned distribution, compute the percentile with permissive DF (e.g., `min_df=1, max_df=1.0`).
- The script supports both 'unique' and 'total' token counting; choose `--count-mode` to match
  whether per-author frequency should matter. Should we only consider unique urls or repeating urls should be considered as well?

Next steps (suggested)
- compare actual full url without normalizing it too much
- Compute threshold without the hyperparameter, but would that be helpful?
