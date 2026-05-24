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

Recent changes in this branch
- Added explicit domain exclusion before counting tokens via `--exclude-domain` (can be repeated).
- When `--post-mode full_url` the script can now use raw URLs (no normalization) for token features —
  this preserves the original URL strings you observed during testing.
- Filtering order changed: domain exclusions are applied first, then per-author token counting (`--min-urls`),
  then TF‑IDF vocabulary pruning (`--min-df`/`--max-df`), and finally authors with fewer than 2 non-zero
  TF‑IDF features are removed before computing cosine similarities.

These changes were introduced to make exclusion behavior explicit and to avoid degenerate
similarity thresholds caused by one-token author vectors.

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
 - `--exclude-domain` DOMAIN : Exclude tokens coming from this domain before counting tokens.
   This flag can be repeated. Default exclusions: `i.imgur.com`, `api.redgifs.com`,
   `external-preview.redd.it`, `reddit.com`, `v.redd.it`, `i.redd.it`, `redgifs.com`.
- `--null-method` observed|sampled_pairs : How to derive threshold from observed pairs (sample or complete data).
- `--sample-size` INT : When using `sampled_pairs` the number of observed pairs to sample.
- `--seed` INT : RNG seed for sampling.
- `--percentile` INT : Percentile for threshold (default 99).

Filtering pipeline
- Row-level validation: invalid JSON rows are skipped.
- Author validation: rows with missing authors or `[deleted]` authors are skipped.
- Post-type validation: self posts are skipped for the post loader, where is_self = true (meaning URLs are not external, rather links the post itself). 
- Domain exclusion: `--exclude-domain` removes tokens from blocked domains before they are added.
- Duplicate URL suppression: URLs extracted from `title`/`selftext` are skipped if they match the post’s main token.
- Author count filtering: authors with `<= --min-urls` tokens are removed, using `--count-mode` to choose unique vs total counting.
- TF-IDF vocabulary pruning: `--min-df` and `--max-df` prune rare/common tokens from the vector space.
- TF-IDF row pruning: authors with fewer than 2 non-zero TF-IDF features after pruning are removed.
- Threshold sampling: `--null-method sampled_pairs` uses only a sampled subset of observed author pairs to estimate the percentile threshold.

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
