# src_3 Debug Harness

This folder contains a controlled debugging path for the semantic similarity pipeline.

## Why this exists

The production detection stage in `src_2` uses:

1. multilingual sentence embeddings
2. random-hyperplane LSH to generate candidate buckets
3. FAISS exact inner-product search inside each bucket
4. a similarity threshold to decide which pairs are suspicious

That is useful for scale, but it makes debugging hard when false positives appear in the high 0.80 range. This `src_3` harness removes LSH and evaluates exact cosine similarity on a tiny labeled sample so you can isolate the source of the error.

## What the scripts do

`debug_semantic_similarity.py`:

1. loads a CSV with labeled text pairs
2. embeds every unique text with the multilingual SentenceTransformer model
3. computes exact cosine similarity for every labeled pair
4. compares those scores against a chosen threshold
5. sweeps a range of thresholds and reports precision, recall, F1, and confusion counts
6. writes review files so you can inspect false positives and false negatives directly

`debug_faiss_range_search.py`:

1. loads the same labeled CSV format
2. embeds all unique texts (normalized vectors)
3. builds one global FAISS `IndexFlatIP` (no LSH buckets)
4. runs `range_search(query, radius_threshold)` to return all neighbors above threshold
5. annotates returned neighbors with known labels when those pairs exist in your labeled set
6. exports neighbor tables and threshold metrics for inspection

## Input format

The input CSV must contain:

- `text_a`
- `text_b`
- `label`

Optional columns:

- `pair_id`
- `note`

Label meaning:

- `1` = should be similar
- `0` = should not be similar

A template is provided in [sample_labeled_pairs.csv](sample_labeled_pairs.csv).

## Run

```bash
python code/src_3/debug_semantic_similarity.py --input-file code/src_3/sample_labeled_pairs.csv --output-dir code/src_3/output
```

```bash
python code/src_3/debug_faiss_range_search.py --input-file code/src_3/sample_labeled_pairs.csv --output-dir code/src_3/output/faiss_range --radius-threshold 0.80
```

## Output files

The script writes:

- `debug_pair_scores.csv`: exact similarity for every labeled pair
- `debug_threshold_sweep.csv`: threshold sweep results
- `debug_pair_review.csv`: error-focused review table
- `debug_similarity_distribution.png`: score distribution plot
- `debug_summary.json`: primary metrics and best threshold summary

The FAISS range-search script writes:

- `faiss_labeled_pair_scores.csv`: exact scores/metrics for labeled pairs at current threshold
- `faiss_range_neighbors.csv`: all neighbors returned by FAISS range search
- `faiss_range_neighbors_labeled_only.csv`: only returned neighbors that also exist in labeled pairs
- `faiss_query_summary.csv`: neighbor-count summary per query text
- `faiss_range_summary.json`: run settings and aggregate metrics

## How to use this for diagnosis

Start with a tiny sample where you already know the expected behavior.

1. If unrelated pairs still score very high here, the embedding model or text normalization is the likely problem.
2. If the exact scores look reasonable here but production still gives bad matches, the issue is likely in LSH candidate generation or the current thresholding strategy.
3. If the best threshold is far above the current production threshold, the threshold calibration is too permissive.

## Production script: bot_detection_faiss_no_lsh.py

Once you validate on the small sample, the production script `bot_detection_faiss_no_lsh.py` uses the same FAISS global range-search logic on the full sampled dataset.

### Key differences from src_2/bot_detection_semantic.py:

- **No LSH**: Removes random-hyperplane bucketing entirely
- **Global FAISS**: Builds one exact IndexFlatIP over all embeddings
- **Simpler**: Fewer hyperparameters and simpler code path
- **99% percentile threshold**: Maintained (dynamically calculated from random pairs, not fixed)
- **Same outputs**: Identical CSV/JSONL format for downstream compatibility

### Run

```bash
python code/src_3/bot_detection_faiss_no_lsh.py \
  --input-file sampled_data/sample.jsonl \
  --output-dir code/src_3/output \
  --quantile 0.99 \
  --embedding-model distiluse-base-multilingual-cased-v2
```

### Key CLI arguments

- `--input-file`: path to sampled JSONL
- `--output-dir`: directory for outputs (suspicious_pairs.csv, suspicious_accounts.csv, suspicious_posts.jsonl, detection_summary.json)
- `--quantile`: percentile for threshold (default 0.99 = 99th percentile)
- `--embedding-model`: SentenceTransformer model name (default distiluse-base-multilingual-cased-v2)
- `--random-pairs`: how many random pairs to sample for threshold estimation (default 50000)
- `--min-cross-account-matches`: minimum suspicious links to flag an account (default 1)
- `--min-words`: minimum words per post (default 10)

### Outputs

Same as src_2:

- `suspicious_pairs.csv`: cross-account pairs above threshold
- `suspicious_accounts.csv`: aggregated account-level suspiciousness
- `suspicious_posts.jsonl`: original JSON for flagged authors (feeds into clustering)
- `detection_summary.json`: metadata and diagnostics
- `random_similarity_kde.png`: visualization of 99th percentile threshold

### Comparison: debug scripts vs production script

| Feature | debug_semantic_similarity | debug_faiss_range_search | bot_detection_faiss_no_lsh |
|---------|---------------------------|--------------------------|----------------------------|
| Input | Labeled CSV pairs | Labeled CSV pairs | Full JSONL sample |
| Purpose | Validate raw pair scores | Validate retrieval | Full detection pipeline |
| Scale | Tiny (10s pairs) | Tiny (10s pairs) | Large (10000s posts) |
| Use case | Diagnosis | Diagnosis | Production |
| Threshold | User-specified | User-specified | Auto-calculated (99%) |

## Next steps

1. Use `debug_semantic_similarity.py` to validate raw model scoring on known pairs.
2. Use `debug_faiss_range_search.py` to validate retrieval behavior without LSH.
3. Run `bot_detection_faiss_no_lsh.py` on your full sample and compare results against `src_2`.
4. Check if false positives are reduced by removing LSH candidate generation.
5. If results improve, consider replacing src_2 with this approach.


## Scripts A–E in this folder

This repository now includes a set of diagnostic scripts (A–E) useful for
zeroing in on where retrieval or scoring differences arise.

- Script A: `script_a_compare_small_sklearn_faiss.py` (small-sample exact comparison)
  - Purpose: Compare sklearn cosine_similarity (pairwise exact) against FAISS
    `IndexFlatIP` global range_search on a labeled CSV of pairs.
  - Outputs: per-pair CSV with both scores, metric summary CSV/JSON for threshold evaluation.

- Script B: `script_b_bot_detection_lsh_faiss_range.py` (LSH + FAISS range per-bucket)
  - Purpose: Production-style pipeline that uses LSH buckets but runs
    `IndexFlatIP.range_search` inside each bucket (reduces candidate growth).
  - Outputs: `suspicious_pairs_range_lsh.csv`, `suspicious_accounts_range_lsh.csv`, JSON summary, KDE plot.

- Script C: `script_c_compare_topk_vs_range.py` (added)
  - Purpose: Run both the original top-k-in-bucket retrieval (as in
    `src_2/bot_detection_semantic.py`) and the LSH+range_search method
    on the exact same embeddings and LSH tables; compare which pairs each
    method produces and whether shared pairs have score differences.
  - Outputs: `pair_presence_comparison.csv`, `score_differences_both_methods.csv`, sample CSVs, `comparison_summary.json`.

- Script D: `script_d_indexlsh_vs_indexflatip.py` (added)
  - Purpose: Directly compare FAISS `IndexLSH` neighbor sets to `IndexFlatIP`
    top-k neighbors on a sampled subset (IndexLSH uses LSH-bit similarity
    and does not natively provide cosine scores).
  - Outputs: `per_query_summary.csv`, `mismatch_samples.csv`, `lsh_vs_flatip_summary.json`.

- Script E: `script_e_threshold_range_flatip_vs_lsh_rescore.py` (new redesign)
  - Purpose: Threshold-based comparison that matches detection semantics:
    1) exact `IndexFlatIP.range_search(radius=threshold)` baseline and
    2) `IndexLSH.search(k)` candidate generation followed by manual cosine re-score
    and threshold filter.
  - Also computes threshold from random-pair quantile and writes a KDE plot.
  - Outputs: `random_similarity_samples.csv`, `random_similarity_kde.png`,
    `per_query_threshold_comparison.csv`, `overlap_score_differences.csv`,
    `false_negative_examples.csv`, `false_positive_examples.csv`,
    `pairs_full_comparison.csv` (all pairs with full text and both scores),
    `threshold_range_summary.json`.

- Script F: `script_f_bucketed_threshold_exact_vs_topk.py` (production-bucketing comparison)
  - Purpose: Use the same explicit random-hyperplane LSH buckets as the production pipeline,
    then compare within-bucket exact `IndexFlatIP.range_search(radius=threshold)` against
    within-bucket top-k retrieval + manual cosine re-score + threshold filter.
  - Important: this script does **not** use FAISS `IndexLSH`; the bucketing itself is the LSH stage.
  - Outputs: `random_similarity_samples.csv`, `random_similarity_kde.png`,
    `bucket_pair_comparison.csv` (full text + both scores), `per_query_bucket_comparison.csv`,
    `overlap_score_differences.csv`, `false_negative_examples.csv`, `false_positive_examples.csv`,
    `exact_bucket_stats.json`, `topk_bucket_stats.json`, `bucket_summary.json`.

Use Scripts C, D, E, and F when you need to answer the professor's question about
whether different FAISS index types and retrieval semantics produce the same
neighbor sets and scores. Run them on a representative sample (1k–5k rows)
before attempting the full 30k run to save time.

## Example run commands

These examples assume you have a Python environment with `faiss`, `sentence-transformers`, `pandas`, and `numpy` installed (we recommend using the project's `scamdetect` conda env).

Activate your environment (example):

```bash
conda activate scamdetect
```

Script C (compare top-k vs range inside LSH buckets):

```bash
python code/src_3/script_c_compare_topk_vs_range.py \
  --input-file sampled_data/sample.jsonl \
  --output-dir code/src_3/output/script_c_run \
  --embedding-model distiluse-base-multilingual-cased-v2 \
  --nbits 32 --ntables 4 --k 50 --threshold 0.80
```

Script D (compare IndexLSH vs IndexFlatIP on a sampled subset):

```bash
python code/src_3/script_d_indexlsh_vs_indexflatip.py \
  --input-file sampled_data/sample.jsonl \
  --output-dir code/src_3/output/script_d_run \
  --embedding-model distiluse-base-multilingual-cased-v2 \
  --sample-size 2000 --nbits 32 --k 10
```

Script E (threshold/range redesign: random-pair threshold + KDE + exact-vs-LSH comparison):

```bash
python code/src_3/script_e_threshold_range_flatip_vs_lsh_rescore.py \
  --input-file sampled_data/sample.jsonl \
  --output-dir code/src_3/output/script_e_run \
  --embedding-model distiluse-base-multilingual-cased-v2 \
  --sample-size 2000 --random-pairs 50000 --quantile 0.99 \
  --nbits 32 --lsh-candidate-k 200
```

Script F (bucketed threshold comparison using the production LSH bucketing stage):

```bash
python code/src_3/script_f_bucketed_threshold_exact_vs_topk.py \
  --input-file sampled_data/sample_comments_2024.jsonl \
  --output-dir code/src_3/output/script_f_run \
  --embedding-model distiluse-base-multilingual-cased-v2 \
  --sample-size 0 --lsh-bits 16 --lsh-tables 4 --faiss-top-k 40 \
  --random-pairs 50000 --quantile 0.999
```

Quick production-style runs

Run the LSH+range Search pipeline (Script B) on the full sample:

```bash
python code/src_3/script_b_bot_detection_lsh_faiss_range.py \
  --input-file sampled_data/sample.jsonl \
  --output-dir code/src_3/output/bot_detection_lsh_range \
  --embedding-model distiluse-base-multilingual-cased-v2 --quantile 0.99
```

Run the global FAISS production script (no LSH):

```bash
python code/src_3/bot_detection_faiss_no_lsh.py \
  --input-file sampled_data/sample.jsonl \
  --output-dir code/src_3/output/bot_detection_no_lsh \
  --quantile 0.99
```

## Static checks (quick)

Before running heavy diagnostics, run a quick syntax/import check on the new scripts:

```bash
python -m py_compile code/src_3/script_c_compare_topk_vs_range.py
python -m py_compile code/src_3/script_d_indexlsh_vs_indexflatip.py
python -m py_compile code/src_3/script_e_threshold_range_flatip_vs_lsh_rescore.py
python -m py_compile code/src_3/script_f_bucketed_threshold_exact_vs_topk.py
```

To run a light lint (if you have `ruff` installed):

```bash
ruff check code/src_3/script_c_compare_topk_vs_range.py code/src_3/script_d_indexlsh_vs_indexflatip.py code/src_3/script_e_threshold_range_flatip_vs_lsh_rescore.py code/src_3/script_f_bucketed_threshold_exact_vs_topk.py
```

## How to interpret Script E outputs

`threshold_range_summary.json`

- `threshold`: cosine threshold estimated from random-pair quantile.
- `micro_recall_vs_exact`: out of all exact threshold neighbors from FlatIP range,
  how many were recovered by the LSH candidate + re-score path.
- `micro_precision_vs_exact`: out of all LSH+re-score threshold neighbors,
  how many are truly in the exact FlatIP threshold set.
- `mean_abs_score_diff_overlap`: cosine score drift for shared pairs; this should be
  near zero (floating-point tolerance only).

`pairs_full_comparison.csv`

- **Full pair inspection**: Every pair from either method with query text, neighbor text, exact score, LSH score, and a source label:
  - `overlap`: pair found by both methods
  - `false_negative`: pair found by exact but missing from LSH
  - `false_positive`: pair found by LSH but not in exact
- Use this to inspect concrete text examples and score comparisons.
- Sort by `abs_diff` descending to find largest score discrepancies on overlaps.
- Filter by `source='false_positive'` to inspect noisy LSH candidates.

`per_query_threshold_comparison.csv`

- Per-query `recall_vs_exact` and `precision_vs_exact` identify where LSH candidate
  retrieval misses true neighbors or introduces extras.
- High `false_negative_count` means LSH missed exact threshold neighbors for that query.
- High `false_positive_count` means LSH admitted neighbors that are not in exact range set.

`overlap_score_differences.csv`

- Row-level score check on shared neighbors.
- `abs_diff` should be tiny; if it is not, verify embeddings are normalized and no
  post-processing changed vectors between methods.

`false_negative_examples.csv` / `false_positive_examples.csv`

- These are manual inspection tables to understand failure modes by text content.
- Start with top rows to see whether misses are semantic near-duplicates, noisy text,
  or LSH hashing artifacts.

## Note: Bucketing in Script E

Script E uses pure FAISS indexes (`IndexFlatIP` and `IndexLSH`) **without explicit random-hyperplane bucketing**. This differs from `src_2/bot_detection_semantic.py`, which uses random-hyperplane LSH to partition vectors into buckets first, then runs per-bucket FAISS scoring.

**Why the difference?**
- Script E aims to isolate and compare the two FAISS index types themselves.
- If you want to compare the exact bucketing strategy from `src_2`, use Script B (`script_b_bot_detection_lsh_faiss_range.py`) or Script C.
- Script D (top-k comparison) also uses pure FAISS without bucketing.

**When to use each:**
- Script E: "Do IndexFlatIP and IndexLSH return the same neighbors under a threshold?"
- Script B/C: "Does LSH bucketing + per-bucket range_search match the original LSH+top-k approach?"
