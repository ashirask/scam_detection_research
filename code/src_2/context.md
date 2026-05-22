# src_2 Methodology Context

This folder contains updated pipeline scripts that implement the methodology changes discussed in this session.

## Why src_2 exists

The previous pipeline in src_1 was useful for a first pass, but we observed two practical issues:

1. Bot detection with TF-IDF produced low similarity thresholds and many noisy cross-author pairs.
2. Clustering with Qwen embeddings created repeated GPU memory pressure and required runtime memory workarounds.

The scripts in src_2 address those issues directly.

## Updated bot detection approach

File: `bot_detection_semantic.py`

### Changes from src_1

- Replaced TF-IDF vectors with multilingual semantic embeddings.
- Default embedding model is `distiluse-base-multilingual-cased-v2` to better handle non-English posts (for example Turkish and mixed-language content).
- Added Locality-Sensitive Hashing (LSH) using random hyperplanes to prune candidate pairs.
- Added FAISS nearest-neighbor search inside LSH buckets for efficient similarity retrieval.
- Kept quantile-based thresholding, but now computed over semantic cosine similarities.

### Core design

- Embeddings are normalized, so dot product equals cosine similarity.
- LSH creates candidate neighborhoods quickly without global O(n^2) comparisons.
- Multi-table LSH improves recall.
- FAISS performs fast top-k retrieval in each bucket.
- Only cross-author pairs are retained as suspicious candidates.

### Main outputs

- `suspicious_pairs.csv`
- `suspicious_accounts.csv`
- `suspicious_posts.jsonl`
- `random_similarity_kde.png`
- `detection_summary.json`

## Updated clustering approach

File: `cluster_suspicious_accounts.py`

### Changes from src_1

- Replaced Qwen embedding model with `sentence-transformers/all-mpnet-base-v2`.
- Removed fp16 and model.half() usage (not needed for this model in this workflow).
- Added long-text chunk embedding:
  - split each post into 512-token chunks,
  - embed each chunk,
  - average chunk vectors,
  - normalize final post embedding.
- Added cluster-topic mapping output so each detected cluster clearly shows which BERTopic topics it contains.

### Why chunk averaging

`all-mpnet-base-v2` has practical sequence-length limits. Instead of truncating long posts to a single 512-token window, chunk averaging preserves information across the full post and gives an "unlimited length" approximation.

### Main outputs

Per post type (`comment` and `submission` when available):

- `{post_type}_cluster_assignments.csv`
- `{post_type}_clusters_umap.png`
- `{post_type}_clusters_preview.txt`
- `{post_type}_bertopic_topics.csv`
- `{post_type}_bertopic_top_words.txt`
- `{post_type}_bertopic_doc_topics.csv`
- `{post_type}_cluster_topic_breakdown.csv`
- `{post_type}_cluster_dominant_topic.csv`

Combined:

- `suspicious_cluster_assignments_all.csv`

## Post-processing summary utility

File: `summarize_cluster_topics.py`

This script creates compact cluster-to-topic summaries and is tolerant to malformed CSV rows by extracting numeric values from row tails.

Outputs:

- `{post_type}_cluster_topic_summary.csv`
- `cluster_topic_summary_all.csv`

## Recommended dependencies for src_2

Install these into your selected environment:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `sentence-transformers`
- `transformers`
- `bertopic`
- `umap-learn`
- `faiss-cpu` (or `faiss-gpu` if your HPC environment supports it)

## Example commands

### 1) Semantic bot detection

```bash
python code/src_2/bot_detection_semantic.py \
  --input-file sampled_data/sample_20000.jsonl \
  --output-dir code/output/semantic_detection_2024 \
  --embedding-model distiluse-base-multilingual-cased-v2 \
  --quantile 0.995 \
  --lsh-bits 16 \
  --lsh-tables 4 \
  --faiss-top-k 40
```

### 2) Clustering with chunked mpnet embeddings

```bash
python code/src_2/cluster_suspicious_accounts.py \
  --input-file sampled_data/sample_comments_2024.jsonl \
  --suspicious-accounts-file code/output/comments_2024/suspicious_accounts.csv \
  --output-dir code/output/clustering_comments_2024 \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --chunk-size-tokens 512 \
  --batch-size 32 \
  --bertopic-min-topic-size 15
```

### 3) Optional compact summaries

```bash
python code/src_2/summarize_cluster_topics.py \
  --output-dir code/output/clustering_comments_2024 \
  --top-k 3
```

## Notes

- `cluster_topic_breakdown` is the direct per-cluster topic inclusion table requested in this update.
- Keep `src_1` intact for reproducibility/comparison.
- Tune `quantile`, `lsh-bits`, `lsh-tables`, and `faiss-top-k` based on dataset size and desired precision-recall tradeoff.
