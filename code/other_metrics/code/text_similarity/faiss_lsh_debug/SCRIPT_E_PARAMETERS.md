# Script E Parameter Recommendations for 30K Comments

## Dataset Profile
- Input JSONL: 50,000 records
- After filtering (min_words=10, skip deleted/removed): ~30,000 usable
- Expected: 2-3K unique authors

## Recommended Parameters

### For Initial Diagnostic Run (Fast, ~15 min)
```bash
--sample-size 5000          # Test with 5K first
--nbits 32                  # LSH bit length (good quality/speed tradeoff)
--lsh-candidate-k 200       # Retrieve 200 candidates before re-score
--random-pairs 50000        # You have this many, use them
--quantile 0.99             # Conservative threshold
--max-error-samples 500     # Inspect error patterns
```

### For Full Validation Run (Slower, ~45 min)
```bash
--sample-size 0             # Use all 30K (0 = don't sample)
--nbits 32
--lsh-candidate-k 200
--random-pairs 50000
--quantile 0.99
--max-error-samples 500
```

## Parameter Explanation

| Parameter | Value | Why? |
|-----------|-------|------|
| `--sample-size` | 5000 (initial) or 0 (full) | 5K is representative but fast; use 0 for final validation |
| `--nbits` | 32 | Sweet spot for 30K vectors: LSH works well, not too slow |
| `--lsh-candidate-k` | 200 | Retrieve 200 LSH candidates, then manual re-score to threshold. Balances recall vs speed. |
| `--random-pairs` | 50000 | More pairs = more stable threshold estimate. 50K is plenty for 30K vectors. |
| `--quantile` | 0.99 | 99th percentile = strict threshold for suspicious pairs (matches production) |
| `--max-error-samples` | 500 | Keep 500 FN/FP examples for manual review (up from default 300) |

## What to Inspect in Outputs

1. **threshold_range_summary.json**
   - `threshold`: the cosine cutoff (expect ~0.75-0.85 for comments)
   - `micro_recall_vs_exact`: should be >90% if LSH candidate-k is high enough
   - `micro_precision_vs_exact`: >95% means few false positives from LSH

2. **pairs_full_comparison.csv** (~5K-50K rows depending on sample size)
   - Sort by `abs_diff` desc → see score mismatches
   - Filter `source='false_positive'` → inspect noisy LSH additions
   - Filter `source='false_negative'` → see what exact method caught but LSH missed

3. **per_query_threshold_comparison.csv**
   - Check `recall_vs_exact` distribution: should be clustered high (>0.8)
   - Low recall queries → LSH bucketing limitations

## Quick Local Test (Before SLURM)

```bash
conda activate scamdetect
python code/src_3/script_e_threshold_range_flatip_vs_lsh_rescore.py \
  --input-file sampled_data/sample_comments_2024.jsonl \
  --output-dir code/src_3/output/script_e_test \
  --sample-size 1000 \
  --nbits 32 \
  --lsh-candidate-k 200 \
  --random-pairs 10000 \
  --quantile 0.99
```

This runs in ~3-5 minutes on CPU and validates setup before SLURM submission.
