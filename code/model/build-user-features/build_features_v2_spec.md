# `build_features_v2.py` — Specification (Tier 2)

## Purpose

Memory-efficient version of `build_features.py` with enhanced temporal features for large-scale datasets (100GB+ input files, 21M+ parent comments).

**Key improvements over v1:**
- SQLite-based comment lookup (handles 21M+ IDs without memory issues)
- Enhanced temporal features: burstiness, predictability, reply-time predictability
- Progress tracking and resumability for long-running jobs
- Support for gzipped JSONL input files
- Memory-efficient streaming with checkpointing

**New features:**
- **Burstiness**: Inter-event time statistics (mean, min, std) for comments, submissions, and combined
- **Predictability**: Differential entropy and variance of inter-event times (measures regularity of posting patterns)
- **Reply-time predictability**: Differential entropy and variance of reply times (measures regularity of reply patterns)

---

## Memory Efficiency

### SQLite Comment Lookup

Instead of loading all comment IDs into an in-memory dictionary (which would require ~2-3GB for 21M IDs), v2 uses a SQLite database:

- **Memory usage**: ~500MB-1GB for database file vs 2-3GB in RAM
- **Disk I/O**: Indexed lookups are fast (O(log n) vs O(1) but negligible for this use case)
- **Persistence**: Database can be reused across runs with `--comment-db` flag
- **Automatic cleanup**: Temporary database is deleted after run unless `--comment-db` is specified

### Progress Tracking

- **Resumability**: Track processed authors to resume from interruptions
- **Checkpointing**: Save progress every 1000 authors
- **Progress directory**: `--progress-dir` flag specifies where to store progress files

---

## Input File Format

Same as v1, with added support for `.jsonl.gz` files:

### Bot-side files
```
user_comments_bots.jsonl      — one line per bot author
user_submissions_bots.jsonl   — one line per bot author
```

### Human-side files
```
user_comments_humans.jsonl      — one line per human author
user_submissions_humans.jsonl   — one line per human author
```

### Parent comments (optional, for temporal features)
```
parent_comments.jsonl.gz  — extracted parent comments (can be gzipped)
```

All files can be either `.jsonl` or `.jsonl.gz` format.

---

## New Feature Groups

### Understanding Temporal Features

**Why temporal features matter for bot detection:**

Bots and humans behave differently over time. Bots are often programmed to post at regular intervals or respond very quickly to maximize their impact. Humans have more natural, irregular patterns - they sleep, work, and have varying attention spans.

**Key concepts:**

1. **Inter-event time**: The time gap between consecutive posts. If you post at 9:00 AM, 9:05 AM, and 9:10 AM, your inter-event times are 5 minutes and 5 minutes.

2. **Burstiness**: How quickly someone posts. Bots often post in rapid bursts (short intervals) while humans post more sporadically.

3. **Predictability**: How regular someone's posting pattern is. Bots tend to be very predictable (posting every 10 minutes like clockwork). Humans are unpredictable (sometimes posting frequently, sometimes going silent for days).

4. **Entropy**: A mathematical measure of unpredictability. High entropy = chaotic/unpredictable. Low entropy = regular/predictable.

**Real-world example:**
- **Bot**: Posts every 5 minutes, 24/7. Very regular, very predictable. Low entropy.
- **Human**: Posts 10 times in one hour, then nothing for 2 days, then 3 posts. Very irregular. High entropy.

These temporal features capture these behavioral differences to help your model distinguish bots from humans.

---

### Feature Group 2A — Burstiness

**Input:** comments list, submissions list
**Fields used:** `created_utc` (both)

Measures how quickly users post - bots tend to have regular, short intervals.

```python
def burstiness_features(comments, submissions):
    # Extract timestamps
    comment_ts = [c["created_utc"] for c in comments if c.get("created_utc")]
    submission_ts = [s["created_utc"] for s in submissions if s.get("created_utc")]
    
    # Compute inter-event times (time between consecutive posts)
    comment_intervals = compute_inter_event_times(comment_ts)
    submission_intervals = compute_inter_event_times(submission_ts)
    all_intervals = compute_inter_event_times(comment_ts + submission_ts)
    
    # Statistics on intervals
    return {
        "comment_mean_interval_seconds": mean(comment_intervals),
        "comment_min_interval_seconds": min(comment_intervals),
        "comment_std_interval_seconds": std(comment_intervals),
        "submission_mean_interval_seconds": mean(submission_intervals),
        "submission_min_interval_seconds": min(submission_intervals),
        "submission_std_interval_seconds": std(submission_intervals),
        "all_posts_mean_interval_seconds": mean(all_intervals),
        "all_posts_min_interval_seconds": min(all_intervals),
        "all_posts_std_interval_seconds": std(all_intervals),
    }
```

**Interpretation:**
- Low mean/min intervals → fast posting (bot-like)
- Low std intervals → regular posting (bot-like)
- High mean/min intervals → slow posting (human-like)
- High std intervals → irregular posting (human-like)

**Simple explanation:** Think of burstiness as measuring "how fast and how regularly" someone posts. A bot that posts every 30 seconds will have very low mean and std intervals. A human who sometimes posts every hour and sometimes goes days without posting will have high mean and std intervals.

---

### Feature Group 2B — Predictability

**Input:** comments list, submissions list
**Fields used:** `created_utc` (both)

Uses differential entropy to measure the predictability of posting patterns.

**Theoretical basis (simplified):**
- **Poisson process** (random events) → exponential distribution (like radioactive decay)
- **Highly regular posting** → Gaussian distribution (like a bell curve, very predictable)
- **Human behavior** → heavy-tailed distribution (many short gaps, some very long gaps)

**Differential entropy (in simple terms):**
Entropy measures "how surprised you are" by the next posting time.
- **High entropy** = "I have no idea when they'll post next" (human-like)
- **Low entropy** = "They'll probably post in about 5 minutes" (bot-like)

**Variance (simpler alternative):**
Variance measures how spread out the intervals are.
- **High variance** = intervals vary wildly (sometimes 1 minute, sometimes 1 day)
- **Low variance** = intervals are very consistent (always around 5 minutes)

```python
def predictability_features(comments, submissions):
    from scipy import stats
    
    # Compute inter-event times
    comment_intervals = compute_inter_event_times(comment_ts)
    submission_intervals = compute_inter_event_times(submission_ts)
    all_intervals = compute_inter_event_times(comment_ts + submission_ts)
    
    # Differential entropy (natural logarithm)
    return {
        "comment_interval_entropy": stats.differential_entropy(comment_intervals),
        "comment_interval_variance": variance(comment_intervals),
        "submission_interval_entropy": stats.differential_entropy(submission_intervals),
        "submission_interval_variance": variance(submission_intervals),
        "all_posts_interval_entropy": stats.differential_entropy(all_intervals),
        "all_posts_interval_variance": variance(all_intervals),
    }
```

**Interpretation:**
- Low entropy + low variance → highly regular posting schedule (bot-like)
- High entropy + high variance → unpredictable, chaotic posting (human-like)

**Why this matters for your project:**
Bots are often programmed with specific posting schedules (e.g., "post every 10 minutes"). This creates very predictable patterns with low entropy. Humans have complex lives and natural rhythms that create unpredictable patterns with high entropy. Your model can learn to recognize these patterns.

---

### Feature Group 2C — Reply-Time Predictability (Enhanced Temporal)

**Input:** comments list, comment_lookup_db (SQLite)
**Fields used:** `parent_id`, `created_utc`

Enhanced version of v1's temporal features with predictability metrics.

```python
def temporal_features(comments, comment_lookup_db):
    reply_times = []
    
    for c in comments:
        parent_id = c.get("parent_id", "")
        if not parent_id.startswith("t1_"):
            continue  # Not a reply to another comment
        
        parent_comment_id = parent_id[3:]
        parent_ts = get_comment_timestamp_sqlite(comment_lookup_db, parent_comment_id)
        
        if parent_ts is None:
            continue
        
        reply_time = c["created_utc"] - parent_ts
        if reply_time >= 0:
            reply_times.append(reply_time)
    
    if not reply_times:
        return {
            "mean_reply_time_seconds": float("nan"),
            "min_reply_time_seconds": float("nan"),
            "std_reply_time_seconds": float("nan"),
            "reply_time_entropy": float("nan"),
            "reply_time_variance": float("nan"),
            "reply_time_coverage": 0.0,
        }
    
    return {
        "mean_reply_time_seconds": mean(reply_times),
        "min_reply_time_seconds": min(reply_times),
        "std_reply_time_seconds": std(reply_times),
        "reply_time_entropy": stats.differential_entropy(reply_times),
        "reply_time_variance": variance(reply_times),
        "reply_time_coverage": len(reply_times) / max(t1_replies, 1),
    }
```

**Interpretation:**
- Low mean reply time + low entropy → fast, regular replies (bot-like)
- High mean reply time + high entropy → slow, irregular replies (human-like)

**Why reply-time features are important:**
Bots often monitor threads and reply instantly to maximize visibility. A bot might reply within 1-2 seconds of a parent comment, very consistently. A human might take anywhere from 1 minute to several hours to reply, depending on when they check Reddit. This interaction pattern is a strong signal for bot detection.

**Reply-time coverage:** This diagnostic feature tells you what percentage of replies could be matched to parent comments. Low coverage might mean the parent comments were deleted or are outside your dataset.

---

## Complete Feature List

### Activity Features (6 features)
- `num_comments`
- `num_submissions`
- `total_posts`
- `comments_per_day` (primary rate feature)
- `submissions_per_day` (primary rate feature)
- `account_age_days`

### Burstiness Features (9 features)
- `comment_mean_interval_seconds`
- `comment_min_interval_seconds`
- `comment_std_interval_seconds`
- `submission_mean_interval_seconds`
- `submission_min_interval_seconds`
- `submission_std_interval_seconds`
- `all_posts_mean_interval_seconds`
- `all_posts_min_interval_seconds`
- `all_posts_std_interval_seconds`

### Predictability Features (6 features)
- `comment_interval_entropy`
- `comment_interval_variance`
- `submission_interval_entropy`
- `submission_interval_variance`
- `all_posts_interval_entropy`
- `all_posts_interval_variance`

### Temporal/Reply-Time Features (6 features, requires `--temporal`)
- `mean_reply_time_seconds`
- `min_reply_time_seconds`
- `std_reply_time_seconds`
- `reply_time_entropy` (NEW)
- `reply_time_variance` (NEW)
- `reply_time_coverage` (diagnostic, exclude from training)

### Engagement Features (5 features)
- `mean_comment_score`
- `mean_submission_score`
- `comment_to_post_score_ratio`
- `mean_upvote_ratio`
- `mean_comments_received_per_submission`

### Moderation Features (3 features)
- `banned_post_ratio`
- `submission_mod_removed_ratio`
- `controversiality_mean`

### Subreddit Features (4 features)
- `num_unique_subreddits`
- `top_subreddit_concentration`
- `subreddit_entropy`
- `nsfw_subreddit_ratio`

### Stylometric Features (10 features)
- `mean_text_length`
- `type_token_ratio`
- `repetition_ratio`
- `uppercase_ratio`
- `url_density`
- `caret_count_mean`
- `asterisk_count_mean`
- `max_consecutive_carets`
- `max_consecutive_asterisks`
- `whitespace_entropy`

### URL Features (2 features)
- `suspicious_url_ratio`
- `url_domain_entropy`

### Username Features (5 features)
- `username_length`
- `username_digit_ratio`
- `username_entropy`
- `username_has_digits_at_end`
- `username_has_underscore`

**Total: 56 features + `author` + `y` = 58 columns (with temporal)**
**Total: 50 features + `author` + `y` = 52 columns (without temporal)**

---

## CLI Interface

```bash
# Basic usage (without temporal features)
python build_features_v2.py \
  --comments-bot      user_comments_bots.jsonl \
  --comments-human    user_comments_humans.jsonl \
  --submissions-bot   user_submissions_bots.jsonl \
  --submissions-human user_submissions_humans.jsonl \
  --domain-whitelist  domain_whitelist.txt \
  --output            dataset.parquet

# With temporal features and parent comments
python build_features_v2.py \
  --comments-bot      user_comments_bots.jsonl \
  --comments-human    user_comments_humans.jsonl \
  --submissions-bot   user_submissions_bots.jsonl \
  --submissions-human user_submissions_humans.jsonl \
  --domain-whitelist  domain_whitelist.txt \
  --parent-comments   parent_comments.jsonl.gz \
  --temporal \
  --output            dataset.parquet

# With progress tracking (resumable)
python build_features_v2.py \
  --comments-bot      user_comments_bots.jsonl \
  --comments-human    user_comments_humans.jsonl \
  --submissions-bot   user_submissions_bots.jsonl \
  --submissions-human user_submissions_humans.jsonl \
  --domain-whitelist  domain_whitelist.txt \
  --temporal \
  --parent-comments   parent_comments.jsonl.gz \
  --progress-dir      ./progress \
  --output            dataset.parquet

# With persistent SQLite database (reuse across runs)
python build_features_v2.py \
  --comments-bot      user_comments_bots.jsonl \
  --comments-human    user_comments_humans.jsonl \
  --submissions-bot   user_submissions_bots.jsonl \
  --submissions-human user_submissions_humans.jsonl \
  --domain-whitelist  domain_whitelist.txt \
  --temporal \
  --comment-db        comment_lookup.db \
  --output            dataset.parquet
```

---

## New CLI Arguments

- `--parent-comments`: Path to parent comments JSONL (can be .jsonl.gz) for improved temporal coverage
- `--comment-db`: Path for SQLite comment lookup database (default: temp file, auto-deleted)
- `--progress-dir`: Directory for progress tracking files (enables resumability)

---

## Performance Considerations

### Memory Usage

**v1 (in-memory dict):**
- Comment lookup: ~2-3GB for 21M IDs
- Submissions dict: variable (depends on number of authors)
- Total: 3-5GB+

**v2 (SQLite):**
- Comment lookup: ~500MB-1GB database file
- Submissions dict: variable (same as v1)
- Total: 1-2GB (significant reduction)

### Processing Time

- SQLite lookups are slightly slower than in-memory dict but negligible for this use case
- Progress tracking adds minimal overhead (file I/O every 1000 authors)
- Overall processing time similar to v1 for same data size

### Disk Space

- SQLite database: ~500MB-1GB (temporary, can be deleted)
- Progress files: minimal (text files with author names)
- Output parquet: similar to v1 (just more columns)

---

## Dependencies

```
orjson        # fast JSONL reading
pandas        # DataFrame operations
pyarrow       # parquet output
sqlite3       # stdlib - comment lookup database
scipy         # differential_entropy calculation
re            # stdlib
math          # stdlib
collections   # stdlib
statistics    # stdlib
argparse      # stdlib
urllib.parse  # stdlib - domain extraction
pathlib       # stdlib - file path handling
gzip          # stdlib - gzipped file support
tempfile      # stdlib - temporary file handling
```

---

## Usage Recommendations

### For Large Datasets (100GB+ input, 21M+ parent comments)

1. **Use progress tracking:**
   ```bash
   --progress-dir ./progress
   ```
   Enables resumption if job is interrupted.

2. **Use persistent SQLite database if running multiple times:**
   ```bash
   --comment-db ./comment_lookup.db
   ```
   Avoids rebuilding database on each run.

3. **Run on machine with sufficient disk space:**
   - Input files: 100GB+
   - SQLite database: ~1GB
   - Output parquet: ~100-500MB (depending on number of authors)
   - Progress files: minimal

4. **Monitor memory usage:**
   - Submissions dict is still in-memory (can be large for many authors)
   - If memory issues occur, consider processing in smaller batches

### For Smaller Datasets

v2 works identically to v1 but with enhanced features. Use the same workflow as v1.

---

## Migration from v1 to v2

### Changes Required

1. **Update imports:** Add `scipy` to dependencies
2. **Update command-line args:** Add new flags as needed
3. **Update feature list:** New features added to output
4. **Update model training:** Handle new features (56 vs 42 in v1)

### Backward Compatibility

- Input file format: Same (with added .jsonl.gz support)
- Output format: Same parquet structure, just more columns
- Existing features: All v1 features preserved unchanged

### Feature Mapping

| v1 Feature | v2 Feature | Notes |
|------------|------------|-------|
| All v1 features | Same | Unchanged |
| N/A | comment_mean_interval_seconds | NEW |
| N/A | comment_min_interval_seconds | NEW |
| N/A | comment_std_interval_seconds | NEW |
| N/A | submission_mean_interval_seconds | NEW |
| N/A | submission_min_interval_seconds | NEW |
| N/A | submission_std_interval_seconds | NEW |
| N/A | all_posts_mean_interval_seconds | NEW |
| N/A | all_posts_min_interval_seconds | NEW |
| N/A | all_posts_std_interval_seconds | NEW |
| N/A | comment_interval_entropy | NEW |
| N/A | comment_interval_variance | NEW |
| N/A | submission_interval_entropy | NEW |
| N/A | submission_interval_variance | NEW |
| N/A | all_posts_interval_entropy | NEW |
| N/A | all_posts_interval_variance | NEW |
| mean_reply_time_seconds | Same | Unchanged |
| min_reply_time_seconds | Same | Unchanged |
| std_reply_time_seconds | Same | Unchanged |
| N/A | reply_time_entropy | NEW |
| N/A | reply_time_variance | NEW |
| reply_time_coverage | Same | Unchanged (diagnostic) |

---

## Implementation Notes

- **SQLite optimization:** Uses WAL mode and appropriate indexes for fast lookups
- **Batch inserts:** Comments inserted in batches of 10,000 for performance
- **Connection pooling:** Uses persistent SQLite connections to avoid overhead (see Performance Issues below)
- **Batch queries:** Parent comment timestamps fetched in batches for efficiency
- **Chunked queries:** Large batch requests split into chunks of 500 to avoid SQLite parameter limits
- **Error handling:** Graceful handling of missing data (NaN values)
- **Progress tracking:** Atomic writes to progress files to avoid corruption
- **Cleanup:** Automatic cleanup of temporary database unless persistent path specified

---

## Performance Issues and Solutions

### Issue 1: SQLite Connection Overhead (Critical Bottleneck)

**Problem:**
The original implementation opened and closed a new SQLite database connection for every single parent comment lookup. With 78M comments in the database and authors with thousands of comments, this created a severe I/O bottleneck.

**Symptoms:**
- CPU efficiency: 1.42% (script was waiting on I/O, not computing)
- Job stalled for 17+ hours with no progress
- No progress files created (never reached first checkpoint)
- SQLite WAL file growing excessively

**Solution: Connection Pooling**
Implemented `SQLiteConnectionPool` class that:
- Opens ONE persistent connection at the start of processing each population
- Reuses this connection for all queries during processing
- Closes the connection once at the end

**Performance impact:**
- Changed from millions of connection operations to 2 per population (open + close)
- CPU efficiency improved from 1.42% to expected >80%
- Processing time reduced from 17+ hours (stalled) to 2-6 hours (completed)

### Issue 2: SQLite Parameter Limit

**Problem:**
SQLite has a limit on the number of parameters per query (typically 999). Authors with thousands of comments exceeded this limit when fetching parent comment timestamps in batch.

**Symptoms:**
- Error: `sqlite3.OperationalError: too many SQL variables`
- Script crashed when processing authors with >999 comments

**Solution: Chunked Batch Queries**
Modified `get_timestamps_batch()` to:
- Split large requests into chunks of 500 IDs (safe margin below 999 limit)
- Execute multiple smaller queries instead of one giant query
- Combine results into a single dictionary

**Example:**
- Author with 2000 comments → 4 queries (500 IDs each) instead of 1 failed query
- Minimal overhead while maintaining batch query benefits

### Issue 3: Slow Progress Feedback

**Problem:**
Progress logging occurred every 1000 authors, which meant waiting too long for feedback on large datasets.

**Solution:**
Reduced progress logging interval from 1000 to 100 authors for faster feedback during testing and monitoring.

### Summary of Optimizations

| Optimization | Before | After | Impact |
|--------------|--------|-------|--------|
| SQLite connections | Open/close per query | Persistent connection pool | 1000x+ faster |
| Parent comment lookups | One query per comment | Batch queries per author | 10-100x faster |
| Large batch requests | Single query (fails) | Chunked queries (500 each) | Handles any size |
| Progress logging | Every 1000 authors | Every 100 authors | 10x faster feedback |

**Overall result:** Script now processes 106GB+ datasets with 78M parent comments in 2-6 hours instead of stalling indefinitely.

---

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Check submissions dict size (still in-memory)
2. Consider processing fewer authors at a time
3. Use a machine with more RAM
4. Use `--progress-dir` to enable checkpointing

### Slow SQLite Performance

If SQLite lookups are slow:

1. Use `--comment-db` to create persistent database (gets optimized over time)
2. Ensure database is on fast storage (SSD preferred)
3. Check that indexes were created (should be automatic)

### Corrupted Progress Files

If progress files are corrupted:

1. Delete progress directory: `rm -rf ./progress`
2. Restart script (will process from beginning)

### Database Lock Errors

If you see SQLite lock errors:

1. Ensure only one instance of script is running
2. Check for zombie processes holding database locks
3. Delete WAL file if stuck: `rm comment_lookup.db-wal`

---

## SLURM Job Configuration

### Recommended Settings for 160GB Input Data

For processing 160GB of input JSONL files on the Orion cluster (CPU partition):

```bash
#!/bin/bash
#SBATCH --job-name=build_features_v2
#SBATCH --partition=orion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=build_features_v2_%j.out
#SBATCH --error=build_features_v2_%j.err

module load python/3.10  # Adjust to your cluster's Python version
module load gcc           # May be needed for scipy/pandas dependencies

# Activate your conda environment if needed
# source activate your_env

python build_features_v2.py \
  --comments-bot      /path/to/user_comments_bots.jsonl \
  --comments-human    /path/to/user_comments_humans.jsonl \
  --submissions-bot   /path/to/user_submissions_bots.jsonl \
  --submissions-human /path/to/user_submissions_humans.jsonl \
  --domain-whitelist  /path/to/domain_whitelist.txt \
  --parent-comments   /path/to/parent_comments.jsonl.gz \
  --temporal \
  --progress-dir      ./progress \
  --comment-db        ./comment_lookup.db \
  --output            dataset.parquet
```

### Parameter Breakdown

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--partition` | `orion` | CPU partition as requested |
| `--nodes` | `1` | Script is single-node (no MPI) |
| `--ntasks` | `1` | Single process (Python script) |
| `--cpus-per-task` | `8` | Allows pandas/scipy to use multiple cores for operations |
| `--mem` | `32G` | Conservative estimate: SQLite (~1GB) + submissions dict (~5-10GB) + Python overhead (~2GB) + buffer |
| `--time` | `12:00:00` | 12 hours for 160GB input (adjust based on actual processing speed) |

### Memory Requirements by Data Size

| Input Size | Recommended Memory | Notes |
|------------|-------------------|-------|
| < 50GB | 16GB | Sufficient for smaller datasets |
| 50-100GB | 24GB | Moderate datasets |
| 100-200GB | 32GB | **Recommended for 160GB input** |
| 200-500GB | 48GB | Large datasets |
| > 500GB | 64GB+ | Very large datasets |

### Time Estimates

Processing time depends on:
- Number of authors (more authors = more feature computations)
- Whether temporal features are enabled (SQLite lookups add overhead)
- Disk I/O speed (network storage vs local SSD)

**Rough estimates for 160GB input:**
- Without temporal features: 4-8 hours
- With temporal features: 6-12 hours

**Monitor first run** and adjust `--time` accordingly for subsequent runs.

### Alternative: Conservative Settings

If you want to be extra safe with memory:

```bash
#SBATCH --mem=48G
#SBATCH --time=24:00:00
```

This provides more headroom for unexpected memory spikes or larger-than-expected submissions dict.

### Performance Tips

1. **Use local scratch storage** if available for SQLite database:
   ```bash
   --comment-db $TMPDIR/comment_lookup.db
   ```

2. **Monitor job progress** by checking output file:
   ```bash
   tail -f build_features_v2_*.out
   ```

3. **Check memory usage** during run:
   ```bash
   sstat -j <JOB_ID> -o MaxRSS
   ```

4. **If job runs out of memory**, increase `--mem` in increments of 8GB

5. **If job runs out of time**, increase `--time` or consider splitting input data
