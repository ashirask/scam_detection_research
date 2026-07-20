# `merge_bot_extractions.py` — Specification

## Purpose

Consolidates all Pass 2 compressed output files (`comments_bots_RC_*.jsonl.gz` and
`submissions_bots_RS_*.jsonl.gz`) into per-author aggregated records, producing:

1. `user_comments_bots.jsonl` — one JSON object per bot author with their full comment list
2. `user_submissions_bots.jsonl` — one JSON object per bot author with their full submission list

These two files are the direct inputs to `build_features.py` alongside the equivalent
human-side files. Label assignment (`y = 1` for bots, `y = 0` for humans) happens inside
`build_features.py` — not here. This script's only job is reshaping compressed per-record
files into per-author JSONL that matches the structure of the existing human files.

---

## Context: Human-side files already exist

The human population files are already available in the correct format from the earlier
random-sample extraction:

```
user_comments_humans.jsonl      ← one line per human author, all their comments
user_submissions_humans.jsonl   ← one line per human author, all their submissions
```

Each line in these files is:
```json
{"author": "username", "comments": [{full_record}, {full_record}, ...]}
{"author": "username", "submissions": [{full_record}, {full_record}, ...]}
```

This script produces bot-side files in exactly the same format so `build_features.py`
can process both populations identically.

---

## Inputs

### 1. Pass 2 output directory (`--input-dir`)
Directory containing all Pass 2 compressed output files:
```
results/
  comments_bots_RC_2024-01.jsonl.gz
  comments_bots_RC_2024-02.jsonl.gz
  ...
  comments_bots_RC_2024-12.jsonl.gz
  submissions_bots_RS_2024-01.jsonl.gz
  ...
  submissions_bots_RS_2024-12.jsonl.gz
```
Globs for `comments_bots_*.jsonl.gz` and `submissions_bots_*.jsonl.gz` separately.
All months processed together — no month-level distinction needed at this stage.

### 2. Bot authors global list (`--authors-file`)
`bot_authors_global.txt` — the capped, deduplicated author list from
`merge_pass1_authors.py`. Used only as a sanity check at the end — Pass 2 already
filtered to this set.

### 3. Minimum posts threshold (`--min-posts`, default 3)
Authors with fewer total posts across all months are dropped. Same threshold used in
Pass 1, applied again as a safety net.

### 4. Output directory (`--output-dir`)
Where output files are written. Created if it does not exist.

---

## Author Skip List (hardcoded, same as extraction script)

```python
SKIP_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}
```

Belt-and-suspenders — Pass 2 already excluded these, but guard again in case of any
edge cases in the compressed output.

---

## Reading Compressed Input Files

All input files are `.jsonl.gz`. Read with `gzip.open()` in text mode:

```python
import gzip, orjson

def stream_jsonl_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except Exception:
                continue  # skip malformed lines, count in stats
```

This is the only structural difference from the human-side files which are plain
(uncompressed) JSONL — everything else is identical.

---

## Processing Steps

### Step 1 — Stream and aggregate all comment files

```python
import glob

user_comments = {}   # {author: [full_record, ...]}

for path in sorted(glob.glob(f"{args.input_dir}/comments_bots_*.jsonl.gz")):
    print(f"  Reading {path}...")
    for record in stream_jsonl_gz(path):
        author = record.get("author", "")
        if author in SKIP_AUTHORS:
            continue
        if author not in user_comments:
            user_comments[author] = []
        user_comments[author].append(record)
        stats["total_comments"] += 1
```

`sorted()` ensures months are processed in chronological order — glob order is
filesystem-dependent and non-deterministic.

### Step 2 — Stream and aggregate all submission files

```python
user_submissions = {}   # {author: [full_record, ...]}

for path in sorted(glob.glob(f"{args.input_dir}/submissions_bots_*.jsonl.gz")):
    print(f"  Reading {path}...")
    for record in stream_jsonl_gz(path):
        author = record.get("author", "")
        if author in SKIP_AUTHORS:
            continue
        if author not in user_submissions:
            user_submissions[author] = []
        user_submissions[author].append(record)
        stats["total_submissions"] += 1
```

### Step 3 — Compute per-author post counts and apply minimum post filter

```python
all_authors = set(user_comments.keys()) | set(user_submissions.keys())

kept_authors = set()
dropped_authors = set()

for author in all_authors:
    n_comments    = len(user_comments.get(author, []))
    n_submissions = len(user_submissions.get(author, []))
    total         = n_comments + n_submissions

    if total >= args.min_posts:
        kept_authors.add(author)
    else:
        dropped_authors.add(author)
        # free memory for dropped authors
        user_comments.pop(author, None)
        user_submissions.pop(author, None)

stats["authors_kept"]    = len(kept_authors)
stats["authors_dropped"] = len(dropped_authors)
```

### Step 4 — Write per-author JSONL files

Output format matches the human-side files exactly:
- One JSON object per line
- Each object: `{"author": str, "comments": [list of full records]}`
- Submissions file same structure with key `"submissions"`

```python
import json

comments_out    = f"{args.output_dir}/user_comments_bots.jsonl"
submissions_out = f"{args.output_dir}/user_submissions_bots.jsonl"

with open(comments_out, 'w', encoding='utf-8') as f:
    for author in sorted(kept_authors):
        records = user_comments.get(author, [])
        f.write(json.dumps({"author": author, "comments": records}) + '\n')

with open(submissions_out, 'w', encoding='utf-8') as f:
    for author in sorted(kept_authors):
        records = user_submissions.get(author, [])
        f.write(json.dumps({"author": author, "submissions": records}) + '\n')
```

Output is **uncompressed** intentionally — `build_features.py` reads these files
multiple times for different feature groups, and uncompressed sequential reads are
faster than repeated gzip decompression.

### Step 5 — Sanity check against authors file

```python
expected = set(open(args.authors_file).read().splitlines())
unexpected = kept_authors - expected
missing    = expected - kept_authors

if unexpected:
    print(f"[WARN] {len(unexpected)} authors in output not in global list")
if missing:
    print(f"[INFO] {len(missing)} expected authors have no records in Pass 2 output")
    print(f"       (likely posted only in corrupted months or below min-posts threshold)")
```

---

## Output Files

```
merged/
  user_comments_bots.jsonl       ← one line per bot author
  user_submissions_bots.jsonl    ← one line per bot author
  merge_bot_extractions_summary.txt
```

### Output line format (must match human-side files exactly)

Comments file — each line:
```json
{"author": "SomeBot_v2", "comments": [{"body": "...", "created_utc": 1704067200, "subreddit": "...", ...}, ...]}
```

Submissions file — each line:
```json
{"author": "SomeBot_v2", "submissions": [{"title": "...", "created_utc": 1704067205, "selftext": "...", ...}, ...]}
```

---

## Summary Output

Printed to stdout and written to `merge_bot_extractions_summary.txt`:

```
=== merge_bot_extractions Summary ===
Input
  Comment files processed  : 12
  Submission files processed: 12
  Total comment records    : 4,201,884
  Total submission records :   312,441

Authors
  Unique authors found     : 5,000
  Dropped (< 3 posts)      :    38
  Final bot authors kept   : 4,962

Output
  user_comments_bots.jsonl    : 4,962 authors
  user_submissions_bots.jsonl : 4,962 authors

Note: label assignment (y=1) happens in build_features.py, not here.
```

---

## CLI Interface

```bash
python merge_bot_extractions.py \
  --input-dir    results/ \
  --authors-file bot_authors_global.txt \
  --output-dir   merged/ \
  [--min-posts   3]
```

---

## Memory Note

All records for all authors are held in `user_comments` and `user_submissions` dicts
simultaneously during aggregation. For 5000 bot authors each with potentially hundreds
of posts, this could reach several GB. On a cluster node with 16GB+ this is fine.
If memory is a concern, process in author batches or use a SQLite intermediate store —
but for the expected scale this should not be necessary.

---

## Dependencies

```
gzip      # stdlib — reading compressed input
orjson    # fast JSON parsing
json      # stdlib — writing output
glob      # stdlib
argparse  # stdlib
```
