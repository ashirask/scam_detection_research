# `extract_bots_from_zst.py` — Specification

## Purpose

Extracts all Reddit records belonging to bot accounts from monthly `.zst` dumps, using a
**two-pass design**:

- **Pass 1 (author discovery):** streams a zst file reading only the `author` field per line.
  Applies matching rules and minimum post threshold. Outputs a small text file of qualifying
  bot usernames for that month — extremely fast, minimal disk usage.
- **Pass 2 (full record extraction):** re-streams the same zst file, loading the full JSON
  record only for authors confirmed in the global bot author list (produced by merging all
  Pass 1 outputs). Writes complete records to compressed output.

Designed to run as **12 independent SLURM array jobs** (one per month), with a lightweight
merge step between passes. Pass 1 and Pass 2 are modes of the same script, selected via
`--pass 1` or `--pass 2`.

---

## Cluster Information

Partition to use: **Orion** (CPU partition — this script uses no GPU)
Do not submit to the GPU or Nebula_GPU partition — this job does streaming I/O and regex
matching only, and requesting GPU nodes adds unnecessary queue delay.

```
Orion*  stdmem,amd,epyc,rome        ← recommended default
Orion*  stdmem,amd,epyc,genoa
Orion*  stdmem,intel,xeon,caslake
```

Recommended resources per job:
```
#SBATCH --partition=Orion
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
```

---

## Inputs

### 1. Zst file (`--zst-file`)
A single compressed file — either `RC_YYYY-MM.zst` (comments) or `RS_YYYY-MM.zst`
(submissions). One job processes exactly one file per pass.

### 2. File type (`--file-type`)
`comments` or `submissions` — controls output filename convention.

### 3. Pass (`--pass-num`)
`1` or `2` — selects which mode the script runs in (see Processing Steps).

### 4. BotRank CSV (`--botrank`, Pass 1 only)
Produced by `fetch_botrank.py`. Expected columns: `bot_name`, `score`, and optionally
`rank`, `good_votes`, `bad_votes`, `comment_karma`, `link_karma`.

```
rank,bot_name,score,good_votes,bad_votes,comment_karma,link_karma
1,AutoModerator,0.99,50000,12,...
2,RemindMeBot,0.98,...
```

### 5. BotRank top-N (`--botrank-top-n`, default 500, Pass 1 only)
Only the top N rows of the BotRank CSV are used. Sort the CSV before passing it in
(e.g. via `fetch_botrank.py --sort bad-votes`).

### 6. Authors file (`--authors-file`, Pass 2 only)
`bot_authors_global.txt` — produced by `merge_pass1_authors.py` between the two passes.
One username per line, deduplicated across all months.

### 7. Minimum posts threshold (`--min-posts`, default 3, Pass 1 only)
Authors with fewer than this many posts seen in this month's file are excluded from the
Pass 1 output. Filters accounts with too little signal for feature engineering.

### 8. Extraction mode (`--mode`, default "bot", Pass 1 only)
`bot` or `human` — controls which accounts are extracted in Pass 1:
- `bot` (default): extracts accounts matching bot detection rules
- `human`: extracts accounts NOT matching bot detection rules (inverse selection)

Pass 2 is mode-agnostic — it simply extracts full records for whatever author list is provided.

### 9. Output directory (`--output-dir`)
Directory where all output files are written. Created if it does not exist.

---

## Author Skip List (hardcoded)

The following authors are unconditionally skipped in **both passes**, before any match
check is applied:

```python
SKIP_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}
```

Rationale:
- `AutoModerator` — volume so dominant it distorts training data (posts in nearly every
  thread); adds no discriminative signal since it is trivially identifiable by name alone.
- `[deleted]` / `[removed]` — no username means no features can be computed.
- `""` — malformed records.

These are hardcoded, not loaded from a file, since they are categorical certainties that
should apply to every run without risk of accidental omission.

---

## Matching Rules (Pass 1 only — must be IDENTICAL across all 12 month-jobs)

Freeze this exact ruleset before launching the SLURM array. Do not change between jobs.

### Rule A — Username regex

Evaluate on `author.lower()` unless noted:

| Pattern | Implementation | Example matches |
|---|---|---|
| Exact `"bot"` | `username.lower() == "bot"` | `bot` |
| Whole-word `bot` | `re.search(r'\bbot\b', username.lower())` | `newsbot`, `Link_Bot` |
| Whole-word `mod` | `re.search(r'\bmod\b', username.lower())` | `ModHelper` |
| Underscore-bounded `_bot` | `re.search(r'(^bot_\|_bot$\|_bot_)', username.lower())` | `link_bot_v2` |

**Excluded rule:** `\brobot\b` has been intentionally removed (low yield, edge-case pattern).

**False-positive guard** — if `\bbot\b` matched, additionally check that the lowercased
username is NOT exactly one of:

```python
BOT_FALSE_POSITIVES = {
    "bottle", "bottom", "botox", "both", "bother", "botanical",
    "botanic", "botswana", "bought", "boots", "booth"
}
```

**Note on `mod` false positives:** the word-boundary anchor correctly excludes
compound words (`module`, `model` — single tokens, never match).
However, it does not exclude legitimate standalone uses like `iron_mod`.
Do not pre-build a guard list for this — run the extraction first, then after Pass 1
completes inspect all unique usernames that matched *only* via the `mod` rule
(identifiable via `_matched_pattern` field). Build exclusion lists from what is actually
observed, and apply them during the merge step — not by re-running the SLURM extraction.

`rule_a_match()` must return `(bool, pattern_name_or_None)` where `pattern_name` is one of:
`"exact"`, `"bot"`, `"mod"`, `"_bot"`. This is required for post-run false-positive review.

### Rule B — BotRank membership

```python
botrank_set = set(df.head(top_n)["bot_name"].str.lower())
is_botrank_match = author.lower() in botrank_set
```

### Rule C — Text-based phrase matching

Checks comment body or submission selftext/title for common bot phrases. This rule
helps identify bots that don't have obvious username patterns but explicitly identify
themselves in their content.

```python
# For comments
text_content = record.get("body", "")

# For submissions (check both selftext and title)
selftext = record.get("selftext", "")
title = record.get("title", "")
text_content = f"{selftext} {title}"

# Case-insensitive substring matching
BOT_PHRASES = [
    "i am a bot",
    "beep beep",
    "beep boop",
    "i'm a bot",
    "im a bot",
    "this is a bot",
    "automated message",
    "auto moderator",
    "automated response",
    "bot message",
]

is_text_match = any(phrase in text_content.lower() for phrase in BOT_PHRASES)
```

### Combined match

```python
rule_hit, matched_pattern = rule_a_match(author)
botrank_hit = author.lower() in botrank_set
text_hit = rule_c_match(text_content)

is_bot = rule_hit or botrank_hit or text_hit

# Apply mode-based selection
if args.mode == "bot":
    is_match = is_bot
else:  # human mode
    is_match = not is_bot
```

---

## Processing Steps

### PASS 1 — Author Discovery

**Goal:** identify qualifying bot authors cheaply. Only `author` field is extracted per line.
No full JSON parse of all fields needed.

#### Step 1 — Load BotRank set (once, before streaming)

```python
botrank_df = pd.read_csv(args.botrank)
botrank_set = set(botrank_df.head(args.botrank_top_n)["bot_name"].str.lower())
```

#### Step 2 — Stream zst file, author-only parse

```python
import zstandard as zstd
import io

author_post_count = {}   # {author: int} — counts posts seen this month
corruption_detected = False

try:
    with open(args.zst_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            for line in text_stream:
                stats["total_lines"] += 1
                try:
                    record = orjson.loads(line)
                except (json.JSONDecodeError, ValueError) as e:
                    # Line-level JSON error - can skip and continue
                    stats["parse_errors"] += 1
                    if stats["parse_errors"] <= 10:  # Only log first 10 to avoid spam
                        print(f"[WARN] JSON parse error at line {stats['total_lines']}: {e}", file=sys.stderr)
                    continue

                author = record.get("author", "")
                if author in SKIP_AUTHORS:
                    stats["skipped"] += 1
                    continue

                rule_hit, matched_pattern = rule_a_match(author)
                botrank_hit = author.lower() in botrank_set
                text_hit = rule_c_match(text_content)

                is_bot = rule_hit or botrank_hit or text_hit

                # Apply mode-based selection
                if (args.mode == "bot" and is_bot) or (args.mode == "human" and not is_bot):
                    author_post_count[author] = author_post_count.get(author, 0) + 1

except zstd.ZstdError as e:
    # File-level corruption - cannot continue past this point
    corruption_detected = True
    print(f"[ERROR] Zstd decompression error at line {stats['total_lines']}: {e}", file=sys.stderr)
    print(f"[ERROR] File may be corrupted or truncated: {args.zst_file}", file=sys.stderr)
    print(f"[INFO] Partial results saved: {len(author_post_count)} authors found before error", file=sys.stderr)
except Exception as e:
    # Unexpected error - treat as corruption
    corruption_detected = True
    print(f"[ERROR] Unexpected error at line {stats['total_lines']}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"[INFO] Partial results saved: {len(author_post_count)} authors found before error", file=sys.stderr)
```

#### Step 3 — Apply minimum post threshold and write author list

```python
qualifying = {
    author for author, count in author_post_count.items()
    if count >= args.min_posts
}

output_path = os.path.join(args.output_dir, f"pass1_authors_{file_type}_{period}.txt")
with open(output_path, 'w') as f:
    for author in sorted(qualifying):
        f.write(author + '\n')
```

#### Step 4 — Write Pass 1 summary

File: `pass1_authors_{file_type}_{YYYY-MM}_summary.txt`

```
=== Pass 1 Summary: comments RC_2024-01 ===
Source file              : RC_2024-01.zst
Total lines scanned      : 147,998,199
Parse errors             : 412
Skipped (skip list)      : 2,841,022
Candidate authors found  : 8,204
Below min-posts (< 3)    : 1,102
Qualifying authors output: 7,102
Output file              : pass1_authors_comments_2024-01.txt
Corrupted                : false
Runtime                  : 6m 14s
```

---

### BETWEEN PASSES — merge_pass1_authors.py (lightweight, runs once)

Reads all `pass1_authors_*.txt` files from all months and file types, takes the set union,
and writes a single deduplicated global author list.

```python
# merge_pass1_authors.py
import glob, argparse, random

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir",   required=True)
parser.add_argument("--output",      default="bot_authors_global.txt")
parser.add_argument("--max-authors", type=int, default=None,
                    help="Cap total unique bot authors via random sample. "
                         "If not set, all qualifying authors are kept.")
parser.add_argument("--seed",        type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
args = parser.parse_args()

# Collect all unique authors across all months and file types
all_authors = set()
for path in glob.glob(f"{args.input_dir}/pass1_authors_*.txt"):
    with open(path) as f:
        all_authors.update(line.strip() for line in f if line.strip())

print(f"Total unique bot authors found: {len(all_authors)}")

# Apply cap via simple random sample if requested
random.seed(args.seed)
if args.max_authors and len(all_authors) > args.max_authors:
    selected = set(random.sample(sorted(all_authors), args.max_authors))
    print(f"Capped to {len(selected)} authors (random sample, seed={args.seed})")
else:
    selected = all_authors
    print(f"No cap applied — keeping all {len(selected)} authors")

with open(args.output, 'w') as f:
    for author in sorted(selected):
        f.write(author + '\n')

print(f"Written to: {args.output}")
```

**Recommended `--max-authors` value:** match your human sample size (e.g. 5000 human
accounts → set `--max-authors 5000`). This produces a balanced 1:1 bot:human ratio for
model training without heavy class reweighting.

If fewer bot authors are found than `--max-authors` (e.g. only 3000 qualifying bots across
all months), the script keeps all of them — the cap is a ceiling, not a target.

`--seed 42` ensures the same random sample is produced every time you re-run the merge,
making the experiment reproducible.

Run manually after all Pass 1 SLURM jobs complete:
```bash
python merge_pass1_authors.py \
  --input-dir results/ --output bot_authors_global.txt --max-authors 5000
```

Expected runtime: seconds. Output size: small (thousands of usernames as plain text).

---

### PASS 2 — Full Record Extraction

**Goal:** collect complete JSON records for every author confirmed in `bot_authors_global.txt`.

#### Step 1 — Load author set (once, before streaming)

```python
with open(args.authors_file) as f:
    bot_authors = set(line.strip() for line in f if line.strip())
```

#### Step 2 — Stream zst file, full JSON parse

```python
corruption_detected = False

try:
    with open(args.zst_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            for line in text_stream:
                stats["total_lines"] += 1
                try:
                    record = orjson.loads(line)
                except (json.JSONDecodeError, ValueError) as e:
                    # Line-level JSON error - can skip and continue
                    stats["parse_errors"] += 1
                    if stats["parse_errors"] <= 10:  # Only log first 10 to avoid spam
                        print(f"[WARN] JSON parse error at line {stats['total_lines']}: {e}", file=sys.stderr)
                    continue

                author = record.get("author", "")
                if author in SKIP_AUTHORS:
                    stats["skipped"] += 1
                    continue
                if author not in bot_authors:
                    continue   # O(1) set lookup — negligible overhead

                out_f.write(orjson.dumps(record).decode() + '\n')
                stats["matched"] += 1

except zstd.ZstdError as e:
    # File-level corruption - cannot continue past this point
    corruption_detected = True
    print(f"[ERROR] Zstd decompression error at line {stats['total_lines']}: {e}", file=sys.stderr)
    print(f"[ERROR] File may be corrupted or truncated: {args.zst_file}", file=sys.stderr)
    print(f"[INFO] Partial results saved: {stats['matched']} matches found before error", file=sys.stderr)
except Exception as e:
    # Unexpected error - treat as corruption
    corruption_detected = True
    print(f"[ERROR] Unexpected error at line {stats['total_lines']}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"[INFO] Partial results saved: {stats['matched']} matches found before error", file=sys.stderr)
finally:
    out_f.close()   # flush all buffered writes regardless of error
```

#### Step 3 — Output format: compressed jsonl written directly during streaming

```python
import gzip

output_path = os.path.join(args.output_dir, f"comments_bots_RC_{period}.jsonl.gz")
out_f = gzip.open(output_path, 'wt', encoding='utf-8', compresslevel=6)
```

Output filenames:
```
results/comments_bots_RC_2024-01.jsonl.gz
results/submissions_bots_RS_2024-01.jsonl.gz
...
```

`compresslevel=6` — good compression ratio without bottlenecking streaming throughput.
`.jsonl` files compress 5–10x since they are repetitive JSON text.

#### Step 4 — Write Pass 2 summary

File: `comments_bots_RC_{YYYY-MM}_summary.txt`

```
=== Pass 2 Summary: comments RC_2024-01 ===
Source file              : RC_2024-01.zst
Authors in global list   : 7,841
Total lines scanned      : 147,998,199
Parse errors             : 412
Skipped (skip list)      : 2,841,022
Matched and written      : 18,402
Unique authors matched   : 3,104
Output file              : comments_bots_RC_2024-01.jsonl.gz
Corrupted                : false
Runtime                  : 12m 08s
```

---

## SLURM Setup — Three sbatch files

The workflow uses three sbatch files submitted in sequence. Each depends on the previous
completing successfully via `--dependency=afterok`.

### How `--dependency=afterok` works

When you submit a job with `--dependency=afterok:JOBID`, SLURM holds it in the queue
(`PD` state, reason `Dependency`) and only releases it once job `JOBID` has completed with
exit code 0 on ALL array tasks. If any task fails (non-zero exit), the dependent job is
never released — it sits as `PD` until you cancel it or the dependency is overridden.

This means you submit all three sbatch files at once at the start, walk away, and SLURM
handles sequencing automatically. You do not need to monitor Pass 1 and manually trigger
Pass 2.

### File 1: `pass1_array.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=bot_pass1
#SBATCH --array=0-11
#SBATCH --partition=Orion
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/pass1_%a.out

# Edit to your exact months — array range must equal len(MONTHS)-1
MONTHS=(2024-01 2024-02 2024-03 2024-04 2024-05 2024-06
        2024-07 2024-08 2024-09 2024-10 2024-11 2024-12)
MONTH=${MONTHS[$SLURM_ARRAY_TASK_ID]}

RC_FILE="/data/reddit/RC_${MONTH}.zst"
RS_FILE="/data/reddit/RS_${MONTH}.zst"

echo "=== Pass 1 | Task ${SLURM_ARRAY_TASK_ID} | Month ${MONTH} ==="

# Pre-flight check
MISSING=0
[ ! -f "$RC_FILE" ] && echo "[FATAL] Missing: $RC_FILE" && MISSING=1
[ ! -f "$RS_FILE" ] && echo "[FATAL] Missing: $RS_FILE" && MISSING=1
[ "$MISSING" -eq 1 ] && exit 1

python extract_bots_from_zst.py \
  --pass-num 1 --zst-file "$RC_FILE" --file-type comments \
  --botrank botrank_top500.csv --botrank-top-n 500 \
  --min-posts 3 --output-dir results/
COMMENTS_EXIT=$?

python extract_bots_from_zst.py \
  --pass-num 1 --zst-file "$RS_FILE" --file-type submissions \
  --botrank botrank_top500.csv --botrank-top-n 500 \
  --min-posts 3 --output-dir results/
SUBMISSIONS_EXIT=$?

echo "=== Done | comments exit=${COMMENTS_EXIT} | submissions exit=${SUBMISSIONS_EXIT} ==="
# Exit non-zero if either failed (triggers afterok dependency to hold)
[ "$COMMENTS_EXIT" -ne 0 ] || [ "$SUBMISSIONS_EXIT" -ne 0 ] && exit 1
exit 0
```

### File 2: `merge_pass1.sbatch`

Single job (no array), runs after all 12 Pass 1 tasks complete.

```bash
#!/bin/bash
#SBATCH --job-name=bot_merge
#SBATCH --partition=Orion
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/merge_pass1.out

echo "=== Merging Pass 1 author lists ==="
python merge_pass1_authors.py \
  --input-dir   results/ \
  --output      bot_authors_global.txt \
  --max-authors 5000 \
  --seed        42
echo "Merge complete: $(wc -l < bot_authors_global.txt) unique bot authors selected"
```

### File 3: `pass2_array.sbatch`

Identical structure to `pass1_array.sbatch` but runs Pass 2.

```bash
#!/bin/bash
#SBATCH --job-name=bot_pass2
#SBATCH --array=0-11
#SBATCH --partition=Orion
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/pass2_%a.out

MONTHS=(2024-01 2024-02 2024-03 2024-04 2024-05 2024-06
        2024-07 2024-08 2024-09 2024-10 2024-11 2024-12)
MONTH=${MONTHS[$SLURM_ARRAY_TASK_ID]}

RC_FILE="/data/reddit/RC_${MONTH}.zst"
RS_FILE="/data/reddit/RS_${MONTH}.zst"

echo "=== Pass 2 | Task ${SLURM_ARRAY_TASK_ID} | Month ${MONTH} ==="

MISSING=0
[ ! -f "$RC_FILE" ] && echo "[FATAL] Missing: $RC_FILE" && MISSING=1
[ ! -f "$RS_FILE" ] && echo "[FATAL] Missing: $RS_FILE" && MISSING=1
[ "$MISSING" -eq 1 ] && exit 1

python extract_bots_from_zst.py \
  --pass-num 2 --zst-file "$RC_FILE" --file-type comments \
  --authors-file bot_authors_global.txt \
  --output-dir results/
COMMENTS_EXIT=$?

python extract_bots_from_zst.py \
  --pass-num 2 --zst-file "$RS_FILE" --file-type submissions \
  --authors-file bot_authors_global.txt \
  --output-dir results/
SUBMISSIONS_EXIT=$?

echo "=== Done | comments exit=${COMMENTS_EXIT} | submissions exit=${SUBMISSIONS_EXIT} ==="
[ "$COMMENTS_EXIT" -ne 0 ] || [ "$SUBMISSIONS_EXIT" -ne 0 ] && exit 1
exit 0
```

### How to submit all three at once

```bash
mkdir -p logs results

# Submit Pass 1 — runs immediately
PASS1_JOB=$(sbatch --parsable pass1_array.sbatch)
echo "Pass 1 submitted: job $PASS1_JOB"

# Submit merge — held until ALL Pass 1 tasks succeed
MERGE_JOB=$(sbatch --parsable --dependency=afterok:${PASS1_JOB} merge_pass1.sbatch)
echo "Merge submitted: job $MERGE_JOB (waiting on $PASS1_JOB)"

# Submit Pass 2 — held until merge succeeds
PASS2_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_JOB} pass2_array.sbatch)
echo "Pass 2 submitted: job $PASS2_JOB (waiting on $MERGE_JOB)"

echo "All submitted. Monitor with: squeue -u $USER"
```

Paste these five lines into your terminal as a block — they run instantly since SLURM just
registers the jobs and returns. All three jobs will appear immediately in `squeue` with
Pass 1 as `R` (or `PD` if queued) and the other two as `PD (Dependency)`.

### Monitoring

```bash
squeue -u $USER                          # see all your jobs and their states
tail -f logs/pass1_0.out                 # watch a specific task live
grep -l "FATAL\|ERROR" logs/pass1_*.out  # find failed tasks after run
sacct -j <JOBID> --format=MaxRSS,Elapsed # check actual memory and runtime used
```

---

## CLI Interface Summary

```bash
# Pass 1 (bot mode - default)
python extract_bots_from_zst.py \
  --pass-num       1 \
  --zst-file       /data/reddit/RC_2024-01.zst \
  --file-type      comments \
  --botrank        botrank_top500.csv \
  --botrank-top-n  500 \
  --min-posts      3 \
  --mode           bot \
  --output-dir     results/

# Pass 1 (human mode - extracts non-bots)
python extract_bots_from_zst.py \
  --pass-num       1 \
  --zst-file       /data/reddit/RC_2024-01.zst \
  --file-type      comments \
  --botrank        botrank_top500.csv \
  --botrank-top-n  500 \
  --min-posts      3 \
  --mode           human \
  --output-dir     results/

# Pass 2 (mode-agnostic - works with bot or human author lists)
python extract_bots_from_zst.py \
  --pass-num       2 \
  --zst-file       /data/reddit/RC_2024-01.zst \
  --file-type      comments \
  --authors-file   bot_authors_global.txt \
  --output-dir     results/
```

---

## Output Structure (after all passes complete)

```
results/
  pass1_authors_comments_2024-01.txt       ← one per month per file-type (Pass 1)
  pass1_authors_submissions_2024-01.txt
  ...
  pass1_authors_*_summary.txt              ← Pass 1 summaries

bot_authors_global.txt                     ← merged global list (merge step)

results/
  comments_bots_RC_2024-01.jsonl.gz        ← full records, compressed (Pass 2)
  submissions_bots_RS_2024-01.jsonl.gz
  ...
  comments_bots_RC_*_summary.txt           ← Pass 2 summaries
```

---

## Dependencies

```
zstandard     # streaming zst decompression
orjson        # fast JSON parsing (fallback: stdlib json)
pandas        # BotRank CSV loading (Pass 1 only)
gzip          # stdlib — compressed output (Pass 2)
re            # stdlib
argparse      # stdlib
```

---

## Implementation Notes

- **Pass 1 is extremely fast** — it only reads one field per line and writes a small text
  file. Expect it to run 2–3x faster than Pass 2 for the same month.
- **Pass 2 total time** is comparable to the original single-pass script — the extra scan
  is cheap since the bottleneck is always decompression throughput, not the match logic.
- **`AutoModerator` is hardcoded in SKIP_AUTHORS** — do not remove this. It posts in nearly
  every subreddit and would generate millions of records that add no discriminative signal
  to the model.
- **Keep all JSON fields in Pass 2 output** — no field stripping at this stage. Feature
  selection happens in `build_features.py`.
- **Freeze the BotRank file and `--botrank-top-n` before launching Pass 1.** Both must be
  identical across all 12 array tasks or different months will use different bot definitions,
  invalidating the global merge.
- **If a Pass 1 task is corrupted mid-file**, its partial author list is still valid for
  the authors discovered before the failure. The global merge will include them. The Pass 2
  job for that month will then extract their full records from whatever portion of the file
  was readable. Corrupted months are flagged in summaries (`Corrupted: true`) for review.
- **Error handling distinguishes between file-level and line-level errors:**
  - JSON parse errors (line-level): skipped and logged (first 10 only to avoid spam), processing continues
  - zstd decompression errors (file-level): cannot continue past corruption point, partial results saved with detailed logging
  - This ensures maximum data recovery while clearly distinguishing recoverable vs fatal errors
- **Human extraction mode (`--mode human`)** inverts all bot detection rules to extract non-bot accounts.
  This enables balanced dataset creation without needing separate human extraction scripts.
