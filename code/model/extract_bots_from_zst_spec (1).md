# `extract_bots_from_zst.py` — Specification

## Purpose

Streams a single month's `.zst` dump (comments OR submissions) line by line, and writes out
every record whose author matches either:
1. The rule-based bot username pattern (same rules as `build_labels.py`), or
2. The BotRank known-bot list (scraped via `fetch_botrank.py`)

Designed to run as **one independent job per month per file-type** (e.g. via SLURM array job),
with no coordination between jobs. Running this across all 12 months and merging results is
equivalent to a global "discover all bots, then collect all their history" pass — because the
matching rule is static and does not depend on results from other months.

---

## Why single-pass works here

A two-pass design (discover bot usernames first, then re-scan to collect all their history)
would require either re-reading every zst file twice, or serializing all discovery jobs before
any collection job can start — both break the "one month per parallel SLURM job" model.

Since the match criteria (regex pattern + static BotRank username set) do not change based on
what's found in other months, each month's job independently catches every bot record that
exists in that month. Concatenating the 12 months' outputs after the fact gives the same
result as a global discovery pass — there is no information lost by not coordinating across
jobs, as long as every job uses the identical, frozen matching ruleset.

**One known limitation, accepted for this phase:** if a bot only posts in a month where its
username happens not to match the regex (e.g. it changed its username, or the BotRank score
threshold changes later), that month's activity won't be caught. This is acceptable for the
one-year, first-pass goal. A future second pass (re-scanning with a finalized bot list) can
backfill this gap if needed.

---

## Inputs

### 1. Zst file (`--zst-file`)
A single compressed file — either `RC_YYYY-MM.zst` (comments) or `RS_YYYY-MM.zst` (submissions).
One job processes exactly one file.

### 2. File type (`--file-type`)
`comments` or `submissions` — determines which JSON schema to expect and which output
filename convention to use.

### 3. BotRank CSV (`--botrank`)
Same file produced by `fetch_botrank.py`. Expected columns: `bot_name`, `score` (and
optionally `rank`, `good_votes`, `bad_votes`, `comment_karma`, `link_karma` — unused here).

```
rank,bot_name,score,good_votes,bad_votes,comment_karma,link_karma
1,AutoModerator,0.99,50000,12,...
2,RemindMeBot,0.98,...
```

### 4. BotRank top-N (`--botrank-top-n`, default 500)
Only the top N rows (by whatever order the CSV is already sorted in — sort before passing in,
e.g. via `fetch_botrank.py --sort bad-votes`) are used to build the match set.

---

## Matching Rules (must be IDENTICAL across all 12 month-jobs)

Freeze this exact ruleset before launching the SLURM array — do not change it between jobs,
since consistency across months is what makes single-pass valid.

### Rule A — Username regex (same as `build_labels.py` Step 5a)

Evaluate on `author.lower()` unless noted:

| Pattern | Regex / check |
|---|---|
| Exact `"bot"` | `username.lower() == "bot"` |
| Whole-word `bot` | `re.search(r'\bbot\b', username.lower())` |
| Starts with `Auto` (case-sensitive) | `username.startswith("Auto")` |
| Whole-word `auto` | `re.search(r'\bauto\b', username.lower())` |
| Whole-word `mod` | `re.search(r'\bmod\b', username.lower())` |
| Underscore-bounded `_bot` | `re.search(r'(^bot_|_bot$|_bot_)', username.lower())` |

**Note:** the `\brobot\b` rule has been removed from this ruleset (dropped per project decision
— low yield, edge-case pattern). Do not include it.

False-positive guard — exclude if lowercased username is exactly one of:
```python
BOT_FALSE_POSITIVES = {
    "bottle", "bottom", "botox", "both", "bother", "botanical",
    "botanic", "botswana", "bought", "boots", "booth"
}
```

### Rule A — known limitation: `auto` and `mod` whole-word matches

The word-boundary anchor (`\bauto\b`, `\bmod\b`) correctly excludes compound words like
`automatic`, `automotive`, `module`, `model` — those are single tokens and never match.

However, the anchor does **not** exclude legitimate standalone uses of these words by humans —
e.g. `auto_enthusiast`, `classic_auto_fan`, `iron_mod`, `mod_squad_fan`. These are whole-word
matches that are still false positives, just a different category than incidental substrings.

**Do not pre-build a guard list for `auto`/`mod` before running the extraction** — the false
positives in this category depend on what real usernames actually look like in the data, which
isn't knowable in advance. Instead:

1. Run the extraction as specified (no `auto`/`mod` guard list yet).
2. After the run, build a review file: every unique username that matched **only** via the
   `auto` or `mod` rule (i.e. `_rule_matched: true` but did not also match `bot`/`_bot`/exact
   patterns, and did not match BotRank). This is the highest-risk-of-false-positive subset.
3. Manually inspect this list and build `AUTO_FALSE_POSITIVES` / `MOD_FALSE_POSITIVES` sets
   from what's actually observed.
4. Apply these guard lists in the **merge step** (`merge_bot_extractions.py`), not by re-running
   the SLURM extraction — this avoids re-scanning the zst files and keeps the frozen-ruleset
   guarantee intact for the raw extraction pass.

**Flag note on `Auto`-pattern noise (raised by user):** words like `Automatic`,
`Automotive`, `Automation` will trigger the `\bauto\b` whole-word check only if `auto`
appears as its own word — `Automatic` does NOT match `\bauto\b` since it's one continuous
token, not a separate word. So `automatic_jay` does not match, but `auto_jay` does. This
should already prevent most of the willy-nilly `Automatic` false positives observed. If
false positives are still showing up, log the matched username + which rule fired (see
Output section) so they can be reviewed and the ruleset refined for the *next* SLURM run —
but do not change rules mid-run across the 12 months.

### Rule B — BotRank membership

```python
botrank_set = set(df.head(top_n)["bot_name"].str.lower())
is_botrank_match = author.lower() in botrank_set
```

### Combined match condition

```python
is_match = rule_a_match(author) or is_botrank_match(author)
match_reason = "rule" if rule_a_match(author) else "botrank"
# if both match, prefer "rule" as the reported reason, but log both flags
```

---

## Processing Steps

### Step 1 — Load BotRank set (once, before streaming)

```python
botrank_df = pd.read_csv(args.botrank)
botrank_top = botrank_df.head(args.botrank_top_n)
botrank_set = set(botrank_top["bot_name"].str.lower())
```

### Step 2 — Stream the zst file

Use `zstandard` library to decompress in streaming mode (do not decompress the whole file to
disk first — these files can be large).

```python
import zstandard as zstd
import io

with open(args.zst_file, 'rb') as fh:
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
        for line in text_stream:
            process_line(line)
```

### Step 3 — Per-line processing

```python
def process_line(line, botrank_set, writer, stats):
    try:
        record = orjson.loads(line)
    except Exception:
        stats["parse_errors"] += 1
        return

    author = record.get("author", "")
    if author == "[deleted]" or author == "":
        stats["skipped_deleted"] += 1
        return
    # NOTE: do NOT skip AutoModerator here like build_labels.py's general aggregation does —
    # AutoModerator IS a bot and should be CAPTURED, not filtered out.

    rule_hit, matched_pattern = rule_a_match(author)  # returns (bool, pattern_name or None)
    botrank_hit = author.lower() in botrank_set

    if rule_hit or botrank_hit:
        reason = "rule" if rule_hit else "botrank"
        record["_match_reason"] = reason
        record["_rule_matched"] = rule_hit
        record["_matched_pattern"] = matched_pattern  # e.g. "bot", "auto", "mod", "_bot", "exact"
        record["_botrank_matched"] = botrank_hit
        writer.write(record)
        stats["matched"] += 1

    stats["total_lines"] += 1
```

**Important:** `rule_a_match()` must return *which specific pattern fired* (`"bot"`, `"auto"`,
`"mod"`, `"_bot"`, `"exact"`), not just a boolean. This is required for the post-run
`auto`/`mod` false-positive review step described above — without knowing which rule matched,
you can't isolate the highest-risk subset for manual review.

**Important deviation from `build_labels.py`:** that script's Step 2 explicitly skips
`AutoModerator` when aggregating the random-sample population (since you don't want
AutoModerator polluting your human-sample stats). This script does the **opposite** —
AutoModerator and all other bot-pattern matches are exactly what you want to capture. Do not
carry over the `[deleted]`/`AutoModerator` skip logic from `build_labels.py` wholesale.

### Step 4 — Write matched records incrementally

Do not buffer all matches in memory for the whole file — append to output as you go, since a
single month could still have thousands of bot-matching lines (especially from
AutoModerator-style accounts that post very frequently).

Write to a `.jsonl` file (not parquet) for simplicity and append-friendliness during
streaming; convert to parquet in a separate fast batch step after the SLURM run completes if
desired.

```
{file_type}_bots_{YYYY-MM}.jsonl
```

e.g. `comments_bots_2024-01.jsonl`, `submissions_bots_2024-03.jsonl`

### Step 5 — Per-job summary log

Print and write to `{file_type}_bots_{YYYY-MM}_summary.txt`:

```
=== Bot Extraction Summary: comments_bots_2024-01 ===
Source file          : RC_2024-01.zst
Total lines processed: 48,213,902
Parse errors          : 412
Skipped (deleted)     : 1,204,331
Matched (total)       : 18,402
  via rule            : 17,890
  via botrank         : 612
  via both             : 100
Unique authors matched: 3,104
Output                : comments_bots_2024-01.jsonl
Runtime               : 14m 32s
```

---

## SLURM Array Job Setup

One job per (month, file-type) pair — 12 months × 2 file types = 24 jobs, or run comments and
submissions sequentially within one job per month for 12 jobs total. Recommend the latter if
your cluster has per-job overhead, since it halves job count.

### Example `extract_bots_array.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=bot_extract
#SBATCH --array=0-11
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/bot_extract_%a.out

MONTHS=(2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12)
MONTH=${MONTHS[$SLURM_ARRAY_TASK_ID]}

python extract_bots_from_zst.py \
  --zst-file /data/reddit/RC_${MONTH}.zst \
  --file-type comments \
  --botrank botrank_top500.csv \
  --botrank-top-n 500 \
  --output-dir results/

python extract_bots_from_zst.py \
  --zst-file /data/reddit/RS_${MONTH}.zst \
  --file-type submissions \
  --botrank botrank_top500.csv \
  --botrank-top-n 500 \
  --output-dir results/
```
Corruption handling inside extract_bots_from_zst.py

The Python script itself must catch decompression errors mid-stream and exit cleanly with
partial results saved, rather than crashing uncaught. Add this wrapping around the
streaming loop:

```python
import zstandard as zstd

records_written_before_failure = 0
corruption_detected = False

try:
    with open(args.zst_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            for line in text_stream:
                process_line(line, botrank_set, writer, stats)
except zstd.ZstdError as e:
    corruption_detected = True
    print(f"[ERROR] Zstd decompression error after {stats['total_lines']} lines: {e}", file=sys.stderr)
    print(f"[ERROR] File may be corrupted or truncated: {args.zst_file}", file=sys.stderr)
except Exception as e:
    corruption_detected = True
    print(f"[ERROR] Unexpected error after {stats['total_lines']} lines: {e}", file=sys.stderr)
finally:
    writer.close()  # ensure all buffered matches written to disk so far are flushed

# Write summary regardless of whether corruption occurred — partial results are still useful
write_summary(stats, corruption_detected=corruption_detected)

if corruption_detected:
    print(f"[WARN] Exiting with partial results: {stats['matched']} matches found before failure.")
    sys.exit(1)  # non-zero exit so SLURM/sbatch script logs this as a failure, but doesn't kill sibling steps
else:
    sys.exit(0)

```

The summary file (Step 5) should also gain a corrupted: true/false field so you know at a
glance, when reviewing 12 months of summaries later, which months had incomplete coverage and
may be worth re-downloading and re-running individually.

**Critical:** the `--botrank` file and `--botrank-top-n` value must be byte-for-byte identical
across all 24 sub-jobs. Freeze `botrank_top500.csv` before launching the array — do not
re-scrape or re-sort it mid-run, or different months will use different bot definitions.

---

## Output (after all 12 months complete)

### Per-month files
```
results/comments_bots_2024-01.jsonl
results/comments_bots_2024-02.jsonl
...
results/submissions_bots_2024-12.jsonl
```

### Merge step (separate lightweight script, `merge_bot_extractions.py`)

Not a re-scan — just concatenation + author-level dedup/aggregation, structurally identical
to `build_labels.py` Steps 2–4 but applied to the bot-only extraction output instead of the
random sample:

1. Concatenate all `comments_bots_*.jsonl` → group by author → `user_comments_bots.jsonl`
2. Concatenate all `submissions_bots_*.jsonl` → group by author → `user_submissions_bots.jsonl`
3. Build `labels_bots.parquet` — same schema as `build_labels.py`'s `labels.parquet`, but
   every row has `label = "bot"`, `label_source` = `"rule"` or `"botrank"` (whichever matched
   first for that author across any month — use OR logic, since an author matched by rule in
   any month is a rule-match overall).
4. Concatenate with the existing random-sample `labels.parquet` (humans) to form the final
   combined training label set. Watch for authors appearing in BOTH sets (a random-sampled
   human account that also got swept into the bot extraction) — these should be resolved
   toward `bot`, since the targeted extraction is higher-precision than random sampling.

---

## CLI Interface

```bash
python extract_bots_from_zst.py \
  --zst-file       /data/reddit/RC_2024-01.zst \
  --file-type      comments \
  --botrank        botrank_top500.csv \
  --botrank-top-n  500 \
  --output-dir     results/
```

---

## Dependencies

```
zstandard     # pip install zstandard — streaming zst decompression
orjson        # fast JSON parsing
pandas        # botrank CSV loading
pyarrow       # if converting to parquet later
re            # stdlib
argparse      # stdlib
```

---

## Implementation Notes

- **Stream, never buffer the full file.** A month's zst can decompress to tens of GB.
- **Do not skip AutoModerator** in this script (unlike `build_labels.py`'s random-sample
  aggregation) — AutoModerator and similar accounts are exactly the high-volume bot signal
  you want.
- **Freeze the matching ruleset** (regex patterns + BotRank top-N file) before launching the
  SLURM array. Changing it mid-run breaks the single-pass equivalence-to-global-discovery
  argument.
- **Log every match with its reason** so you can audit for false positives (like the
  `Automatic`-name concern) after the run, without needing to re-run the extraction —
  just filter the output `.jsonl` for `_rule_matched: true` and inspect usernames.
- Expect highly skewed output: a handful of extremely high-volume bots (AutoModerator,
  RemindMeBot) will dominate line counts. This is fine and expected — for the supervised
  model, you'll likely want to **cap or sample posts per bot author** later (e.g. max 500
  posts/comments per bot) so one hyperactive bot doesn't dominate your embedding-based
  features. That capping decision belongs in `build_features.py`, not here — keep this
  script's job purely as "capture everything that matches."
