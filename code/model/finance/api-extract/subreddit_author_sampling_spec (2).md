# Spec: Subreddit → Sampled Author Extraction Pipeline
**Arctic Shift API** (`https://arctic-shift.photon-reddit.com/api`)

## 0.0 Instructions for the implementing coding agent

This spec is written to be implemented directly, with minimal further clarification needed. Follow these ground rules:

- **Target**: Python 3.9+, stdlib + `requests` only (no new dependencies — matches the existing reference script).
- **Reference script**: `download_arctic_shift_V2_3.py` (provided alongside this spec) is the existing, battle-tested author-level downloader. Do **not** reimplement its HTTP/logging/concurrency primitives from scratch — port these functions **verbatim or near-verbatim** (rename only where the subreddit-based scripts need a different signature):
  - `polite_get(session, url, params, min_sleep, max_retries=6)` — rate-limit-aware GET with 429/422/5xx backoff. Copy unchanged.
  - `extract_records(payload)` — unwraps `{"data": [...]}`. Copy unchanged.
  - The `logging.basicConfig(...)` setup block in `main()` (StreamHandler + optional FileHandler). Copy unchanged.
  - The heartbeat-logger pattern (`_active_workers` dict + lock, `heartbeat_logger()` thread, `_completed_authors` counter pattern) — copy the pattern, rename `_completed_authors` → `_completed_subreddits` for Stage 1.
  - The atomic-write pattern (`tmp_path` → `.replace(out_path)`) and `.done` marker with `{"count": N, "last_cursor": X}` JSON — copy unchanged.
  - The cursor-advance safety check (`if cursor is not None and new_cursor <= cursor and new_count == 0: break`) — copy unchanged, this prevents infinite loops on API edge cases.
- **Four separate scripts**, one per stage (§3), each independently runnable and independently resumable. Do not merge them into one monolithic script — Stage 1 runs for a long time on a cluster; Stages 2–4 are local, instant, and will be re-run repeatedly (e.g. every time you want a new resample batch).
- **Deliverables**: `extract_subreddit_authors.py`, `build_author_pools.py`, `sample_authors.py`, `merge_master_list.py`, each with `argparse` CLI matching §4–§7 exactly (flag names, types, defaults).
- Every script must be safely re-runnable (idempotent given the same inputs/flags) and must log to both stdout and an optional `--log-file`.
- If anything in this spec is genuinely ambiguous when you sit down to implement it, prefer the behavior of the reference script over inventing something new — the two pipelines are meant to feel like one system.

## 0. Goal

Given a list of ~200 finance-related subreddit names (already filtered by keyword/min-posts/subscribers from 2025 metadata), produce a **random sample of ~1000 unique authors per subreddit**, ready to feed into the existing `download_arctic_shift_V2_3.py` (author-level full history downloader).

This spec covers everything *before* that script runs: subreddit → raw author-bearing records → deduped author pool → random sample → merged master list.

## 1. Decisions locked in for this version

| Decision | Choice |
|---|---|
| History depth | **Full available history**, no date bound, per subreddit (posts + comments) |
| Author pool composition | **Combined**: posts and comments feed one pooled author set per subreddit, sampled together |
| Noise filtering | **Minimal**: drop only `None` / `"[deleted]"` / empty-string authors. Bots, `AutoModerator`, etc. are *kept* in the pool (filter later if needed, since bot-detection may itself be a downstream research question) |

Because "full history" was chosen deliberately, note the risk up front: a handful of your 200 subreddits (e.g. very large, long-running ones like r/wallstreetbets-scale communities, if any survived your filters) could have millions of comments. That doesn't break the pipeline, but it does mean per-subreddit runtime is unbounded and heavily skewed — a small number of subreddits could dominate total wall-clock time. The spec includes a **runaway-safety knob** (§6) purely as an operational safeguard (loud logging + optional page cap you control), not as a silent data-quality compromise.

## 2. Why not use raw full records or the `aggregate` endpoint

- **Field-limiting is API-sanctioned**: the docs explicitly say selecting only needed fields reduces both response size and server time. We only need `author`, `created_utc`, and `id` per record (`id` for pagination/dedup safety, `created_utc` to drive the cursor). We do **not** need `body`, `score`, `title`, etc. This alone should cut payload size by an order of magnitude vs. your existing per-author script (which correctly stores full records — that's a different use case, full user history, where you want everything).
- **`/api/posts/search/aggregate?aggregate=author`** returns authors ranked by activity volume. Sampling from that would bias your 1000 users toward the subreddit's heaviest posters, not a representative cross-section of participants. We use the plain `search` endpoints and dedupe/sample ourselves instead.

## 3. Pipeline stages

```
subreddits.txt (200 names)
      │
      ▼
┌─────────────────────────────────────────────┐
│ STAGE 1 — extract_subreddit_authors.py       │
│ Pull minimal (author, created_utc, id)       │
│ records from /posts/search and                │
│ /comments/search per subreddit, full history  │
└─────────────────────────────────────────────┘
      │  raw_meta/<subreddit>/{posts,comments}.jsonl.gz
      ▼
┌─────────────────────────────────────────────┐
│ STAGE 2 — build_author_pools.py               │
│ Dedupe authors per subreddit (posts+comments  │
│ combined), drop None/[deleted]/empty,         │
│ compute per-author counts                     │
└─────────────────────────────────────────────┘
      │  author_pools/<subreddit>_authors.csv
      ▼
┌─────────────────────────────────────────────┐
│ STAGE 3 — sample_authors.py                   │
│ Seeded random.sample(pool, 1000) per subreddit│
└─────────────────────────────────────────────┘
      │  samples/<subreddit>_sample1000.txt
      ▼
┌─────────────────────────────────────────────┐
│ STAGE 4 — merge_master_list.py                │
│ Global dedupe across all 200 subreddits,      │
│ preserve subreddit membership mapping         │
└─────────────────────────────────────────────┘
      │  master/all_sampled_authors.txt
      │  master/author_subreddit_membership.csv
      ▼
   feeds into existing download_arctic_shift_V2_3.py
```

Stages 1 and 2/3/4 are deliberately separate scripts. Stage 1 is the only one that touches the network and is the only one that needs to run on a long HPC job; Stages 2–4 are pure local post-processing over small files and can run in seconds on a login node.

## 4. Stage 1 — `extract_subreddit_authors.py`

### 4.1 What it does

For each subreddit, for `kind` in `{posts, comments}`:
- Call `posts/search` or `comments/search` with:
  - `subreddit=<name>`
  - `fields=author,created_utc,id`
  - `limit=100`
  - `sort=asc`
  - `after=<cursor>` (cursor-based pagination, same pattern as your author script)
- Write each minimal record as one line of gzip JSONL.
- Advance cursor by `last_record.created_utc + 1`, same safety checks your existing script has (cursor-not-advancing guard, dedup via `seen_ids` within a page-batch window — full-run dedup happens in Stage 2, not here).
- On completion, write a `.done` marker with `{"count": N, "last_cursor": X}` — identical resumability contract to your existing script.

### 4.2 Concurrency model

- `ThreadPoolExecutor` over **subreddits**, not authors (each worker owns one subreddit, and within it does posts then comments sequentially — mirrors your existing `process_author` structure almost exactly, just swap the "author" dimension for "subreddit").
- Default `--workers 6` (same default as your existing script). Do not scale this up aggressively — this is still one shared free API regardless of how many subreddits you spread across.
- Reuse `polite_get` **verbatim** from `download_arctic_shift_V2_3.py`: it already correctly handles `X-RateLimit-Remaining`/`X-RateLimit-Reset`, 429/422/5xx backoff. No changes needed there.

### 4.3 Output layout (scratch-friendly)

```
<scratch>/subreddit_extraction/
    raw_meta/
        <subreddit>/
            posts.jsonl.gz          # {"author":..., "created_utc":..., "id":...}
            comments.jsonl.gz
    _done/
        <subreddit>/
            posts.done              # {"count": N, "last_cursor": X}
            comments.done
    logs/
        run.log
        heartbeat.log              # or same file, same as existing script
    failures.log
```

**This raw_meta data is permanent, not scratch-temp-and-delete.** It is kept intentionally so that Stage 3 can be re-run later to draw additional, non-overlapping batches of sampled authors from the same subreddit without re-hitting the API (see §6.3). No script in this pipeline should ever delete `raw_meta/`.

Example line from `raw_meta/wallstreetbets/comments.jsonl.gz` (illustrative subreddit name):
```json
{"author": "some_user_123", "created_utc": 1717200045, "id": "kx8f2la"}
```

Example `_done/wallstreetbets/comments.done`:
```json
{"count": 284193, "last_cursor": 1735689600}
```

### 4.4 CLI

```
python extract_subreddit_authors.py \
    --subreddits-file subreddits_200plus.json \
    --output-dir /scratch/$USER/subreddit_extraction \
    --workers 6 \
    --min-sleep 0.2 \
    --heartbeat-interval 300 \
    --log-file /scratch/$USER/subreddit_extraction/logs/run.log \
    --max-pages-safety 500000   # see §6, safety net only
```

**`--subreddits-file` must accept both `.txt` and `.json`** (the user's actual list is a JSON file with slightly more than 200 subreddits — treat 200 as approximate, not a hardcoded count anywhere in the pipeline). Implement a `load_subreddits(path) -> list[str]` function with this contract:

- **Input**: path to a `.txt` or `.json` file.
- **Output**: a flat, deduped, order-preserving list of subreddit name strings (no `r/` prefix, no leading/trailing whitespace).
- **Behavior**:
  - `.txt`: one subreddit name per line, blank lines skipped (same pattern as the existing script's `load_authors()`).
  - `.json`: must handle, in order of attempt:
    1. A flat JSON array of strings: `["wallstreetbets", "investing", ...]`.
    2. A JSON array of objects, trying common keys in this order: `subreddit`, `name`, `display_name`.
    3. If neither shape matches, raise a clear `ValueError` naming the file and the unexpected shape — do not guess silently.
  - Log the final count of subreddits loaded at INFO level right after loading, before any network calls start, so a bad input file fails fast and visibly.
- This loader is used by Stage 1 only; Stages 2–4 operate on whatever subreddit subdirectories/files actually exist under `raw_meta/` / `author_pools/` / `samples/`, so they never need to re-read the original subreddit list.

Note: `--after`/`--before` are still exposed as optional flags (unset by default) purely so you can re-run a targeted backfill or narrow one problem subreddit later without code changes — they don't change the default full-history behavior.

### 4.5 Logging conventions (matches your existing script)

- One line per page fetched at INFO level with running counts, exactly like `fetch_author_records`.
- Heartbeat thread every 5 min listing active workers: `<subreddit>/<kind> (page N, M records)`.
- `failures.log`: `subreddit \t kind \t error` for anything that exhausted retries.
- Progress line every N subreddits completed with current rate-limit-remaining.

## 5. Stage 2 — `build_author_pools.py`

- For each subreddit, stream `raw_meta/<subreddit>/posts.jsonl.gz` and `comments.jsonl.gz`.
- Drop records where `author` is `None`, `""`, or `"[deleted]"`.
- Maintain an in-memory `dict[author] -> {n_posts, n_comments, first_seen_utc, last_seen_utc}`. This is safe memory-wise: even a subreddit with hundreds of thousands of unique authors is a few tens of MB as a dict of short strings + 4 ints.
- Write `author_pools/<subreddit>_authors.csv` with columns: `author, n_posts, n_comments, first_seen_utc, last_seen_utc`.
- **Dedup guarantee**: because the pool is a `dict` keyed by `author`, every author appears exactly once in this CSV regardless of how many posts/comments they made, and regardless of whether they appeared in the posts file, the comments file, or both. This is what makes Stage 3's `random.sample()` produce 1000 *distinct* authors per subreddit with no further dedup logic needed there.
- Log pool size per subreddit — this is your first real QA signal (e.g., a subreddit whose "200 first" placement was purely subscriber-count-driven but has a tiny active-author pool will show up here before you waste sampling effort on it).

## 6. Stage 3 — `sample_authors.py`

### 6.1 Default behavior (first run / batch 1)

- Read `author_pools/<subreddit>_authors.csv`, take the `author` column as the pool.
- **Reproducible per-subreddit seed**: `seed = hash_to_int(f"{GLOBAL_SEED}:{subreddit}:{batch_id}")` (e.g. via `int(hashlib.sha256(seed_string.encode()).hexdigest(), 16) % (2**32)`), so reruns are deterministic and independent per subreddit *and per batch*, but the whole run is reproducible from one `--seed` flag (default `42`, see §10 for what this means).
- `rng = random.Random(seed); sample = rng.sample(available_pool, min(1000, len(available_pool)))`.
- If `len(available_pool) < 1000`: sample everything, flag subreddit in the summary as **insufficient** — do not silently pad or oversample duplicates.
- Output: `samples/<subreddit>_batch1_sample1000.txt` (one username per line) + append a row to `sampling_summary.csv`: `subreddit, batch_id, pool_size, available_pool_size, n_sampled, seed_used, insufficient_flag, timestamp`.

### 6.2 CLI

```
python sample_authors.py \
    --pools-dir author_pools/ \
    --output-dir samples/ \
    --sample-size 1000 \
    --seed 42 \
    --batch-id 1
```

### 6.3 Repeat/resample support (batch 2+)

Because `raw_meta/` and `author_pools/` are persisted permanently (§4.3), you can draw additional non-overlapping batches later without touching the network:

- Maintain a per-subreddit registry file `samples/_registry/<subreddit>_sampled_so_far.txt` — the union of every author ever sampled for that subreddit across all batches.
- On each run: `available_pool = pool_authors - already_sampled_authors` (read from the registry), then sample from `available_pool` as in §6.1, using `batch_id` in the seed string so batch 2's draw is a different deterministic sequence from batch 1's, not just "whatever's left in iteration order."
- After writing `samples/<subreddit>_batch{N}_sample1000.txt`, append the newly sampled authors to the registry file.
- `--batch-id` must be supplied explicitly (no auto-increment guessing) so a re-run with the same `--batch-id` is idempotent (regenerates the same batch) rather than silently drawing a new one.
- If `len(available_pool) < sample-size` on a batch-2+ run, this means the subreddit's author pool is close to exhausted — flag it in the summary the same way as the insufficient-pool case in §6.1, and log a clear message stating how many unsampled authors remain.

**Runaway-safety knob (§4.4's `--max-pages-safety`)**: purely an operational guard — if a single subreddit/kind pagination exceeds this many pages, Stage 1 logs a loud WARNING and marks that subreddit/kind `.done` with whatever it has so far, rather than one pathological subreddit silently consuming the whole job's walltime. Default should be set high enough it basically never fires under normal conditions (e.g. 500,000 pages = 50M records) — it's a tripwire, not a limit you're expected to hit.

## 7. Stage 4 — `merge_master_list.py`

- Concatenate all `samples/<subreddit>_batch*_sample1000.txt` files present (accept a `--batches` filter, e.g. `--batches 1` to merge only batch 1 across all subreddits, or `--batches all` to include every batch ever drawn).
- Because finance subreddits overlap heavily in userbase, the **same author will appear under multiple subreddits**. For the downstream full-history author script you don't want to re-download the same user twice.
- Output:
  - `master/all_sampled_authors.txt` — globally deduped, ready to hand to `download_arctic_shift_V2_3.py` as `--humans-file` (per your confirmed workflow, §8.1).
  - `master/author_subreddit_membership.csv` — `author, subreddit, batch_id` (long format), so you don't lose which subreddit(s)/batch qualified each author for downstream stratified analysis.
- This script should be safe to re-run any time a new batch is drawn — it always regenerates `master/` fresh from whatever `samples/` currently contains, it does not incrementally append.

### 7.1 CLI

```
python merge_master_list.py \
    --samples-dir samples/ \
    --output-dir master/ \
    --batches 1
```

## 8. Feasibility / rate-limit summary

- No fixed published rate limit — it's dynamic via response headers, and your existing `polite_get` already handles it correctly. Reusing it here is the right call, not a rewrite.
- The actual bottleneck is **volume, not rate limit**: 200 subreddits × up to full history × posts+comments. With field-limiting this is far more tractable than pulling full records, but a few large subreddits could still take a long time. Since you've chosen full history, my main recommendation is: **run Stage 1 first, watch the per-subreddit pool-size logs from Stage 2 as they land, and only intervene (e.g., manually re-run a specific subreddit with `--before` to cap it) if one subreddit is clearly an outlier** — don't pre-optimize for a problem you may not actually have across all 200.
- Run on the login/data-transfer node exactly as your existing script's docstring already instructs — same reasoning applies (compute nodes often block outbound internet; hammering the free API from many parallel nodes is against its spirit anyway).

## 8. Downstream integration (confirmed)

### 8.1 Feeding `download_arctic_shift_V2_3.py`

No patch to the existing script. Pass:
```
python download_arctic_shift_V2_3.py \
    --bots-file empty_bots.txt \
    --humans-file master/all_sampled_authors.txt \
    --output-dir <scratch>/author_full_history \
    --after 2025-01-01 --before 2026-01-01 \
    --workers 6
```
`empty_bots.txt` is just a zero-byte (or whitespace-only) file — `load_authors()` already handles an empty file gracefully (returns `[]`), so no code change is needed there.

## 9. Confirmed decisions (resolved)

1. **Integration**: thin-wrapper approach (§8.1) — no patch to `download_arctic_shift_V2_3.py`.
2. **Seed**: default `--seed 42` for Stage 3, documented per-run in `sampling_summary.csv`. See §10 for what the seed does and why the value itself doesn't matter as long as it's fixed and recorded.
3. **Raw data retention**: `raw_meta/` and `author_pools/` are permanent, not deleted by any script (§4.3). This exists specifically to support drawing additional non-overlapping author batches later (§6.3) without re-hitting the API.

## 10. Appendix: what the random seed does

A pseudo-random number generator (PRNG) needs a starting value — the "seed" — to produce its sequence of "random" numbers. Same seed + same input list, in the same order, → **identical** output every single time. No seed (or a seed derived from wall-clock time) → a different sample on every run, even with identical inputs, which makes the pipeline unreproducible: you (or a reviewer) couldn't regenerate the same 1000 users to verify or extend results later.

Concretely, in Stage 3: `random.Random(seed).sample(pool, 1000)` — the `seed` is computed from `(GLOBAL_SEED, subreddit, batch_id)`, so:
- Two different subreddits get two different, independent draws (no correlation between them).
- Batch 2 for the same subreddit gets a *different* deterministic draw than batch 1 (not just "next N in some order").
- Re-running Stage 3 with the same `--seed`/`--batch-id` on unchanged input pools always reproduces the exact same sample — this is what lets you regenerate `master/all_sampled_authors.txt` from scratch at any time and get back the same list.

The specific integer value of `GLOBAL_SEED` (`42` here) has no special meaning — any fixed integer produces an equally valid, equally random-looking, equally reproducible sample. It only matters that it's (a) fixed rather than time-based, and (b) recorded in `sampling_summary.csv` so the run is auditable later.
