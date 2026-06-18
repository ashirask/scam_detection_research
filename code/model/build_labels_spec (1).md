# `build_labels.py` — Specification

## Purpose

Reads raw JSONL files (comments and submissions from Reddit Pushshift dumps) plus an optional
sampled-authors `.txt` file and a pre-downloaded BotRank CSV, aggregates all activity per
author, and produces a `labels.parquet` file with one row per user containing:
- a binary label (`bot` or `human`) — no unknown category
- the source of that label (`rule`, `botrank`)
- a numeric `y` column (1 = bot, 0 = human)
- basic per-user post counts used downstream to filter sparse accounts

---

## Inputs

### 1. Comments JSONL (`--comments`)
One JSON object per line. **Store all fields** from the raw record during aggregation — do not
pre-filter columns at this stage, since `build_features.py` may need fields not listed here.
Key fields used in labeling and aggregation:

| Field | Type | Description |
|---|---|---|
| `author` | str | Reddit username (skip if `[deleted]` or empty) |
| `author_fullname` | str | Internal Reddit user ID (`t2_...`) |
| `created_utc` | float | Unix timestamp of comment |
| `subreddit` | str | Subreddit name |
| `body` | str | Comment text |
| `link_id` | str | Parent submission ID (`t3_...`) |
| `parent_id` | str | Parent comment or submission ID |
| `score` | int | Net upvotes |
| `distinguished` | str\|null | `"moderator"` if mod-distinguished |
| `controversiality` | int | 0 or 1 |
| `collapsed` | bool | Whether comment was collapsed |
| `collapsed_reason` | str\|null | Reason for collapse (e.g. spam) |
| `is_submitter` | bool | Whether commenter is also the OP |
| `source_period` | str | e.g. `"2024-01"` — month of source file |

Example record:
```json
{
  "author": "daddylikeabosss",
  "author_fullname": "t2_efdb5jqxt",
  "created_utc": 1704067200.0,
  "subreddit": "SluttyConfessions",
  "body": "Kyle, it's Morgan.  Don't fuck up",
  "link_id": "t3_18vj1sf",
  "parent_id": "t1_kfrmzi5",
  "score": 3,
  "distinguished": null,
  "controversiality": 0,
  "collapsed": false,
  "collapsed_reason": null,
  "is_submitter": false,
  "source_period": "2024-01"
}
```

### 2. Submissions JSONL (`--submissions`)
One JSON object per line. **Store all fields** from the raw record. Key fields:

| Field | Type | Description |
|---|---|---|
| `author` | str | Reddit username |
| `author_fullname` | str | Internal Reddit user ID (`t2_...`) |
| `created_utc` | float | Unix timestamp of submission |
| `subreddit` | str | Subreddit name |
| `title` | str | Post title |
| `selftext` | str | Body text (empty string for link posts) |
| `url` | str | Link URL or Reddit image URL |
| `domain` | str | Domain of the linked URL |
| `score` | int | Net upvotes |
| `upvote_ratio` | float | Fraction of upvotes (0.0–1.0) |
| `num_comments` | int | Number of comments on the post |
| `num_crossposts` | int | Number of times crossposted |
| `is_self` | bool | True = text post, False = link post |
| `over_18` | bool | NSFW flag |
| `spoiler` | bool | Spoiler flag |
| `locked` | bool | Whether post was locked |
| `distinguished` | str\|null | `"moderator"` if mod-distinguished |
| `source_period` | str | e.g. `"2024-01"` |

Example record:
```json
{
  "author": "rtbot2",
  "author_fullname": "t2_11rjpo",
  "created_utc": 1704067205.0,
  "subreddit": "realtech",
  "title": "Chief justice centers Supreme Court annual report on AI's dangers",
  "selftext": "",
  "url": "https://thehill.com/regulation/...",
  "domain": "thehill.com",
  "score": 1,
  "upvote_ratio": 1.0,
  "num_comments": 1,
  "num_crossposts": 0,
  "is_self": false,
  "over_18": false,
  "spoiler": false,
  "locked": false,
  "distinguished": null,
  "source_period": "2024-01"
}
```

### 3. Sampled authors `.txt` (`--authors-file`, optional but recommended)
Plain text file, one Reddit username per line. Represents the random user sample produced
during data extraction. If provided, **filter to only these authors at parse time** — skip any
record whose `author` is not in this set. This is the primary scoping mechanism and avoids
accumulating records for users outside the study population.

```
rtbot2
MarfromMke
daddylikeabosss
some_user_123
...
```

### 4. BotRank CSV (`--botrank`, optional)
Pre-downloaded from `https://botrank.pastimes.eu/` (see Section: Obtaining BotRank Data).
Expected columns: `username`, `score` (float 0–1, higher = more likely bot).
Only the **top 500 by score** are used (configurable via `--botrank-top-n`, default 500).
This follows the BotBusters paper methodology and avoids noisy mid-confidence entries.

---

## Obtaining BotRank Data (one-time setup)

Run the standalone helper script `fetch_botrank.py` before running `build_labels.py`.

## Processing Steps

### Step 1 — Load authors filter

If `--authors-file` is provided, load all usernames into a `set[str]` (case-sensitive, as
Reddit usernames are case-sensitive in practice). This set is used to gate all record
processing in Steps 2–3.

### Step 2 — Stream and aggregate comments JSONL

Parse line by line using `orjson` (fallback: `json`). For each line:
1. Parse JSON.
2. Extract `author`. Skip if `author` is `[deleted]`, `AutoModerator`, or empty string `""`.
3. If authors filter is active and `author` not in filter set: skip.
4. Store the **full parsed record** (all fields) into a per-author accumulator:

```python
user_comments[author].append(record)   # list of full comment dicts
```

### Step 3 — Stream and aggregate submissions JSONL

Same as Step 2 but for submissions:

```python
user_submissions[author].append(record)   # list of full submission dicts
```

### Step 4 — Compute per-author summary stats

For each author seen across both files, compute:

```python
{
  "author": str,
  "author_fullname": str,          # first seen value from either file
  "num_comments": int,             # len(user_comments[author])
  "num_submissions": int,          # len(user_submissions[author])
  "total_posts": int,              # num_comments + num_submissions
  "first_seen_utc": float,         # min created_utc across all records
  "last_seen_utc": float,          # max created_utc across all records
  "account_span_days": float,      # (last_seen_utc - first_seen_utc) / 86400
}
```

### Step 5 — Username-based rule labeling

Apply to every author. **Label priority: bot patterns are checked first. If any bot pattern
matches, label = `bot`. If no bot pattern matches, label = `human`.** There is no unknown
category.

#### 5a. Bot patterns (label = `bot`, source = `rule`)

Evaluate all of the following on `username.lower()` unless noted:

| Pattern | Implementation | Example matches |
|---|---|---|
| Exact match `"bot"` | `username.lower() == "bot"` | `bot` |
| Whole-word `bot` | `re.search(r'\bbot\b', username.lower())` | `newsbot`, `Link_Bot`, `reddit_bot` |
| Starts with `Auto` (case-sensitive) | `username.startswith("Auto")` | `AutoModerator`, `AutoTLDR` |
| Whole-word `auto` | `re.search(r'\bauto\b', username.lower())` | `auto_poster`, `the_auto` |
| Whole-word `mod` | `re.search(r'\bmod\b', username.lower())` | `ModHelper`, `sub_mod` |
| Underscore-bounded `_bot` or `bot_` | `re.search(r'(^bot_|_bot$|_bot_)', username.lower())` | `link_bot_v2` |

#### 5b. False-positive guard

Before finalizing a `bot` label from the `\bbot\b` pattern only (not the other patterns),
verify the match is not an incidental substring. If the lowercased username exactly equals
one of these words, do **not** label as bot:

```python
BOT_FALSE_POSITIVES = {
    "bottle", "bottom", "botox", "both", "bother", "botanical",
    "botanic", "botswana", "bought", "boots", "booth"
}
```

Also: if `\bbot\b` matched but it's part of a compound like `robotics` or `robots`, the
match is already blocked by the word-boundary anchor. No extra check needed for those.

#### 5c. Default (no bot pattern matched)

```
label = "human"
label_source = "rule"
label_confidence = 1.0
```

### Step 6 — BotRank augmentation (if `--botrank` provided)

1. Load the CSV. Sort by `score` descending. Take top `--botrank-top-n` rows (default 500).
2. Build a lookup dict: `{username.lower(): score}`.
3. For each author:
   - Look up `author.lower()` in the BotRank dict.
   - If found and score ≥ `--botrank-threshold` (default 0.8):
     - Override label to `bot`, source to `botrank`, confidence to the score value.
     - Set `botrank_score = score`.
   - If found but score < threshold: store `botrank_score` but do not change label.
   - If not found: `botrank_score = NaN`.
4. **BotRank only asserts `bot` — it never flips a username-matched bot to `human`.**
   No label conflict flag needed since BotRank only augments the bot class.

### Step 7 — Compute numeric Y

```python
y = 1   if label == "bot"
y = 0   if label == "human"
# No NaN — every author gets a label
```

### Step 8 — Filter sparse accounts

Drop authors with `total_posts < --min-posts` (default 3). Log count dropped. These accounts
have insufficient signal for feature engineering.

---

## Output

### `labels.parquet`

One row per author. All columns:

| Column | Type | Description |
|---|---|---|
| `author` | str | Reddit username (join key for all downstream files) |
| `author_fullname` | str | Reddit internal ID (`t2_...`) |
| `label` | str | `"bot"` or `"human"` |
| `label_source` | str | `"rule"` or `"botrank"` |
| `label_confidence` | float | 1.0 for rule; BotRank score for botrank matches |
| `y` | int | 1 (bot) or 0 (human) |
| `num_comments` | int | Total comments in dataset |
| `num_submissions` | int | Total submissions in dataset |
| `total_posts` | int | num_comments + num_submissions |
| `first_seen_utc` | float | Earliest activity timestamp |
| `last_seen_utc` | float | Latest activity timestamp |
| `account_span_days` | float | Span of observed activity in days |
| `botrank_score` | float | Raw BotRank score (NaN if not in BotRank) |

### `label_summary.txt`

Printed to stdout and optionally written to file (`--summary`):

```
=== Label Summary ===
Total authors in sample        : 5000
  Authors in JSONL             : 4980    (20 in txt not found in JSONL)
  Dropped (sparse < 3 posts)   : 142

Final labeled dataset          : 4838
  bot  (rule)                  : 280
  bot  (botrank override)      : 38
  human                        : 4520

BotRank top-N used             : 500
  BotRank matches in sample    : 38
Output written to              : labels.parquet
```

---

## CLI Interface

```bash
python build_labels.py \
  --comments       /path/to/comments.jsonl \
  --submissions    /path/to/submissions.jsonl \
  [--authors-file  /path/to/sampled_authors.txt] \
  [--botrank       /path/to/botrank_full.csv] \
  [--botrank-top-n        500] \
  [--botrank-threshold    0.8] \
  [--min-posts            3] \
  [--output        labels.parquet] \
  [--summary       label_summary.txt]
```

---

## Dependencies

```
orjson        # fast JSONL parsing (pip install orjson); fallback: stdlib json
pandas
pyarrow       # parquet output (pip install pyarrow)
re            # stdlib
argparse      # stdlib
# For fetch_botrank.py only:
requests
beautifulsoup4
```

---

## Implementation Notes

- Stream JSONL files line by line — never `json.load()` the whole file at once.
- Store full raw records per user (not just selected fields) so `build_features.py` has
  access to everything without re-reading the JSONL.
- After `build_labels.py` completes, the raw per-user comment/submission lists are written
  to disk as `user_comments.jsonl` and `user_submissions.jsonl` (one JSON object per line,
  keyed by author) so `build_features.py` can read them without re-parsing the full dump.
- Use `author` (username) as the universal join key — this matches the sampled-authors txt
  and is the key used in BotRank.
- Log every label decision at DEBUG level for auditability:
  `DEBUG: rtbot2 → bot [rule: startswith('Auto') match]`
  `DEBUG: some_news_bot → bot [rule: \bbot\b whole-word match]`
  `DEBUG: MarfromMke → human [rule: no bot pattern]`
  `DEBUG: covert_bot_99 → bot [botrank: score=0.97, top-500]`
