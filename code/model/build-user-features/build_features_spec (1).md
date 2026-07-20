# `build_features.py` — Specification (Tier 1)

## Purpose

Computes per-author features (X) and assigns labels (y) for all authors in both the bot
and human populations. Outputs a single `dataset.parquet` file — one row per author, all
features plus label — ready for model training.

Labels are assigned inside this script based on which input files an author came from:
- Authors from bot input files → `y = 1`
- Authors from human input files → `y = 0`

No separate labels file is needed.

---

## Input File Format

### Bot-side files (from `merge_bot_extractions.py`)
```
user_comments_bots.jsonl      — one line per bot author
user_submissions_bots.jsonl   — one line per bot author
```

### Human-side files (from earlier random-sample extraction)
```
user_comments_humans.jsonl      — one line per human author
user_submissions_humans.jsonl   — one line per human author
```

### Line format — comments file
Each line is one JSON object:
```json
{
  "author": "SomeBot_v2",
  "comments": [
    {
      "author": "SomeBot_v2",
      "author_fullname": "t2_11rjpo",
      "created_utc": 1704067200.0,
      "body": "DLPA. And mushies (psilocybin).",
      "subreddit": "Biohackers",
      "score": 1,
      "controversiality": 0,
      "edited": false,
      "collapsed": false,
      "link_id": "t3_1l6trks",
      "parent_id": "t3_1l6trks",
      "id": "mwtxkjg",
      "no_follow": true,
      "banned_at_utc": null,
      "banned_by": null,
      "distinguished": null,
      "author_premium": false,
      "author_patreon_flair": false,
      ...all other fields preserved from extraction...
    },
    ...
  ]
}
```

### Line format — submissions file
Each line is one JSON object:
```json
{
  "author": "SomeUser",
  "submissions": [
    {
      "author": "SomeUser",
      "author_fullname": "t2_1pdfrf7o5j",
      "created_utc": 1756392700.0,
      "title": "Chlorine",
      "selftext": "[removed]",
      "subreddit": "Biohackers",
      "score": 1,
      "upvote_ratio": 1.0,
      "num_comments": 0,
      "num_crossposts": 0,
      "over_18": false,
      "edited": false,
      "is_self": true,
      "domain": "self.Biohackers",
      "url": "https://www.reddit.com/r/Biohackers/...",
      "banned_at_utc": null,
      "banned_by": null,
      "removed_by_category": "moderator",
      "distinguished": null,
      "no_follow": true,
      "author_premium": false,
      ...all other fields preserved from extraction...
    },
    ...
  ]
}
```

All fields from the original Pushshift records are preserved — the extraction scripts did
not strip any fields. Do not use fields inside `_meta` as features — they are
extractor-added metadata with inconsistent keys across different extractions and carry
no bot detection signal. Simply ignore the `_meta` key when accessing record fields;
all feature-relevant fields are at the top level of each record.

---

## Pre-computation Step: Domain Whitelist (`build_domain_whitelist.py`)

Must be run ONCE before `build_features.py`. Scans all four input JSONL files, extracts
every URL from `body` (comments), `selftext`, and `url` fields (submissions), counts
domain frequency across the full bot+human dataset, and writes the top-N domains to a
text file.

### URL extraction

```python
import re
URL_PATTERN = re.compile(r'https?://[^\s\)\]\"\']+')

def extract_urls(text):
    return URL_PATTERN.findall(text or "")

def extract_domain(url):
    # extract domain only, strip www.
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        return domain.lstrip("www.")
    except Exception:
        return None
```

### Sources per record type
- Comments: extract URLs from `body` field
- Submissions: extract URLs from both `selftext` field AND `url` field (the submission
  link itself)

### Output: `domain_whitelist.txt`
One domain per line, top-N by frequency across all authors (default N=500).

```bash
python build_domain_whitelist.py \
  --comments-bot    user_comments_bots.jsonl \
  --comments-human  user_comments_humans.jsonl \
  --submissions-bot user_submissions_bots.jsonl \
  --submissions-human user_submissions_humans.jsonl \
  --top-n           500 \
  --output          domain_whitelist.txt
```

`build_domain_whitelist.py` is a standalone script (~50 lines). It does not need a
separate spec — implement it as a simple counter over all records in all four files,
extract domains, write top-N to text file.

---

## Pre-computation Step: Comment Lookup Dict (temporal features only)

Required only when `--temporal` flag is passed. Built once in memory at script startup
before per-author feature computation begins.

Scans ALL comment records from both bot and human comment files, builds:
```python
comment_lookup = {}   # {comment_id: created_utc}
# key: record["id"]  (e.g. "mwtxkjg")
# value: record["created_utc"]  (e.g. 1749477202.0)
```

This enables reply-time computation: for a comment whose `parent_id` starts with `t1_`,
strip the prefix to get the parent comment ID, look it up in `comment_lookup` to get its
timestamp, subtract to get reply latency.

```python
def build_comment_lookup(comments_bot_path, comments_human_path):
    lookup = {}
    for path in [comments_bot_path, comments_human_path]:
        for author_record in stream_jsonl(path):
            for comment in author_record.get("comments", []):
                cid = comment.get("id")
                ts  = comment.get("created_utc")
                if cid and ts:
                    lookup[cid] = ts
    return lookup
```

Memory estimate: ~10M comments × ~30 bytes per entry ≈ ~300MB. Acceptable on a standard
compute node. If memory is a concern, use a SQLite on-disk dict instead.

---

## Processing Flow

```
1. Load domain whitelist → set
2. If --temporal: build comment_lookup dict
3. For each population (bots, humans):
     For each author in the JSONL file:
       a. Load author's comments list and submissions list
       b. Call each feature-group function
       c. Collect results into one flat dict per author
       d. Assign y = 1 (bot) or y = 0 (human)
4. Combine all author dicts into DataFrame
5. Write dataset.parquet
```

Both populations are processed by identical feature functions — no branching on bot vs
human inside any feature computation. The only difference is the `y` assignment at step 3d.

---

## Feature Functions

Each feature group is implemented as a standalone function taking the author's comment
list and/or submission list and returning a flat dict of feature_name → value.

Missing values (e.g. no submissions → submission features are NaN) are returned as
`float("nan")` — not 0, not -1. Tree-based models (LightGBM, XGBoost) handle NaN
natively. For models that don't (neural nets, TabPFN), impute with median in the
dataset assembly step.

---

### Feature Group 1 — Activity / Volume

**Input:** comments list, submissions list
**No fields required beyond list lengths and timestamps**

```python
def activity_features(comments, submissions):
    n_c = len(comments)
    n_s = len(submissions)
    all_ts = [r["created_utc"] for r in comments + submissions if r.get("created_utc")]

    if len(all_ts) >= 2:
        span_days = (max(all_ts) - min(all_ts)) / 86400
    else:
        span_days = 0.0

    span_safe = max(span_days, 1.0)   # avoid division by zero

    return {
        "num_comments":        n_c,
        "num_submissions":     n_s,
        "total_posts":         n_c + n_s,
        "comments_per_day":    n_c / span_safe,
        "submissions_per_day": n_s / span_safe,
        "account_age_days":    span_days,
    }
```

---

### Feature Group 2 — Temporal / Reply Time (optional)

**Input:** comments list, comment_lookup dict
**Only computed when `--temporal` flag is passed**
**Only uses comment→comment replies — NOT submission→comment**

A comment is a reply to another comment when `parent_id` starts with `"t1_"`.
Fast-reply-to-submission (parent_id starts with `"t3_"`) is excluded per methodology.

```python
def temporal_features(comments, comment_lookup):
    reply_times = []

    for c in comments:
        parent_id = c.get("parent_id", "")
        if not parent_id.startswith("t1_"):
            continue   # top-level comment, not a reply to another comment
        parent_comment_id = parent_id[3:]   # strip "t1_" prefix
        parent_ts = comment_lookup.get(parent_comment_id)
        if parent_ts is None:
            continue   # parent not in dataset
        reply_time = c["created_utc"] - parent_ts
        if reply_time >= 0:   # guard against timestamp anomalies
            reply_times.append(reply_time)

    if not reply_times:
        return {
            "mean_reply_time_seconds": float("nan"),
            "min_reply_time_seconds":  float("nan"),
            "std_reply_time_seconds":  float("nan"),
            "reply_time_coverage":     0.0,   # diagnostic only, not a model feature
        }

    import statistics
    t1_replies = sum(1 for c in comments if c.get("parent_id","").startswith("t1_"))

    return {
        "mean_reply_time_seconds": sum(reply_times) / len(reply_times),
        "min_reply_time_seconds":  min(reply_times),
        "std_reply_time_seconds":  statistics.pstdev(reply_times) if len(reply_times) > 1
                                   else 0.0,
        "reply_time_coverage":     len(reply_times) / max(t1_replies, 1),
    }
```

`reply_time_coverage` is written to the dataset for diagnostics but should be excluded
from model training features (it measures data availability, not bot behavior).

---

### Feature Group 3 — Engagement / Karma

**Input:** comments list, submissions list
**Fields used:** `score` (both), `upvote_ratio` (submissions only),
`num_comments` (submissions only)

```python
def engagement_features(comments, submissions):
    c_scores = [c["score"] for c in comments if "score" in c]
    s_scores = [s["score"] for s in submissions if "score" in s]
    upvote_ratios = [s["upvote_ratio"] for s in submissions
                     if s.get("upvote_ratio") is not None]
    comments_received = [s["num_comments"] for s in submissions
                         if "num_comments" in s]

    mean_c_score = sum(c_scores) / len(c_scores) if c_scores else float("nan")
    mean_s_score = sum(s_scores) / len(s_scores) if s_scores else float("nan")

    if mean_c_score is not float("nan") and mean_s_score is not float("nan"):
        ratio = mean_c_score / (abs(mean_s_score) + 1)
    else:
        ratio = float("nan")

    return {
        "mean_comment_score":                mean_c_score,
        "mean_submission_score":             mean_s_score,
        "comment_to_post_score_ratio":       ratio,
        "mean_upvote_ratio":                 sum(upvote_ratios) / len(upvote_ratios)
                                             if upvote_ratios else float("nan"),
        "mean_comments_received_per_submission": sum(comments_received) / len(comments_received)
                                                 if comments_received else float("nan"),
    }
```

---

### Feature Group 4 — Moderation Signals

**Input:** comments list, submissions list
**Fields used:**
- `banned_at_utc` — present on both comments and submissions; not null if account was
  banned at time of post
- `removed_by_category` — submissions only; value `"moderator"` means mod-removed
- `controversiality` — comments only; 0 or 1

```python
def moderation_features(comments, submissions):
    all_records = comments + submissions
    total = max(len(all_records), 1)

    banned_count = sum(1 for r in all_records if r.get("banned_at_utc") is not None)

    mod_removed = sum(1 for s in submissions
                      if s.get("removed_by_category") == "moderator")

    controversiality_vals = [c.get("controversiality", 0) for c in comments]

    return {
        "banned_post_ratio":             banned_count / total,
        "submission_mod_removed_ratio":  mod_removed / max(len(submissions), 1),
        "controversiality_mean":         sum(controversiality_vals) / len(controversiality_vals)
                                         if controversiality_vals else float("nan"),
    }
```

---

### Feature Group 5 — Subreddit Activity

**Input:** comments list, submissions list
**Fields used:** `subreddit` (both), `over_18` (submissions only)

```python
import math
from collections import Counter

def subreddit_features(comments, submissions):
    all_subreddits = (
        [c["subreddit"] for c in comments   if c.get("subreddit")] +
        [s["subreddit"] for s in submissions if s.get("subreddit")]
    )

    if not all_subreddits:
        return {
            "num_unique_subreddits":      float("nan"),
            "top_subreddit_concentration": float("nan"),
            "subreddit_entropy":          float("nan"),
            "nsfw_subreddit_ratio":       float("nan"),
        }

    counts = Counter(all_subreddits)
    total  = sum(counts.values())

    # entropy of subreddit distribution
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )

    # fraction of posts in single most-used subreddit
    top_concentration = counts.most_common(1)[0][1] / total

    # nsfw ratio — from over_18 field on submissions only
    nsfw_count = sum(1 for s in submissions if s.get("over_18") is True)
    nsfw_ratio = nsfw_count / max(len(submissions), 1)

    return {
        "num_unique_subreddits":       len(counts),
        "top_subreddit_concentration": top_concentration,
        "subreddit_entropy":           entropy,
        "nsfw_subreddit_ratio":        nsfw_ratio,
    }
```

---

### Feature Group 6 — Text Stylometrics

**Input:** comments list, submissions list
**Text sources per record type:**
- Comments: `body` field
- Submissions: `title` field + `selftext` field (concatenated with a space if both present)

```python
import re, math
from collections import Counter

def get_texts(comments, submissions):
    texts = []
    for c in comments:
        body = c.get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            texts.append(body)
    for s in submissions:
        title    = s.get("title", "")
        selftext = s.get("selftext", "")
        if selftext in ("[deleted]", "[removed]", ""):
            selftext = ""
        combined = (title + " " + selftext).strip()
        if combined:
            texts.append(combined)
    return texts

def stylometric_features(comments, submissions):
    texts = get_texts(comments, submissions)
    if not texts:
        return {k: float("nan") for k in [
            "mean_text_length", "type_token_ratio", "repetition_ratio",
            "uppercase_ratio", "url_density", "caret_count_mean",
            "asterisk_count_mean", "max_consecutive_carets",
            "max_consecutive_asterisks", "whitespace_entropy"
        ]}

    # mean text length
    mean_len = sum(len(t) for t in texts) / len(texts)

    # type-token ratio across all text combined
    all_text = " ".join(texts).lower()
    tokens   = re.findall(r'\b[a-z]+\b', all_text)
    ttr = len(set(tokens)) / max(len(tokens), 1)

    # repetition ratio — fraction of posts that are exact duplicates of another
    normalized = [t.strip().lower() for t in texts]
    counts     = Counter(normalized)
    dup_count  = sum(c - 1 for c in counts.values() if c > 1)
    rep_ratio  = dup_count / max(len(texts), 1)

    # uppercase ratio — over all alphabetic characters
    alpha_chars = [ch for ch in all_text if ch.isalpha()]
    upper_count = sum(1 for ch in "".join(texts) if ch.isupper())
    upper_ratio = upper_count / max(len(alpha_chars), 1)

    # URL density — urls per post
    URL_RE = re.compile(r'https?://\S+')
    total_urls = sum(len(URL_RE.findall(t)) for t in texts)
    url_density = total_urls / len(texts)

    # caret and asterisk features
    caret_counts    = [t.count("^") for t in texts]
    asterisk_counts = [t.count("*") for t in texts]

    def max_consecutive(text, char):
        max_run = 0
        current = 0
        for ch in text:
            if ch == char:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    max_carets    = max((max_consecutive(t, "^") for t in texts), default=0)
    max_asterisks = max((max_consecutive(t, "*") for t in texts), default=0)

    # whitespace entropy — entropy of whitespace token distribution per post, then mean
    def ws_entropy(text):
        tokens = re.findall(r'\s+', text)
        if not tokens:
            return 0.0
        c = Counter(tokens)
        total = sum(c.values())
        return -sum((v / total) * math.log2(v / total) for v in c.values())

    mean_ws_entropy = sum(ws_entropy(t) for t in texts) / len(texts)

    return {
        "mean_text_length":        mean_len,
        "type_token_ratio":        ttr,
        "repetition_ratio":        rep_ratio,
        "uppercase_ratio":         upper_ratio,
        "url_density":             url_density,
        "caret_count_mean":        sum(caret_counts) / len(caret_counts),
        "asterisk_count_mean":     sum(asterisk_counts) / len(asterisk_counts),
        "max_consecutive_carets":  max_carets,
        "max_consecutive_asterisks": max_asterisks,
        "whitespace_entropy":      mean_ws_entropy,
    }
```

---

### Feature Group 7 — URL Features

**Input:** comments list, submissions list, domain_whitelist set
**Requires:** `domain_whitelist.txt` loaded before per-author processing

```python
def url_features(comments, submissions, domain_whitelist):
    URL_RE = re.compile(r'https?://\S+')

    all_urls = []
    for c in comments:
        all_urls.extend(URL_RE.findall(c.get("body", "") or ""))
    for s in submissions:
        all_urls.extend(URL_RE.findall(s.get("selftext", "") or ""))
        link_url = s.get("url", "")
        if link_url and link_url.startswith("http"):
            all_urls.append(link_url)

    if not all_urls:
        return {
            "suspicious_url_ratio": float("nan"),
            "url_domain_entropy":   float("nan"),
        }

    domains = [extract_domain(u) for u in all_urls]
    domains = [d for d in domains if d]   # filter None

    suspicious = sum(1 for d in domains if d not in domain_whitelist)
    susp_ratio = suspicious / len(domains)

    # domain entropy
    domain_counts = Counter(domains)
    total = sum(domain_counts.values())
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in domain_counts.values()
    )

    return {
        "suspicious_url_ratio": susp_ratio,
        "url_domain_entropy":   entropy,
    }
```

---

### Feature Group 8 — Username Features

**Input:** author string only — no post records needed

```python
import math
from collections import Counter

def username_features(author):
    length = len(author)
    digits = sum(1 for ch in author if ch.isdigit())
    alpha  = sum(1 for ch in author if ch.isalpha())

    # character entropy
    counts = Counter(author.lower())
    total  = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())

    return {
        "username_length":             length,
        "username_digit_ratio":        digits / max(length, 1),
        "username_entropy":            entropy,
        "username_has_digits_at_end":  int(bool(author and author[-1].isdigit())),
        "username_has_underscore":     int("_" in author),
    }
```

---

## Main Processing Loop

```python
def process_population(comments_path, submissions_path, label,
                        domain_whitelist, comment_lookup, args):
    results = []

    # load submissions into a dict keyed by author for O(1) lookup
    sub_by_author = {}
    for record in stream_jsonl(submissions_path):
        sub_by_author[record["author"]] = record.get("submissions", [])

    for record in stream_jsonl(comments_path):
        author    = record["author"]
        comments  = record.get("comments", [])
        submissions = sub_by_author.get(author, [])

        feats = {"author": author, "y": label}
        feats.update(activity_features(comments, submissions))
        feats.update(engagement_features(comments, submissions))
        feats.update(moderation_features(comments, submissions))
        feats.update(subreddit_features(comments, submissions))
        feats.update(stylometric_features(comments, submissions))
        feats.update(url_features(comments, submissions, domain_whitelist))
        feats.update(username_features(author))

        if args.temporal and comment_lookup is not None:
            feats.update(temporal_features(comments, comment_lookup))

        results.append(feats)

    return results
```

---

## Output: `dataset.parquet`

One row per author. Columns:

| Column | Type | Notes |
|---|---|---|
| `author` | str | Join key |
| `y` | int | 1 = bot, 0 = human |
| `num_comments` | int | |
| `num_submissions` | int | |
| `total_posts` | int | |
| `comments_per_day` | float | |
| `submissions_per_day` | float | |
| `account_age_days` | float | |
| `mean_reply_time_seconds` | float | NaN if --temporal not used or no parents found |
| `min_reply_time_seconds` | float | NaN if --temporal not used |
| `std_reply_time_seconds` | float | NaN if --temporal not used |
| `reply_time_coverage` | float | Diagnostic — exclude from model training |
| `mean_comment_score` | float | |
| `mean_submission_score` | float | |
| `comment_to_post_score_ratio` | float | |
| `mean_upvote_ratio` | float | |
| `mean_comments_received_per_submission` | float | |
| `banned_post_ratio` | float | |
| `submission_mod_removed_ratio` | float | |
| `controversiality_mean` | float | |
| `num_unique_subreddits` | int | |
| `top_subreddit_concentration` | float | |
| `subreddit_entropy` | float | |
| `nsfw_subreddit_ratio` | float | |
| `mean_text_length` | float | |
| `type_token_ratio` | float | |
| `repetition_ratio` | float | |
| `uppercase_ratio` | float | |
| `url_density` | float | |
| `caret_count_mean` | float | |
| `asterisk_count_mean` | float | |
| `max_consecutive_carets` | int | |
| `max_consecutive_asterisks` | int | |
| `whitespace_entropy` | float | |
| `suspicious_url_ratio` | float | NaN if no URLs |
| `url_domain_entropy` | float | NaN if no URLs |
| `username_length` | int | |
| `username_digit_ratio` | float | |
| `username_entropy` | float | |
| `username_has_digits_at_end` | int | 0 or 1 |
| `username_has_underscore` | int | 0 or 1 |

Total: **42 features** + `author` + `y` = 44 columns.

---

## CLI Interface

```bash
# Step 1 — build domain whitelist (run once)
python build_domain_whitelist.py \
  --comments-bot      user_comments_bots.jsonl \
  --comments-human    user_comments_humans.jsonl \
  --submissions-bot   user_submissions_bots.jsonl \
  --submissions-human user_submissions_humans.jsonl \
  --top-n             500 \
  --output            domain_whitelist.txt

# Step 2 — build features (without temporal)
python build_features.py \
  --comments-bot      user_comments_bots.jsonl \
  --comments-human    user_comments_humans.jsonl \
  --submissions-bot   user_submissions_bots.jsonl \
  --submissions-human user_submissions_humans.jsonl \
  --domain-whitelist  domain_whitelist.txt \
  --output            dataset.parquet

# Step 2 — build features (with temporal)
python build_features.py \
  --comments-bot      user_comments_bots.jsonl \
  --comments-human    user_comments_humans.jsonl \
  --submissions-bot   user_submissions_bots.jsonl \
  --submissions-human user_submissions_humans.jsonl \
  --domain-whitelist  domain_whitelist.txt \
  --temporal \
  --output            dataset.parquet
```

---

## Summary Output

Printed to stdout on completion:

```
=== build_features Summary ===
Authors processed
  Bot authors    : 4,962
  Human authors  : 4,998
  Total          : 9,960

Features computed : 43
  Temporal        : yes / no (--temporal flag)
  NaN rates (top 5 by frequency):
    mean_reply_time_seconds      : 62.4%  (parents not in dataset)
    suspicious_url_ratio         : 18.2%  (no URLs posted)
    mean_submission_score        : 8.1%   (no submissions)
    ...

Class balance
  y=1 (bot)   : 4,962  (49.8%)
  y=0 (human) : 4,998  (50.2%)

Output : dataset.parquet  (9,960 rows × 45 columns)
```

---

## Dependencies

```
orjson        # fast JSONL reading
pandas
pyarrow       # parquet output
re            # stdlib
math          # stdlib
collections   # stdlib
statistics    # stdlib
argparse      # stdlib
urllib.parse  # stdlib — domain extraction
```

---

## Implementation Notes

- **Process one author at a time** — do not load all comments/submissions for all authors
  into memory simultaneously. Load submissions into a dict by author once at startup,
  then stream through the comments file author by author.
- **NaN not 0 for missing values** — an author with no submissions should have NaN for
  submission features, not 0. 0 would be misleading (it looks like they submitted and got
  zero score, rather than never submitted at all).
- **Text from `[deleted]`/`[removed]` posts** — skip these strings when building text
  features. A `body` or `selftext` of `"[deleted]"` or `"[removed]"` is not real text
  and should not contribute to TTR, length, or any other text statistic.
- **Sorting glob results** when reading multiple files — not applicable here since input
  files are explicitly named via CLI args, not globbed.
