# Inference Pipeline Design — Bot Detection on New Authors

> **Note for implementing agents:** every function below has an explicit
> signature with types. Every script has an explicit CLI argument table.
> Every output file has an explicit column/field schema. Treat these as
> contracts — if something needed to implement a function isn't specified
> here, that's a gap to flag, not to guess at.

## Overview

This document specifies the inference pipeline that scores new, unlabeled Reddit
authors (~175k, growing) using a model already trained and selected via
`make_splits.py` → `train_model.py` → `test_ratio_experiments.py`. It is a
companion to `DOCUMENTATION_SUMMARY.md` and assumes that pipeline's outputs
(`splits/`, `artifacts/{strategy}/`, `results/{strategy}/`) already exist on disk.

**Scope boundary:** nothing in this pipeline fits, trains, or tunes anything.
Every transform, imputer, and scaler is a saved artifact from training, loaded
and applied as-is. The only new computation is `model.predict_proba(...)`.

## Confirmed Facts

1. **Schema matches, but `y` is a placeholder, not a label.** The new
   175k-author parquet is produced by running the same `build_features_v2.py`
   used for training, so its feature schema matches exactly. However, because
   the whole new pool was passed in as the "human" population argument to
   `build_features_v2.py`, every row got `label=0` hardcoded — `y=0` here
   means "came from the unlabeled extraction pool," **not** "confirmed
   human." This column must never be used for evaluation, must-not be
   confused with ground truth, and should be treated as an artifact of the
   feature-building process, not a target.
2. **Model + strategy is fixed: `xgboost` / `none`.** From
   `get_model_registry`, `xgboost` uses preprocessing variant **A**
   (passthrough — no imputer or scaler). Phase A for this specific choice
   therefore only needs `artifacts/none/xgboost_model.pkl`; no
   `imputer_median.pkl` or `scaler_mlp.pkl` load is required. The scripts
   below still take `--model`/`--strategy` as CLI args (so this stays true if
   the choice ever changes), but the defaults and examples now reflect
   `xgboost`/`none`.
3. **Zero-activity filtering is not done by `build_features_v2.py`.** It only
   happens downstream, inside `load_and_clean_data` in `make_splits.py`/
   `train_model.py`, which drops authors with `num_comments==0 AND
   num_submissions==0` *after* features are built. `build_features_v2.py`
   computes `num_comments`/`num_submissions` (see `activity_features`) but
   does not filter on them. **This means the new 175k-author parquet still
   contains zero-activity authors, and Phase A must apply the identical
   filter** — otherwise you're scoring a population the model was never
   validated against (degenerate, mostly-NaN feature rows for accounts with
   no captured posts).
4. **The comments corpus is two author-keyed JSONL files** —
   `comments.jsonl` and `submissions.jsonl` — where each line is one author's
   full record: `{"author": ..., "submissions": [...]}` /
   `{"author": ..., "comments": [...]}`. This is the same shape
   `build_features_v2.py` already consumes. It is not partitioned or
   indexed — a lookup for one author currently means scanning the file
   linearly. This shapes Phase C's design (see below).

## Key Design Principles

### 1. Score once, decide thresholds many times
Model inference (Phase A) and threshold decisions (Phase B) are two separate
scripts. Phase A is the only step that loads the model and touches all 175k
rows; it's run once per model/strategy choice (or when new authors are added)
and its output is a small cached parquet of probabilities. Phase B never loads
the model — it only reads that cache, so you can try as many thresholds as you
want in seconds.

### 2. Never re-fit anything
`transform_params.pkl`, `categorical_features.json`, and the strategy-specific
imputer/scaler are loaded and *applied*, never fit on the new data. Re-fitting
on 175k new authors would silently produce a different feature distribution
than the model was trained on and invalidate the model's learned decision
boundary.

### 3. Hard schema lock
The new dataset must be forced into the exact `feature_cols` the model was
trained on (same names, same set). Missing a training feature is a hard
failure, not a warning — the model cannot score without it. Extra columns are
dropped with a logged warning (they're probably from an upstream feature-set
change that hasn't been backported to this model).

### 4. Decouple cheap compute from expensive I/O
Scoring 175k rows of tabular features takes seconds. Reading from a 300GB
comments corpus does not. Phase C is deliberately scoped to only the
shortlisted authors from Phase B (typically hundreds to a few thousand), never
the full 175k, and never the full corpus per experiment.

### 5. Prevalence-aware thresholding
The threshold saved in `results/{strategy}/train_val_model_metrics.csv` was
tuned against a validation set with a curated 1:10 bot:human ratio. Real-world
prevalence in 175k freshly pulled authors is almost certainly far lower — this
is exactly the degradation `test_ratio_experiments.py` was built to quantify.
Phase B should surface the training-tuned threshold as a *default starting
point*, not a fixed answer, and should let you cross-reference against the
ratio-experiment curves (`results/ratio_experiments/{model}_{strategy}/`) for
the chosen model.

## How This Differs From the Training Pipeline

| Aspect | `train_model.py` / `test_ratio_experiments.py` | Inference pipeline |
|---|---|---|
| Feature transform | `fit_feature_transform` on train fold | `apply_feature_transform` only, using saved `transform_params.pkl` |
| Imputer / scaler | Fit per resample strategy in `create_preprocessing_variants` | Loaded from `artifacts/{strategy}/imputer_median.pkl` or `imputer_mlp.pkl` + `scaler_mlp.pkl` — must match the strategy the chosen model came from |
| Categorical columns | Detected via `nunique <= 2` on train | Fixed list from `categorical_features.json`, passed through unchanged |
| Zero-variance columns | Detected and dropped | Already absent from `feature_cols` — new data just excludes them too |
| Labels | Present; drive threshold tuning and every metric | Absent — output is a bare probability, no precision/recall/F1 possible |
| Splitting | `train_test_split` × 2, stratified | None — every row gets scored |
| Resampling | Applied to training fold per strategy | Not applicable — resampling only ever touched training data |
| Threshold | Tuned against `y_val` inside the script | An external decision made against cached scores, in a separate script |
| Scale bottleneck | Model training / SHAP | I/O against the 300GB comments corpus, not the model |

## Known Gotchas to Guard Against

- **`y` is a placeholder, not ground truth.** Every row in the new author
  parquet has `y=0` purely because of how `build_features_v2.py` was invoked
  (whole pool passed as the "human" argument), not because these are
  confirmed humans. Phase A must (a) never compute or report any metric
  (accuracy, precision, recall, "% correct") against this column for the new
  data, and (b) drop it immediately after a sanity assertion that it's
  uniformly 0 — a non-zero value would indicate the extraction step was
  invoked incorrectly. It exists in the file only because it's a required
  column in `build_features_v2.py`'s output schema, not because it carries
  information about these authors.
- **Silent negative-clipping.** `apply_feature_transform`'s log branch clips
  inputs to `>= 0` before taking the log. The transform *method* per column
  (`log` vs `zscore`) is fixed once, from `transform_params.pkl`, at train
  time — it never changes based on new data. The clip only matters for
  columns assigned `log` (i.e. columns that had zero negatives in the
  training split): if such a column ever produces a negative value in new
  data, it gets silently floored to 0 instead of raising an error.
  **Verification status:** a proxy check was run comparing *which columns
  have any negative value* across train/val/test vs. the new 175k-author
  data, and found the same column names in both. That's a reasonable signal
  but one inferential step removed from the actual question — it doesn't
  read the `log`/`zscore` tags themselves. **The precise check has not been
  run yet** and should be, before treating this as resolved:
  ```python
  import pickle, pandas as pd
  with open("splits/transform_params.pkl", "rb") as f:
      transform_params = pickle.load(f)
  log_columns = [c for c, p in transform_params.items() if p["method"] == "log"]
  new_raw = pd.read_parquet("new_authors_175k.parquet")
  for col in log_columns:
      n_neg = (new_raw[col] < 0).sum()
      if n_neg > 0:
          print(f"{col}: {n_neg} negative values ({100*n_neg/len(new_raw):.3f}%) will be silently zeroed")
  ```
  No output means the `log`-tagged columns are clean in the current batch.
  This exact logic is what `count_log_clips` in Phase A implements as an
  automated, per-run check going forward — see `CLIPPING_TRANSFORM_QA.md`
  for the full walkthrough of why this check (and not the column-set
  comparison) is the correct one.
- **Feature drift / extrapolation.** The model's decision boundary only
  reflects the value ranges it saw in training. An author whose features fall
  far outside that range isn't scored *incorrectly* — the probability is
  just a guess made outside the region the model learned from. Concrete
  example: if `num_comments` in `train_raw.parquet` has p1=0, p99=850, and
  max=3,200, and 400 of 175,000 new authors (0.23%) exceed 850 while 15 of
  those (0.009%) exceed 3,200 (a value never seen in training at all), those
  15 authors' predictions deserve a second look before being trusted at face
  value — especially if any of them end up in the final bot shortlist. See
  `run_drift_checks` in Phase A for the exact computation.
- **Zero-activity authors are still in the raw file.** `build_features_v2.py`
  computes `num_comments`/`num_submissions` but does not filter on them —
  that filter lives in `make_splits.py`'s `load_and_clean_data`. The new
  175k-author parquet will contain some authors with zero captured posts;
  Phase A must apply the same filter before scoring, and should report how
  many authors were excluded this way (they should be tracked as "no
  activity to classify," not silently discarded from bookkeeping).
- **Strategy/preprocessor mismatch.** The median/scale statistics inside
  `imputer_median.pkl` / `imputer_mlp.pkl` / `scaler_mlp.pkl` differ across
  `none` / `downsample` / `smotenc` because each strategy resamples the
  training fold differently before those are fit. Always load the
  preprocessor from the *same* `artifacts/{strategy}/` directory as the model.
- **Unseen categorical values.** Categorical columns are fixed by name from
  training, not re-detected. If upstream feature engineering ever produces a
  third value in a column that was binary at train time, it will pass through
  unchanged into the model — usually harmless for tree models, worth a
  sanity check for the MLP.
- **`feature_cols` is not a literal field anywhere — it must be derived
  correctly.** `split_manifest.json` stores `categorical_columns`,
  `continuous_columns`, and `zero_variance_columns_dropped` as three
  separate lists; it never saves a single combined `feature_cols` list.
  Reconstructing one by concatenating `categorical_columns +
  continuous_columns` is **not** safe — nothing guarantees that order
  matches what the model was actually fit on. The correct source is
  `splits/train_transformed.parquet`'s column order (minus `author` and
  `y`), since that file's columns are exactly, and in the exact order, what
  was passed to `model.fit(...)`.
- **Zero-variance columns need no special handling at inference —** they're
  excluded via the same "drop any column not in `feature_cols`" step as
  `reply_time_coverage`, once `feature_cols` above is derived correctly.
  `build_features_v2.py` still computes these columns for every author
  (it has no knowledge of what training later deemed zero-variance), so
  they'll be present as raw columns in the new 175k-author parquet and need
  dropping — `enforce_feature_schema` already does this, nothing extra is
  needed. Note this determination was made only on the *training* split; a
  column constant in training could show real variation in the new pool,
  but the model was never trained with that signal and genuinely can't use
  it regardless.
- **`author` is dropped as a feature but preserved as the DataFrame index.**
  "Dropped" means it never appears among the columns passed to
  `model.predict_proba(X)` (same exclusion as training). "Preserved" means
  `enforce_feature_schema` moves it via `df.set_index("author")` rather than
  deleting it — pandas carries the index through every subsequent
  operation automatically. At the end,
  `pd.DataFrame({"author": X.index, "probability": probabilities})` lines up
  correctly with no merge needed, precisely because nothing in between ever
  calls `.reset_index()`. This is why the Execution Order section is
  explicit that no intermediate function may reset the index — that's the
  one rule that keeps this alignment guaranteed.

---

## Phase A — `score_authors.py`

### Purpose
Load the chosen trained model, replay the exact training-time feature
preparation on new, unlabeled authors, and persist probabilities so scoring
never has to be repeated.

### CLI Arguments

| Arg | Type | Required | Default | Description |
|---|---|---|---|---|
| `--new-data` | path | yes | — | Parquet from `build_features_v2.py` on the new author pool. Contains a placeholder `y` column (see gotchas). |
| `--project-root` | path | yes | — | Root containing `splits/`, `artifacts/`, `results/`. |
| `--strategy` | str | yes | — | `none` \| `downsample` \| `smotenc`. Selects `artifacts/{strategy}/`. |
| `--model` | str | yes | — | `lightgbm` \| `xgboost` \| `randomforest` \| `mlp` \| `tabpfn`. Selects `{model}_model.pkl`. |
| `--filter-zero-activity` | flag | no | `True` | Apply `num_comments==0 AND num_submissions==0` exclusion. |
| `--drift-lo-pct` / `--drift-hi-pct` | float | no | `0.01` / `0.99` | Percentile band used for the drift check. |
| `--drift-warn-pct` | float | no | `1.0` | Warn if more than this % of rows fall outside the band for a given feature. |
| `--clip-warn-pct` | float | no | `0.1` | Warn if more than this % of rows get clipped in a `log`-method column. |
| `--output-dir` | path | no | `{project_root}/inference/scores/{model}_{strategy}/` | Override output location. |

No `--random-seed`: Phase A scores every row deterministically (no
splitting, resampling, or sampling), so a seed has nothing to act on.

### Inputs
- `--new-data`: parquet of new authors, produced by the same
  `build_features_v2.py` run as training — same feature schema, plus a
  placeholder `y` column (see Gotchas: this is **not** ground truth and must
  be dropped, never evaluated against).
- `splits/categorical_features.json` (list of categorical column names) and
  `splits/transform_params.pkl` (per-column transform method/params).
  **`feature_cols` itself is not a field in `split_manifest.json`** — that
  file only stores `categorical_columns`, `continuous_columns`, and
  `zero_variance_columns_dropped` separately (see `make_splits.py`, where
  these are three independent list variables, never concatenated into one
  saved list). The authoritative `feature_cols`, in the exact order the
  model was trained on, must instead be read as the column order of
  `splits/train_transformed.parquet` with `author` and `y` excluded — this
  is the one source that's guaranteed to match what the model actually saw,
  since reconstructing it from `categorical_columns + continuous_columns`
  would not necessarily preserve the original column order.
- `splits/split_manifest.json` for `categorical_columns` / `continuous_columns`
  (needed to split `feature_cols` into the two subsets `apply_saved_transform`
  and `run_drift_checks` operate on) and for informational context
  (`zero_variance_columns_dropped` — see Gotchas: these columns are already
  absent from `feature_cols`, no special handling needed beyond the normal
  schema-drop step).
- `splits/train_raw.parquet` (for the drift check reference distribution).
- `artifacts/{strategy}/{model}_model.pkl` and, depending on the model's
  preprocessing variant (A/B/C, same mapping as `test_ratio_experiments.py`):
  `imputer_median.pkl` (B) or `imputer_mlp.pkl` + `scaler_mlp.pkl` (C). **For
  the current choice (`xgboost`/`none`), the variant is A — passthrough, no
  imputer/scaler artifact needs to be loaded at all.** The lookup logic stays
  generic so this keeps working if the model choice changes later.

### Execution Order

`score_authors.py`'s `main()` calls these in exactly this sequence — no
step is optional or reorderable:

1. `df, excluded = load_and_clean_new_data(new_data_path, filter_zero_activity)`
2. `df = strip_placeholder_label(df)`
3. `X = enforce_feature_schema(df, feature_cols)` — `X` is now indexed by
   `author`, columns are exactly `feature_cols`. Every step from here on
   receives and returns this same author-indexed `X`; none of them reset
   the index.
4. `X = sanitize_infinities(X)`
5. `drift_df, drift_summary = run_drift_checks(X, train_raw, continuous_cols, ...)`
   — run on **raw** values, before any transform, since `train_raw.parquet`
   is itself untransformed.
6. `X, clip_report = apply_saved_transform(X, transform_params, continuous_cols, categorical_cols)`
7. `X = apply_preprocessing_variant(X, model_name, artifacts_dir)`
8. `probabilities = score(model, X)`
9. Assemble `scores.parquet` from `X.index` (author), `probabilities`,
   `drift_df["n_features_out_of_range"]` (joined on the same index),
   plus the constant `model`/`strategy`/`scored_at` fields.
10. Write `excluded_authors.json` from `excluded` (step 1) and
    `inference_manifest.json` from `clip_report` (step 6), `drift_summary`
    (step 5), and the row counts collected along the way.

### Key Functions

```python
def load_and_clean_new_data(
    dataset_path: str, filter_zero_activity: bool = True
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Loads the parquet, drops rows with missing/empty "author".
    Asserts the remaining "author" values are unique — raises ValueError
    listing the duplicates if not (an author must map to exactly one row
    for downstream joins to be safe).
    If filter_zero_activity: also drops rows where
    (num_comments == 0) & (num_submissions == 0) — required here because
    build_features_v2.py computes these counts but does not filter on them.
    Returns (cleaned_df, excluded) where excluded is
    {"missing_author": [author, ...], "zero_activity": [author, ...]}.
    """

def strip_placeholder_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asserts df["y"] is uniformly 0 (raises ValueError otherwise — a non-zero
    value means build_features_v2.py was invoked against the wrong
    population files). Drops the "y" column. Returns df without "y".
    Nothing downstream of this function may ever reference "y" again.
    """

def enforce_feature_schema(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Sets df's index to "author" first (df = df.set_index("author")) so the
    identifier travels through every subsequent step without being treated
    as a model input feature, and is never at risk of being dropped as an
    "extra column."
    Raises ValueError listing any name in feature_cols missing from
    df.columns (checked before the reindex).
    Drops (with a logged warning) any remaining column not in feature_cols
    (e.g. reply_time_coverage, same exclusion as training).
    Returns df reindexed to exactly feature_cols, in that order, with
    "author" as the index.
    """

def sanitize_infinities(X: pd.DataFrame) -> pd.DataFrame:
    """Replace +inf/-inf with NaN. Identical to the training-pipeline version."""

def run_drift_checks(
    X_new_raw: pd.DataFrame,
    train_raw: pd.DataFrame,
    continuous_cols: list[str],
    lo_pct: float = 0.01,
    hi_pct: float = 0.99,
    warn_pct: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Scoped to continuous_cols only — categorical/binary columns (0/1 flags)
    aren't meaningful to run a percentile check against.
    For each column in continuous_cols:
        lo, hi = train_raw[col].quantile(lo_pct), train_raw[col].quantile(hi_pct)
        oor_mask = (X_new_raw[col] < lo) | (X_new_raw[col] > hi)
        pct_oor = 100 * oor_mask.mean()
        if pct_oor > warn_pct: log a warning
    Returns:
      - per_author_df: DataFrame indexed like X_new_raw (i.e. by "author")
        with one bool column f"{col}__oor" per continuous column, plus an
        int column "n_features_out_of_range" (row-wise sum across all
        f"{col}__oor" columns).
      - summary: {col: {"pct_out_of_range": pct_oor, "train_lo": lo,
        "train_hi": hi, "new_min": ..., "new_max": ...}} for every column
        checked.
    Non-blocking — never raises, only logs and returns data for the manifest.
    """

def apply_saved_transform(
    X: pd.DataFrame,
    transform_params: dict,
    continuous_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Applies feature_transform.apply_feature_transform to X[continuous_cols]
    only; categorical_cols are reattached unchanged (mirrors
    apply_transform_split in make_splits.py). Preserves the "author" index.
    Internally calls count_log_clips(X[continuous_cols], transform_params)
    BEFORE applying the transform, so the clip report reflects what would be
    zeroed, not what already was.
    Returns (X_transformed, clip_report) where clip_report is
    {col: {"n_clipped": int, "pct_clipped": float}} for every log-method column.
    """

def count_log_clips(
    X_raw: pd.DataFrame, transform_params: dict
) -> dict[str, dict]:
    """
    For every column where transform_params[col]["method"] == "log":
        n_clipped = (X_raw[col] < 0).sum()
        pct_clipped = 100 * n_clipped / len(X_raw)
    Returns {col: {"n_clipped": n_clipped, "pct_clipped": pct_clipped}}.
    This is the precise check discussed in CLIPPING_TRANSFORM_QA.md — it
    reads the method tag directly from transform_params rather than
    inferring it from which columns show negatives elsewhere.
    """

def apply_preprocessing_variant(
    X: pd.DataFrame, model_name: str, artifacts_dir: str
) -> pd.DataFrame:
    """
    variant = {"lightgbm": "A", "xgboost": "A", "randomforest": "B",
               "tabpfn": "B", "mlp": "C"}[model_name]
    A: returns X unchanged.
    B: loads artifacts_dir/imputer_median.pkl, returns imputer.transform(X)
       as a DataFrame with the original columns and the "author" index
       preserved.
    C: loads artifacts_dir/imputer_mlp.pkl and artifacts_dir/scaler_mlp.pkl,
       applies imputer then scaler, same column/index reconstruction.
    Never calls .fit(...) — only .transform(...).
    """

def score(model, X: pd.DataFrame) -> np.ndarray:
    """Returns model.predict_proba(X)[:, 1]."""
```

### Outputs

**`{output_dir}/scores.parquet`**
| Column | Type | Notes |
|---|---|---|
| `author` | string | |
| `probability` | float64 | model's P(bot) |
| `model` | string | constant, e.g. `"xgboost"` |
| `strategy` | string | constant, e.g. `"none"` |
| `scored_at` | string (ISO 8601) | run timestamp |
| `n_features_out_of_range` | int | from `run_drift_checks`, for later filtering |

**`{output_dir}/excluded_authors.json`**
```json
{
  "missing_author": {"count": 3, "authors": ["...", "..."]},
  "zero_activity":  {"count": 812, "authors": ["...", "..."]}
}
```

**`{output_dir}/inference_manifest.json`**
```json
{
  "run_timestamp": "2026-09-01T12:00:00Z",
  "new_data_path": "...",
  "project_root": "...",
  "model": "xgboost",
  "strategy": "none",
  "model_artifact_path": "artifacts/none/xgboost_model.pkl",
  "n_input_rows": 175000,
  "n_excluded_missing_author": 3,
  "n_excluded_zero_activity": 812,
  "n_scored": 174185,
  "feature_schema_check": {"missing_features": [], "extra_features_dropped": []},
  "clip_report": {"<col>": {"n_clipped": 0, "pct_clipped": 0.0}},
  "drift_summary": {"<col>": {"pct_out_of_range": 0.23, "train_lo": 0, "train_hi": 850, "new_min": 0, "new_max": 3200}},
  "n_authors_with_any_drift_flag": 512
}
```

---

## Phase B — `explore_thresholds.py`

### Purpose
Turn cached probabilities into classifications at whatever threshold(s) you
want to inspect, without touching the model or the 175k-row feature table
again. **One run of this script reads one `scores.parquet` (one probability
per author) and applies as many threshold cutoffs to that same column as you
give it** — it never rescoring anything per threshold, and it never needs
more than one Phase A output as input.

### CLI Arguments

| Arg | Type | Required | Default | Description |
|---|---|---|---|---|
| `--scores` | path | yes | — | Path to Phase A's `scores.parquet`. |
| `--project-root` | path | yes | — | For loading reference threshold/ratio context and `feature_cols`. |
| `--strategy` | str | yes | — | Must match the strategy used to produce `--scores`. |
| `--model` | str | yes | — | Must match the model used to produce `--scores`. |
| `--thresholds` | float, one or more (`nargs='+'`) | no | training-tuned threshold from `results/{strategy}/train_val_model_metrics.csv` | One run can evaluate 1 or many cutoffs against the same cached probabilities. |
| `--new-raw-data` | path | no | value of Phase A's `--new-data` (read from `inference_manifest.json`) | The raw new-author parquet, unprocessed — needed for `attach_raw_features`. |
| `--output-dir` | path | no | `{project_root}/inference/review/{model}_{strategy}/` | |

### Inputs
- `--scores`: Phase A's `scores.parquet` (`author`, `probability`,
  `n_features_out_of_range`, ...).
- `results/{strategy}/train_val_model_metrics.csv` — training-tuned
  threshold for `--model`.
- `results/ratio_experiments/{model}_{strategy}/ratio_metrics.csv` if
  present — optional context, not required to run.
- `splits/split_manifest.json` — for `feature_cols`, needed by
  `attach_raw_features`.
- `--new-raw-data` — the same raw parquet Phase A scored, read fresh (plain
  `pd.read_parquet`, not the author-indexed pipeline object from Phase A).

### Execution Order

1. `scores_df = load_scores(scores_path)`
2. `default_threshold, ratio_df = load_reference_threshold(project_root, strategy, model)`
   — used only if `--thresholds` wasn't provided.
3. `results = classify_at_thresholds(scores_df, thresholds)` → `{t: df}`
4. `new_raw_df = pd.read_parquet(new_raw_data_path)`
5. For each `t, df in results.items()`:
   a. `flagged = df[df["predicted_label"] == 1]`
   b. `flagged = attach_raw_features(flagged, new_raw_df, feature_cols)`
   c. write `flagged` to `{output_dir}/threshold_{t}/flagged_authors.csv`
6. `plot_score_distribution(scores_df, thresholds, f"{output_dir}/score_distribution.png")`
7. Write `{output_dir}/threshold_summary.csv` — one row per threshold, from
   the `n_flagged`/`flagged_rate_pct` computed in step 3, plus
   `is_training_tuned_threshold` set from step 2.

### Key Functions

```python
def load_scores(scores_path: str) -> pd.DataFrame:
    """Reads scores.parquet. Must contain ['author', 'probability']."""

def load_reference_threshold(
    project_root: str, strategy: str, model: str
) -> tuple[float, pd.DataFrame | None]:
    """
    Reads results/{strategy}/train_val_model_metrics.csv, returns the row for
    `model`'s tuned threshold as a float.
    Also attempts to read
    results/ratio_experiments/{model}_{strategy}/ratio_metrics.csv if it
    exists (returns None if not) — this is surfaced as context in the summary
    output, never applied automatically.
    """

def classify_at_thresholds(
    scores_df: pd.DataFrame, thresholds: list[float]
) -> dict[float, pd.DataFrame]:
    """
    For each t in thresholds:
        out = scores_df.copy()
        out["predicted_label"] = (out["probability"] >= t).astype(int)
    Returns {t: out} — same scores_df, one predicted_label column per
    threshold value. No precision/recall/F1 — there is no ground truth for
    the new authors, only "how many, and who."
    """

def attach_raw_features(
    flagged_df: pd.DataFrame, new_raw_df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Left-joins flagged_df (author, probability, predicted_label) to
    new_raw_df[["author"] + feature_cols] on "author". Returns the merged
    dataframe so a reviewer can see raw (untransformed) feature values
    next to the prediction, not log/z-score units.
    """

def plot_score_distribution(
    scores_df: pd.DataFrame, thresholds: list[float], output_path: str
) -> None:
    """
    One histogram of scores_df["probability"] (single distribution — there is
    only one probability per author), with a vertical line at each value in
    thresholds. Saved once per script run, not once per threshold.
    """
```

### Outputs

**`{output_dir}/threshold_summary.csv`**
| Column | Type | Notes |
|---|---|---|
| `threshold` | float | |
| `n_flagged` | int | |
| `flagged_rate_pct` | float | |
| `is_training_tuned_threshold` | bool | true for the value pulled from `train_val_model_metrics.csv` |

**`{output_dir}/threshold_{t}/flagged_authors.csv`** (one such file per threshold in `--thresholds`)
| Column | Type | Notes |
|---|---|---|
| `author` | string | |
| `probability` | float64 | |
| `predicted_label` | int (0/1) | |
| `n_features_out_of_range` | int | passed through from Phase A |
| `<feature_col>` | (raw dtype) | one column per training feature, raw units, from `attach_raw_features` |

**`{output_dir}/score_distribution.png`** — single histogram, all thresholds overlaid.

---

## Phase C — Two Standalone Scripts

Split into two scripts since the index only needs building once and is
reused across every future review:

### C1 — `build_author_offset_index.py`

Run manually, once, against each corpus file. Not integrated into
`build_features_v2.py` (that run has already happened) — this is a
dedicated one-time pass.

**CLI Arguments**

| Arg | Type | Required | Description |
|---|---|---|---|
| `--jsonl-path` | path | yes | e.g. `comments.jsonl` or `submissions.jsonl` |
| `--index-output` | path | yes | output parquet path, e.g. `comments_index.parquet` |

```python
def build_author_offset_index(jsonl_path: str, index_output_path: str) -> None:
    """
    Opens jsonl_path in binary mode ("rb"). For each line:
        offset = f.tell()          # position BEFORE reading the line
        line = f.readline()
        length = len(line)
        author = orjson.loads(line)["author"]
    Writes a parquet file with columns:
        author (str), byte_offset (int64), byte_length (int64)
    Also writes {index_output_path}.manifest.json:
        {"source_file": jsonl_path, "file_size_bytes": ..., "n_lines_indexed": ...,
         "built_at": "<ISO8601>"}
    The manifest lets a later run detect staleness by comparing
    file_size_bytes against the current file size before trusting the index.
    """
```

Run once per corpus file:
```bash
python inference/build_author_offset_index.py --jsonl-path comments.jsonl --index-output comments_index.parquet
python inference/build_author_offset_index.py --jsonl-path submissions.jsonl --index-output submissions_index.parquet
```

### C2 — `inspect_flagged_comments.py`

**CLI Arguments**

| Arg | Type | Required | Description |
|---|---|---|---|
| `--flagged-authors` | path | yes | a `flagged_authors.csv` from Phase B |
| `--comments-jsonl` / `--comments-index` | path | yes | corpus file + its index |
| `--submissions-jsonl` / `--submissions-index` | path | yes | corpus file + its index |
| `--output-dir` | path | yes | e.g. `{phase_b_threshold_dir}/raw_posts/` |

```python
def fetch_author_record(
    author: str, index_df: pd.DataFrame, jsonl_path: str
) -> dict | None:
    """
    Looks up author in index_df (columns: author, byte_offset, byte_length).
    If not found, returns None (log a warning — means the corpus predates
    or postdates this author, or the index is stale).
    Otherwise: open jsonl_path in "rb", f.seek(byte_offset),
    line = f.read(byte_length), return orjson.loads(line).
    """

def summarize_for_review(
    author: str, comments_record: dict | None, submissions_record: dict | None
) -> dict:
    """
    Extracts a reviewer-friendly subset of fields from the raw Reddit-style
    records — e.g. for each comment/submission: body/selftext, title
    (submissions only), subreddit, created_utc, score. Returns:
        {"author": author, "n_comments": ..., "n_submissions": ...,
         "comments": [...trimmed...], "submissions": [...trimmed...]}
    """
```

**Output:** `{output_dir}/raw_posts_bundle.jsonl` — one line per flagged
author, each line the dict returned by `summarize_for_review`.

### Design Constraints
- The index (C1) is the only step that reads the full corpus files; C2 only
  ever does targeted seeks against the flagged-author list from Phase B.
- Before trusting an index in C2, compare the corpus file's current size to
  `file_size_bytes` in its manifest; mismatch means rebuild.

---

## Data Flow

```
new_authors_175k.parquet (y is a placeholder — always 0, not ground truth)
    ↓
score_authors.py  ←── splits/transform_params.pkl, categorical_features.json,
    │                  split_manifest.json (feature_cols), train_raw.parquet (drift ref)
    │              ←── artifacts/none/xgboost_model.pkl  (variant A: no imputer/scaler needed)
    ↓
inference/scores/xgboost_none/scores.parquet   (author, probability, n_features_out_of_range)
    ↓
explore_thresholds.py  ←── results/none/train_val_model_metrics.csv (default threshold)
    │                  ←── results/ratio_experiments/xgboost_none/ (context, if present)
    ↓
inference/review/xgboost_none/threshold_{t}/flagged_authors.csv
    ↓
build_author_offset_index.py  ──→  comments_index.parquet, submissions_index.parquet
    │                                (one-time full read of comments.jsonl / submissions.jsonl,
    │                                 run manually, already decided as standalone — not
    │                                 folded into build_features_v2.py)
    ↓
inspect_flagged_comments.py  ←── flagged_authors.csv + offset indexes (targeted seeks only)
    ↓
qualitative review of flagged accounts' posts/comments
```

## Reproducibility / Bookkeeping Checklist

- [ ] `inference_manifest.json` records exactly which model/strategy artifact
      file was loaded, so a scores file is always traceable to a specific
      trained model.
- [ ] Excluded-author counts (missing author, zero activity) are reported
      in `excluded_authors.json`, not silently dropped.
- [ ] Any threshold used for a "final" flagged list is recorded in
      `threshold_summary.csv` alongside the training-tuned value, so the
      choice is auditable later.
- [ ] Drift-flag counts (`n_features_out_of_range`) are reviewed before
      trusting predictions for any author that shows up in the final
      shortlist with a high count.
- [ ] Log-clip rates (`clip_report` in the manifest) are checked as an
      automated regression, not just the one-off manual comparison already
      done — a future extraction run could reintroduce the issue.
- [ ] Offset-index manifests (`file_size_bytes`) are checked against current
      corpus file sizes before trusting a lookup in Phase C.

---

## Diagnostic Phase D — Non-Flagged Author Review

`sample_non_flagged_authors.py` selects authors from the cached Phase A scores;
it does not rescore or rebuild features. An author is non-flagged when
`probability < threshold`. Authors in `excluded_authors.json` are not
non-flagged because they were never scored.

The script supports reproducible random sampling and score-stratified sampling
from near-threshold, middle, and very-low score bands. It joins the selected
authors to the original raw feature parquet and writes
`sampled_non_flagged_authors.csv`. The `y` column, when present in the feature
source, is a placeholder from feature construction and is never used as a
label.

The selected author CSV is the input to the existing indexed retrieval phase.
Retrieval should use the merged comments/submissions indexes and their main and
manual JSONL sources. The resulting review bundle uses the same capped format
as flagged-author review: at most 20 comments and 20 submissions per author.
Missing content is a corpus/index coverage result, not evidence that an author
is human. Human review of a non-flagged author can identify a candidate false
negative, but it does not create ground truth automatically.

Example:

```bash
conda activate scamdetect
python inference/sample_non_flagged_authors.py \
  --scores inference/scores/xgboost_none/scores.parquet \
  --raw-features inference/data/merged_authors.parquet \
  --threshold 0.77 \
  --mode stratified \
  --per-band 20 \
  --seed 20260908 \
  --output-dir inference/review/xgboost_none/non_flagged/
```

## Diagnostic Phase E — Flagged Rule Coverage

`diagnose_flagged_rule_coverage.py` measures overlap between flagged authors and
the live extraction heuristics in `zst-extraction/extract_bots_from_zst.py`.
Rule A evaluates username patterns, Rule B evaluates membership in the first
`--botrank-top-n` rows of the exact supplied BotRank CSV, and Rule C evaluates
the live phrase list against comment bodies and submission selftext plus title.

The authoritative mode uses complete records retrieved through the merged
offset indexes. The existing `raw_posts_bundle.jsonl` may also be supplied for
a fast observed-content mode, but because it contains only the first 20
comments/submissions and truncated text, a missing Rule C match there is not a
complete negative result.

The script writes one row per flagged author to
`flagged_rule_evidence.csv`, aggregate counts and percentages to
`flagged_rule_summary.csv`, and provenance to `flagged_rule_manifest.json`.
The per-author evidence includes independent Rule A/B/C flags, matched Rule C
phrases and a few samples, scanned-record counts, coverage status, and a rule
category such as `A`, `B+C`, or `none`.

The percentage for authors with no observed rule match is always reported with
its denominator. In complete mode:

$$
	ext{no-rule-match percentage} =
\frac{\text{flagged authors with no Rule A/B/C match}}
     {\text{all flagged authors}} \times 100
$$

This is evidence that the model flags accounts beyond the extraction
heuristics; it is not model accuracy, precision, recall, or confirmed bot
status. The live extractor is the ruleset authority; older specification and
legacy diagnostic phrase lists are not silently combined with it.
