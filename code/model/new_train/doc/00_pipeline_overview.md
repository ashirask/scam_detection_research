# Bot Detection Pipeline v2 — Architecture Overview

This document is the shared contract for three scripts. Read this fully before
implementing any of the other spec files, since they all depend on the
conventions defined here.

## Why this redesign exists

The previous pipeline (`train_val_model.py` + `test_ratio_experiments.py`) had
two structural problems:

1. Splits were built by pulling from a main dataset **and** a supplemental
   extra-humans dataset, with index-based bookkeeping to avoid double-counting
   rows across train/val/test. Diagnostic checks found large overlaps between
   splits — i.e. the same author could appear in more than one split.
2. Validation was forced to an artificial 1:30 ratio (via topping up from the
   extra-humans file) while train ratio was configurable, and test ratio
   experiments used yet another topped-up pool. Three different ratio
   mechanisms, three different data sources, one bug surface.

The data now lives in a **single parquet file** at a fixed 1:10 bot:human
ratio (6,478 bots / 64,780 humans). The extra-humans file is retired
entirely — do not reference it anywhere in the new scripts.

Two more bugs were found in the old code and must be fixed here (not
reproduced):

- **Threshold tuning silently falls back to 0.5.** The old
  `find_optimal_threshold()` initialized `best_threshold = 0.5` and only
  updated it if some threshold in the sweep grid cleared the `min_precision`
  floor. For models that never cleared that floor, the function returned 0.5
  without any indication that the constraint failed — indistinguishable from
  a model whose genuinely optimal threshold happened to be 0.5. Fix: always
  report the threshold that gets **closest to the precision floor** (or the
  max-precision threshold if none clear it), and add an explicit
  `precision_target_met: bool` field to the metrics output.
- **SHAP was hardcoded to LightGBM only.** `shap.TreeExplainer` works on any
  tree ensemble (LightGBM, XGBoost, RandomForest) — it was never a SHAP
  limitation, the old code just never looped over the model registry. Fix:
  compute SHAP for every trained model, picking the explainer by model type
  (see "SHAP / importance" section below).

## Directory layout

```
project/
  data/
    dataset.parquet                  # combined, 1:10, the only data input
  splits/
    train_raw.parquet                # real rows, untransformed, natural 1:10
    val_raw.parquet
    test_raw.parquet
    train_transformed.parquet        # after log/CDF transform (see below)
    val_transformed.parquet
    test_transformed.parquet
    transform_params.pkl
    categorical_features.json        # list of column names treated as categorical
    split_manifest.json              # counts, ratios, seed, overlap-check result
  artifacts/
    none/
      lightgbm_model.pkl, xgboost_model.pkl, randomforest_model.pkl,
      mlp_model.pkl, tabpfn_model.pkl
      imputer_median.pkl, imputer_mlp.pkl, scaler_mlp.pkl
    downsample/
      (same file set)
    smotenc/
      (same file set)
  results/
    none/
      train_val_model_metrics.csv, shap_bar_<model>.png, shap_beeswarm_<model>.png,
      shap_feature_ranking_<model>.csv, permutation_importance_<model>.csv,
      correlation_heatmap.png
    downsample/  (same file set)
    smotenc/     (same file set)
    ratio_experiments/
      <model>_<strategy>/
        ratio_metrics.csv, recall_vs_ratio.png, precision_vs_ratio.png,
        pr_curves_by_ratio.png
  scripts/
    make_splits.py
    feature_transform.py             # unchanged, existing module — reuse as-is
    train_model.py
    test_ratio_experiments.py
```

Every script takes `--project-root` (default `.`) and derives the paths above
from it, so nothing is hardcoded to one machine.

## Data contract

- Label column: `y` (1 = bot, 0 = human).
- Identifier column: `author` (excluded from features).
- `reply_time_coverage` is excluded from features (carried over from the old
  pipeline — confirm this is still correct, but do not silently drop other
  columns).
- Row filtering: drop rows with null/empty `author`; drop rows where
  `num_comments == 0 AND num_submissions == 0`. Reuse the existing
  `load_and_clean_data()` / `prepare_features_and_labels()` /
  `sanitize_infinities()` functions from the old script verbatim — they are
  not part of what broke.

## Canonical split (fit once, shared by everything downstream)

Built once, in `make_splits.py`, and never rebuilt inside the training or
ratio scripts. All three training strategies and the ratio-robustness script
must load the exact same val/test rows.

1. `train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)` →
   `(train_val, test)`.
2. `train_test_split(train_val_X, train_val_y, test_size=0.20, stratify=y_train_val, random_state=SEED)`
   → `(train, val)`. (0.20 of the 80% remainder = 16% of the total, giving the
   64/16/20 split via two stratified calls — do not use a single three-way
   custom split; sklearn's two-call approach is simpler and this was
   explicitly the point of the redesign.)
3. Stratifying by `y` preserves the natural 1:10 ratio in all three splits
   automatically — no ratio-forcing logic is needed anywhere in this step.
4. **Overlap guard-rail:** after splitting, assert
   `set(train.author) & set(val.author) & set(test.author) == set()`
   pairwise, and print the counts. This check existed in diagnostics before
   but should now be a permanent, non-optional assertion in the script (it
   should trivially pass with `train_test_split`, but it's cheap insurance
   given that overlap was the original failure mode).
5. `random_state=SEED` must be a CLI arg (default 42) and must be identical
   across all scripts for a given experiment run.

## Feature transform (fit once, on real training rows only)

- Fit `feature_transform.fit_feature_transform()` on **`train_raw`** —
  i.e. the real rows in the 64% training split, before any resampling exists.
  This is the only fit call in the entire pipeline. Persist as
  `transform_params.pkl`.
- Apply `apply_feature_transform()` to `train_raw`, `val_raw`, `test_raw` to
  produce the `*_transformed` parquet files. These transformed files are what
  every downstream script reads — `train_model.py` and
  `test_ratio_experiments.py` should never call `fit_feature_transform` again.
- **Categorical/binary columns are excluded from the log/CDF transform by
  orchestration, not by modifying `feature_transform.py`.** `make_splits.py`
  calls `fit_feature_transform` / `apply_feature_transform` on a
  continuous-columns-only slice of the dataframe, then reattaches the
  untouched categorical columns afterward. `feature_transform.py` itself is
  never modified.
- **Why passthrough, given it was already confirmed safe to transform them
  (see below):** this is a modeling-design choice, not a bug fix. The
  log/CDF transform exists to reshape continuous distributions — compress a
  heavy right tail (log) or robustly standardize scale against outliers
  (CDF/MAD). Neither rationale applies to a two-point categorical variable;
  transforming one doesn't change model performance (monotonic, so tree
  splits and MLP scaling are unaffected either way) but does cost
  interpretability — SHAP dependence plots, partial dependence, and manual
  feature audits would show relabeled values (e.g. `-9.21`/`0.0001`) instead
  of the original `0`/`1` for zero benefit. Apply the transform only to the
  features it was designed for.
- **Detect categorical/binary columns automatically:**
  `is_categorical = X[col].nunique(dropna=True) <= 2`, computed on the raw
  training split. Save the resulting column name list to
  `categorical_features.json`. This list is used two places downstream: (a)
  `make_splits.py`'s passthrough logic above, and (b) `train_model.py`'s
  SMOTENC `categorical_features` index argument — always compute that index
  by looking up each saved column name's position in the transformed
  dataframe's column order at load time, never hardcode or persist a
  positional index.
- Note for completeness: transforming these columns would not have been
  incorrect — `log()` is deterministic and one-to-one, so SMOTENC's
  exact-match/mode-based categorical handling would still have worked
  correctly on the transformed values. Passthrough is chosen for
  interpretability and conceptual clarity, not because the alternative was
  broken.
- This threshold only catches strictly-binary columns. If the feature set has
  any low-cardinality integer-coded categoricals beyond true/false flags
  (e.g. a 3-5 value code), they will not be caught by `nunique <= 2` and will
  be treated as continuous by both the transform and SMOTENC unless added to
  `categorical_features.json` manually. Flag this to the user rather than
  silently assuming binary-only.

## Resampling strategies (train fold only — val/test are never resampled)

All three strategies read the **same** `train_transformed.parquet` /
`val_transformed.parquet` / `test_transformed.parquet`. They differ only in
what happens to the training fold immediately before `model.fit()`:

- `none`: use `train_transformed` as-is (natural 1:10, ~4,146 bots / 41,459
  humans at 64% of the full dataset).
- `downsample`: randomly sample humans from the training fold down to the bot
  count (1:1). Simple `df.sample(n=n_bots, random_state=SEED)` on the human
  subset — no need for `imblearn.RandomUnderSampler` unless the agent prefers
  it for consistency with the SMOTENC import; either is fine as long as the
  seed is respected and the result is logged (n_bots, n_humans_before,
  n_humans_after).
- `smotenc`: apply `imblearn.over_sampling.SMOTENC` to the training fold to
  upsample bots to 1:1. `categorical_features` must be the index positions
  (not names) of the columns listed in `categorical_features.json`, mapped
  against the training fold's column order. Fit and resample on the
  **already-transformed** training features (continuous columns
  log/CDF-scaled, categorical columns raw {0,1}) — do not run SMOTENC on raw
  untransformed continuous features, since heavy-tailed raw counts distort
  the k-NN distance SMOTE relies on for interpolation.
- Log the pre/post class counts for every strategy in the run's console
  output and in a `resample_manifest.json` next to that strategy's results.

## Model registry (unchanged from the old script, reuse as-is)

Five models: `lightgbm`, `xgboost`, `randomforest`, `mlp`, `tabpfn`, with the
same hyperparameters and preprocessing-variant assignment (A = raw/imputed
passthrough for LightGBM/XGBoost, B = median-imputed for RandomForest/TabPFN,
C = median-imputed + standard-scaled for MLP) as the existing
`get_model_registry()` function. Carry this function over unchanged except:
add a `--models` CLI arg (space-separated subset, default = all 5) so a given
run can train a subset without editing code.

**TabPFN caveat to flag to the user, not silently work around:** TabPFN has a
row-count ceiling (historically ~10k rows for the public model). The
`downsample` strategy (~8,300 rows at 1:1) and `none` strategy (~45k rows)
should both be checked against whatever ceiling the installed TabPFN version
enforces, and the script should raise a clear error (not silently truncate
data) if a strategy's training fold exceeds it, telling the user which
strategy/row-count triggered it.

## Threshold tuning (fixed version)

Replace the old `find_optimal_threshold` with a version that:

1. Sweeps thresholds `np.arange(0.05, 0.96, 0.05)` (widen slightly from the
   old 0.1–0.9 grid so extremes aren't missed).
2. Tracks the best F1 among thresholds that clear `min_precision` (unchanged
   goal), **and separately** tracks the threshold with the single highest
   precision across the whole grid, regardless of whether it clears the
   floor.
3. If at least one threshold clears the floor: return the best-F1 threshold
   among those, with `precision_target_met=True`.
4. If none clear the floor: return the max-precision threshold instead of a
   hardcoded 0.5, with `precision_target_met=False`, and print a visible
   warning naming the model and the best precision actually achieved.
5. `train_val_model_metrics.csv` gains a `precision_target_met` column so
   this is visible in the output table, not just the console log.

## SHAP / permutation importance (generalized to all models)

- Loop over every trained model in the registry, not just LightGBM.
- Explainer selection by model type:
  - `lightgbm`, `xgboost`, `randomforest` → `shap.TreeExplainer`.
  - `mlp`, `tabpfn` → `shap.Explainer(model.predict_proba, background_sample)`
    where `background_sample` is a small random subset of the training fold
    (e.g. `shap.sample(X_train, 100)`) — full-validation-set Kernel/Permutation
    SHAP on a non-tree model is too slow to run on the full val set.
- Output naming becomes per-model: `shap_bar_<model>.png`,
  `shap_beeswarm_<model>.png`, `shap_feature_ranking_<model>.csv` (the old
  code had no model suffix since only one model was ever run).
- Permutation importance: loop over **all** models in the registry (the old
  code hardcoded `["lightgbm", "randomforest"]` — same class of bug as SHAP,
  fix identically).

## Ratio-robustness construction (bot-subsampling, extra-humans fully retired)

Since there is no supplemental human source anymore, ratio changes are
achieved by holding the **test fold's human count fixed** and subsampling
**bots** down to hit the target ratio (this is also more realistic — it
models bots becoming rarer against a fixed human population, not humans
appearing from nowhere):

- 1:10 (natural): use the full reserved test fold as-is, no subsampling.
- 1:30: `n_bots_target = round(n_humans_test / 30)`, sample that many bots
  from the test fold's bots without replacement, keep all humans.
- 1:100: `n_bots_target = round(n_humans_test / 100)`, same approach.
- If `n_bots_target` at some ratio would be small enough to make metrics
  noisy (e.g. under ~50), print a visible warning in the output rather than
  silently reporting a metric on a tiny sample.
- Use a fixed `random_state=SEED` for the bot subsampling draw at each ratio
  so runs are reproducible.

## Cross-script conventions

- Every script accepts `--random-seed` (default 42) and uses it for every
  stochastic operation (splitting, downsampling, SMOTENC, bot subsampling,
  MLP/TabPFN internals where applicable).
- Every script writes a short JSON manifest of its own run config (args,
  timestamp, row counts in/out) alongside its outputs, so results can be
  traced back to the exact command that produced them.
- Every script creates its own output directories automatically
  (`os.makedirs(path, exist_ok=True)`) rather than assuming they already
  exist. The only directories that must exist before running anything are
  `data/` (containing `dataset.parquet`) and `scripts/` (containing
  `feature_transform.py`) — `splits/`, `artifacts/`, and `results/` and all
  their strategy/model subfolders are created on demand.
- No script should ever reference an extra-humans file, path, or CLI flag —
  remove that concept entirely rather than leaving unused flags around.
