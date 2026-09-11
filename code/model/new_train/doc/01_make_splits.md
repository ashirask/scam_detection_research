# Spec: `make_splits.py`

Reads `00_pipeline_overview.md` first — this script implements the "Canonical
split" and "Feature transform" sections of that document. Run this **once**
per experiment (i.e. once per random seed you want to test), before any
training script runs. Its outputs are the single source of truth every other
script reads.

## Purpose

Load the combined dataset, produce a stratified 64/16/20 train/val/test split
with natural 1:10 ratio preserved in every split, fit the log/CDF feature
transform once on the real training rows, and persist everything needed
downstream. Does not train any model.

## CLI arguments

```
--dataset            required, path to dataset.parquet
--project-root        default "."  (splits/ dir is created under this)
--random-seed         default 42
--filter-zero-posts   flag, default True (reuse existing behavior)
--run-transform-sanity-check   flag, reuse existing plot_transform_sanity_check
```

## Steps

1. **Load + clean.** Reuse `load_and_clean_data()`, `prepare_features_and_labels()`,
   `sanitize_infinities()` from the old `train_val_model.py` unchanged. Print
   the resulting bot/human counts and confirm they're close to 6,478 / 64,780
   (warn, don't fail, if they've drifted — the dataset may have been updated).

2. **Detect categorical/binary columns.**
   `categorical_cols = [c for c in feature_cols if X[c].nunique(dropna=True) <= 2]`.
   Print the list. Save to `splits/categorical_features.json` as a plain JSON
   list of column names.

3. **Split.**
   ```python
   X_trainval, X_test, y_trainval, y_test = train_test_split(
       X, y, test_size=0.20, stratify=y, random_state=SEED)
   X_train, X_val, y_train, y_val = train_test_split(
       X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=SEED)
   ```
   (0.20 of the 80% remainder → 16% of total train_val goes to val, 64% to
   train — confirm the arithmetic in code with a printed sanity check of
   final row counts / fractions, don't just trust the math silently.)

4. **Overlap assertion.** After splitting, pull `author` back in (join on
   index) and assert zero intersection between all three splits' author sets.
   This must be a hard assertion (raise, not warn) — this exact failure mode
   is why the old pipeline was scrapped, so this script should never silently
   ship overlapping splits.

5. **Report per-split ratio.** Print bot:human counts and ratio for train,
   val, test. All three should be close to 1:10 — if any deviates by more
   than a rounding amount, warn loudly (this would indicate a stratification
   bug).

6. **Fit transform on train only, continuous columns only.** Split
   `feature_cols` into `continuous_cols` (everything not in the categorical
   list from step 2) and `categorical_cols`. Call
   `fit_feature_transform(X_train[continuous_cols])` (the raw, untransformed
   training features, continuous columns only) → `transform_params`. Save as
   `splits/transform_params.pkl`. Do not modify `feature_transform.py` — the
   passthrough is achieved by never handing it the categorical columns in the
   first place, not by changing what it does internally.

   When building each split's transformed output, reassemble:
   ```python
   X_transformed = pd.concat([
       apply_feature_transform(X[continuous_cols], transform_params),
       X[categorical_cols],   # untouched, byte-identical to raw
   ], axis=1)
   ```
   Assert `X_transformed[categorical_cols].equals(X_raw[categorical_cols])`
   after reassembly as a sanity check that the orchestration wiring is
   correct.

7. **Apply transform to all three splits.**
   `apply_feature_transform(X_val, transform_params)`,
   `apply_feature_transform(X_test, transform_params)` — never re-fit.

8. **Persist outputs** (see directory layout in the overview doc):
   - `splits/train_raw.parquet`, `val_raw.parquet`, `test_raw.parquet`
     (features + `y` + `author`, untransformed)
   - `splits/train_transformed.parquet`, `val_transformed.parquet`,
     `test_transformed.parquet` (features + `y` + `author`, transformed;
     categorical columns unchanged)
   - `splits/transform_params.pkl`
   - `splits/categorical_features.json`
   - `splits/split_manifest.json`: `{seed, n_total, n_train, n_val, n_test,
     bot_human_ratio_train, bot_human_ratio_val, bot_human_ratio_test,
     overlap_check_passed: true, categorical_columns: [...], timestamp}`

9. **Optional sanity-check plots.** If `--run-transform-sanity-check`, reuse
   the existing `plot_transform_sanity_check()` logic against
   `train_raw` vs `train_transformed`, saved to `splits/transform_sanity_check.png`.

## What this script must NOT do

- Must not reference an extra-humans dataset in any form.
- Must not train any model.
- Must not do any ratio-forcing or extra-human topping-up logic — the whole
  point of the redesign is that stratified `train_test_split` makes this
  unnecessary.
- Must not tune any threshold or compute any importance metric — that's
  `train_model.py`'s job.
