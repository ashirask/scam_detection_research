# Spec: `train_model.py`

Reads `00_pipeline_overview.md` first. Reads splits produced by
`make_splits.py` — never re-splits or re-fits the feature transform. Run this
script **three times** (once per `--resample-strategy`) to produce the three
models-under-comparison. All three runs read identical val/test data.

## Purpose

Given a resampling strategy, apply it to the training fold only, train all
requested models, tune each model's decision threshold on the (always 1:10)
validation set, evaluate, and compute SHAP + permutation importance +
correlation analysis for every trained model.

## CLI arguments

```
--project-root         default "."
--resample-strategy     required, one of {none, downsample, smotenc}
--models                space-separated subset of
                        {lightgbm, xgboost, randomforest, mlp, tabpfn},
                        default = all five
--min-precision         default 0.90
--random-seed           default 42 (must match make_splits.py's seed)
--skip-shap             flag
--skip-permutation      flag
--skip-tabpfn           flag (independent of --models, for convenience)
```

Output paths are auto-namespaced by strategy:
`artifacts/<strategy>/...`, `results/<strategy>/...` — do not let the three
strategy runs overwrite each other.

## Steps

1. **Load splits.** Read `splits/train_transformed.parquet`,
   `splits/val_transformed.parquet` (test is not touched by this script at
   all — it's reserved for `test_ratio_experiments.py`). Read
   `splits/categorical_features.json` for the SMOTENC branch.

2. **Apply the resampling strategy to the training fold only:**

   - `none`: use the training fold as loaded, no changes. Log bot/human
     counts (should be the natural ~1:10 from the 64% split).

   - `downsample`: separate training rows into bot/human subsets. Sample
     humans down to `n_bots` using `random_state=SEED`. Concatenate back into
     the balanced 1:1 training set. Log before/after counts.

   - `smotenc`: build the `categorical_features` index list by mapping the
     column names in `categorical_features.json` to their positional index in
     the training fold's column order (exclude `author`/`y` from the index
     count — indices refer to the feature matrix only). Instantiate
     `imblearn.over_sampling.SMOTENC(categorical_features=cat_idx,
     random_state=SEED)`. Call `fit_resample(X_train, y_train)` to upsample
     bots to 1:1. Log before/after counts and confirm the categorical columns
     in the synthetic rows still only contain values from the original
     column's value set (SMOTENC should guarantee this, but assert it as a
     sanity check rather than trusting silently).

   Write a `resample_manifest.json` in this strategy's results dir recording
   the before/after class counts and the strategy used.

3. **Preprocessing variants.** Reuse `create_preprocessing_variants()`
   unchanged — variant A (passthrough) for LightGBM/XGBoost, variant B
   (median-imputed) for RandomForest/TabPFN, variant C (median-imputed +
   scaled) for MLP. Fit imputers/scalers on the **resampled** training fold
   (this is strategy-specific, unlike the shared feature_transform — median
   imputation stats and scaling stats should reflect whatever data the model
   actually trains on) and apply to validation.

4. **Model registry.** Reuse `get_model_registry()` unchanged, filtered to
   `--models`. Before training TabPFN, check the resampled training row count
   against the installed TabPFN version's documented row ceiling; raise a
   clear error (naming the strategy and row count) rather than silently
   proceeding or truncating if it's exceeded. Note: if `--models` doesn't
   include `tabpfn`, none of this fires — leaving `tabpfn` defined in the
   registry but excluded via `--models` is safe and requires no code removal.

5. **Train + tune threshold + evaluate**, per model:
   - Fit as in the old `train_models()` loop (unchanged fit logic / eval_set
     wiring for LightGBM/XGBoost early stopping).
   - Tune threshold using the **fixed** `find_optimal_threshold` described in
     the overview doc (returns `precision_target_met` alongside the
     threshold).
   - Evaluate on validation at the tuned threshold, same metrics as before
     (`roc_auc`, `pr_auc`, `f1`, `precision`, `recall`, `cohen_kappa`, `fpr`,
     `fnr`, confusion matrix counts).
   - Save the fitted model to `artifacts/<strategy>/<model>_model.pkl`.

6. **Results summary.** `results/<strategy>/train_val_model_metrics.csv` —
   same columns as the old `save_results_summary()` output, plus
   `precision_target_met` and `resample_strategy`.

7. **SHAP**, generalized per the overview doc's explainer-selection rule.
   Output per model: `results/<strategy>/shap_bar_<model>.png`,
   `shap_beeswarm_<model>.png`, `shap_feature_ranking_<model>.csv`. Print top
   15 features per model, same as before.

8. **Permutation importance**, looped over all trained models (not just two).
   Output per model: `results/<strategy>/permutation_importance_<model>.csv`.

9. **Correlation analysis**, unchanged, computed once on the (strategy's
   resampled) preprocessing-variant-B training features:
   `results/<strategy>/correlation_heatmap.png`.

## Cross-strategy comparison note (for the write-up, not the code)

Because SHAP/permutation/correlation are all computed on the strategy's own
*resampled* training data, differences between strategies in these plots
partly reflect the resampling itself (e.g. SMOTENC's synthetic bot rows may
shift feature correlations slightly vs. real-only downsampling). This is
expected and worth calling out explicitly in the final report — it is not a
bug to reconcile in code.

## What this script must NOT do

- Must not re-run `train_test_split` or re-fit `fit_feature_transform` —
  both come from `make_splits.py`'s output.
- Must not touch `test_transformed.parquet` at all.
- Must not hardcode SHAP or permutation importance to a single model.
- Must not silently default a threshold to 0.5 without flagging
  `precision_target_met=False`.
