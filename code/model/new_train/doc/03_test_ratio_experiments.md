# Spec: `test_ratio_experiments.py`

Reads `00_pipeline_overview.md` first. Reads the reserved test split from
`make_splits.py` and one trained model from `train_model.py`'s output. Run
this **after** you've picked a winning strategy+model from the three
`train_model.py` runs' validation metrics — it's meant to be pointed at one
model at a time, chosen via `--model` / `--strategy`, not run automatically
across all combinations.

## Purpose

Evaluate one already-trained model across multiple bot:human **test** ratios
(1:10, 1:30, 1:100 by default) to check how precision/recall/threshold
behavior holds up as bots become rarer, while validation ratio (used during
training) stays fixed at 1:10 throughout — this script never touches or
re-tunes against validation.

## CLI arguments

```
--project-root      default "."
--strategy           required, one of {none, downsample, smotenc}
                     (selects which artifacts/<strategy>/ directory to load from)
--model              required, one of
                     {lightgbm, xgboost, randomforest, mlp, tabpfn}
--ratios             space-separated bot:human ratios expressed as bot_fraction,
                     default: 0.0909 0.0323 0.0099   (i.e. 1:10, 1:30, 1:100)
--random-seed        default 42
--min-precision      default 0.90 (for the per-ratio threshold re-sweep)
```

## Steps

1. **Load test data.** Read `splits/test_transformed.parquet` (already
   transformed by `make_splits.py` — do not re-apply or re-fit any
   transform here). Split into bot rows and human rows.

2. **Load the trained model** from
   `artifacts/<strategy>/<model>_model.pkl`, plus whatever preprocessing
   artifacts that model's variant needs (imputer/scaler from
   `artifacts/<strategy>/`, matching the variant assignment in the model
   registry — e.g. RandomForest needs `imputer_median.pkl` from that same
   strategy directory, not a shared one, since imputer stats were fit on
   that strategy's resampled training data).

3. **For each target ratio**, construct the test subsample by holding humans
   fixed and subsampling bots (per the overview doc's "Ratio-robustness
   construction" section):
   ```python
   n_humans = len(human_rows)          # fixed at every ratio
   n_bots_target = round(n_humans * bot_fraction / (1 - bot_fraction))
   n_bots_target = min(n_bots_target, len(bot_rows))  # can't exceed what exists
   bots_sample = bot_rows.sample(n=n_bots_target, random_state=SEED)
   ratio_test_set = pd.concat([bots_sample, human_rows])
   ```
   If `n_bots_target < 50`, print a visible warning that metrics at this
   ratio will be noisy given the small bot sample size, but still compute
   and report them (don't silently skip).

4. **Per-ratio threshold behavior.** For each ratio, do two things (both are
   useful and answer different questions — keep both):
   - Evaluate using the model's **training-time tuned threshold** (loaded
     from that strategy's `train_val_model_metrics.csv`) — this shows how a
     threshold picked at 1:10 validation degrades as real-world prevalence
     shifts.
   - Separately, re-run the fixed `find_optimal_threshold` sweep **on this
     ratio's test set** — this shows what the threshold *would* need to be if
     you could re-tune per prevalence level. Report both side by side; do not
     conflate them into one number.

5. **Metrics per ratio** (both threshold modes): `roc_auc`, `pr_auc`,
   `precision`, `recall`, `f1`, `fpr`, `fnr`, confusion matrix counts, plus
   `precision_target_met` for the re-tuned-threshold mode.

6. **Outputs**, all under `results/ratio_experiments/<model>_<strategy>/`:
   - `ratio_metrics.csv` — one row per ratio per threshold-mode
     (`threshold_mode` column: `training_tuned` or `retuned_per_ratio`).
   - `recall_vs_ratio.png` — recall on y-axis, ratio (as bot:human label, not
     raw fraction) on x-axis, one line per threshold mode.
   - `precision_vs_ratio.png` — same layout, precision on y-axis.
   - `f1_vs_ratio.png` — same layout, F1 on y-axis.
   - `pr_curves_by_ratio.png` — overlaid precision-recall curves, one per
     ratio, to show how the whole curve shifts (not just the operating
     point) as prevalence changes.
   - `run_manifest.json` — strategy, model, ratios requested, actual bot
     counts achieved per ratio, seed, timestamp.

## What this script must NOT do

- Must not reference an extra-humans dataset — ratios are built entirely from
  bot-subsampling within the reserved test fold, per the overview doc.
- Must not touch the validation set or re-tune anything against it.
- Must not run across all five models automatically — it's a single
  `--model`/`--strategy` pointer by design, so you can inspect one
  model/strategy combination at a time after picking a winner from the three
  `train_model.py` runs.
