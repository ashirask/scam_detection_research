#!/usr/bin/env python3
"""
score_authors.py

Phase A of the inference pipeline: Load the chosen trained model, replay the exact
training-time feature preparation on new, unlabeled authors, and persist probabilities
so scoring never has to be repeated.

This script is HPC-cluster compatible:
- Uses non-interactive matplotlib backend (Agg) to avoid display issues
- Handles paths properly across different filesystems
- Includes comprehensive error handling and logging
- All imports are at top level for cluster compatibility

Usage:
  python score_authors.py \
    --new-data data/new_authors_175k.parquet \
    --project-root . \
    --strategy none \
    --model xgboost \
    [--no-filter-zero-activity] \
    [--drift-lo-pct 0.01] \
    [--drift-hi-pct 0.99] \
    [--drift-warn-pct 1.0] \
    [--clip-warn-pct 0.1] \
    [--output-dir inference/scores/xgboost_none/]

HPC Cluster Example:
  sbatch score_authors.sbatch  # Or your cluster's job submission system
"""

import os
import sys
import argparse
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy/pandas types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add scripts directory to path for feature_transform import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.feature_transform import apply_feature_transform


# ---------------------------------------------------------------------------
# Logging setup for HPC cluster compatibility
# ---------------------------------------------------------------------------

def setup_logging():
    """
    Configure logging for HPC cluster output to .out and .err files.
    
    INFO and DEBUG messages go to stdout (.out file)
    WARNING and ERROR messages go to stderr (.err file)
    All messages include timestamps for debugging.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # stdout handler for INFO/DEBUG (goes to .out file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                          datefmt='%Y-%m-%d %H:%M:%S')
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)
    
    # stderr handler for WARNING/ERROR (goes to .err file)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                          datefmt='%Y-%m-%d %H:%M:%S')
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)
    
    return logger


# Initialize logging
logger = setup_logging()


# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------

def load_and_clean_new_data(
    dataset_path: str, filter_zero_activity: bool = True
) -> tuple:
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
    logger.info(f"Loading new data from {dataset_path}")
    df = pd.read_parquet(dataset_path)
    initial_count = len(df)
    logger.info(f"  Initial rows: {initial_count}")
    
    excluded = {"missing_author": [], "zero_activity": []}
    
    # Drop rows with missing/empty author names
    missing_author_mask = df["author"].isna() | (df["author"] == "")
    missing_authors = df.loc[missing_author_mask, "author"].tolist()
    if missing_authors:
        excluded["missing_author"] = missing_authors
        logger.warning(f"  Dropping {len(missing_authors)} rows with missing/empty author")
        df = df[~missing_author_mask]
    
    # Assert author uniqueness
    duplicates = df[df["author"].duplicated(keep=False)]
    if not duplicates.empty:
        duplicate_authors = duplicates["author"].unique().tolist()
        raise ValueError(
            f"Found {len(duplicate_authors)} duplicate authors in dataset. "
            f"Each author must map to exactly one row. Duplicates: {duplicate_authors[:10]}..."
        )
    
    # Filter authors with 0 comments AND 0 submissions
    if filter_zero_activity:
        zero_activity_mask = (df["num_comments"] == 0) & (df["num_submissions"] == 0)
        zero_activity_authors = df.loc[zero_activity_mask, "author"].tolist()
        if zero_activity_authors:
            excluded["zero_activity"] = zero_activity_authors
            logger.info(f"  Filtering {len(zero_activity_authors)} authors with 0 comments AND 0 submissions")
            df = df[~zero_activity_mask]
    
    logger.info(f"  After cleaning: {len(df)} rows")
    logger.info(f"  Excluded: {len(excluded['missing_author'])} missing author, "
                f"{len(excluded['zero_activity'])} zero activity")
    
    return df, excluded


def strip_placeholder_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asserts df["y"] is uniformly 0 (raises ValueError otherwise — a non-zero
    value means build_features_v2.py was invoked against the wrong
    population files). Drops the "y" column. Returns df without "y".
    Nothing downstream of this function may ever reference "y" again.
    """
    if "y" not in df.columns:
        raise ValueError("Input dataframe must contain 'y' column (placeholder label)")
    
    unique_values = df["y"].unique()
    if len(unique_values) != 1 or unique_values[0] != 0:
        raise ValueError(
            f"Placeholder label 'y' must be uniformly 0 for unlabeled inference data. "
            f"Found values: {unique_values}. This indicates build_features_v2.py was "
            f"invoked incorrectly against the wrong population files."
        )
    
    logger.info("Verified placeholder label 'y' is uniformly 0 (as expected for unlabeled data)")
    df = df.drop(columns=["y"])
    return df


def enforce_feature_schema(
    df: pd.DataFrame, feature_cols: list
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
    logger.info("Enforcing feature schema")
    
    # Check for missing features
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required features in new data: {missing_features}. "
            f"Model cannot score without these features."
        )
    
    # Check for extra columns
    extra_columns = [c for c in df.columns if c not in feature_cols + ["author", "y"]]
    if extra_columns:
        logger.warning(f"  Dropping extra columns not in training feature set: {extra_columns}")
    
    # Set author as index before dropping columns
    if "author" not in df.columns:
        raise ValueError("Dataframe must contain 'author' column")
    
    df = df.set_index("author")
    logger.info(f"  Set 'author' as index (now {len(df)} authors)")
    
    # Reindex to exact feature columns in correct order
    df = df[feature_cols]
    logger.info(f"  Enforced schema: {len(feature_cols)} features")
    
    return df


def sanitize_infinities(X: pd.DataFrame) -> pd.DataFrame:
    """Replace +inf/-inf with NaN. Identical to the training-pipeline version."""
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    inf_count = (X.isin([np.inf, -np.inf])).sum().sum()
    if inf_count > 0:
        logger.info(f"Replaced {inf_count} infinity values with NaN")
    return X_clean


# ---------------------------------------------------------------------------
# Drift checking
# ---------------------------------------------------------------------------

def run_drift_checks(
    X_new_raw: pd.DataFrame,
    train_raw: pd.DataFrame,
    continuous_cols: list,
    lo_pct: float = 0.01,
    hi_pct: float = 0.99,
    warn_pct: float = 1.0,
) -> tuple:
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
    logger.info("Running drift checks against training distribution")
    logger.info(f"  Percentile band: {lo_pct*100:.0f}% to {hi_pct*100:.0f}%")
    logger.info(f"  Warning threshold: {warn_pct}% out of range")
    
    per_author_data = {}
    summary = {}
    
    for col in continuous_cols:
        if col not in X_new_raw.columns or col not in train_raw.columns:
            logger.warning(f"  Skipping drift check for {col} (missing from one dataset)")
            continue
        
        train_lo = train_raw[col].quantile(lo_pct)
        train_hi = train_raw[col].quantile(hi_pct)
        new_min = X_new_raw[col].min()
        new_max = X_new_raw[col].max()
        
        oor_mask = (X_new_raw[col] < train_lo) | (X_new_raw[col] > train_hi)
        pct_oor = 100 * oor_mask.mean()
        
        per_author_data[f"{col}__oor"] = oor_mask
        
        summary[col] = {
            "pct_out_of_range": float(pct_oor),
            "train_lo": float(train_lo),
            "train_hi": float(train_hi),
            "new_min": float(new_min),
            "new_max": float(new_max)
        }
        
        if pct_oor > warn_pct:
            logger.warning(
                f"  {col}: {pct_oor:.2f}% out of range "
                f"(train range: [{train_lo:.3f}, {train_hi:.3f}], "
                f"new range: [{new_min:.3f}, {new_max:.3f}])"
            )
    
    # Build per-author DataFrame
    per_author_df = pd.DataFrame(per_author_data, index=X_new_raw.index)
    per_author_df["n_features_out_of_range"] = per_author_df.sum(axis=1).astype(int)
    
    n_authors_with_drift = (per_author_df["n_features_out_of_range"] > 0).sum()
    logger.info(f"  Authors with any drift flags: {n_authors_with_drift} "
                f"({100*n_authors_with_drift/len(per_author_df):.2f}%)")
    
    return per_author_df, summary


# ---------------------------------------------------------------------------
# Transform application
# ---------------------------------------------------------------------------

def count_log_clips(
    X_raw: pd.DataFrame, transform_params: dict
) -> dict:
    """
    For every column where transform_params[col]["method"] == "log":
        n_clipped = (X_raw[col] < 0).sum()
        pct_clipped = 100 * n_clipped / len(X_raw)
    Returns {col: {"n_clipped": n_clipped, "pct_clipped": pct_clipped}}.
    This is the precise check discussed in CLIPPING_TRANSFORM_QA.md — it
    reads the method tag directly from transform_params rather than
    inferring it from which columns show negatives elsewhere.
    """
    logger.info("Checking for negative values in log-transformed columns")
    
    clip_report = {}
    log_columns = [col for col, params in transform_params.items() 
                   if params.get("method") == "log"]
    
    for col in log_columns:
        if col not in X_raw.columns:
            continue
        
        n_clipped = (X_raw[col] < 0).sum()
        pct_clipped = 100 * n_clipped / len(X_raw)
        
        clip_report[col] = {
            "n_clipped": int(n_clipped),
            "pct_clipped": float(pct_clipped)
        }
        
        if n_clipped > 0:
            logger.warning(
                f"  {col}: {n_clipped} negative values ({pct_clipped:.3f}%) will be silently zeroed"
            )
    
    total_clipped = sum(r["n_clipped"] for r in clip_report.values())
    if total_clipped == 0:
        logger.info("  No negative values found in log-transformed columns")
    else:
        logger.warning(f"  Total: {total_clipped} values will be clipped across {len(log_columns)} columns")
    
    return clip_report


def apply_saved_transform(
    X: pd.DataFrame,
    transform_params: dict,
    continuous_cols: list,
    categorical_cols: list,
) -> tuple:
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
    logger.info("Applying saved feature transform")
    
    # Run clip check BEFORE applying transform
    clip_report = count_log_clips(X[continuous_cols], transform_params)
    
    # Apply transform to continuous columns only
    X_continuous_transformed = apply_feature_transform(X[continuous_cols], transform_params)
    
    # Reattach categorical columns unchanged
    X_transformed = pd.concat([
        X_continuous_transformed,
        X[categorical_cols]
    ], axis=1)
    
    # Verify categorical columns are unchanged
    if not X_transformed[categorical_cols].equals(X[categorical_cols]):
        raise ValueError("Categorical columns were modified during transform")
    
    logger.info("  Transform applied, categorical columns verified unchanged")
    
    return X_transformed, clip_report


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
    logger.info(f"Applying preprocessing variant for model: {model_name}")
    
    variant_map = {
        "lightgbm": "A",
        "xgboost": "A",
        "randomforest": "B",
        "tabpfn": "B",
        "mlp": "C"
    }
    
    if model_name not in variant_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    variant = variant_map[model_name]
    logger.info(f"  Preprocessing variant: {variant}")
    
    if variant == "A":
        # Passthrough - no preprocessing needed
        logger.info("  Variant A: passthrough (no imputer/scaler)")
        return X
    
    elif variant == "B":
        # Median imputation
        imputer_path = os.path.join(artifacts_dir, "imputer_median.pkl")
        if not os.path.exists(imputer_path):
            raise FileNotFoundError(f"Imputer not found at {imputer_path}")
        
        imputer = joblib.load(imputer_path)
        logger.info("  Variant B: median imputation")
        
        X_imputed = pd.DataFrame(
            imputer.transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_imputed
    
    elif variant == "C":
        # Median imputation + standard scaling
        imputer_path = os.path.join(artifacts_dir, "imputer_mlp.pkl")
        scaler_path = os.path.join(artifacts_dir, "scaler_mlp.pkl")
        
        if not os.path.exists(imputer_path):
            raise FileNotFoundError(f"Imputer not found at {imputer_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        logger.info("  Variant C: median imputation + standard scaling")
        
        X_imputed = pd.DataFrame(
            imputer.transform(X),
            columns=X.columns,
            index=X.index
        )
        X_scaled = pd.DataFrame(
            scaler.transform(X_imputed),
            columns=X.columns,
            index=X.index
        )
        return X_scaled
    
    else:
        raise ValueError(f"Unknown variant: {variant}")


def score(model, X: pd.DataFrame) -> np.ndarray:
    """Returns model.predict_proba(X)[:, 1]."""
    logger.info("Running model inference")
    probabilities = model.predict_proba(X)[:, 1]
    logger.info(f"  Generated probabilities for {len(probabilities)} authors")
    return probabilities


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for score_authors.py.
    
    This function orchestrates the entire scoring process:
    1. Parse command-line arguments
    2. Load and clean new author data
    3. Strip placeholder label
    4. Enforce feature schema and set author as index
    5. Sanitize infinities
    6. Run drift checks against training distribution
    7. Apply saved feature transforms
    8. Apply preprocessing variant (imputer/scaler)
    9. Score with model
    10. Assemble and write outputs
    
    The function ensures HPC compatibility by:
    - Using logger for structured output to .out/.err files
    - Proper error handling and assertions
    - Comprehensive logging for debugging
    - Path handling that works across different filesystems
    
    Returns:
        None (saves outputs to disk and logs progress)
    
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If data validation fails or model/strategy mismatch
    """
    parser = argparse.ArgumentParser(
        description="Score new unlabeled authors using a trained bot detection model"
    )
    parser.add_argument("--new-data", required=True,
                        help="Parquet from build_features_v2.py on the new author pool")
    parser.add_argument("--project-root", required=True,
                        help="Root containing splits/, artifacts/, results/")
    parser.add_argument("--strategy", required=True,
                        choices=["none", "downsample", "smotenc"],
                        help="Resampling strategy used for training")
    parser.add_argument("--model", required=True,
                        choices=["lightgbm", "xgboost", "randomforest", "mlp", "tabpfn"],
                        help="Model to use for scoring")
    parser.add_argument("--no-filter-zero-activity", action="store_true",
                        help="Disable filtering of authors with 0 comments AND 0 submissions")
    parser.add_argument("--drift-lo-pct", type=float, default=0.01,
                        help="Lower percentile for drift check (default: 0.01)")
    parser.add_argument("--drift-hi-pct", type=float, default=0.99,
                        help="Upper percentile for drift check (default: 0.99)")
    parser.add_argument("--drift-warn-pct", type=float, default=1.0,
                        help="Warn if more than this % of rows fall outside drift band (default: 1.0)")
    parser.add_argument("--clip-warn-pct", type=float, default=0.1,
                        help="Warn if more than this % of rows get clipped in log-method columns (default: 0.1)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output location (default: {project_root}/inference/scores/{model}_{strategy}/)")
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = args.project_root
    splits_dir = os.path.join(project_root, "splits")
    artifacts_dir = os.path.join(project_root, "artifacts", args.strategy)
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(project_root, "inference", "scores", 
                                   f"{args.model}_{args.strategy}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    filter_zero_activity = not args.no_filter_zero_activity
    
    logger.info("="*60)
    logger.info("Phase A: Score Authors")
    logger.info("="*60)
    logger.info(f"New data path: {args.new_data}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Filter zero activity: {filter_zero_activity}")
    
    # Step 1: Load and clean new data
    logger.info("\n" + "="*60)
    logger.info("Step 1: Load and clean new data")
    logger.info("="*60)
    df, excluded = load_and_clean_new_data(args.new_data, filter_zero_activity)
    
    # Step 2: Strip placeholder label
    logger.info("\n" + "="*60)
    logger.info("Step 2: Strip placeholder label")
    logger.info("="*60)
    df = strip_placeholder_label(df)
    
    # Load required artifacts
    logger.info("\n" + "="*60)
    logger.info("Loading training artifacts")
    logger.info("="*60)
    
    # Load split manifest
    split_manifest_path = os.path.join(splits_dir, "split_manifest.json")
    if not os.path.exists(split_manifest_path):
        raise FileNotFoundError(f"Split manifest not found at {split_manifest_path}")
    with open(split_manifest_path, "r") as f:
        split_manifest = json.load(f)
    logger.info(f"Loaded split manifest from {split_manifest_path}")
    
    categorical_cols = split_manifest["categorical_columns"]
    continuous_cols = split_manifest["continuous_columns"]
    logger.info(f"  Categorical columns: {len(categorical_cols)}")
    logger.info(f"  Continuous columns: {len(continuous_cols)}")
    
    # Load transform params
    transform_params_path = os.path.join(splits_dir, "transform_params.pkl")
    if not os.path.exists(transform_params_path):
        raise FileNotFoundError(f"Transform params not found at {transform_params_path}")
    transform_params = joblib.load(transform_params_path)
    logger.info(f"Loaded transform params from {transform_params_path}")
    
    # Load training raw data for drift reference
    train_raw_path = os.path.join(splits_dir, "train_raw.parquet")
    if not os.path.exists(train_raw_path):
        raise FileNotFoundError(f"Training raw data not found at {train_raw_path}")
    train_raw = pd.read_parquet(train_raw_path)
    logger.info(f"Loaded training raw data from {train_raw_path}")
    
    # Load training transformed data for authoritative feature_cols
    train_transformed_path = os.path.join(splits_dir, "train_transformed.parquet")
    if not os.path.exists(train_transformed_path):
        raise FileNotFoundError(f"Training transformed data not found at {train_transformed_path}")
    train_transformed = pd.read_parquet(train_transformed_path)
    logger.info(f"Loaded training transformed data from {train_transformed_path}")
    
    # Derive authoritative feature_cols from training transformed data
    LABEL_COL = "y"
    AUTHOR_COL = "author"
    feature_cols = [c for c in train_transformed.columns 
                    if c not in [LABEL_COL, AUTHOR_COL]]
    logger.info(f"  Authoritative feature_cols: {len(feature_cols)} features")
    
    # Load model
    model_path = os.path.join(artifacts_dir, f"{args.model}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Step 3: Enforce feature schema
    logger.info("\n" + "="*60)
    logger.info("Step 3: Enforce feature schema")
    logger.info("="*60)
    X = enforce_feature_schema(df, feature_cols)
    
    # Step 4: Sanitize infinities
    logger.info("\n" + "="*60)
    logger.info("Step 4: Sanitize infinities")
    logger.info("="*60)
    X = sanitize_infinities(X)
    
    # Step 5: Run drift checks
    logger.info("\n" + "="*60)
    logger.info("Step 5: Run drift checks")
    logger.info("="*60)
    drift_df, drift_summary = run_drift_checks(
        X, train_raw, continuous_cols,
        lo_pct=args.drift_lo_pct,
        hi_pct=args.drift_hi_pct,
        warn_pct=args.drift_warn_pct
    )
    
    # Step 6: Apply saved transform
    logger.info("\n" + "="*60)
    logger.info("Step 6: Apply saved transform")
    logger.info("="*60)
    X, clip_report = apply_saved_transform(X, transform_params, continuous_cols, categorical_cols)
    
    # Step 7: Apply preprocessing variant
    logger.info("\n" + "="*60)
    logger.info("Step 7: Apply preprocessing variant")
    logger.info("="*60)
    X = apply_preprocessing_variant(X, args.model, artifacts_dir)
    
    # Step 8: Score
    logger.info("\n" + "="*60)
    logger.info("Step 8: Score with model")
    logger.info("="*60)
    probabilities = score(model, X)
    
    # Step 9: Assemble scores
    logger.info("\n" + "="*60)
    logger.info("Step 9: Assemble scores")
    logger.info("="*60)
    
    run_timestamp = datetime.utcnow().isoformat() + "Z"
    
    scores_df = pd.DataFrame({
        "author": X.index,
        "probability": probabilities,
        "model": args.model,
        "strategy": args.strategy,
        "scored_at": run_timestamp,
        "n_features_out_of_range": drift_df["n_features_out_of_range"]
    })
    
    logger.info(f"  Assembled scores for {len(scores_df)} authors")
    
    # Step 10: Write outputs
    logger.info("\n" + "="*60)
    logger.info("Step 10: Write outputs")
    logger.info("="*60)
    
    # Write scores.parquet
    scores_path = os.path.join(output_dir, "scores.parquet")
    scores_df.to_parquet(scores_path, index=False)
    logger.info(f"  Saved scores to {scores_path}")
    
    # Write excluded_authors.json
    excluded_output = {
        "missing_author": {
            "count": int(len(excluded["missing_author"])),
            "authors": excluded["missing_author"]
        },
        "zero_activity": {
            "count": int(len(excluded["zero_activity"])),
            "authors": excluded["zero_activity"]
        }
    }
    excluded_path = os.path.join(output_dir, "excluded_authors.json")
    with open(excluded_path, "w") as f:
        json.dump(excluded_output, f, indent=2, cls=NumpyEncoder)
    logger.info(f"  Saved excluded authors to {excluded_path}")
    
    # Write inference_manifest.json
    n_authors_with_drift = (drift_df["n_features_out_of_range"] > 0).sum()
    
    # Check for extra/missing features
    feature_schema_check = {
        "missing_features": [],
        "extra_features_dropped": []
    }
    
    # Get actual columns from input df (before schema enforcement)
    # We need to reload the original df to check this
    original_df = pd.read_parquet(args.new_data)
    original_feature_cols = [c for c in original_df.columns 
                             if c not in ["author", "y", "reply_time_coverage"]]
    
    missing_features = [f for f in feature_cols if f not in original_feature_cols]
    extra_features = [f for f in original_feature_cols if f not in feature_cols]
    
    feature_schema_check["missing_features"] = missing_features
    feature_schema_check["extra_features_dropped"] = extra_features
    
    manifest = {
        "run_timestamp": run_timestamp,
        "new_data_path": args.new_data,
        "project_root": project_root,
        "model": args.model,
        "strategy": args.strategy,
        "model_artifact_path": model_path,
        "n_input_rows": int(len(original_df)),
        "n_excluded_missing_author": int(len(excluded["missing_author"])),
        "n_excluded_zero_activity": int(len(excluded["zero_activity"])),
        "n_scored": int(len(scores_df)),
        "feature_schema_check": feature_schema_check,
        "clip_report": clip_report,
        "drift_summary": drift_summary,
        "n_authors_with_any_drift_flag": int(n_authors_with_drift)
    }
    
    manifest_path = os.path.join(output_dir, "inference_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    logger.info(f"  Saved inference manifest to {manifest_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Phase A complete")
    logger.info("="*60)
    logger.info(f"Scores: {scores_path}")
    logger.info(f"Excluded authors: {excluded_path}")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
