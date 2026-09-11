#!/usr/bin/env python3
"""
make_splits.py

Implements the "Canonical split" and "Feature transform" sections of 00_pipeline_overview.md.
Run this once per experiment (i.e. once per random seed you want to test), before any
training script runs. Its outputs are the single source of truth every other script reads.

This script is HPC-cluster compatible:
- Uses non-interactive matplotlib backend (Agg) to avoid display issues
- Handles paths properly across different filesystems
- Includes comprehensive error handling and logging
- All imports are at top level for cluster compatibility

Usage:
  python make_splits.py \
    --dataset data/dataset.parquet \
    --project-root . \
    --random-seed 42 \
    [--no-filter-zero-posts] \
    [--run-transform-sanity-check]

HPC Cluster Example:
  sbatch train_model.sbatch  # Or your cluster's job submission system
"""

import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# Set matplotlib to non-interactive backend for HPC compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add scripts directory to path for feature_transform import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_transform import fit_feature_transform, apply_feature_transform


# ---------------------------------------------------------------------------
# Data loading / cleaning (reused from old script, unchanged)
# ---------------------------------------------------------------------------

def load_and_clean_data(dataset_path, filter_zero_posts=True):
    """
    Load a parquet dataset and drop invalid rows. Prints class distribution.
    
    This function performs data cleaning steps:
    1. Drops rows with missing/empty author names
    2. Optionally filters authors with 0 comments AND 0 submissions
    
    Args:
        dataset_path (str): Path to parquet file containing the dataset
        filter_zero_posts (bool): If True, filter authors with 0 comments AND 0 submissions
                                  Default: True
    
    Returns:
        pd.DataFrame: Cleaned dataframe with valid author rows
    
    Output:
        Prints initial row count, cleaned row count, and bot/human distribution
    """
    df = pd.read_parquet(dataset_path)
    initial_count = len(df)
    
    # Drop rows with missing/empty author names
    df = df[df["author"].notna() & (df["author"] != "")]
    
    # Filter authors with 0 comments AND 0 submissions
    if filter_zero_posts:
        zero_posts_mask = (df["num_comments"] == 0) & (df["num_submissions"] == 0)
        zero_posts_count = zero_posts_mask.sum()
        if zero_posts_count > 0:
            df = df[~zero_posts_mask]
            print(f"Filtered {zero_posts_count} authors with 0 comments AND 0 submissions")
    
    print(f"Loaded {dataset_path}")
    print(f"  Initial rows : {initial_count}")
    print(f"  After cleaning: {len(df)}")
    print(f"  Bot authors   : {(df['y']==1).sum()}")
    print(f"  Human authors : {(df['y']==0).sum()}")
    return df


def prepare_features_and_labels(df):
    """
    Separate features and labels from the dataframe.
    
    Excludes identifier columns (author) and non-feature columns (reply_time_coverage).
    The label column 'y' is separated from features.
    
    Args:
        df (pd.DataFrame): Input dataframe containing features, labels, and metadata
    
    Returns:
        tuple: (X, y, feature_cols)
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Label series (1=bot, 0=human)
            feature_cols (list): List of feature column names
    
    Output:
        Prints NaN rates for features with missing values
    """
    EXCLUDE_COLS = ["author", "reply_time_coverage"]
    LABEL_COL = "y"
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS + [LABEL_COL]]
    X = df[feature_cols]
    y = df[LABEL_COL]
    nan_rates = X.isnull().mean().sort_values(ascending=False)
    print("\nNaN rates before training:")
    print(nan_rates[nan_rates > 0])
    return X, y, feature_cols


def sanitize_infinities(X):
    """
    Replace inf/-inf values with NaN.
    
    This must run on every split before any transform to handle extreme values
    that could break downstream processing.
    
    Args:
        X (pd.DataFrame): Feature matrix that may contain infinity values
    
    Returns:
        pd.DataFrame: Feature matrix with inf/-inf replaced by NaN
    
    Output:
        Prints count of infinity values replaced (if any)
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    inf_count = (X.isin([np.inf, -np.inf])).sum().sum()
    if inf_count > 0:
        print(f"Replaced {inf_count} infinity values with NaN")
    return X_clean


# ---------------------------------------------------------------------------
# Transform sanity check (reused from old script, unchanged)
# ---------------------------------------------------------------------------

def plot_transform_sanity_check(X_train_raw, X_train_transformed, y_train, columns, output_dir):
    """
    Generate side-by-side raw vs. transformed histograms for visual verification.
    
    This creates a manual-review artifact to verify that:
    1. Bot vs. human separation in the tail is preserved, not flattened by the transform
    2. Z-score routed columns produce a sane, non-degenerate spread rather than
       extreme values (which would signal a bad MAD estimate)
    
    Args:
        X_train_raw (pd.DataFrame): Raw training features before transformation
        X_train_transformed (pd.DataFrame): Training features after transformation
        y_train (pd.Series): Training labels (1=bot, 0=human)
        columns (list): List of column names to visualize (subset of all features)
        output_dir (str): Directory path where the plot will be saved
    
    Returns:
        None (saves plot file to disk)
    
    Output:
        Saves transform_sanity_check.png to output_dir
        Prints confirmation message
    """
    print(f"\n{'='*50}\nGenerating Transform Sanity Check\n{'='*50}")
    
    fig, axes = plt.subplots(len(columns), 2, figsize=(10, 3 * len(columns)))
    bots = y_train == 1
    humans = y_train == 0

    for i, col in enumerate(columns):
        for ax, data, title in [
            (axes[i, 0], X_train_raw[col], f"{col} (raw)"),
            (axes[i, 1], X_train_transformed[col], f"{col} (transformed)"),
        ]:
            ax.hist(data[humans].dropna(), bins=40, alpha=0.5, label="human", density=True)
            ax.hist(data[bots].dropna(), bins=40, alpha=0.5, label="bot", density=True)
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/transform_sanity_check.png", dpi=150)
    plt.close()
    print(f"Saved transform sanity check to {output_dir}/transform_sanity_check.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for make_splits.py.
    
    This function orchestrates the entire split creation and feature transformation process:
    1. Parse command-line arguments
    2. Load and clean the dataset
    3. Drop zero-variance columns
    4. Detect categorical/binary columns
    5. Create stratified train/val/test splits (64/16/20)
    6. Check for author overlap between splits
    7. Report per-split bot:human ratios
    8. Fit feature transform on training data only
    9. Apply transform to all splits
    10. Persist all outputs (raw and transformed splits, parameters, manifests)
    11. Optionally generate transform sanity check plots
    
    The function ensures HPC compatibility by:
    - Using non-interactive matplotlib backend
    - Proper error handling and assertions
    - Comprehensive logging for debugging
    - Path handling that works across different filesystems
    
    Returns:
        None (saves outputs to disk and prints progress)
    
    Raises:
        AssertionError: If author overlap is detected between splits
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If data validation fails
    """
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits and fit feature transform once"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset.parquet")
    parser.add_argument("--project-root", default=".", help="Project root directory (default: .)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-filter-zero-posts", action="store_true", 
                        help="Disable filtering of authors with 0 comments AND 0 submissions")
    parser.add_argument("--run-transform-sanity-check", action="store_true",
                        help="Generate transform sanity check plots")
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = args.project_root
    splits_dir = os.path.join(project_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    filter_zero_posts = not args.no_filter_zero_posts
    SEED = args.random_seed
    
    print(f"Random seed: {SEED}")
    print(f"Project root: {project_root}")
    print(f"Splits directory: {splits_dir}")
    
    # Step 1: Load + clean
    print("\n" + "="*60)
    print("Step 1: Load and clean data")
    print("="*60)
    df = load_and_clean_data(args.dataset, filter_zero_posts=filter_zero_posts)
    X, y, feature_cols = prepare_features_and_labels(df)
    X = sanitize_infinities(X)
    
    # Check bot/human counts are close to expected
    n_bots = (y == 1).sum()
    n_humans = (y == 0).sum()
    expected_bots, expected_humans = 6478, 64780
    if abs(n_bots - expected_bots) > 100 or abs(n_humans - expected_humans) > 1000:
        print(f"\nWARNING: Bot/human counts differ from expected")
        print(f"  Expected: {expected_bots} bots, {expected_humans} humans")
        print(f"  Actual:   {n_bots} bots, {n_humans} humans")
        print(f"  Dataset may have been updated")
    
    # Step 2: Drop zero-variance columns
    print("\n" + "="*60)
    print("Step 2: Drop zero-variance columns")
    print("="*60)
    zero_variance_cols = []
    for col in feature_cols:
        if X[col].nunique(dropna=True) <= 1:
            zero_variance_cols.append(col)
    
    if zero_variance_cols:
        print(f"Dropping {len(zero_variance_cols)} zero-variance columns:")
        for col in zero_variance_cols:
            print(f"  - {col}")
        X = X.drop(columns=zero_variance_cols)
        feature_cols = [c for c in feature_cols if c not in zero_variance_cols]
    else:
        print("No zero-variance columns found")
    
    # Step 3: Detect categorical/binary columns
    print("\n" + "="*60)
    print("Step 3: Detect categorical/binary columns")
    print("="*60)
    categorical_cols = [c for c in feature_cols if X[c].nunique(dropna=True) <= 2]
    print(f"Found {len(categorical_cols)} categorical/binary columns:")
    for col in categorical_cols:
        unique_vals = X[col].dropna().unique()
        print(f"  - {col}: {unique_vals}")
    
    # Save categorical features list
    categorical_features_path = os.path.join(splits_dir, "categorical_features.json")
    with open(categorical_features_path, "w") as f:
        json.dump(categorical_cols, f, indent=2)
    print(f"Saved categorical features to {categorical_features_path}")
    
    # Split into continuous and categorical columns
    continuous_cols = [c for c in feature_cols if c not in categorical_cols]
    print(f"Continuous columns: {len(continuous_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Step 4: Split data 64/16/20 using two stratified calls
    print("\n" + "="*60)
    print("Step 4: Create stratified train/val/test splits")
    print("="*60)
    
    # First split: 80% train_val, 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    
    # Second split: 80% of train_val (64% total) to train, 20% of train_val (16% total) to val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=SEED
    )
    
    print(f"Train: {len(X_train)} rows ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val:   {len(X_val)} rows ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test)} rows ({len(X_test)/len(X)*100:.1f}%)")
    
    # Verify arithmetic
    expected_train_pct = 0.64
    expected_val_pct = 0.16
    expected_test_pct = 0.20
    actual_train_pct = len(X_train) / len(X)
    actual_val_pct = len(X_val) / len(X)
    actual_test_pct = len(X_test) / len(X)
    
    print(f"\nSanity check on split percentages:")
    print(f"  Train: expected {expected_train_pct:.2f}, actual {actual_train_pct:.2f}")
    print(f"  Val:   expected {expected_val_pct:.2f}, actual {actual_val_pct:.2f}")
    print(f"  Test:  expected {expected_test_pct:.2f}, actual {actual_test_pct:.2f}")
    
    if (abs(actual_train_pct - expected_train_pct) > 0.01 or
        abs(actual_val_pct - expected_val_pct) > 0.01 or
        abs(actual_test_pct - expected_test_pct) > 0.01):
        print("WARNING: Split percentages deviate from expected values")
    
    # Step 5: Overlap assertion - critical for data integrity
    print("\n" + "="*60)
    print("Step 5: Check for author overlap between splits")
    print("="*60)
    
    # Add author back to X splits for overlap check
    train_authors = set(df.loc[X_train.index, "author"])
    val_authors = set(df.loc[X_val.index, "author"])
    test_authors = set(df.loc[X_test.index, "author"])
    
    train_val_overlap = train_authors & val_authors
    train_test_overlap = train_authors & test_authors
    val_test_overlap = val_authors & test_authors
    
    print(f"Train authors: {len(train_authors)}")
    print(f"Val authors:   {len(val_authors)}")
    print(f"Test authors:  {len(test_authors)}")
    
    if train_val_overlap:
        print(f"ERROR: {len(train_val_overlap)} authors in both train and val")
        raise AssertionError(f"Train/val overlap detected: {len(train_val_overlap)} authors")
    if train_test_overlap:
        print(f"ERROR: {len(train_test_overlap)} authors in both train and test")
        raise AssertionError(f"Train/test overlap detected: {len(train_test_overlap)} authors")
    if val_test_overlap:
        print(f"ERROR: {len(val_test_overlap)} authors in both val and test")
        raise AssertionError(f"Val/test overlap detected: {len(val_test_overlap)} authors")
    
    print("No overlap detected between splits")
    
    # Step 6: Report per-split ratio
    print("\n" + "="*60)
    print("Step 6: Report per-split bot:human ratios")
    print("="*60)
    
    for split_name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        n_bots_split = (split_y == 1).sum()
        n_humans_split = (split_y == 0).sum()
        ratio = n_humans_split / n_bots_split if n_bots_split > 0 else float('inf')
        print(f"{split_name}: {n_bots_split} bots : {n_humans_split} humans (1:{ratio:.1f})")
        
        # Warn if ratio deviates significantly from 1:10
        if abs(ratio - 10) > 2:
            print(f"  WARNING: Ratio deviates from expected 1:10")
    
    # Step 7: Fit transform on train only, continuous columns only
    print("\n" + "="*60)
    print("Step 7: Fit feature transform on train (continuous columns only)")
    print("="*60)
    
    transform_params = fit_feature_transform(X_train[continuous_cols])
    transform_params_path = os.path.join(splits_dir, "transform_params.pkl")
    joblib.dump(transform_params, transform_params_path)
    print(f"Saved transform params to {transform_params_path}")
    
    # Step 8: Apply transform to all three splits
    print("\n" + "="*60)
    print("Step 8: Apply transform to all splits")
    print("="*60)
    
    def apply_transform_split(X_split, split_name):
        """
        Apply transform to continuous columns only, then reattach categorical columns.
        
        Args:
            X_split (pd.DataFrame): Feature matrix for a split (train/val/test)
            split_name (str): Name of the split for error messages
        
        Returns:
            pd.DataFrame: Transformed feature matrix with categorical columns reattached
        
        Raises:
            AssertionError: If categorical columns are modified by the transform
        """
        X_continuous_transformed = apply_feature_transform(X_split[continuous_cols], transform_params)
        X_transformed = pd.concat([
            X_continuous_transformed,
            X_split[categorical_cols]
        ], axis=1)
        
        # Sanity check: categorical columns should be unchanged
        assert X_transformed[categorical_cols].equals(X_split[categorical_cols]), \
            f"Categorical columns changed after transform in {split_name}"
        
        return X_transformed
    
    X_train_transformed = apply_transform_split(X_train, "train")
    X_val_transformed = apply_transform_split(X_val, "val")
    X_test_transformed = apply_transform_split(X_test, "test")
    
    print("Transform applied to all splits")
    print("Categorical columns verified unchanged")
    
    # Step 9: Persist outputs
    print("\n" + "="*60)
    print("Step 9: Persist outputs")
    print("="*60)
    
    # Create dataframes with author, y, and features
    def create_output_df(X_transformed, split_y, split_name):
        """
        Create output dataframe with author, y, and transformed features.
        
        Args:
            X_transformed (pd.DataFrame): Transformed feature matrix
            split_y (pd.Series): Labels for the split
            split_name (str): Name of the split for error messages
        
        Returns:
            pd.DataFrame: Complete dataframe with features, labels, and author
        """
        df_out = X_transformed.copy()
        df_out["y"] = split_y
        df_out["author"] = df.loc[split_y.index, "author"]
        return df_out
    
    train_df = create_output_df(X_train_transformed, y_train, "train")
    val_df = create_output_df(X_val_transformed, y_val, "val")
    test_df = create_output_df(X_test_transformed, y_test, "test")
    
    # Save raw splits (untransformed)
    train_raw_df = create_output_df(X_train, y_train, "train")
    val_raw_df = create_output_df(X_val, y_val, "val")
    test_raw_df = create_output_df(X_test, y_test, "test")
    
    train_raw_df.to_parquet(os.path.join(splits_dir, "train_raw.parquet"), index=False)
    val_raw_df.to_parquet(os.path.join(splits_dir, "val_raw.parquet"), index=False)
    test_raw_df.to_parquet(os.path.join(splits_dir, "test_raw.parquet"), index=False)
    print("Saved raw splits")
    
    # Save transformed splits
    train_df.to_parquet(os.path.join(splits_dir, "train_transformed.parquet"), index=False)
    val_df.to_parquet(os.path.join(splits_dir, "val_transformed.parquet"), index=False)
    test_df.to_parquet(os.path.join(splits_dir, "test_transformed.parquet"), index=False)
    print("Saved transformed splits")
    
    # Save split manifest with comprehensive metadata
    split_manifest = {
        "seed": SEED,
        "n_total": len(X),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "bot_human_ratio_train": f"1:{(y_train==0).sum()/(y_train==1).sum():.1f}",
        "bot_human_ratio_val": f"1:{(y_val==0).sum()/(y_val==1).sum():.1f}",
        "bot_human_ratio_test": f"1:{(y_test==0).sum()/(y_test==1).sum():.1f}",
        "overlap_check_passed": True,
        "categorical_columns": categorical_cols,
        "continuous_columns": continuous_cols,
        "zero_variance_columns_dropped": zero_variance_cols,
        "timestamp": datetime.now().isoformat()
    }
    
    manifest_path = os.path.join(splits_dir, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"Saved split manifest to {manifest_path}")
    
    # Step 10: Optional sanity-check plots
    if args.run_transform_sanity_check:
        print("\n" + "="*60)
        print("Step 10: Generate transform sanity check plots")
        print("="*60)
        
        # Pick a few representative columns for visualization
        plot_cols = continuous_cols[:5] if len(continuous_cols) >= 5 else continuous_cols
        plot_transform_sanity_check(X_train, X_train_transformed, y_train, plot_cols, splits_dir)
    
    print("\n" + "="*60)
    print("make_splits.py completed successfully")
    print("="*60)
    print(f"\nOutputs saved to {splits_dir}:")
    print("  - train_raw.parquet, val_raw.parquet, test_raw.parquet")
    print("  - train_transformed.parquet, val_transformed.parquet, test_transformed.parquet")
    print("  - transform_params.pkl")
    print("  - categorical_features.json")
    print("  - split_manifest.json")
    if args.run_transform_sanity_check:
        print("  - transform_sanity_check.png")


if __name__ == "__main__":
    main()
