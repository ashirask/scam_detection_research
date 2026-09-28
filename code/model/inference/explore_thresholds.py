#!/usr/bin/env python3
"""
explore_thresholds.py

Phase B of the inference pipeline: Turn cached probabilities into classifications
at whatever threshold(s) you want to inspect, without touching the model or the
175k-row feature table again.

This script is HPC-cluster compatible:
- Uses non-interactive matplotlib backend (Agg) to avoid display issues
- Handles paths properly across different filesystems
- Includes comprehensive error handling and logging
- All imports are at top level for cluster compatibility

Usage:
  python explore_thresholds.py \
    --scores inference/scores/xgboost_none/scores.parquet \
    --project-root . \
    --strategy none \
    --model xgboost \
    [--thresholds 0.5 0.7 0.9] \
    [--new-raw-data data/new_authors_175k.parquet] \
    [--output-dir inference/review/xgboost_none/]

HPC Cluster Example:
  sbatch explore_thresholds.sbatch  # Or your cluster's job submission system
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Set matplotlib to non-interactive backend for HPC compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
# Key functions
# ---------------------------------------------------------------------------

def load_scores(scores_path: str) -> pd.DataFrame:
    """
    Reads scores.parquet. Must contain ['author', 'probability'].
    
    Args:
        scores_path: Path to Phase A's scores.parquet
    
    Returns:
        DataFrame with columns including 'author' and 'probability'
    
    Raises:
        FileNotFoundError: If scores file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading scores from {scores_path}")
    
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Scores file not found at {scores_path}")
    
    scores_df = pd.read_parquet(scores_path)
    
    required_cols = ['author', 'probability']
    missing_cols = [c for c in required_cols if c not in scores_df.columns]
    if missing_cols:
        raise ValueError(f"Scores file missing required columns: {missing_cols}")
    
    logger.info(f"  Loaded {len(scores_df)} scores")
    logger.info(f"  Probability range: [{scores_df['probability'].min():.4f}, "
                f"{scores_df['probability'].max():.4f}]")
    
    return scores_df


def load_reference_threshold(
    project_root: str, strategy: str, model: str
) -> tuple:
    """
    Reads results/{strategy}/train_val_model_metrics.csv, returns the row for
    `model`'s tuned threshold as a float.
    Also attempts to read
    results/ratio_experiments/{model}_{strategy}/ratio_metrics.csv if it
    exists (returns None if not) — this is surfaced as context in the summary
    output, never applied automatically.
    
    Args:
        project_root: Project root directory
        strategy: Resampling strategy (none, downsample, smotenc)
        model: Model name (lightgbm, xgboost, etc.)
    
    Returns:
        tuple: (default_threshold, ratio_df)
            default_threshold: Training-tuned threshold as float
            ratio_df: DataFrame from ratio experiments if exists, else None
    """
    logger.info("Loading reference threshold from training metrics")
    
    # Load training-tuned threshold
    metrics_path = os.path.join(project_root, "results", strategy, "train_val_model_metrics.csv")
    if not os.path.exists(metrics_path):
        logger.warning(f"Training metrics not found at {metrics_path}")
        return 0.5, None  # Fallback to 0.5
    
    metrics_df = pd.read_csv(metrics_path)
    model_row = metrics_df[metrics_df["model"] == model]
    
    if model_row.empty:
        logger.warning(f"Model {model} not found in training metrics")
        return 0.5, None
    
    default_threshold = model_row.iloc[0]["optimal_threshold"]
    logger.info(f"  Training-tuned threshold: {default_threshold:.4f}")
    
    # Attempt to load ratio experiments data
    ratio_path = os.path.join(project_root, "results", "ratio_experiments", 
                               f"{model}_{strategy}", "ratio_metrics.csv")
    ratio_df = None
    if os.path.exists(ratio_path):
        logger.info(f"  Found ratio experiments data at {ratio_path}")
        ratio_df = pd.read_csv(ratio_path)
    else:
        logger.info(f"  No ratio experiments data found at {ratio_path}")
    
    return default_threshold, ratio_df


def classify_at_thresholds(
    scores_df: pd.DataFrame, thresholds: list
) -> dict:
    """
    For each t in thresholds:
        out = scores_df.copy()
        out["predicted_label"] = (out["probability"] >= t).astype(int)
    Returns {t: out} — same scores_df, one predicted_label column per
    threshold value. No precision/recall/F1 — there is no ground truth for
    the new authors, only "how many, and who."
    
    Args:
        scores_df: DataFrame with 'probability' column
        thresholds: List of threshold values to apply
    
    Returns:
        Dictionary mapping threshold -> DataFrame with predicted_label column
    """
    logger.info(f"Classifying at {len(thresholds)} threshold(s): {thresholds}")
    
    results = {}
    for t in thresholds:
        out = scores_df.copy()
        out["predicted_label"] = (out["probability"] >= t).astype(int)
        
        n_flagged = (out["predicted_label"] == 1).sum()
        flagged_rate_pct = 100 * n_flagged / len(out)
        
        logger.info(f"  Threshold {t:.4f}: {n_flagged} flagged ({flagged_rate_pct:.2f}%)")
        results[t] = out
    
    return results


def attach_raw_features(
    flagged_df: pd.DataFrame, new_raw_df: pd.DataFrame, feature_cols: list
) -> pd.DataFrame:
    """
    Left-joins flagged_df (author, probability, predicted_label) to
    new_raw_df[["author"] + feature_cols] on "author". Returns the merged
    dataframe so a reviewer can see raw (untransformed) feature values
    next to the prediction, not log/z-score units.
    
    Args:
        flagged_df: DataFrame with author, probability, predicted_label columns
        new_raw_df: Raw feature dataframe (untransformed)
        feature_cols: List of feature column names to attach
    
    Returns:
        Merged DataFrame with raw feature values attached
    """
    logger.info(f"Attaching raw features for {len(flagged_df)} flagged authors")
    
    # Select only author and feature columns from raw data
    raw_cols = ["author"] + feature_cols
    raw_subset = new_raw_df[raw_cols]
    
    # Left join on author
    merged = flagged_df.merge(raw_subset, on="author", how="left")
    
    # Check for any authors not found in raw data
    missing_authors = merged[merged[feature_cols[0]].isna()]["author"].tolist()
    if missing_authors:
        logger.warning(f"  {len(missing_authors)} flagged authors not found in raw data")
    
    logger.info(f"  Attached {len(feature_cols)} raw feature columns")
    
    return merged


def plot_score_distribution(
    scores_df: pd.DataFrame, thresholds: list[float], output_path: str
) -> None:
    """
    One histogram of scores_df["probability"] (single distribution — there is
    only one probability per author), with a vertical line at each value in
    thresholds. Saved once per script run, not once per threshold.
    
    Args:
        scores_df: DataFrame with 'probability' column
        thresholds: List of threshold values to mark on plot
        output_path: Path where to save the PNG file
    """
    logger.info(f"Plotting score distribution to {output_path}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(scores_df["probability"], bins=50, alpha=0.7, edgecolor='black')
    
    # Add vertical lines for each threshold
    colors = plt.cm.rainbow(np.linspace(0, 1, len(thresholds)))
    for t, color in zip(thresholds, colors):
        ax.axvline(t, color=color, linestyle='--', linewidth=2, 
                  label=f'Threshold {t:.3f}')
    
    ax.set_xlabel('Probability (Bot Score)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Bot Scores with Thresholds', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved score distribution plot to {output_path}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for explore_thresholds.py.
    
    This function orchestrates the threshold exploration process:
    1. Parse command-line arguments
    2. Load scores from Phase A
    3. Load reference threshold from training metrics
    4. Classify at specified thresholds
    5. Load raw data for feature attachment
    6. For each threshold: attach raw features and write flagged authors
    7. Plot score distribution
    8. Write threshold summary
    
    The function ensures HPC compatibility by:
    - Using logger for structured output to .out/.err files
    - Proper error handling and assertions
    - Comprehensive logging for debugging
    - Path handling that works across different filesystems
    
    Returns:
        None (saves outputs to disk and logs progress)
    
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If data validation fails
    """
    parser = argparse.ArgumentParser(
        description="Explore threshold cutoffs on cached model probabilities"
    )
    parser.add_argument("--scores", required=True,
                        help="Path to Phase A's scores.parquet")
    parser.add_argument("--project-root", required=True,
                        help="For loading reference threshold/ratio context and feature_cols")
    parser.add_argument("--strategy", required=True,
                        choices=["none", "downsample", "smotenc"],
                        help="Must match the strategy used to produce --scores")
    parser.add_argument("--model", required=True,
                        choices=["lightgbm", "xgboost", "randomforest", "mlp", "tabpfn"],
                        help="Must match the model used to produce --scores")
    parser.add_argument("--thresholds", nargs='+', type=float, default=None,
                        help="One or more threshold cutoffs (default: training-tuned threshold)")
    parser.add_argument("--new-raw-data", default=None,
                        help="Raw new-author parquet for attaching features "
                             "(default: read from inference_manifest.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output location "
                             "(default: {project_root}/inference/review/{model}_{strategy}/)")
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = args.project_root
    splits_dir = os.path.join(project_root, "splits")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(project_root, "inference", "review", 
                                   f"{args.model}_{args.strategy}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Phase B: Explore Thresholds")
    logger.info("="*60)
    logger.info(f"Scores path: {args.scores}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Load scores
    logger.info("\n" + "="*60)
    logger.info("Step 1: Load scores")
    logger.info("="*60)
    scores_df = load_scores(args.scores)
    
    # Step 2: Load reference threshold
    logger.info("\n" + "="*60)
    logger.info("Step 2: Load reference threshold")
    logger.info("="*60)
    default_threshold, ratio_df = load_reference_threshold(project_root, args.strategy, args.model)
    
    # Determine thresholds to use
    if args.thresholds:
        thresholds = args.thresholds
        logger.info(f"Using user-specified thresholds: {thresholds}")
    else:
        thresholds = [default_threshold]
        logger.info(f"Using training-tuned threshold: {default_threshold:.4f}")
    
    # Step 3: Classify at thresholds
    logger.info("\n" + "="*60)
    logger.info("Step 3: Classify at thresholds")
    logger.info("="*60)
    results = classify_at_thresholds(scores_df, thresholds)
    
    # Load new raw data path from manifest if not provided
    if args.new_raw_data is None:
        # Try to read from inference manifest
        scores_dir = os.path.dirname(args.scores)
        manifest_path = os.path.join(scores_dir, "inference_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            args.new_raw_data = manifest["new_data_path"]
            logger.info(f"Read new-raw-data path from manifest: {args.new_raw_data}")
        else:
            raise ValueError(
                "Must specify --new-raw-data or provide inference_manifest.json "
                "in the same directory as scores.parquet"
            )
    
    # Step 4: Load raw data
    logger.info("\n" + "="*60)
    logger.info("Step 4: Load raw data")
    logger.info("="*60)
    new_raw_df = pd.read_parquet(args.new_raw_data)
    logger.info(f"Loaded raw data from {args.new_raw_data}")
    logger.info(f"  {len(new_raw_df)} rows")
    
    # Load feature_cols from split manifest
    split_manifest_path = os.path.join(splits_dir, "split_manifest.json")
    if not os.path.exists(split_manifest_path):
        raise FileNotFoundError(f"Split manifest not found at {split_manifest_path}")
    with open(split_manifest_path, "r") as f:
        split_manifest = json.load(f)
    
    # Derive feature_cols from split manifest
    categorical_cols = split_manifest["categorical_columns"]
    continuous_cols = split_manifest["continuous_columns"]
    feature_cols = categorical_cols + continuous_cols
    logger.info(f"  {len(feature_cols)} features from split manifest")
    
    # Step 5: For each threshold, attach raw features and write output
    logger.info("\n" + "="*60)
    logger.info("Step 5: Process each threshold")
    logger.info("="*60)
    
    summary_rows = []
    
    for t, df in results.items():
        logger.info(f"\n--- Processing threshold {t:.4f} ---")
        
        # Create threshold-specific output directory
        threshold_dir = os.path.join(output_dir, f"threshold_{t:.4f}")
        os.makedirs(threshold_dir, exist_ok=True)
        
        # Get flagged authors
        flagged = df[df["predicted_label"] == 1].copy()
        n_flagged = len(flagged)
        flagged_rate_pct = 100 * n_flagged / len(df)
        
        logger.info(f"  Flagged authors: {n_flagged} ({flagged_rate_pct:.2f}%)")
        
        # Attach raw features
        flagged_with_features = attach_raw_features(flagged, new_raw_df, feature_cols)
        
        # Write flagged authors CSV
        flagged_path = os.path.join(threshold_dir, "flagged_authors.csv")
        flagged_with_features.to_csv(flagged_path, index=False)
        logger.info(f"  Saved flagged authors to {flagged_path}")
        
        # Add to summary
        is_training_tuned = (abs(t - default_threshold) < 1e-6)
        summary_rows.append({
            "threshold": t,
            "n_flagged": n_flagged,
            "flagged_rate_pct": flagged_rate_pct,
            "is_training_tuned_threshold": is_training_tuned
        })
    
    # Step 6: Plot score distribution
    logger.info("\n" + "="*60)
    logger.info("Step 6: Plot score distribution")
    logger.info("="*60)
    plot_path = os.path.join(output_dir, "score_distribution.png")
    plot_score_distribution(scores_df, thresholds, plot_path)
    
    # Step 7: Write threshold summary
    logger.info("\n" + "="*60)
    logger.info("Step 7: Write threshold summary")
    logger.info("="*60)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "threshold_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved threshold summary to {summary_path}")
    
    # Log summary table
    logger.info("\nThreshold Summary:")
    logger.info(summary_df.to_string(index=False))
    
    logger.info("\n" + "="*60)
    logger.info("Phase B complete")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Threshold summary: {summary_path}")
    logger.info(f"Score distribution plot: {plot_path}")


if __name__ == "__main__":
    main()
