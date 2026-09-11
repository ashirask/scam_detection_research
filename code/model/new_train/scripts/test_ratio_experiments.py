#!/usr/bin/env python3
"""
test_ratio_experiments.py

Evaluates one already-trained model across multiple bot:human test ratios.
Test-only script -- no training happens here.

This script is HPC-cluster compatible:
- Uses non-interactive matplotlib backend (Agg) to avoid display issues
- Handles paths properly across different filesystems
- Includes comprehensive error handling and logging
- All imports are at top level for cluster compatibility

Usage:
  python test_ratio_experiments.py \
    --project-root . \
    --strategy none \
    --model lightgbm \
    --ratios 0.0909 0.0323 0.0099 \
    --random-seed 42 \
    --min-precision 0.90

HPC Cluster Example:
  sbatch test_ratio_experiments.sbatch  # Or your cluster's job submission system
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

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, average_precision_score,
    roc_curve, precision_recall_curve
)

# Set matplotlib to non-interactive backend for HPC compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging for HPC cluster compatibility
def setup_logging():
    """
    Configure logging for HPC cluster output to .out and .err files.
    
    INFO and DEBUG messages go to stdout (.out file)
    WARNING and ERROR messages go to stderr (.err file)
    All messages include timestamps for debugging.
    """
    # Create logger
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
# Threshold tuning (same fixed version as train_model.py)
# ---------------------------------------------------------------------------

def find_optimal_threshold(model, X_val, y_val, metric="f1", min_precision=0.90, 
                          threshold_range=None):
    """
    Find optimal decision threshold with FIXED behavior per specification.
    
    This function implements the same fixed threshold tuning logic as train_model.py:
    1. Sweeps thresholds across the specified range
    2. Tracks the best F1 among thresholds that clear min_precision
    3. Separately tracks the max-precision threshold across all thresholds
    4. Returns the best-F1 threshold if any cleared min_precision, otherwise returns max-precision threshold
    5. Explicitly reports whether precision target was met via precision_target_met flag
    
    Args:
        model: Trained classifier with predict_proba method
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels (1=bot, 0=human)
        metric (str): Metric to optimize among thresholds clearing min_precision
                     Options: "f1", "precision", "recall". Default: "f1"
        min_precision (float): Minimum precision threshold. Default: 0.90
        threshold_range (np.ndarray): Array of threshold values to evaluate.
                                      If None, uses default range 0.05 to 0.95 step 0.05
    
    Returns:
        tuple: (optimal_threshold, threshold_metrics, precision_target_met)
            optimal_threshold (float): Best threshold found
            threshold_metrics (dict): Metrics at optimal threshold
            precision_target_met (bool): Whether any threshold cleared min_precision
    
    Output:
        Logs warning if no threshold cleared min_precision
    """
    if threshold_range is None:
        threshold_range = np.arange(0.05, 0.96, 0.05)
    
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Track best F1 among thresholds that clear min_precision
    best_threshold_f1, best_score_f1 = 0.5, 0
    # Track max precision across all thresholds
    best_threshold_max_prec, max_precision = 0.5, 0
    
    for thresh in threshold_range:
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        # Track max precision
        if prec > max_precision:
            max_precision = prec
            best_threshold_max_prec = thresh
        
        # Track best F1 among thresholds clearing min_precision
        if prec >= min_precision:
            score = {"f1": f1, "precision": prec, "recall": rec}.get(metric, f1)
            if score > best_score_f1:
                best_score_f1, best_threshold_f1 = score, thresh
    
    # Determine which threshold to use
    if best_score_f1 > 0:  # At least one threshold cleared min_precision
        optimal_threshold = best_threshold_f1
        precision_target_met = True
    else:  # No threshold cleared min_precision
        optimal_threshold = best_threshold_max_prec
        precision_target_met = False
        logger.warning(f"No threshold cleared min_precision={min_precision}")
        logger.warning(f"Using max-precision threshold: {optimal_threshold:.2f} (precision={max_precision:.3f})")
    
    y_pred_opt = (y_proba >= optimal_threshold).astype(int)
    threshold_metrics = {
        "threshold": optimal_threshold,
        "precision": precision_score(y_val, y_pred_opt, zero_division=0),
        "recall": recall_score(y_val, y_pred_opt, zero_division=0),
        "f1": f1_score(y_val, y_pred_opt, zero_division=0),
    }
    
    return optimal_threshold, threshold_metrics, precision_target_met


def evaluate_at_threshold(model, X, y_true, threshold):
    """
    Evaluate a trained model on given data at a specified threshold.
    
    Computes comprehensive classification metrics including:
    - ROC-AUC and PR-AUC (threshold-independent)
    - Precision, Recall, F1 (threshold-dependent)
    - Cohen's Kappa (agreement metric)
    - False Positive Rate and False Negative Rate
    - Confusion matrix components
    
    Args:
        model: Trained classifier with predict_proba method
        X (pd.DataFrame): Feature matrix to evaluate on
        y_true (pd.Series): True labels (1=bot, 0=human)
        threshold (float): Decision threshold for classification
    
    Returns:
        dict: Dictionary of computed metrics
            - roc_auc: float
            - pr_auc: float
            - precision: float
            - recall: float
            - f1: float
            - cohen_kappa: float
            - fpr: float
            - fnr: float
            - tp, fp, tn, fn: int
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    
    metrics = {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, pred),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return metrics


def compute_threshold_curve(model, X, y_true, threshold_range=None):
    """
    Compute metrics across a range of thresholds for threshold sensitivity analysis.
    
    This shows how F1 and precision change as the decision threshold varies,
    independent of any specific threshold choice.
    
    Args:
        model: Trained classifier with predict_proba method
        X (pd.DataFrame): Feature matrix
        y_true (pd.Series): True labels (1=bot, 0=human)
        threshold_range (np.ndarray): Array of threshold values to evaluate
                                      If None, uses default range 0.05-0.95 step 0.02
    
    Returns:
        pd.DataFrame: DataFrame with metrics for each threshold
            - threshold: float
            - precision: float
            - recall: float
            - f1: float
    """
    if threshold_range is None:
        threshold_range = np.arange(0.05, 0.96, 0.02)
    
    y_proba = model.predict_proba(X)[:, 1]
    results = []
    
    for thresh in threshold_range:
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        results.append({"threshold": thresh, "precision": prec, "recall": rec, "f1": f1})
    
    threshold_df = pd.DataFrame(results)
    logger.info(f"Computed threshold curve with {len(threshold_df)} threshold points")
    return threshold_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for test_ratio_experiments.py.
    
    This function orchestrates the ratio robustness evaluation:
    1. Parse command-line arguments
    2. Load test data from make_splits.py
    3. Load trained model and preprocessing artifacts
    4. Evaluate model at multiple test ratios
    5. Compare training-tuned vs per-ratio re-tuned thresholds
    6. Generate comprehensive performance plots:
       - Combined metric vs ratio grid (2x2: Recall, Precision, F1, PR curves)
       - ROC curves by ratio (threshold-independent)
       - Confusion matrices by ratio (training-tuned threshold)
       - Threshold sensitivity analysis (full threshold sweep)
    7. Save results, threshold curves, and manifest
    
    The function ensures HPC compatibility by:
    - Using logger for structured output to .out/.err files
    - Proper error handling and assertions
    - Comprehensive logging for debugging
    - Path handling that works across different filesystems
    
    Returns:
        None (saves outputs to disk and logs progress)
    
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If model or strategy configuration is invalid
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained model across multiple test ratios"
    )
    parser.add_argument("--project-root", default=".", help="Project root directory (default: .)")
    parser.add_argument("--strategy", required=True,
                        choices=["none", "downsample", "smotenc"],
                        help="Resampling strategy used for training")
    parser.add_argument("--model", required=True,
                        choices=["lightgbm", "xgboost", "randomforest", "mlp", "tabpfn"],
                        help="Model to evaluate")
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.0909, 0.0323, 0.0099],
                        help="Bot fractions for test ratios (default: 0.0909 0.0323 0.0099)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed (must match make_splits.py and train_model.py)")
    parser.add_argument("--min-precision", type=float, default=0.90,
                        help="Minimum precision for threshold re-tuning (default: 0.90)")
    parser.add_argument("--threshold-min", type=float, default=0.05,
                        help="Minimum threshold for sweep (default: 0.05)")
    parser.add_argument("--threshold-max", type=float, default=0.95,
                        help="Maximum threshold for sweep (default: 0.95)")
    parser.add_argument("--threshold-step", type=float, default=0.02,
                        help="Step size for threshold sweep (default: 0.02)")
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = args.project_root
    splits_dir = os.path.join(project_root, "splits")
    artifacts_dir = os.path.join(project_root, "artifacts", args.strategy)
    results_dir = os.path.join(project_root, "results", "ratio_experiments", 
                                f"{args.model}_{args.strategy}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    SEED = args.random_seed
    strategy = args.strategy
    model_name = args.model
    
    logger.info(f"Random seed: {SEED}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Ratios (bot fractions): {args.ratios}")
    logger.info(f"Results directory: {results_dir}")
    
    # Step 1: Load test data
    logger.info("\n" + "="*60)
    logger.info("Step 1: Load test data from make_splits.py")
    logger.info("="*60)
    
    test_df = pd.read_parquet(os.path.join(splits_dir, "test_transformed.parquet"))
    logger.info(f"Loaded test: {len(test_df)} rows")
    
    # Separate features and labels
    LABEL_COL = "y"
    AUTHOR_COL = "author"
    feature_cols = [c for c in test_df.columns if c not in [LABEL_COL, AUTHOR_COL]]
    
    X_test = test_df[feature_cols]
    y_test = test_df[LABEL_COL]
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Test bot/human: {(y_test==1).sum()} / {(y_test==0).sum()}")
    
    # Split into bot and human rows
    bot_mask = y_test == 1
    human_mask = y_test == 0
    
    bot_rows = test_df[bot_mask]
    human_rows = test_df[human_mask]
    
    logger.info(f"Bot rows: {len(bot_rows)}")
    logger.info(f"Human rows: {len(human_rows)}")
    
    # Step 2: Load trained model and preprocessing artifacts
    logger.info("\n" + "="*60)
    logger.info("Step 2: Load trained model and preprocessing artifacts")
    logger.info("="*60)
    
    model_path = os.path.join(artifacts_dir, f"{model_name}_model.pkl")
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Load preprocessing artifacts based on model type
    # From get_model_registry: A=passthrough (LGBM/XGB), B=median (RF/TabPFN), C=median+scale (MLP)
    preprocessing_variant = None
    if model_name in ["lightgbm", "xgboost"]:
        preprocessing_variant = "A"
    elif model_name in ["randomforest", "tabpfn"]:
        preprocessing_variant = "B"
    elif model_name == "mlp":
        preprocessing_variant = "C"
    
    if preprocessing_variant == "B":
        imputer = joblib.load(os.path.join(artifacts_dir, "imputer_median.pkl"))
        logger.info("Loaded imputer_median.pkl")
    elif preprocessing_variant == "C":
        imputer = joblib.load(os.path.join(artifacts_dir, "imputer_mlp.pkl"))
        scaler = joblib.load(os.path.join(artifacts_dir, "scaler_mlp.pkl"))
        logger.info("Loaded imputer_mlp.pkl and scaler_mlp.pkl")
    
    # Load training-time tuned threshold from train_val_model_metrics.csv
    metrics_path = os.path.join(project_root, "results", strategy, "train_val_model_metrics.csv")
    metrics_df = pd.read_csv(metrics_path)
    model_metrics = metrics_df[metrics_df["model"] == model_name].iloc[0]
    training_tuned_threshold = model_metrics["optimal_threshold"]
    logger.info(f"Loaded training-time tuned threshold: {training_tuned_threshold:.3f}")
    
    # Step 3: Build threshold range for re-tuning
    threshold_range = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, 
                                args.threshold_step)
    
    # Step 4: Evaluate at each ratio
    logger.info("\n" + "="*60)
    logger.info("Step 3: Evaluate at each test ratio")
    logger.info("="*60)
    
    results = []
    ratio_labels = []
    threshold_curves = {}  # Store threshold curves for each ratio
    roc_data = {}  # Store ROC curve data for each ratio
    confusion_data = {}  # Store confusion matrix data for each ratio
    
    for bot_fraction in args.ratios:
        # Convert bot fraction to ratio label (e.g., 0.0909 -> "1:10")
        ratio_label = f"1:{int((1-bot_fraction)/bot_fraction)}" if bot_fraction > 0 else "1:inf"
        ratio_labels.append(ratio_label)
        
        logger.info(f"\n--- Ratio {ratio_label} (bot_fraction={bot_fraction:.4f}) ---")
        
        # Build test subsample by holding humans fixed and subsampling bots
        n_humans = len(human_rows)
        n_bots_target = round(n_humans * bot_fraction / (1 - bot_fraction))
        n_bots_target = min(n_bots_target, len(bot_rows))  # Can't exceed available bots
        
        if n_bots_target < 50:
            logger.warning(f"Small bot sample size ({n_bots_target}) - metrics will be noisy")
        
        logger.info(f"Target bots: {n_bots_target}, Humans: {n_humans}")
        
        # Sample bots
        rng = np.random.RandomState(SEED)
        bots_sample = bot_rows.sample(n=n_bots_target, random_state=SEED)
        ratio_test_set = pd.concat([bots_sample, human_rows])
        
        # Prepare features
        X_ratio = ratio_test_set[feature_cols]
        y_ratio = ratio_test_set[LABEL_COL]
        
        # Apply preprocessing if needed
        if preprocessing_variant == "B":
            X_ratio = pd.DataFrame(
                imputer.transform(X_ratio), columns=feature_cols, index=X_ratio.index
            )
        elif preprocessing_variant == "C":
            X_ratio = pd.DataFrame(
                scaler.transform(imputer.transform(X_ratio)), 
                columns=feature_cols, index=X_ratio.index
            )
        
        logger.info(f"Actual test set: {len(X_ratio)} rows ({(y_ratio==1).sum()} bots, {(y_ratio==0).sum()} humans)")
        
        # Compute ROC curve data (threshold-independent)
        y_proba = model.predict_proba(X_ratio)[:, 1]
        fpr, tpr, roc_thresholds = roc_curve(y_ratio, y_proba)
        roc_auc = roc_auc_score(y_ratio, y_proba)
        roc_data[ratio_label] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
            "roc_auc": roc_auc
        }
        
        # Compute threshold curve for analysis
        threshold_curve = compute_threshold_curve(model, X_ratio, y_ratio, threshold_range)
        threshold_curves[ratio_label] = threshold_curve
        
        # Evaluate with training-time tuned threshold
        logger.info(f"\nEvaluating with training-time tuned threshold: {training_tuned_threshold:.3f}")
        metrics_training = evaluate_at_threshold(model, X_ratio, y_ratio, training_tuned_threshold)
        metrics_training["ratio_label"] = ratio_label
        metrics_training["bot_fraction"] = bot_fraction
        metrics_training["n_bots"] = (y_ratio == 1).sum()
        metrics_training["n_humans"] = (y_ratio == 0).sum()
        metrics_training["threshold"] = training_tuned_threshold
        metrics_training["threshold_mode"] = "training_tuned"
        metrics_training["precision_target_met"] = None  # Not applicable for training-tuned
        results.append(metrics_training)
        
        # Store confusion matrix data for training-tuned threshold
        confusion_data[ratio_label] = {
            "tp": metrics_training["tp"],
            "fp": metrics_training["fp"],
            "tn": metrics_training["tn"],
            "fn": metrics_training["fn"]
        }
        
        logger.info(f"  Precision: {metrics_training['precision']:.3f}")
        logger.info(f"  Recall: {metrics_training['recall']:.3f}")
        logger.info(f"  F1: {metrics_training['f1']:.3f}")
        
        # Re-tune threshold on this ratio's test set
        logger.info(f"\nRe-tuning threshold on this ratio's test set")
        optimal_threshold, _, precision_target_met = find_optimal_threshold(
            model, X_ratio, y_ratio, metric="f1", 
            min_precision=args.min_precision, threshold_range=threshold_range
        )
        
        # Evaluate with re-tuned threshold
        logger.info(f"Evaluating with re-tuned threshold: {optimal_threshold:.3f}")
        metrics_retuned = evaluate_at_threshold(model, X_ratio, y_ratio, optimal_threshold)
        metrics_retuned["ratio_label"] = ratio_label
        metrics_retuned["bot_fraction"] = bot_fraction
        metrics_retuned["n_bots"] = (y_ratio == 1).sum()
        metrics_retuned["n_humans"] = (y_ratio == 0).sum()
        metrics_retuned["threshold"] = optimal_threshold
        metrics_retuned["threshold_mode"] = "retuned_per_ratio"
        metrics_retuned["precision_target_met"] = precision_target_met
        results.append(metrics_retuned)
        
        logger.info(f"  Precision: {metrics_retuned['precision']:.3f}")
        logger.info(f"  Recall: {metrics_retuned['recall']:.3f}")
        logger.info(f"  F1: {metrics_retuned['f1']:.3f}")
        logger.info(f"  Precision target met: {precision_target_met}")
    
    # Step 5: Save results
    logger.info("\n" + "="*60)
    logger.info("Step 4: Save results")
    logger.info("="*60)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, "ratio_metrics.csv"), index=False)
    logger.info("Saved ratio_metrics.csv")
    
    # Step 6: Generate plots
    logger.info("\n" + "="*60)
    logger.info("Step 5: Generate plots")
    logger.info("="*60)
    
    # Separate results by threshold mode
    training_tuned_results = results_df[results_df["threshold_mode"] == "training_tuned"]
    retuned_results = results_df[results_df["threshold_mode"] == "retuned_per_ratio"]
    
    # 1. Combined metric vs ratio grid plot (2x2)
    logger.info("Generating combined metric vs ratio grid plot")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Metric vs Test Ratio - {model_name} ({strategy})', fontsize=16, fontweight='bold')
    
    # Top-left: Recall vs Ratio
    ax = axes[0, 0]
    ax.plot(ratio_labels, training_tuned_results["recall"], 
         marker='o', label='Training-tuned threshold')
    ax.plot(ratio_labels, retuned_results["recall"], 
         marker='s', label='Re-tuned per ratio')
    ax.set_xlabel('Bot:Human Ratio')
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs Test Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: Precision vs Ratio
    ax = axes[0, 1]
    ax.plot(ratio_labels, training_tuned_results["precision"], 
         marker='o', label='Training-tuned threshold')
    ax.plot(ratio_labels, retuned_results["precision"], 
         marker='s', label='Re-tuned per ratio')
    ax.set_xlabel('Bot:Human Ratio')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Test Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: F1 vs Ratio
    ax = axes[1, 0]
    ax.plot(ratio_labels, training_tuned_results["f1"], 
         marker='o', label='Training-tuned threshold')
    ax.plot(ratio_labels, retuned_results["f1"], 
         marker='s', label='Re-tuned per ratio')
    ax.set_xlabel('Bot:Human Ratio')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs Test Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: PR Curves by Ratio
    ax = axes[1, 1]
    for i, (bot_fraction, ratio_label) in enumerate(zip(args.ratios, ratio_labels)):
        n_humans = len(human_rows)
        n_bots_target = round(n_humans * bot_fraction / (1 - bot_fraction))
        n_bots_target = min(n_bots_target, len(bot_rows))
        
        rng = np.random.RandomState(SEED)
        bots_sample = bot_rows.sample(n=n_bots_target, random_state=SEED)
        ratio_test_set = pd.concat([bots_sample, human_rows])
        
        X_ratio = ratio_test_set[feature_cols]
        y_ratio = ratio_test_set[LABEL_COL]
        
        if preprocessing_variant == "B":
            X_ratio = pd.DataFrame(
                imputer.transform(X_ratio), columns=feature_cols, index=X_ratio.index
            )
        elif preprocessing_variant == "C":
            X_ratio = pd.DataFrame(
                scaler.transform(imputer.transform(X_ratio)), 
                columns=feature_cols, index=X_ratio.index
            )
        
        y_proba = model.predict_proba(X_ratio)[:, 1]
        precision, recall, _ = precision_recall_curve(y_ratio, y_proba)
        pr_auc = average_precision_score(y_ratio, y_proba)
        
        ax.plot(recall, precision, marker=None, label=f'{ratio_label} (AUC={pr_auc:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves by Test Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "metric_vs_ratio_combined.png"), dpi=150)
    plt.close()
    logger.info("Saved metric_vs_ratio_combined.png")
    
    # 2. ROC Curves by Ratio
    logger.info("Generating ROC curves by ratio")
    plt.figure(figsize=(10, 8))
    
    for ratio_label, roc_info in roc_data.items():
        plt.plot(roc_info["fpr"], roc_info["tpr"], 
                marker=None, label=f'{ratio_label} (AUC={roc_info["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves by Test Ratio - {model_name} ({strategy})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curves_by_ratio.png"), dpi=150)
    plt.close()
    logger.info("Saved roc_curves_by_ratio.png")
    
    # 3. Confusion Matrices by Ratio (using training-tuned threshold)
    logger.info("Generating confusion matrices by ratio")
    n_ratios = len(ratio_labels)
    n_cols = min(4, n_ratios)  # Max 4 columns per row
    n_rows = (n_ratios + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f'Confusion Matrices by Test Ratio (Training-Tuned Threshold) - {model_name} ({strategy})', 
               fontsize=16, fontweight='bold')
    
    for idx, (ratio_label, conf_data) in enumerate(confusion_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        cm = np.array([[conf_data["tn"], conf_data["fp"]], 
                       [conf_data["fn"], conf_data["tp"]]])
        
        # Normalize confusion matrix for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'}, ax=ax)
        ax.set_title(f'{ratio_label}\nTN={conf_data["tn"]}, FP={conf_data["fp"]}\nFN={conf_data["fn"]}, TP={conf_data["tp"]}', 
                    fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide unused subplots
    for idx in range(n_ratios, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrices_by_ratio.png"), dpi=150)
    plt.close()
    logger.info("Saved confusion_matrices_by_ratio.png")
    
    # 4. Threshold Analysis Plot
    logger.info("Generating threshold analysis plot")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Threshold Sensitivity Analysis - {model_name} ({strategy})', 
               fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratio_labels)))
    
    # Plot 1: F1 vs Threshold for each ratio
    ax1 = axes[0]
    for idx, (ratio_label, threshold_curve) in enumerate(sorted(threshold_curves.items())):
        ax1.plot(threshold_curve["threshold"], threshold_curve["f1"],
                'o-', linewidth=2, markersize=5, color=colors[idx],
                label=ratio_label)
    ax1.axvline(training_tuned_threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Training-tuned={training_tuned_threshold:.2f}')
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Decision Threshold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: Precision vs Threshold for each ratio
    ax2 = axes[1]
    for idx, (ratio_label, threshold_curve) in enumerate(sorted(threshold_curves.items())):
        ax2.plot(threshold_curve["threshold"], threshold_curve["precision"],
                'o-', linewidth=2, markersize=5, color=colors[idx],
                label=ratio_label)
    ax2.axvline(training_tuned_threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Training-tuned={training_tuned_threshold:.2f}')
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Decision Threshold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Plot 3: Recall vs Threshold for each ratio
    ax3 = axes[2]
    for idx, (ratio_label, threshold_curve) in enumerate(sorted(threshold_curves.items())):
        ax3.plot(threshold_curve["threshold"], threshold_curve["recall"],
                'o-', linewidth=2, markersize=5, color=colors[idx],
                label=ratio_label)
    ax3.axvline(training_tuned_threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Training-tuned={training_tuned_threshold:.2f}')
    ax3.set_xlabel('Decision Threshold')
    ax3.set_ylabel('Recall')
    ax3.set_title('Recall vs Decision Threshold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "threshold_analysis.png"), dpi=150)
    plt.close()
    logger.info("Saved threshold_analysis.png")
    
    # Step 7: Save run manifest
    run_manifest = {
        "strategy": strategy,
        "model": model_name,
        "ratios_requested": args.ratios,
        "ratios_achieved": [
            {
                "ratio_label": r["ratio_label"],
                "bot_fraction": r["bot_fraction"],
                "n_bots": int(r["n_bots"]),
                "n_humans": int(r["n_humans"])
            }
            for r in results
        ],
        "random_seed": SEED,
        "min_precision": args.min_precision,
        "threshold_range": {
            "min": args.threshold_min,
            "max": args.threshold_max,
            "step": args.threshold_step
        },
        "training_tuned_threshold": float(training_tuned_threshold),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(results_dir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=2)
    logger.info("Saved run_manifest.json")
    
    # Step 8: Save threshold curves for additional analysis
    logger.info("Saving threshold curves to CSV")
    for ratio_label, threshold_curve in threshold_curves.items():
        threshold_curve_path = os.path.join(results_dir, f"threshold_curve_{ratio_label.replace(':', '-')}.csv")
        threshold_curve.to_csv(threshold_curve_path, index=False)
    logger.info(f"Saved threshold curves to CSV files")
    
    logger.info("\n" + "="*60)
    logger.info("test_ratio_experiments.py completed successfully")
    logger.info("="*60)
    logger.info(f"\nResults saved to {results_dir}")
    logger.info("Generated plots:")
    logger.info("  - metric_vs_ratio_combined.png (2x2 grid)")
    logger.info("  - roc_curves_by_ratio.png")
    logger.info("  - confusion_matrices_by_ratio.png")
    logger.info("  - threshold_analysis.png")


if __name__ == "__main__":
    main()
