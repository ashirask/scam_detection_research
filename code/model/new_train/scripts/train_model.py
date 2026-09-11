#!/usr/bin/env python3
"""
train_model.py

Trains multiple classifiers on the bot/human dataset with a specified resampling strategy.
Threshold tuning is done on the validation set. Test evaluation happens in test_ratio_experiments.py.

This script is HPC-cluster compatible:
- Uses non-interactive matplotlib backend (Agg) to avoid display issues
- Handles TabPFN row ceiling errors gracefully
- Includes comprehensive error handling and logging
- All imports are at top level for cluster compatibility
- Memory-efficient processing for large datasets

Usage:
  python train_model.py \
    --project-root . \
    --resample-strategy none \
    --models lightgbm xgboost randomforest mlp tabpfn \
    --min-precision 0.90 \
    --random-seed 42 \
    [--skip-shap] [--skip-permutation] [--skip-tabpfn]

HPC Cluster Example:
  sbatch train_model.sbatch  # Or your cluster's job submission system
"""

import os
import sys
import argparse
import json
import time
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, cohen_kappa_score,
    confusion_matrix, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.inspection import permutation_importance

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tabpfn import TabPFNClassifier

# Set matplotlib to non-interactive backend for HPC compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import shap

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

# Suppress SHAP TreeExplainer warning about list output format
warnings.filterwarnings('ignore', message='.*TreeExplainer shap values output has changed.*')


# ---------------------------------------------------------------------------
# Data loading / cleaning (reused from old script, unchanged)
# ---------------------------------------------------------------------------

def load_and_clean_data(dataset_path, filter_zero_posts=True):
    """
    Load a parquet dataset and drop invalid rows. Logs class distribution.
    
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
        Logs initial row count, cleaned row count, and bot/human distribution
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
            logger.info(f"Filtered {zero_posts_count} authors with 0 comments AND 0 submissions")
    
    logger.info(f"Loaded {dataset_path}")
    logger.info(f"  Initial rows : {initial_count}")
    logger.info(f"  After cleaning: {len(df)}")
    logger.info(f"  Bot authors   : {(df['y']==1).sum()}")
    logger.info(f"  Human authors : {(df['y']==0).sum()}")
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
        Logs NaN rates for features with missing values
    """
    EXCLUDE_COLS = ["author", "reply_time_coverage"]
    LABEL_COL = "y"
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS + [LABEL_COL]]
    X = df[feature_cols]
    y = df[LABEL_COL]
    nan_rates = X.isnull().mean().sort_values(ascending=False)
    logger.info("NaN rates before training:")
    logger.info(f"\n{nan_rates[nan_rates > 0]}")
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
        Logs count of infinity values replaced (if any)
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    inf_count = (X.isin([np.inf, -np.inf])).sum().sum()
    if inf_count > 0:
        logger.info(f"Replaced {inf_count} infinity values with NaN")
    return X_clean


# ---------------------------------------------------------------------------
# Preprocessing variants (unchanged from original pipeline)
# ---------------------------------------------------------------------------

def create_preprocessing_variants(X_train, X_val, feature_cols, artifact_dir):
    X_train_A, X_val_A = X_train, X_val

    imputer_B = SimpleImputer(strategy="median")
    X_train_B = pd.DataFrame(imputer_B.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_B = pd.DataFrame(imputer_B.transform(X_val), columns=feature_cols, index=X_val.index)

    imputer_C = SimpleImputer(strategy="median")
    scaler_C = StandardScaler()
    X_train_C = pd.DataFrame(
        scaler_C.fit_transform(imputer_C.fit_transform(X_train)), columns=feature_cols, index=X_train.index
    )
    X_val_C = pd.DataFrame(
        scaler_C.transform(imputer_C.transform(X_val)), columns=feature_cols, index=X_val.index
    )

    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(imputer_B, f"{artifact_dir}/imputer_median.pkl")
    joblib.dump(imputer_C, f"{artifact_dir}/imputer_mlp.pkl")
    joblib.dump(scaler_C, f"{artifact_dir}/scaler_mlp.pkl")
    logger.info(f"Saved preprocessors to {artifact_dir}/")

    return {
        "A": (X_train_A, X_val_A),
        "B": (X_train_B, X_val_B),
        "C": (X_train_C, X_val_C),
    }


def get_model_registry(skip_tabpfn=False):
    MODELS = {
        "lightgbm": {
            "model": LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1,
            ),
            "preprocessing": "A",
            "fit_kwargs": {
                "eval_set": [(None, None)],
                "callbacks": [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
            }
        },
        "xgboost": {
            "model": XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42,
                eval_metric="logloss", early_stopping_rounds=50, verbosity=0,
            ),
            "preprocessing": "A",
            "fit_kwargs": {"eval_set": [(None, None)], "verbose": False}
        },
        "randomforest": {
            "model": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
            "preprocessing": "B",
            "fit_kwargs": {}
        },
        "tabpfn": {
            "model": TabPFNClassifier(device="cuda"),
            "preprocessing": "B",
            "fit_kwargs": {}
        } if not skip_tabpfn else None,
        "mlp": {
            "model": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=200,
                early_stopping=True, validation_fraction=0.1, random_state=42,
            ),
            "preprocessing": "C",
            "fit_kwargs": {}
        },
    }
    if skip_tabpfn and "tabpfn" in MODELS:
        del MODELS["tabpfn"]
    return MODELS


# ---------------------------------------------------------------------------
# Threshold tuning + evaluation (FIXED version per spec)
# ---------------------------------------------------------------------------

def find_optimal_threshold(model, X_val, y_val, metric="f1", min_precision=0.90, 
                          threshold_range=None):
    """
    Find optimal decision threshold with FIXED behavior per specification.
    
    This function implements the fixed threshold tuning logic that resolves the
    silent fallback bug from the old pipeline. It:
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
                - threshold: float
                - precision: float
                - recall: float
                - f1: float
            precision_target_met (bool): Whether any threshold cleared min_precision
    
    Output:
        Logs optimal threshold, metrics at that threshold, and whether precision target was met
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
    logger.info(f"Optimal threshold: {optimal_threshold:.2f} (target: {metric}, min_precision: {min_precision})")
    logger.info(f"Metrics at optimal threshold: Precision={threshold_metrics['precision']:.3f}, "
          f"Recall={threshold_metrics['recall']:.3f}, F1={threshold_metrics['f1']:.3f}")
    logger.info(f"Precision target met: {precision_target_met}")
    
    return optimal_threshold, threshold_metrics, precision_target_met


def evaluate(model, X, y_true, name="model", threshold=0.5):
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
        name (str): Name for logging purposes. Default: "model"
        threshold (float): Decision threshold for classification. Default: 0.5
    
    Returns:
        dict: Dictionary of computed metrics
            - roc_auc: float (Area under ROC curve)
            - pr_auc: float (Area under precision-recall curve)
            - f1: float (F1 score)
            - precision: float (Precision score)
            - recall: float (Recall score)
            - cohen_kappa: float (Cohen's kappa coefficient)
            - fpr: float (False positive rate)
            - fnr: float (False negative rate)
            - tp: int (True positives)
            - fp: int (False positives)
            - tn: int (True negatives)
            - fn: int (False negatives)
    
    Output:
        Logs classification report and key metrics
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    metrics = {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "cohen_kappa": cohen_kappa_score(y_true, pred),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    logger.info(f"--- {name} ---")
    logger.info(f"\n{classification_report(y_true, pred, target_names=['human', 'bot'])}")
    logger.info(f"ROC-AUC : {metrics['roc_auc']:.4f}")
    logger.info(f"PR-AUC  : {metrics['pr_auc']:.4f}")
    logger.info(f"Kappa   : {metrics['cohen_kappa']:.4f}")
    logger.info(f"FPR     : {metrics['fpr']:.4f}  (humans wrongly flagged)")
    logger.info(f"FNR     : {metrics['fnr']:.4f}  (bots missed)")
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_models(MODELS, PREPROCESSING_DATA, y_train, y_val, artifact_dir, 
                 tune_threshold=True, min_precision=0.90, threshold_range=None):
    """
    Train all models in the registry with threshold tuning and evaluation.
    
    This function orchestrates the training process for all models:
    1. Iterates through model registry
    2. Applies appropriate preprocessing variant
    3. Fits model with early stopping where applicable
    4. Tunes decision threshold on validation set
    5. Evaluates model at optimal threshold
    6. Saves model artifact and records results
    
    Args:
        MODELS (dict): Model registry from get_model_registry()
        PREPROCESSING_DATA (dict): Preprocessing variants from create_preprocessing_variants()
        y_train (pd.Series): Training labels
        y_val (pd.Series): Validation labels
        artifact_dir (str): Directory to save model artifacts
        tune_threshold (bool): Whether to perform threshold tuning. Default: True
        min_precision (float): Minimum precision for threshold tuning. Default: 0.90
        threshold_range (np.ndarray): Threshold values to sweep. Default: 0.05-0.95 step 0.05
    
    Returns:
        dict: Results dictionary with keys as model names and values containing:
            - val_metrics: dict of validation metrics
            - train_time_s: training time in seconds
            - optimal_threshold: best threshold found
            - threshold_metrics: metrics at optimal threshold
            - precision_target_met: whether precision target was achieved
    
    Output:
        Logs training progress for each model
        Saves model artifacts to artifact_dir
    """
    os.makedirs(artifact_dir, exist_ok=True)
    all_results = {}

    for name, config in MODELS.items():
        logger.info(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
        Xtr, Xv = PREPROCESSING_DATA[config["preprocessing"]]
        model = config["model"]

        fit_kwargs = config.get("fit_kwargs", {}).copy()
        if "eval_set" in fit_kwargs:
            fit_kwargs["eval_set"] = [(Xv, y_val)]

        t0 = time.time()
        model.fit(Xtr, y_train, **fit_kwargs)
        train_time = time.time() - t0

        optimal_threshold, threshold_metrics, precision_target_met = 0.5, {}, False
        if tune_threshold:
            logger.info("Tuning threshold on validation set...")
            optimal_threshold, threshold_metrics, precision_target_met = find_optimal_threshold(
                model, Xv, y_val, metric="f1", min_precision=min_precision, 
                threshold_range=threshold_range
            )

        val_metrics = evaluate(model, Xv, y_val, name=f"{name}_val", threshold=optimal_threshold)
        joblib.dump(model, f"{artifact_dir}/{name}_model.pkl")

        all_results[name] = {
            "val_metrics": val_metrics,
            "train_time_s": round(train_time, 2),
            "optimal_threshold": optimal_threshold,
            "threshold_metrics": threshold_metrics,
            "precision_target_met": precision_target_met,
        }
        logger.info(f"Training time: {train_time:.1f}s")

    return all_results


def save_results_summary(all_results, output_dir, resample_strategy):
    """
    Save training results to CSV and log comparison table.
    
    Creates a comprehensive results CSV with all validation metrics,
    including the new precision_target_met flag, and logs a
    formatted comparison table for easy model selection.
    
    Args:
        all_results (dict): Results dictionary from train_models()
        output_dir (str): Directory to save results CSV
        resample_strategy (str): Name of resampling strategy used
    
    Returns:
        None (saves CSV file and logs table)
    
    Output:
        Saves train_val_model_metrics.csv to output_dir
        Logs formatted comparison table
    """
    rows = []
    for name, res in all_results.items():
        metrics = res.get("val_metrics")
        if metrics:
            row = {
                "model": name, 
                "split": "val",
                "resample_strategy": resample_strategy,
                "train_time_s": res.get("train_time_s", ""),
                "optimal_threshold": res.get("optimal_threshold", 0.5),
                "precision_target_met": res.get("precision_target_met", False),
            }
            row.update({k: v for k, v in metrics.items() if k not in ("tp", "fp", "tn", "fn")})
            rows.append(row)

    results_df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/train_val_model_metrics.csv", index=False)

    logger.info("=== Model Comparison (Validation) ===")
    logger.info(f"""\n{results_df[['model', 'split', 'resample_strategy', 'optimal_threshold', 
                     'precision_target_met', 'roc_auc', 'pr_auc', 'f1',
                     'precision', 'recall', 'cohen_kappa', 'fpr', 'fnr']].to_string(index=False)}""")


# ---------------------------------------------------------------------------
# SHAP importance (generalized to all models per spec)
# ---------------------------------------------------------------------------

def compute_shap_importance(MODELS, PREPROCESSING_DATA, feature_cols, artifact_dir, 
                            output_dir, shap_background_size=100):
    """
    Compute SHAP feature importance for all trained models.
    
    Explainer selection by model type:
      - lightgbm, xgboost, randomforest -> shap.TreeExplainer
      - mlp, tabpfn -> shap.Explainer with background sample
    
    Args:
        MODELS (dict): Model registry
        PREPROCESSING_DATA (dict): Preprocessing variants
        feature_cols (list): Feature column names
        artifact_dir (str): Directory containing trained models
        output_dir (str): Directory to save SHAP outputs
        shap_background_size (int): Background sample size for non-tree models. Default: 100
    
    Returns:
        None (saves SHAP plots and CSV files)
    
    Output:
        Logs progress and top features for each model
        Saves SHAP bar plots, beeswarm plots, and feature rankings
    """
    logger.info(f"\n{'='*50}\nComputing SHAP Feature Importance\n{'='*50}")
    
    for name, config in MODELS.items():
        logger.info(f"\n--- SHAP for {name} ---")
        model = joblib.load(f"{artifact_dir}/{name}_model.pkl")
        Xtr, Xv = PREPROCESSING_DATA[config["preprocessing"]]
        
        # Check for empty training data
        if len(Xtr) == 0:
            logger.warning(f"Skipping SHAP for {name}: training data is empty")
            continue
        
        # Select explainer by model type
        if name in ["lightgbm", "xgboost", "randomforest"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xv)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:  # mlp, tabpfn
            # Use background sample for non-tree models
            background_sample = shap.sample(Xtr, min(shap_background_size, len(Xtr)))
            explainer = shap.Explainer(model.predict_proba, background_sample)
            sv = explainer(Xv)
            if hasattr(sv, 'values'):
                sv = sv.values[:, :, 1]  # Take positive class for multi-output
            else:
                sv = sv[:, :, 1] if len(sv.shape) == 3 else sv[:, 1]  # Handle different SHAP versions
        
        # Plot bar chart
        shap.summary_plot(sv, Xv, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_bar_{name}.png", dpi=150)
        plt.close()
        
        # Plot beeswarm
        shap.summary_plot(sv, Xv, show=False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_beeswarm_{name}.png", dpi=150)
        plt.close()
        
        # Feature ranking
        mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols).sort_values(ascending=False)
        mean_abs_shap.to_csv(f"{output_dir}/shap_feature_ranking_{name}.csv")
        logger.info(f"\nTop 15 features by SHAP ({name}):")
        logger.info(f"\n{mean_abs_shap.head(15).to_string()}")
        
        top_shap = mean_abs_shap.iloc[0]
        drop_candidates = mean_abs_shap[mean_abs_shap < 0.01 * top_shap].index.tolist()
        logger.info(f"\nDrop candidates for Tier 2 ({len(drop_candidates)} features):")
        logger.info(f"{drop_candidates}")


# ---------------------------------------------------------------------------
# Permutation importance (generalized to all models per spec)
# ---------------------------------------------------------------------------

def compute_permutation_importance(MODELS, PREPROCESSING_DATA, feature_cols, y_val, 
                                  artifact_dir, output_dir):
    """
    Compute permutation importance for all trained models.
    
    Args:
        MODELS (dict): Model registry
        PREPROCESSING_DATA (dict): Preprocessing variants
        feature_cols (list): Feature column names
        y_val (pd.Series): Validation labels
        artifact_dir (str): Directory containing trained models
        output_dir (str): Directory to save permutation importance outputs
    
    Returns:
        None (saves permutation importance CSV files)
    
    Output:
        Logs progress and bottom features for each model
        Saves permutation importance CSV files
    """
    logger.info(f"\n{'='*50}\nComputing Permutation Importance\n{'='*50}")
    
    for name, config in MODELS.items():
        logger.info(f"\n--- Permutation importance for {name} ---")
        model = joblib.load(f"{artifact_dir}/{name}_model.pkl")
        _, Xv = PREPROCESSING_DATA[config["preprocessing"]]
        result = permutation_importance(model, Xv, y_val, n_repeats=10, random_state=42, scoring="roc_auc")
        perm_df = pd.DataFrame({
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(f"{output_dir}/permutation_importance_{name}.csv", index=False)
        logger.info(f"\nBottom 10 features — {name}:")
        logger.info(f"\n{perm_df.tail(10).to_string(index=False)}")


# ---------------------------------------------------------------------------
# Correlation analysis (unchanged)
# ---------------------------------------------------------------------------

def compute_correlation_analysis(X_train, output_dir):
    """
    Compute correlation matrix on transformed features.
    
    Args:
        X_train (pd.DataFrame): Training features (variant B - median imputed)
        output_dir (str): Directory to save correlation heatmap
    
    Returns:
        None (saves correlation heatmap)
    
    Output:
        Logs highly correlated feature pairs
        Saves correlation heatmap PNG
    """
    logger.info(f"\n{'='*50}\nComputing Correlation Analysis\n{'='*50}")
    corr = X_train.corr()
    high_corr_pairs = [
        (corr.columns[i], corr.columns[j], corr.iloc[i, j])
        for i in range(len(corr.columns)) for j in range(i + 1, len(corr.columns))
        if abs(corr.iloc[i, j]) > 0.95
    ]
    logger.info(f"\nHighly correlated pairs (>0.95): {len(high_corr_pairs)}")
    for a, b, r in high_corr_pairs:
        logger.info(f"  {a} <-> {b}: {r:.3f}")

    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for train_model.py.
    
    This function orchestrates the entire model training process:
    1. Parse command-line arguments
    2. Load splits from make_splits.py
    3. Apply resampling strategy to training fold
    4. Create preprocessing variants
    5. Build model registry
    6. Train models with threshold tuning
    7. Save results summary
    8. Compute SHAP and permutation importance (optional)
    9. Compute correlation analysis
    
    The function ensures HPC compatibility by:
    - Using logger for structured output to .out/.err files
    - Proper error handling and assertions
    - Comprehensive logging for debugging
    - Path handling that works across different filesystems
    
    Returns:
        None (saves outputs to disk and logs progress)
    
    Raises:
        ValueError: If categorical column missing from training data
        ValueError: If TabPFN row ceiling exceeded
        ValueError: If SMOTENC introduces new categorical values
    """
    parser = argparse.ArgumentParser(
        description="Train bot detection models with specified resampling strategy"
    )
    parser.add_argument("--project-root", default=".", help="Project root directory (default: .)")
    parser.add_argument("--resample-strategy", required=True, 
                        choices=["none", "downsample", "smotenc"],
                        help="Resampling strategy for training fold")
    parser.add_argument("--models", nargs="+", 
                        choices=["lightgbm", "xgboost", "randomforest", "mlp", "tabpfn"],
                        default=["lightgbm", "xgboost", "randomforest", "mlp", "tabpfn"],
                        help="Models to train (default: all five)")
    parser.add_argument("--min-precision", type=float, default=0.90,
                        help="Minimum precision for threshold tuning (default: 0.90)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed (must match make_splits.py)")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP computation")
    parser.add_argument("--skip-permutation", action="store_true", 
                        help="Skip permutation importance")
    parser.add_argument("--skip-tabpfn", action="store_true", 
                        help="Skip TabPFN (independent of --models)")
    parser.add_argument("--threshold-min", type=float, default=0.05,
                        help="Minimum threshold for sweep (default: 0.05)")
    parser.add_argument("--threshold-max", type=float, default=0.95,
                        help="Maximum threshold for sweep (default: 0.95)")
    parser.add_argument("--threshold-step", type=float, default=0.02,
                        help="Step size for threshold sweep (default: 0.02)")
    parser.add_argument("--shap-background-size", type=int, default=100,
                        help="Background sample size for non-tree SHAP (default: 100)")
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = args.project_root
    splits_dir = os.path.join(project_root, "splits")
    artifacts_dir = os.path.join(project_root, "artifacts", args.resample_strategy)
    results_dir = os.path.join(project_root, "results", args.resample_strategy)
    
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    SEED = args.random_seed
    resample_strategy = args.resample_strategy
    
    logger.info(f"Random seed: {SEED}")
    logger.info(f"Resample strategy: {resample_strategy}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Artifacts directory: {artifacts_dir}")
    logger.info(f"Results directory: {results_dir}")
    
    # Step 1: Load splits
    logger.info("\n" + "="*60)
    logger.info("Step 1: Load splits from make_splits.py")
    logger.info("="*60)
    
    train_df = pd.read_parquet(os.path.join(splits_dir, "train_transformed.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "val_transformed.parquet"))
    
    logger.info(f"Loaded train: {len(train_df)} rows")
    logger.info(f"Loaded val: {len(val_df)} rows")
    
    # Load categorical features
    with open(os.path.join(splits_dir, "categorical_features.json"), "r") as f:
        categorical_features = json.load(f)
    logger.info(f"Loaded categorical features: {len(categorical_features)} columns")
    
    # Separate features and labels
    LABEL_COL = "y"
    AUTHOR_COL = "author"
    feature_cols = [c for c in train_df.columns if c not in [LABEL_COL, AUTHOR_COL]]
    
    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[LABEL_COL]
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Train bot/human: {(y_train==1).sum()} / {(y_train==0).sum()}")
    logger.info(f"Val bot/human: {(y_val==1).sum()} / {(y_val==0).sum()}")
    
    # Step 2: Apply resampling strategy to training fold only
    logger.info("\n" + "="*60)
    logger.info(f"Step 2: Apply resampling strategy: {resample_strategy}")
    logger.info("="*60)
    
    n_bots_before = (y_train == 1).sum()
    n_humans_before = (y_train == 0).sum()
    
    # Apply median imputation before SMOTENC (it doesn't handle NaN values)
    if resample_strategy == "smotenc":
        logger.info("Applying median imputation before SMOTENC (required for NaN handling)")
        imputer_smote = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer_smote.fit_transform(X_train), 
                              columns=feature_cols, index=X_train.index)
        # Save the SMOTENC imputer for reproducibility
        joblib.dump(imputer_smote, f"{artifacts_dir}/imputer_smotenc.pkl")
    
    if resample_strategy == "none":
        logger.info("Using training fold as-is (natural ratio)")
        X_train_resampled = X_train
        y_train_resampled = y_train
        n_bots_after = n_bots_before
        n_humans_after = n_humans_before
        
    elif resample_strategy == "downsample":
        logger.info("Downsampling humans to match bot count (1:1)")
        rng = np.random.RandomState(SEED)
        
        bot_mask = y_train == 1
        human_mask = y_train == 0
        
        X_bots = X_train[bot_mask]
        y_bots = y_train[bot_mask]
        X_humans = X_train[human_mask]
        y_humans = y_train[human_mask]
        
        n_bots = len(X_bots)
        X_humans_sampled = X_humans.sample(n=n_bots, random_state=SEED)
        y_humans_sampled = y_humans.loc[X_humans_sampled.index]
        
        X_train_resampled = pd.concat([X_bots, X_humans_sampled])
        y_train_resampled = pd.concat([y_bots, y_humans_sampled])
        
        n_bots_after = n_bots
        n_humans_after = n_bots
        
    elif resample_strategy == "smotenc":
        logger.info("Applying SMOTENC to upsample bots to 1:1")
        from imblearn.over_sampling import SMOTENC
        
        # Build categorical features index list
        cat_idx = []
        for cat_col in categorical_features:
            if cat_col not in X_train.columns:
                raise ValueError(f"Categorical column '{cat_col}' from categorical_features.json "
                               f"not found in training data columns")
            cat_idx.append(X_train.columns.get_loc(cat_col))
        
        logger.info(f"Categorical feature indices: {cat_idx}")
        
        smote = SMOTENC(categorical_features=cat_idx, random_state=SEED)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Verify categorical columns still only contain original values
        for cat_col in categorical_features:
            original_vals = set(X_train[cat_col].unique())  # No NaN after imputation
            resampled_vals = set(X_train_resampled[cat_col].unique())
            if not resampled_vals.issubset(original_vals):
                raise ValueError(f"SMOTENC introduced new values in categorical column '{cat_col}'")
        
        n_bots_after = (y_train_resampled == 1).sum()
        n_humans_after = (y_train_resampled == 0).sum()
    
    logger.info(f"Training fold before resampling: {n_bots_before} bots, {n_humans_before} humans "
          f"(1:{n_humans_before/n_bots_before:.1f})")
    logger.info(f"Training fold after resampling: {n_bots_after} bots, {n_humans_after} humans "
          f"(1:{n_humans_after/n_bots_after:.1f})")
    
    # Save resample manifest
    resample_manifest = {
        "strategy": resample_strategy,
        "before": {"bots": int(n_bots_before), "humans": int(n_humans_before)},
        "after": {"bots": int(n_bots_after), "humans": int(n_humans_after)},
        "seed": SEED,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(results_dir, "resample_manifest.json"), "w") as f:
        json.dump(resample_manifest, f, indent=2)
    
    # Step 3: Preprocessing variants
    logger.info("\n" + "="*60)
    logger.info("Step 3: Create preprocessing variants")
    logger.info("="*60)
    
    PREPROCESSING_DATA = create_preprocessing_variants(
        X_train_resampled, X_val, feature_cols, artifacts_dir
    )
    
    # Step 4: Model registry
    logger.info("\n" + "="*60)
    logger.info("Step 4: Build model registry")
    logger.info("="*60)
    
    MODELS = get_model_registry(skip_tabpfn=args.skip_tabpfn)
    
    # Filter to requested models
    if args.models:
        MODELS = {k: v for k, v in MODELS.items() if k in args.models}
    
    logger.info(f"Models to train: {list(MODELS.keys())}")
    
    # Check TabPFN row ceiling
    if "tabpfn" in MODELS:
        n_train_rows = len(X_train_resampled)
        # TabPFN historically has ~10k row ceiling
        tabpfn_ceiling = 10000  # Adjust if different for installed version
        if n_train_rows > tabpfn_ceiling:
            raise ValueError(
                f"TabPFN row ceiling exceeded: strategy '{resample_strategy}' has "
                f"{n_train_rows} training rows, but TabPFN supports max {tabpfn_ceiling}. "
                f"Use --skip-tabpfn or a different resampling strategy."
            )
        logger.info(f"TabPFN row count check passed: {n_train_rows} rows <= {tabpfn_ceiling}")
    
    # Step 5: Train + tune threshold + evaluate
    logger.info("\n" + "="*60)
    logger.info("Step 5: Train models and tune thresholds")
    logger.info("="*60)
    
    threshold_range = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, 
                                args.threshold_step)
    logger.info(f"Threshold sweep range: {args.threshold_min:.2f} to {args.threshold_max:.2f} "
          f"step {args.threshold_step:.2f}")
    
    all_results = train_models(
        MODELS, PREPROCESSING_DATA, y_train_resampled, y_val, artifacts_dir,
        tune_threshold=True, min_precision=args.min_precision, 
        threshold_range=threshold_range
    )
    
    # Step 6: Results summary
    logger.info("\n" + "="*60)
    logger.info("Step 6: Save results summary")
    logger.info("="*60)
    
    save_results_summary(all_results, results_dir, resample_strategy)
    
    # Step 7: SHAP
    if not args.skip_shap:
        logger.info("\n" + "="*60)
        logger.info("Step 7: Compute SHAP importance")
        logger.info("="*60)
        compute_shap_importance(MODELS, PREPROCESSING_DATA, feature_cols, artifacts_dir, 
                              results_dir, args.shap_background_size)
    
    # Step 8: Permutation importance
    if not args.skip_permutation:
        logger.info("\n" + "="*60)
        logger.info("Step 8: Compute permutation importance")
        logger.info("="*60)
        compute_permutation_importance(MODELS, PREPROCESSING_DATA, feature_cols, y_val, 
                                       artifacts_dir, results_dir)
    
    # Step 9: Correlation analysis
    logger.info("\n" + "="*60)
    logger.info("Step 9: Compute correlation analysis")
    logger.info("="*60)
    # Use preprocessing variant B (median-imputed) for correlation
    X_train_B, _ = PREPROCESSING_DATA["B"]
    compute_correlation_analysis(X_train_B, results_dir)
    
    logger.info("\n" + "="*60)
    logger.info("train_model.py completed successfully")
    logger.info("="*60)
    logger.info(f"\nArtifacts saved to {artifacts_dir}")
    logger.info(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
