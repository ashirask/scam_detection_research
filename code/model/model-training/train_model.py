#!/usr/bin/env python3
"""
train_model.py

Trains multiple classifiers on the Tier 1 feature dataset, evaluates each on a held-out
test set, produces SHAP feature importance analysis, and saves all results for comparison.
Takes dataset.parquet as input and outputs model artifacts, metrics, and plots.

Usage:
  python train_model.py \
    --dataset      dataset.parquet \
    --output-dir   results/ \
    --artifact-dir artifacts/ \
    [--skip-shap]        # skip SHAP analysis (faster run)
    [--skip-permutation] # skip permutation importance (slower step)
"""

import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, cohen_kappa_score,
    confusion_matrix, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.inspection import permutation_importance

# Model imports
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tabpfn import TabPFNClassifier

# SHAP for feature importance
import shap


def load_and_clean_data(dataset_path):
    """
    Load dataset.parquet and drop invalid rows.
    
    Args:
        dataset_path: Path to the parquet file
    
    Returns:
        Cleaned DataFrame
    """
    # Load the parquet file
    df = pd.read_parquet(dataset_path)
    
    # Drop null or empty author names (invalid rows)
    df = df[df["author"].notna() & (df["author"] != "")]
    
    # Confirm class distribution
    print(f"Bot authors   : {(df['y']==1).sum()}")
    print(f"Human authors : {(df['y']==0).sum()}")
    print(f"Total         : {len(df)}")
    
    return df


def prepare_features_and_labels(df):
    """
    Separate features (X) and labels (y) from the DataFrame.
    
    Args:
        df: Cleaned DataFrame
    
    Returns:
        X: Feature DataFrame
        y: Label Series
        feature_cols: List of feature column names
    """
    # Columns to exclude from features
    EXCLUDE_COLS = [
        "author",             # identifier, not a feature
        "reply_time_coverage" # diagnostic only — measures data availability, not bot behavior
    ]
    LABEL_COL = "y"
    
    # Get feature column names (all except excluded and label)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS + [LABEL_COL]]
    
    # Extract features and labels
    X = df[feature_cols]
    y = df[LABEL_COL]
    
    # Print NaN rates before training
    nan_rates = X.isnull().mean().sort_values(ascending=False)
    print("\nNaN rates before training:")
    print(nan_rates[nan_rates > 0])
    
    return X, y, feature_cols


def create_data_splits(X, y, df, artifact_dir):
    """
    Create stratified train/validation/test splits (70%/15%/15%).
    Save test authors to prevent leakage in pseudolabeling.
    
    Args:
        X: Feature DataFrame
        y: Label Series
        df: Original DataFrame (to extract author names)
        artifact_dir: Directory to save test_authors.csv
    
    Returns:
        X_train, X_val, X_test: Feature splits
        y_train, y_val, y_test: Label splits
    """
    # First split: 85% train+val, 15% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    
    # Second split: 85% of 85% = 72.25% train, 15% of 85% = 12.75% val
    # This gives approximately 70% train, 15% val, 15% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.176, stratify=y_train, random_state=42
    )
    
    print(f"\nData splits:")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Save test author list — pseudolabeling rounds must never touch these authors
    test_authors = df.loc[X_test.index, "author"]
    os.makedirs(artifact_dir, exist_ok=True)
    test_authors.to_csv(f"{artifact_dir}/test_authors.csv", index=False)
    print(f"Saved test authors to {artifact_dir}/test_authors.csv")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessing_variants(X_train, X_val, X_test, feature_cols, artifact_dir):
    """
    Create three preprocessing variants for different model types.
    All preprocessors are fit on training data only.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        feature_cols: List of feature column names
        artifact_dir: Directory to save preprocessor artifacts
    
    Returns:
        Dictionary mapping variant names to (train, val, test) tuples
    """
    # Variant A — no preprocessing (NaN-native tree models: LightGBM, XGBoost)
    X_train_A = X_train
    X_val_A = X_val
    X_test_A = X_test
    
    # Variant B — median imputation only (RandomForest, TabPFN)
    imputer_B = SimpleImputer(strategy="median")
    X_train_B = pd.DataFrame(imputer_B.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_B = pd.DataFrame(imputer_B.transform(X_val), columns=feature_cols, index=X_val.index)
    X_test_B = pd.DataFrame(imputer_B.transform(X_test), columns=feature_cols, index=X_test.index)
    
    # Variant C — median imputation + z-score normalization (MLP)
    imputer_C = SimpleImputer(strategy="median")
    scaler_C = StandardScaler()
    X_train_C = pd.DataFrame(
        scaler_C.fit_transform(imputer_C.fit_transform(X_train)), 
        columns=feature_cols, 
        index=X_train.index
    )
    X_val_C = pd.DataFrame(
        scaler_C.transform(imputer_C.transform(X_val)), 
        columns=feature_cols, 
        index=X_val.index
    )
    X_test_C = pd.DataFrame(
        scaler_C.transform(imputer_C.transform(X_test)), 
        columns=feature_cols, 
        index=X_test.index
    )
    
    # Save preprocessors for reuse in pseudolabeling and inference
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(imputer_B, f"{artifact_dir}/imputer_median.pkl")
    joblib.dump(imputer_C, f"{artifact_dir}/imputer_mlp.pkl")
    joblib.dump(scaler_C, f"{artifact_dir}/scaler_mlp.pkl")
    print(f"Saved preprocessors to {artifact_dir}/")
    
    return {
        "A": (X_train_A, X_val_A, X_test_A),
        "B": (X_train_B, X_val_B, X_test_B),
        "C": (X_train_C, X_val_C, X_test_C),
    }


def get_model_registry(skip_tabpfn=False):
    """
    Define the model registry with configurations for each model.
    Each model specifies the model class, preprocessing variant, and fit kwargs.
    
    Args:
        skip_tabpfn: If True, exclude TabPFN from model registry (requires license)
    
    Returns:
        Dictionary of model configurations
    """
    MODELS = {
        "lightgbm": {
            # LightGBM gradient boosting classifier
            "model": LGBMClassifier(
                n_estimators=500,      # Number of boosting rounds
                learning_rate=0.05,     # Step size shrinkage
                num_leaves=31,          # Maximum number of leaves in one tree
                random_state=42,        # Random seed for reproducibility
                verbose=-1,             # Suppress LightGBM output
            ),
            "preprocessing": "A",      # No preprocessing needed (handles NaN natively)
            "fit_kwargs": {
                "eval_set": [(None, None)],  # Placeholder, will be set in training loop
                "callbacks": [
                    lgb.early_stopping(50, verbose=False),  # Stop if no improvement for 50 rounds
                    lgb.log_evaluation(100)  # Log every 100 rounds
                ],
            }
        },
        "xgboost": {
            # XGBoost gradient boosting classifier
            "model": XGBClassifier(
                n_estimators=500,      # Number of boosting rounds
                learning_rate=0.05,     # Step size shrinkage
                max_depth=6,            # Maximum tree depth
                random_state=42,        # Random seed for reproducibility
                eval_metric="logloss",  # Evaluation metric
                early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
                verbosity=0,            # Suppress XGBoost output
            ),
            "preprocessing": "A",      # No preprocessing needed (handles NaN natively)
            "fit_kwargs": {
                "eval_set": [(None, None)],  # Placeholder, will be set in training loop
                "verbose": False,
            }
        },
        "randomforest": {
            # Random forest classifier
            "model": RandomForestClassifier(
                n_estimators=500,      # Number of trees
                random_state=42,        # Random seed for reproducibility
                n_jobs=-1,              # Use all available cores
            ),
            "preprocessing": "B",      # Median imputation needed
            "fit_kwargs": {}          # No special fit kwargs
        },
        "tabpfn": {
            # TabPFN transformer-based classifier (optional - requires license)
            "model": TabPFNClassifier(
                device="cuda",                      # Use CPU (change to 'cuda' if GPU available)
            ),
            "preprocessing": "B",      # Median imputation needed
            "fit_kwargs": {}          # No special fit kwargs
            # Note: TabPFN row limit ~10k, feature limit ~100.
            # ~7k train rows and 42 features fits within limits.
            # Requires TABPFN_TOKEN environment variable or license acceptance.
        } if not skip_tabpfn else None,
        "mlp": {
            # Multi-layer perceptron classifier
            "model": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),  # Three hidden layers with decreasing neurons
                activation="relu",                 # ReLU activation function
                max_iter=200,                     # Maximum number of epochs
                early_stopping=True,              # Stop if validation score doesn't improve
                validation_fraction=0.1,          # Use 10% of training for validation
                random_state=42,                  # Random seed for reproducibility
            ),
            "preprocessing": "C",      # Median imputation + scaling needed
            "fit_kwargs": {}          # No special fit kwargs
        },
    }
    
    # Remove TabPFN entry if skip_tabpfn is True
    if skip_tabpfn and "tabpfn" in MODELS:
        del MODELS["tabpfn"]
    
    return MODELS


def evaluate(model, X, y_true, name="model", threshold=0.5):
    """
    Evaluate a model on a dataset and compute comprehensive metrics.
    
    Args:
        model: Trained model with predict_proba method
        X: Feature DataFrame
        y_true: True labels
        name: Name for printing
        threshold: Decision threshold for binary classification
    
    Returns:
        Dictionary of metrics
    """
    # Get probability predictions for the positive class
    proba = model.predict_proba(X)[:, 1]
    
    # Convert probabilities to binary predictions using threshold
    pred = (proba >= threshold).astype(int)
    
    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    
    # Compute all metrics
    metrics = {
        "roc_auc":     roc_auc_score(y_true, proba),              # Area under ROC curve
        "pr_auc":      average_precision_score(y_true, proba),    # Area under PR curve
        "f1":          f1_score(y_true, pred),                    # F1 score
        "precision":   precision_score(y_true, pred),             # Precision
        "recall":      recall_score(y_true, pred),                 # Recall
        "cohen_kappa": cohen_kappa_score(y_true, pred),           # Cohen's kappa
        "fpr":         fp / (fp + tn),                            # False positive rate (humans wrongly flagged)
        "fnr":         fn / (fn + tp),                            # False negative rate (bots missed)
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),  # Confusion matrix counts
    }
    
    # Print evaluation results
    print(f"\n--- {name} ---")
    print(classification_report(y_true, pred, target_names=["human", "bot"]))
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC  : {metrics['pr_auc']:.4f}")
    print(f"Kappa   : {metrics['cohen_kappa']:.4f}")
    print(f"FPR     : {metrics['fpr']:.4f}  (humans wrongly flagged)")
    print(f"FNR     : {metrics['fnr']:.4f}  (bots missed)")
    
    return metrics


def train_models(MODELS, PREPROCESSING_DATA, y_train, y_val, artifact_dir):
    """
    Train all models in the registry and evaluate on validation set.
    
    Args:
        MODELS: Dictionary of model configurations
        PREPROCESSING_DATA: Dictionary of preprocessing variants
        y_train: Training labels
        y_val: Validation labels
        artifact_dir: Directory to save trained models
    
    Returns:
        Dictionary of results for each model
    """
    os.makedirs(artifact_dir, exist_ok=True)
    all_results = {}
    
    for name, config in MODELS.items():
        print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
        
        # Get the appropriate preprocessing variant for this model
        Xtr, Xv, _ = PREPROCESSING_DATA[config["preprocessing"]]
        model = config["model"]
        
        # Update eval_set in fit_kwargs for tree models with early stopping
        fit_kwargs = config.get("fit_kwargs", {}).copy()
        if "eval_set" in fit_kwargs:
            fit_kwargs["eval_set"] = [(Xv, y_val)]
        
        # Train the model and measure time
        t0 = time.time()
        model.fit(Xtr, y_train, **fit_kwargs)
        train_time = time.time() - t0
        
        # Evaluate on validation set
        val_metrics = evaluate(model, Xv, y_val, name=f"{name}_val")
        
        # Save the trained model
        joblib.dump(model, f"{artifact_dir}/{name}_model.pkl")
        
        # Store results
        all_results[name] = {
            "val_metrics":  val_metrics,
            "train_time_s": round(train_time, 2),
        }
        print(f"Training time: {train_time:.1f}s")
    
    return all_results


def evaluate_test_set(MODELS, PREPROCESSING_DATA, all_results, y_test, artifact_dir):
    """
    Evaluate all models on the held-out test set (run once at the end).
    
    Args:
        MODELS: Dictionary of model configurations
        PREPROCESSING_DATA: Dictionary of preprocessing variants
        all_results: Dictionary of existing results (will be updated)
        y_test: Test labels
        artifact_dir: Directory where trained models are saved
    
    Returns:
        Updated all_results dictionary with test metrics
    """
    print(f"\n{'='*50}\nEvaluating on Test Set\n{'='*50}")
    
    for name, config in MODELS.items():
        # Get the appropriate preprocessing variant for this model
        _, _, Xte = PREPROCESSING_DATA[config["preprocessing"]]
        
        # Load the trained model
        model = joblib.load(f"{artifact_dir}/{name}_model.pkl")
        
        # Evaluate on test set
        test_metrics = evaluate(model, Xte, y_test, name=f"{name}_TEST")
        
        # Store test metrics
        all_results[name]["test_metrics"] = test_metrics
    
    return all_results


def save_results_summary(all_results, output_dir):
    """
    Save all model metrics to a CSV file and print comparison table.
    
    Args:
        all_results: Dictionary of results for each model
        output_dir: Directory to save results
    """
    rows = []
    for name, res in all_results.items():
        for split, metrics in [("val", res.get("val_metrics")),
                                ("test", res.get("test_metrics"))]:
            if metrics:
                # Create row with model name, split, and training time
                row = {"model": name, "split": split,
                       "train_time_s": res.get("train_time_s", "")}
                # Add all metrics except confusion matrix counts
                row.update({k: v for k, v in metrics.items()
                            if k not in ("tp","fp","tn","fn")})
                rows.append(row)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"{output_dir}/all_model_metrics.csv", index=False)
    
    # Print comparison table
    print("\n=== Model Comparison ===")
    print(results_df[["model","split","roc_auc","pr_auc","f1",
                       "cohen_kappa","fpr","fnr"]].to_string(index=False))


def compute_shap_importance(MODELS, PREPROCESSING_DATA, feature_cols, artifact_dir, output_dir):
    """
    Compute SHAP feature importance for LightGBM (fastest SHAP integration).
    Generate bar plot, beeswarm plot, and feature ranking.
    
    Args:
        MODELS: Dictionary of model configurations
        PREPROCESSING_DATA: Dictionary of preprocessing variants
        feature_cols: List of feature column names
        artifact_dir: Directory where trained models are saved
        output_dir: Directory to save results
    """
    print(f"\n{'='*50}\nComputing SHAP Feature Importance\n{'='*50}")
    
    # Load LightGBM model and get preprocessed test data (variant A)
    lgbm_model = joblib.load(f"{artifact_dir}/lightgbm_model.pkl")
    Xtr, Xv, Xte = PREPROCESSING_DATA["A"]
    
    # Create SHAP explainer for tree models
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(Xte)
    
    # For binary classification, shap_values may be a list [class0, class1] or single array
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # Bar plot — mean |SHAP| per feature (global importance)
    shap.summary_plot(sv, Xte, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_bar.png", dpi=150)
    plt.close()
    print(f"Saved SHAP bar plot to {output_dir}/shap_bar.png")
    
    # Beeswarm plot — direction and spread of each feature's effect
    shap.summary_plot(sv, Xte, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=150)
    plt.close()
    print(f"Saved SHAP beeswarm plot to {output_dir}/shap_beeswarm.png")
    
    # Compute and save feature ranking by mean absolute SHAP value
    mean_abs_shap = pd.Series(
        np.abs(sv).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)
    mean_abs_shap.to_csv(f"{output_dir}/shap_feature_ranking.csv")
    
    print("\nTop 15 features by SHAP:")
    print(mean_abs_shap.head(15).to_string())
    
    # Flag drop candidates — features with < 1% of top feature SHAP value
    top_shap = mean_abs_shap.iloc[0]
    drop_candidates = mean_abs_shap[mean_abs_shap < 0.01 * top_shap].index.tolist()
    print(f"\nDrop candidates for Tier 2 ({len(drop_candidates)} features):")
    print(drop_candidates)
    
    # Save raw SHAP values for further analysis
    pd.DataFrame(sv, columns=feature_cols).to_parquet(f"{output_dir}/shap_values.parquet")
    print(f"Saved raw SHAP values to {output_dir}/shap_values.parquet")


def compute_permutation_importance(MODELS, PREPROCESSING_DATA, feature_cols, y_val, artifact_dir, output_dir):
    """
    Compute permutation importance for LightGBM and RandomForest.
    Cross-validates SHAP findings to identify truly unimportant features.
    
    Args:
        MODELS: Dictionary of model configurations
        PREPROCESSING_DATA: Dictionary of preprocessing variants
        feature_cols: List of feature column names
        y_val: Validation labels
        artifact_dir: Directory where trained models are saved
        output_dir: Directory to save results
    """
    print(f"\n{'='*50}\nComputing Permutation Importance\n{'='*50}")
    
    for name in ["lightgbm", "randomforest"]:
        # Load the trained model
        model = joblib.load(f"{artifact_dir}/{name}_model.pkl")
        
        # Get the appropriate preprocessing variant
        Xtr, Xv, _ = PREPROCESSING_DATA[MODELS[name]["preprocessing"]]
        
        # Compute permutation importance
        result = permutation_importance(
            model, Xv, y_val,
            n_repeats=10,           # Number of permutations per feature
            random_state=42, 
            scoring="roc_auc"       # Metric to evaluate
        )
        
        # Create DataFrame with results
        perm_df = pd.DataFrame({
            "feature":          feature_cols,
            "importance_mean":  result.importances_mean,
            "importance_std":   result.importances_std,
        }).sort_values("importance_mean", ascending=False)
        
        # Save to CSV
        perm_df.to_csv(f"{output_dir}/permutation_importance_{name}.csv", index=False)
        
        print(f"\nBottom 10 features — {name}:")
        print(perm_df.tail(10).to_string(index=False))


def compute_correlation_analysis(X_train, feature_cols, output_dir):
    """
    Compute feature correlation matrix and identify highly correlated pairs.
    Generate and save correlation heatmap.
    
    Args:
        X_train: Training features (use variant B for correlation analysis)
        feature_cols: List of feature column names
        output_dir: Directory to save results
    """
    print(f"\n{'='*50}\nComputing Correlation Analysis\n{'='*50}")
    
    # Compute correlation matrix
    corr = X_train.corr()
    
    # Find highly correlated pairs (|r| > 0.95)
    high_corr_pairs = [
        (corr.columns[i], corr.columns[j], corr.iloc[i, j])
        for i in range(len(corr.columns))
        for j in range(i+1, len(corr.columns))
        if abs(corr.iloc[i, j]) > 0.95
    ]
    
    print(f"\nHighly correlated pairs (>0.95): {len(high_corr_pairs)}")
    for a, b, r in high_corr_pairs:
        print(f"  {a} <-> {b}: {r:.3f}")
    
    # Generate and save correlation heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved correlation heatmap to {output_dir}/correlation_heatmap.png")


def main():
    """
    Main training pipeline:
    1. Load and clean data
    2. Prepare features and labels
    3. Create data splits
    4. Create preprocessing variants
    5. Train all models
    6. Evaluate on test set
    7. Save results summary
    8. Compute SHAP importance (optional)
    9. Compute permutation importance (optional)
    10. Compute correlation analysis
    """
    parser = argparse.ArgumentParser(
        description="Train bot detection models on Tier 1 features"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset.parquet")
    parser.add_argument("--output-dir", default="results/", help="Directory for results")
    parser.add_argument("--artifact-dir", default="artifacts/", help="Directory for model artifacts")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--skip-permutation", action="store_true", help="Skip permutation importance")
    parser.add_argument("--skip-tabpfn", action="store_true", help="Skip TabPFN (requires license/API key)")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)
    
    print("="*60)
    print("Bot Detection Model Training Pipeline")
    print("="*60)
    
    # Step 1: Load and clean data
    print("\n[Step 1] Loading and cleaning data...")
    df = load_and_clean_data(args.dataset)
    
    # Step 2: Prepare features and labels
    print("\n[Step 2] Preparing features and labels...")
    X, y, feature_cols = prepare_features_and_labels(df)
    
    # Step 3: Create data splits
    print("\n[Step 3] Creating data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y, df, args.artifact_dir)
    
    # Step 4: Create preprocessing variants
    print("\n[Step 4] Creating preprocessing variants...")
    PREPROCESSING_DATA = create_preprocessing_variants(X_train, X_val, X_test, feature_cols, args.artifact_dir)
    
    # Step 5: Get model registry
    print("\n[Step 5] Loading model registry...")
    MODELS = get_model_registry(skip_tabpfn=args.skip_tabpfn)
    
    # Step 6: Train all models
    print("\n[Step 6] Training models...")
    all_results = train_models(MODELS, PREPROCESSING_DATA, y_train, y_val, args.artifact_dir)
    
    # Step 7: Evaluate on test set
    print("\n[Step 7] Evaluating on test set...")
    all_results = evaluate_test_set(MODELS, PREPROCESSING_DATA, all_results, y_test, args.artifact_dir)
    
    # Step 8: Save results summary
    print("\n[Step 8] Saving results summary...")
    save_results_summary(all_results, args.output_dir)
    
    # Step 9: Compute SHAP importance (optional)
    if not args.skip_shap:
        print("\n[Step 9] Computing SHAP feature importance...")
        compute_shap_importance(MODELS, PREPROCESSING_DATA, feature_cols, args.artifact_dir, args.output_dir)
    else:
        print("\n[Step 9] Skipping SHAP analysis (--skip-shap flag set)")
    
    # Step 10: Compute permutation importance (optional)
    if not args.skip_permutation:
        print("\n[Step 10] Computing permutation importance...")
        compute_permutation_importance(MODELS, PREPROCESSING_DATA, feature_cols, y_val, args.artifact_dir, args.output_dir)
    else:
        print("\n[Step 10] Skipping permutation importance (--skip-permutation flag set)")
    
    # Step 11: Compute correlation analysis
    print("\n[Step 11] Computing correlation analysis...")
    _, X_train_B, _ = PREPROCESSING_DATA["B"]
    compute_correlation_analysis(X_train_B, feature_cols, args.output_dir)
    
    print("\n" + "="*60)
    print("Training pipeline complete!")
    print(f"Results saved to {args.output_dir}")
    print(f"Artifacts saved to {args.artifact_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
