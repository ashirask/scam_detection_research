# `train_model.py` — Specification (Tier 1)

## Purpose

Trains multiple classifiers on the Tier 1 feature dataset, evaluates each on a held-out
test set, produces SHAP feature importance analysis, and saves all results for comparison.
Takes `dataset.parquet` as input and outputs model artifacts, metrics, and plots.

AutoGluon is handled as a separate script (`train_autogluon.py`) due to its non-sklearn
API — see end of this spec.

---

## Input

### `dataset.parquet`
Produced by `build_features.py`. One row per author, 44 columns.

**Pre-training data guard — drop invalid rows first:**
```python
df = pd.read_parquet("dataset.parquet")

# Drop null or empty author names
df = df[df["author"].notna() & (df["author"] != "")]

# Confirm class distribution
print(f"Bot authors   : {(df['y']==1).sum()}")
print(f"Human authors : {(df['y']==0).sum()}")
print(f"Total         : {len(df)}")
```

---

## Columns

### Feature columns (X)
All columns except the following which are explicitly excluded:
```python
EXCLUDE_COLS = [
    "author",             # identifier, not a feature
    "reply_time_coverage" # diagnostic only — measures data availability, not bot behavior
]
LABEL_COL = "y"

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS + [LABEL_COL]]
X = df[feature_cols]
y = df[LABEL_COL]
```

### NaN summary before training
Print NaN rate per feature so you know what the models are handling:
```python
nan_rates = X.isnull().mean().sort_values(ascending=False)
print(nan_rates[nan_rates > 0])
```

---

## Data Splits

Stratified splits preserve the bot:human ratio in each partition.
70% train / 15% val / 15% test.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.176, stratify=y_train, random_state=42
    # 0.176 of 85% ≈ 15% of total
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Save test author list — pseudolabeling rounds must never touch these authors
test_authors = df.loc[X_test.index, "author"]
test_authors.to_csv("artifacts/test_authors.csv", index=False)
```

**The test set is used ONCE at the very end** — not during model selection or
hyperparameter tuning. Use the validation set for all intermediate evaluation.

---

## Preprocessing Variants

Three variants, all fit on training data only. Never fit on validation or test data.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Variant A — no preprocessing (NaN-native tree models: LightGBM, XGBoost)
X_train_A = X_train
X_val_A   = X_val
X_test_A  = X_test

# Variant B — median imputation only (RandomForest, TabPFN)
imputer_B = SimpleImputer(strategy="median")
X_train_B = pd.DataFrame(imputer_B.fit_transform(X_train), columns=feature_cols)
X_val_B   = pd.DataFrame(imputer_B.transform(X_val),       columns=feature_cols)
X_test_B  = pd.DataFrame(imputer_B.transform(X_test),      columns=feature_cols)

# Variant C — median imputation + z-score normalization (MLP)
imputer_C = SimpleImputer(strategy="median")
scaler_C  = StandardScaler()
X_train_C = pd.DataFrame(
    scaler_C.fit_transform(imputer_C.fit_transform(X_train)), columns=feature_cols
)
X_val_C   = pd.DataFrame(
    scaler_C.transform(imputer_C.transform(X_val)), columns=feature_cols
)
X_test_C  = pd.DataFrame(
    scaler_C.transform(imputer_C.transform(X_test)), columns=feature_cols
)

# Save preprocessors for reuse in pseudolabeling and inference
os.makedirs("artifacts", exist_ok=True)
joblib.dump(imputer_B, "artifacts/imputer_median.pkl")
joblib.dump(imputer_C, "artifacts/imputer_mlp.pkl")
joblib.dump(scaler_C,  "artifacts/scaler_mlp.pkl")
```

---

## Model Registry

```python
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tabpfn import TabPFNClassifier

MODELS = {
    "lightgbm": {
        "model": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        ),
        "preprocessing": "A",
        "fit_kwargs": {
            "eval_set": [(X_val_A, y_val)],
            "callbacks": [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100)
            ],
        }
    },
    "xgboost": {
        "model": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=50,
            verbosity=0,
        ),
        "preprocessing": "A",
        "fit_kwargs": {
            "eval_set": [(X_val_A, y_val)],
            "verbose": False,
        }
    },
    "randomforest": {
        "model": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
        ),
        "preprocessing": "B",
        "fit_kwargs": {}
    },
    "tabpfn": {
        "model": TabPFNClassifier(
            device="cpu",
            N_ensemble_configurations=32,
        ),
        "preprocessing": "B",
        "fit_kwargs": {}
        # Note: TabPFN row limit ~10k, feature limit ~100.
        # ~7k train rows and 42 features fits within limits.
    },
    "mlp": {
        "model": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
        "preprocessing": "C",
        "fit_kwargs": {}
    },
}

PREPROCESSING_DATA = {
    "A": (X_train_A, X_val_A, X_test_A),
    "B": (X_train_B, X_val_B, X_test_B),
    "C": (X_train_C, X_val_C, X_test_C),
}
```

---

## Evaluation Function

```python
from sklearn.metrics import (
    classification_report, roc_auc_score, cohen_kappa_score,
    confusion_matrix, average_precision_score,
    f1_score, precision_score, recall_score
)

def evaluate(model, X, y_true, name="model", threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    metrics = {
        "roc_auc":     roc_auc_score(y_true, proba),
        "pr_auc":      average_precision_score(y_true, proba),
        "f1":          f1_score(y_true, pred),
        "precision":   precision_score(y_true, pred),
        "recall":      recall_score(y_true, pred),
        "cohen_kappa": cohen_kappa_score(y_true, pred),
        "fpr":         fp / (fp + tn),   # humans wrongly flagged as bots
        "fnr":         fn / (fn + tp),   # bots missed
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    print(f"\n--- {name} ---")
    print(classification_report(y_true, pred, target_names=["human", "bot"]))
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC  : {metrics['pr_auc']:.4f}")
    print(f"Kappa   : {metrics['cohen_kappa']:.4f}")
    print(f"FPR     : {metrics['fpr']:.4f}  (humans wrongly flagged)")
    print(f"FNR     : {metrics['fnr']:.4f}  (bots missed)")

    return metrics
```

---

## Training Loop

```python
import os, time
os.makedirs("results", exist_ok=True)

all_results = {}

for name, config in MODELS.items():
    print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")

    Xtr, Xv, Xte = PREPROCESSING_DATA[config["preprocessing"]]
    model = config["model"]

    t0 = time.time()
    model.fit(Xtr, y_train, **config.get("fit_kwargs", {}))
    train_time = time.time() - t0

    val_metrics = evaluate(model, Xv, y_val, name=f"{name}_val")
    joblib.dump(model, f"artifacts/{name}_model.pkl")

    all_results[name] = {
        "val_metrics":  val_metrics,
        "train_time_s": round(train_time, 2),
    }
    print(f"Training time: {train_time:.1f}s")
```

---

## Final Test Set Evaluation

Run ONCE after all model comparison on validation set is complete.
Evaluate ALL models on test — needed for paper reporting.

```python
for name, config in MODELS.items():
    _, _, Xte = PREPROCESSING_DATA[config["preprocessing"]]
    model = joblib.load(f"artifacts/{name}_model.pkl")
    test_metrics = evaluate(model, Xte, y_test, name=f"{name}_TEST")
    all_results[name]["test_metrics"] = test_metrics
```

---

## Results Summary

```python
rows = []
for name, res in all_results.items():
    for split, metrics in [("val",  res.get("val_metrics")),
                            ("test", res.get("test_metrics"))]:
        if metrics:
            row = {"model": name, "split": split,
                   "train_time_s": res.get("train_time_s", "")}
            row.update({k: v for k, v in metrics.items()
                        if k not in ("tp","fp","tn","fn")})
            rows.append(row)

results_df = pd.DataFrame(rows)
results_df.to_csv("results/all_model_metrics.csv", index=False)
print("\n=== Model Comparison ===")
print(results_df[["model","split","roc_auc","pr_auc","f1",
                   "cohen_kappa","fpr","fnr"]].to_string(index=False))
```

---

## SHAP Feature Importance

Run on LightGBM — fastest SHAP integration, most reliable for tree models.

```python
import shap
import matplotlib.pyplot as plt
import numpy as np

lgbm_model  = joblib.load("artifacts/lightgbm_model.pkl")
Xtr, Xv, Xte = PREPROCESSING_DATA["A"]

explainer   = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(Xte)

# For binary classification shap_values may be list [class0, class1] or single array
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# Bar plot — mean |SHAP| per feature
shap.summary_plot(sv, Xte, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("results/shap_bar.png", dpi=150)
plt.close()

# Beeswarm — direction and spread of each feature's effect
shap.summary_plot(sv, Xte, show=False)
plt.tight_layout()
plt.savefig("results/shap_beeswarm.png", dpi=150)
plt.close()

# Save rankings
mean_abs_shap = pd.Series(
    np.abs(sv).mean(axis=0), index=feature_cols
).sort_values(ascending=False)
mean_abs_shap.to_csv("results/shap_feature_ranking.csv")

print("\nTop 15 features by SHAP:")
print(mean_abs_shap.head(15).to_string())

# Flag drop candidates — features with < 1% of top feature SHAP value
top_shap        = mean_abs_shap.iloc[0]
drop_candidates = mean_abs_shap[mean_abs_shap < 0.01 * top_shap].index.tolist()
print(f"\nDrop candidates for Tier 2 ({len(drop_candidates)} features):")
print(drop_candidates)

# Save raw SHAP values
pd.DataFrame(sv, columns=feature_cols).to_parquet("results/shap_values.parquet")
```

---

## Permutation Importance

Cross-validates SHAP findings. Features near-zero across both models are safe to drop.

```python
from sklearn.inspection import permutation_importance

for name in ["lightgbm", "randomforest"]:
    model    = joblib.load(f"artifacts/{name}_model.pkl")
    Xtr, Xv, _ = PREPROCESSING_DATA[MODELS[name]["preprocessing"]]

    result = permutation_importance(
        model, Xv, y_val,
        n_repeats=10, random_state=42, scoring="roc_auc"
    )
    perm_df = pd.DataFrame({
        "feature":          feature_cols,
        "importance_mean":  result.importances_mean,
        "importance_std":   result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    perm_df.to_csv(f"results/permutation_importance_{name}.csv", index=False)
    print(f"\nBottom 10 features — {name}:")
    print(perm_df.tail(10).to_string(index=False))
```

---

## Correlation Analysis

```python
import seaborn as sns

corr = X_train_B.corr()

high_corr_pairs = [
    (corr.columns[i], corr.columns[j], corr.iloc[i, j])
    for i in range(len(corr.columns))
    for j in range(i+1, len(corr.columns))
    if abs(corr.iloc[i, j]) > 0.95
]
print(f"\nHighly correlated pairs (>0.95): {len(high_corr_pairs)}")
for a, b, r in high_corr_pairs:
    print(f"  {a} <-> {b}: {r:.3f}")

plt.figure(figsize=(16, 14))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png", dpi=150)
plt.close()
```

---

## Output Structure

```
artifacts/
  lightgbm_model.pkl
  xgboost_model.pkl
  randomforest_model.pkl
  tabpfn_model.pkl
  mlp_model.pkl
  imputer_median.pkl
  imputer_mlp.pkl
  scaler_mlp.pkl
  test_authors.csv          <- never include these in pseudolabel training

results/
  all_model_metrics.csv     <- all models x all metrics x val/test
  shap_bar.png
  shap_beeswarm.png
  shap_values.parquet
  shap_feature_ranking.csv
  permutation_importance_lightgbm.csv
  permutation_importance_randomforest.csv
  correlation_heatmap.png
```

---

## AutoGluon (separate script: `train_autogluon.py`)

AutoGluon manages its own preprocessing, splitting, and ensembling internally.
It does not follow the sklearn API and cannot be plugged into the model registry loop.

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

df = pd.read_parquet("dataset.parquet")
df = df[~df["author"].str.contains(r"^-+|^=+", regex=True, na=False)]
df = df.drop(columns=["author", "reply_time_coverage"])

train_data = df.sample(frac=0.85, random_state=42)
test_data  = df.drop(train_data.index)

predictor = TabularPredictor(
    label="y",
    eval_metric="roc_auc",
    path="artifacts/autogluon/",
).fit(
    train_data,
    presets="best_quality",    # use "medium_quality" for faster run
    time_limit=3600,           # 1 hour — adjust to cluster availability
    excluded_model_types=["FASTAI"],  # remove if GPU is available
)

test_metrics  = predictor.evaluate(test_data)
leaderboard   = predictor.leaderboard(test_data, silent=True)
leaderboard.to_csv("results/autogluon_leaderboard.csv", index=False)
print(leaderboard[["model","score_test","score_val"]].head(10).to_string(index=False))

fi = predictor.feature_importance(test_data)
fi.to_csv("results/autogluon_feature_importance.csv")
```

```bash
python train_autogluon.py \
  --dataset    dataset.parquet \
  --output-dir results/ \
  --time-limit 3600
```

---

## Pseudolabeling — What It Requires (next step after first model)

Pseudolabeling does NOT require new data extraction. Your existing human-labeled
accounts (y=0) are the unlabeled pool — they contain a mix of true humans and
undetected bots that the model can surface.

Workflow after first model is trained:

1. Apply best trained model to all y=0 accounts in dataset
2. predict_proba > 0.95 → flip to y=1 (pseudolabel bot), tag label_source="pseudolabel"
3. predict_proba < 0.05 → keep as y=0 (confirmed human)
4. Drop accounts in 0.05–0.95 range — too uncertain to pseudolabel
5. Re-train on original labeled set + pseudolabeled accounts
6. Evaluate on same test set — compare to Round 0 metrics
7. Do at most 2 rounds — diminishing returns and noise amplification after that

The test set (saved to artifacts/test_authors.csv) must be excluded from all
pseudolabeling rounds — it stays as pure gold labels throughout.

One short script pseudolabel.py (~60 lines) handles this. No new zst extraction needed.

---

## CLI Interface

```bash
python train_model.py \
  --dataset      dataset.parquet \
  --output-dir   results/ \
  --artifact-dir artifacts/ \
  [--skip-shap]        # skip SHAP analysis (faster run)
  [--skip-permutation] # skip permutation importance (slower step)
```

---

## Dependencies

```
lightgbm
xgboost
scikit-learn
tabpfn            # pip install tabpfn
shap
pandas
pyarrow
matplotlib
seaborn
joblib
numpy
# AutoGluon only:
autogluon         # pip install autogluon  (large install ~1GB)
```

---

## Implementation Notes

- **Threshold 0.5 is a starting point.** After seeing precision-recall curves you may
  want to adjust. A higher threshold (e.g. 0.6) raises precision and lowers FPR —
  fewer humans wrongly flagged. A lower threshold raises recall — fewer bots missed.
  Add a precision-recall curve plot per model to results/.
- **Class balance is near-perfect (5000:4998)** — no class weighting needed. If
  pseudolabeling later shifts this ratio significantly, add class_weight="balanced"
  to sklearn models and scale_pos_weight to XGBoost.
- **TabPFN row limit** — designed for under 10k rows. Your ~7k training rows fit.
  If it errors on row count, reduce N_ensemble_configurations=16.
- **SHAP runtime** — under 1 minute for 42 features on ~1500 test rows. If running
  on larger datasets in later tiers, use shap.sample(X_test, 500) to subsample.
- **Save test_authors.csv immediately after splitting** — before any model training.
  This protects the test set even if the script crashes partway through.
