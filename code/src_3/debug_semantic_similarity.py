import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# This script is intentionally "exact-mode" for debugging.
# It does NOT use LSH or ANN candidate pruning, so every labeled pair score
# comes directly from embedding dot product (cosine, because vectors are normalized).

DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent / "sample_labeled_pairs.csv")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")
DEFAULT_MODEL = "distiluse-base-multilingual-cased-v2"


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    text_a: str
    text_b: str
    label: int
    note: str


def parse_args() -> argparse.Namespace:
    """Define CLI options for the exact semantic-similarity debug harness.

    Why these options exist:
    - input/output locations make runs reproducible and auditable.
    - threshold is the main decision boundary to inspect false positives.
    - sweep options let us test many thresholds on the same labeled set.
    - embedding-model lets us compare model behavior without changing code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Debug semantic similarity with a tiny labeled pair set. "
            "This script removes LSH and uses exact cosine similarity only."
        )
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="CSV with text_a, text_b, label.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for debug outputs.")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer model used to produce exact semantic embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Similarity threshold used for the primary binary decision.",
    )
    parser.add_argument(
        "--sweep-start",
        type=float,
        default=0.40,
        help="Threshold sweep lower bound.",
    )
    parser.add_argument(
        "--sweep-end",
        type=float,
        default=0.95,
        help="Threshold sweep upper bound.",
    )
    parser.add_argument(
        "--sweep-steps",
        type=int,
        default=56,
        help="Number of thresholds to test between sweep start and end.",
    )
    parser.add_argument(
        "--max-preview-length",
        type=int,
        default=200,
        help="Maximum text preview length in the exported review CSV.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize whitespace so trivial formatting does not affect embeddings.

    Example: repeated spaces/newlines collapse into single spaces.
    """
    return " ".join((text or "").split())


def load_labeled_pairs(path: str) -> pd.DataFrame:
    """Load and validate labeled pairs used as controlled test data.

    Required columns:
    - text_a, text_b: the two texts to compare
    - label: 1 if similar, 0 if not similar

    Optional columns:
    - pair_id, note
    """
    if not os.path.isfile(path):
        raise SystemExit(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required = {"text_a", "text_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {sorted(missing)}")

    working = df.copy()
    # Normalize text before embedding so formatting artifacts do not create noise.
    working["text_a"] = working["text_a"].astype(str).map(normalize_text)
    working["text_b"] = working["text_b"].astype(str).map(normalize_text)

    # Auto-generate IDs/notes if the file is minimal.
    if "pair_id" not in working.columns:
        working["pair_id"] = [f"pair_{i}" for i in range(len(working))]
    if "note" not in working.columns:
        working["note"] = ""

    # Parse labels safely so malformed CSV rows surface as actionable messages.
    numeric_labels = pd.to_numeric(working["label"], errors="coerce")
    bad_label_mask = numeric_labels.isna()
    if bad_label_mask.any():
        bad_rows = working.loc[bad_label_mask, ["pair_id", "label"]].head(5)
        examples = "; ".join(
            f"pair_id={row['pair_id']} label={row['label']}" for _, row in bad_rows.iterrows()
        )
        raise SystemExit(
            "Label parsing failed. This often means a CSV quoting issue moved columns. "
            f"Examples: {examples}"
        )
    working["label"] = numeric_labels.astype(int)

    # Drop unusable rows where either side became empty after cleanup.
    working = working[(working["text_a"] != "") & (working["text_b"] != "")].reset_index(drop=True)
    if working.empty:
        raise SystemExit("No usable labeled pairs found after cleaning.")

    # Binary labels are enforced so metrics are interpreted consistently.
    invalid_labels = sorted(set(working["label"].unique()) - {0, 1})
    if invalid_labels:
        raise SystemExit(f"Labels must be 0 or 1. Invalid values found: {invalid_labels}")

    return working


def compute_text_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """Embed texts with SentenceTransformer and return normalized float32 vectors.

    normalize_embeddings=True ensures each vector has unit norm, which means
    dot(vec_a, vec_b) equals cosine similarity.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_unique_texts(df: pd.DataFrame, model_name: str, batch_size: int) -> Dict[str, np.ndarray]:
    """Embed unique text strings once, then reuse vectors across all pairs.

    This avoids re-embedding repeated text appearing in multiple labeled pairs.
    """
    # Combine both columns and deduplicate to minimize compute.
    unique_texts = pd.unique(pd.concat([df["text_a"], df["text_b"]], ignore_index=True)).tolist()
    unique_texts = [str(text) for text in unique_texts]
    vectors = compute_text_embeddings(unique_texts, model_name=model_name, batch_size=batch_size)
    # Map raw text -> vector so pair scoring can be a direct lookup.
    return {text: vectors[idx] for idx, text in enumerate(unique_texts)}


def score_pairs(df: pd.DataFrame, text_vectors: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute exact cosine similarity for every labeled pair.

    Important: no LSH, no ANN index, no bucketing. This is pure exact scoring
    from precomputed normalized embeddings.
    """
    rows: List[Dict] = []
    for _, row in df.iterrows():
        vec_a = text_vectors[row["text_a"]]
        vec_b = text_vectors[row["text_b"]]
        # With normalized vectors, dot product is cosine similarity in [-1, 1].
        similarity = float(np.dot(vec_a, vec_b))
        rows.append(
            {
                "pair_id": row["pair_id"],
                "label": int(row["label"]),
                "similarity": similarity,
                "predicted_flag": 0,
                "text_a": row["text_a"],
                "text_b": row["text_b"],
                "note": row["note"],
            }
        )

    scored = pd.DataFrame(rows)
    return scored


def evaluate_threshold(scored: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """Evaluate binary predictions at one threshold and return core metrics."""
    # Predict similar(1) when score crosses the threshold.
    predicted = (scored["similarity"] >= threshold).astype(int)
    y_true = scored["label"].astype(int)

    # confusion_matrix with fixed label order guarantees TN/FP/FN/TP layout.
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def sweep_thresholds(scored: pd.DataFrame, start: float, end: float, steps: int) -> pd.DataFrame:
    """Run threshold evaluation over a range to locate a better cutoff."""
    thresholds = np.linspace(start, end, steps)
    results = [evaluate_threshold(scored, float(threshold)) for threshold in thresholds]
    return pd.DataFrame(results)


def save_similarity_plot(scored: pd.DataFrame, output_file: str, threshold: float) -> None:
    """Plot similarity distributions for positives vs negatives plus threshold line."""
    plt.figure(figsize=(9, 6))
    # Green curve: known similar pairs (label=1).
    sns.kdeplot(scored.loc[scored["label"] == 1, "similarity"], fill=True, color="#2ca02c", label="label=1")
    # Red curve: known dissimilar pairs (label=0).
    sns.kdeplot(scored.loc[scored["label"] == 0, "similarity"], fill=True, color="#d62728", label="label=0")
    plt.axvline(threshold, color="#1f77b4", linestyle="--", linewidth=2, label=f"threshold={threshold:.3f}")
    plt.title("Exact cosine similarity on labeled pairs")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def save_threshold_report(thresholds: pd.DataFrame, output_file: str) -> None:
    """Persist threshold sweep metrics to CSV for offline analysis."""
    thresholds.to_csv(output_file, index=False)


def save_pair_review(scored: pd.DataFrame, output_file: str, max_preview_length: int, threshold: float) -> None:
    """Export pair-level review table focused on misclassifications.

    The output is sorted so likely errors appear first for fast manual audit.
    """
    working = scored.copy()
    working["predicted_flag"] = (working["similarity"] >= threshold).astype(int)
    # is_error marks both FP and FN cases.
    working["is_error"] = ((working["label"] == 1) & (working["predicted_flag"] == 0)) | (
        (working["label"] == 0) & (working["predicted_flag"] == 1)
    )
    # Short previews keep CSVs readable while preserving original full text columns.
    working["text_a_preview"] = working["text_a"].str.slice(0, max_preview_length)
    working["text_b_preview"] = working["text_b"].str.slice(0, max_preview_length)
    working.sort_values(["is_error", "similarity"], ascending=[False, False]).to_csv(output_file, index=False)


def main() -> None:
    """Run the exact-mode debugging pipeline end to end.

    Execution flow:
    1) load validated labeled pairs
    2) embed unique texts with selected model
    3) score all pairs with exact cosine similarity
    4) evaluate one threshold and sweep many thresholds
    5) save artifacts for audit and calibration
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading labeled pairs from: {args.input_file}")
    df = load_labeled_pairs(args.input_file)
    print(f"Usable labeled pairs: {len(df)}")
    print(f"Positive pairs: {int((df['label'] == 1).sum())}")
    print(f"Negative pairs: {int((df['label'] == 0).sum())}")

    print(f"Embedding unique texts with: {args.embedding_model}")
    text_vectors = embed_unique_texts(df, args.embedding_model, args.batch_size)

    # This is the core debug guarantee: exact pair scoring without LSH/ANN shortcuts.
    print("Scoring pairs with exact cosine similarity; no LSH or approximate retrieval is used.")
    scored = score_pairs(df, text_vectors)

    threshold = float(args.threshold)
    # Primary run uses the user-provided threshold for a direct sanity check.
    scored["predicted_flag"] = (scored["similarity"] >= threshold).astype(int)

    summary = evaluate_threshold(scored, threshold)
    print("Primary threshold summary:")
    print(
        f"  threshold={summary['threshold']:.3f} accuracy={summary['accuracy']:.3f} "
        f"precision={summary['precision']:.3f} recall={summary['recall']:.3f} f1={summary['f1']:.3f}"
    )
    print(
        f"  tn={int(summary['tn'])} fp={int(summary['fp'])} fn={int(summary['fn'])} tp={int(summary['tp'])}"
    )

    thresholds = sweep_thresholds(scored, args.sweep_start, args.sweep_end, args.sweep_steps)
    # score_gap is a convenience column for quick spreadsheet filtering.
    thresholds["score_gap"] = thresholds["f1"] - thresholds["f1"].min()
    best_idx = thresholds["f1"].idxmax()
    best_row = thresholds.loc[best_idx]
    print(
        f"Best sweep threshold: {best_row['threshold']:.3f} "
        f"(precision={best_row['precision']:.3f}, recall={best_row['recall']:.3f}, f1={best_row['f1']:.3f})"
    )

    scored_path = os.path.join(args.output_dir, "debug_pair_scores.csv")
    scored.to_csv(scored_path, index=False)

    threshold_path = os.path.join(args.output_dir, "debug_threshold_sweep.csv")
    save_threshold_report(thresholds, threshold_path)

    review_path = os.path.join(args.output_dir, "debug_pair_review.csv")
    save_pair_review(scored, review_path, args.max_preview_length, threshold)

    plot_path = os.path.join(args.output_dir, "debug_similarity_distribution.png")
    save_similarity_plot(scored, plot_path, threshold)

    summary_path = os.path.join(args.output_dir, "debug_summary.json")
    # Convert numpy scalar values so JSON serialization is always stable.
    best_sweep = {
        key: (float(value) if isinstance(value, (np.floating, np.integer)) else value)
        for key, value in best_row.to_dict().items()
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_file": args.input_file,
                "embedding_model": args.embedding_model,
                "threshold": threshold,
                "summary": summary,
                "best_sweep": best_sweep,
            },
            handle,
            indent=2,
        )

    print(f"Saved pair scores: {scored_path}")
    print(f"Saved threshold sweep: {threshold_path}")
    print(f"Saved review CSV: {review_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
