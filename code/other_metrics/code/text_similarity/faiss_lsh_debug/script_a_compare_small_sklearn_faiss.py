import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent / "sample_labeled_pairs.csv")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output" / "script_a_compare")
DEFAULT_MODEL = "distiluse-base-multilingual-cased-v2"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Script A.

    Script A compares pair-level similarity values from:
    1) sklearn cosine_similarity
    2) global FAISS IndexFlatIP range_search
    """
    parser = argparse.ArgumentParser(
        description=(
            "Script A: small paired comparison using sklearn cosine similarity and "
            "global FAISS range_search on labeled CSV pairs."
        )
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="CSV with text_a, text_b, label")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--threshold", type=float, default=0.20, help="Similarity threshold for hit flags")
    parser.add_argument(
        "--max-preview-length",
        type=int,
        default=200,
        help="Maximum text preview length in output CSV",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize text by collapsing repeated whitespace.

    Built-ins used:
    - str.split(): tokenizes on arbitrary whitespace
    - " ".join(...): rebuilds with single spaces
    """
    return " ".join((text or "").split())


def load_labeled_pairs(path: str) -> pd.DataFrame:
    """Load and validate labeled-pair CSV input.

    Required columns:
    - text_a
    - text_b
    - label (0/1)

    Optional columns (auto-created if missing):
    - pair_id
    - note
    """
    # os.path.isfile prevents less clear pandas file errors and exits early with context.
    if not os.path.isfile(path):
        raise SystemExit("Input file not found: {}".format(path))

    # pd.read_csv parses the paired dataset into a DataFrame for vectorized cleaning.
    df = pd.read_csv(path)
    required = {"text_a", "text_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit("Input CSV missing required columns: {}".format(sorted(missing)))

    out = df.copy()
    # .astype(str) ensures robust normalization even if the CSV has non-string values.
    out["text_a"] = out["text_a"].astype(str).map(normalize_text)
    out["text_b"] = out["text_b"].astype(str).map(normalize_text)

    # Fill standard metadata columns when absent to keep downstream schema stable.
    if "pair_id" not in out.columns:
        out["pair_id"] = ["pair_{}".format(i) for i in range(len(out))]
    if "note" not in out.columns:
        out["note"] = ""

    # pd.to_numeric(..., errors="coerce") converts bad labels to NaN for explicit validation.
    labels = pd.to_numeric(out["label"], errors="coerce")
    bad = labels.isna()
    if bad.any():
        # .loc with .head(5) gives concrete examples to debug CSV quoting/parsing issues.
        bad_rows = out.loc[bad, ["pair_id", "label"]].head(5)
        examples = "; ".join(
            "pair_id={} label={}".format(row["pair_id"], row["label"]) for _, row in bad_rows.iterrows()
        )
        raise SystemExit("Failed to parse labels. Check CSV quoting. Examples: {}".format(examples))

    out["label"] = labels.astype(int)
    invalid_labels = sorted(set(out["label"].unique()) - {0, 1})
    if invalid_labels:
        raise SystemExit("Labels must be 0/1. Invalid: {}".format(invalid_labels))

    # Keep only rows with non-empty texts after normalization.
    out = out[(out["text_a"] != "") & (out["text_b"] != "")].reset_index(drop=True)
    if out.empty:
        raise SystemExit("No usable rows after cleaning")

    return out


def build_text_space(df: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    """Build unique text corpus and text->index lookup.

    Using a deduplicated text space avoids embedding the same sentence repeatedly.
    """
    # pd.concat stacks both columns, pd.unique removes duplicates while preserving order.
    all_texts = pd.unique(pd.concat([df["text_a"], df["text_b"]], ignore_index=True)).tolist()
    texts = [str(t) for t in all_texts]
    # Dictionary lookup gives O(1) mapping from text to embedding row index.
    text_to_idx = {text: idx for idx, text in enumerate(texts)}
    return texts, text_to_idx


def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """Compute normalized sentence embeddings for all unique texts.

    normalize_embeddings=True ensures each vector has unit norm so:
    dot_product == cosine_similarity
    """
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # np.asarray(..., dtype=np.float32) ensures FAISS-compatible memory format.
    return np.asarray(emb, dtype=np.float32)


def compute_pair_scores_sklearn(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    text_to_idx: Dict[str, int],
) -> Dict[str, float]:
    """Score each labeled pair using sklearn cosine_similarity.

    Returns a pair_id -> similarity mapping.
    """
    scores: Dict[str, float] = {}
    # Iterate over labeled rows so output remains aligned with the input annotation set.
    for _, row in df.iterrows():
        ia = text_to_idx[row["text_a"]]
        ib = text_to_idx[row["text_b"]]
        # Slice embeddings[ia:ia+1] to keep 2D shape required by sklearn API.
        sim = float(cosine_similarity(embeddings[ia : ia + 1], embeddings[ib : ib + 1])[0, 0])
        scores[str(row["pair_id"])] = sim
    return scores


def compute_pair_scores_faiss_range(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    text_to_idx: Dict[str, int],
    threshold: float,
) -> Dict[str, Optional[float]]:
    """Score pairs using global FAISS range_search.

    For each query vector, FAISS returns neighbors >= threshold.
    A pair may appear twice (a->b and b->a), so we keep max similarity per unordered pair.
    Pairs not returned by range_search are stored as None.
    """
    dim = embeddings.shape[1]
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

    # IndexFlatIP performs exact inner-product search.
    # With normalized vectors, inner product equals cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Query every unique text once, then map returned neighbors to pair keys.
    lims, scores, neighbors = index.range_search(vectors, float(threshold))

    pair_score_lookup: Dict[Tuple[int, int], float] = {}
    # Outer loop walks each query's neighbor span via lims offsets.
    for q in range(vectors.shape[0]):
        start = int(lims[q])
        end = int(lims[q + 1])
        # Inner loop processes every returned neighbor for this query.
        for pos in range(start, end):
            n = int(neighbors[pos])
            # Skip self-hit from range_search (query matched with itself).
            if n == q:
                continue
            # Canonical unordered key deduplicates (q,n) and (n,q).
            a, b = (q, n) if q < n else (n, q)
            sim = float(scores[pos])
            prev = pair_score_lookup.get((a, b))
            # Keep max in case same pair appears from multiple paths.
            if (prev is None) or (sim > prev):
                pair_score_lookup[(a, b)] = sim

    out: Dict[str, Optional[float]] = {}
    # Map back from text indices to pair_id keys used in labeled CSV.
    for _, row in df.iterrows():
        ia = text_to_idx[row["text_a"]]
        ib = text_to_idx[row["text_b"]]
        key = (ia, ib) if ia < ib else (ib, ia)
        out[str(row["pair_id"])] = pair_score_lookup.get(key)
    return out


def build_comparison_table(
    df: pd.DataFrame,
    sklearn_scores: Dict[str, float],
    faiss_range_scores: Dict[str, Optional[float]],
    threshold: float,
) -> pd.DataFrame:
    """Create per-pair comparison table with both scores and threshold hit flags."""
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        pid = str(row["pair_id"])
        s_exact = float(sklearn_scores[pid])
        s_faiss = faiss_range_scores.get(pid)

        # Convert score-to-flag at the same threshold for apples-to-apples classification.
        found_exact = int(s_exact >= threshold)
        found_faiss = int((s_faiss is not None) and (float(s_faiss) >= threshold))

        rows.append(
            {
                "pair_id": pid,
                "label": int(row["label"]),
                "similarity_sklearn_cosine": s_exact,
                "similarity_faiss_global_range": (np.nan if s_faiss is None else float(s_faiss)),
                # abs_diff is NaN if FAISS didn't return this pair at the chosen threshold.
                "abs_diff": (np.nan if s_faiss is None else abs(s_exact - float(s_faiss))),
                "hit_sklearn": found_exact,
                "hit_faiss_range": found_faiss,
                "hit_mismatch": int(found_exact != found_faiss),
                "text_a": row["text_a"],
                "text_b": row["text_b"],
                "note": row.get("note", ""),
            }
        )

    return pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)


def evaluate_hits(y_true: List[int], y_pred: List[int], threshold: float, method_name: str) -> Dict[str, float]:
    """Compute classification metrics for one method at a fixed threshold."""
    # confusion_matrix(..., labels=[0,1]) guarantees consistent TN/FP/FN/TP ordering.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "method": method_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def add_previews(df: pd.DataFrame, max_preview_length: int) -> pd.DataFrame:
    """Add truncated text preview columns for easier CSV inspection."""
    out = df.copy()
    # Only add preview columns when base columns exist so function is schema-tolerant.
    if "text_a" in out.columns:
        out["text_a_preview"] = out["text_a"].astype(str).str.slice(0, max_preview_length)
    if "text_b" in out.columns:
        out["text_b_preview"] = out["text_b"].astype(str).str.slice(0, max_preview_length)
    return out


def main() -> None:
    """Run end-to-end Script A comparison workflow and save reports."""
    args = parse_args()
    # Ensure output directory exists before writing CSV/JSON artifacts.
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading labeled pairs from: {}".format(args.input_file))
    df = load_labeled_pairs(args.input_file)
    print("Loaded {} labeled pairs".format(len(df)))

    texts, text_to_idx = build_text_space(df)
    print("Unique texts: {}".format(len(texts)))

    print("Embedding with model: {}".format(args.embedding_model))
    embeddings = compute_embeddings(texts, args.embedding_model, args.batch_size)

    threshold = float(args.threshold)

    print("Scoring with sklearn cosine_similarity...")
    sklearn_scores = compute_pair_scores_sklearn(df, embeddings, text_to_idx)

    print("Scoring with global FAISS range_search...")
    faiss_scores = compute_pair_scores_faiss_range(df, embeddings, text_to_idx, threshold)

    comparison_df = build_comparison_table(df, sklearn_scores, faiss_scores, threshold)

    # Build per-method label vectors from identical rows for fair metric comparison.
    y_true = comparison_df["label"].astype(int).tolist()
    y_pred_sklearn = comparison_df["hit_sklearn"].astype(int).tolist()
    y_pred_faiss = comparison_df["hit_faiss_range"].astype(int).tolist()

    metrics_rows = [
        evaluate_hits(y_true, y_pred_sklearn, threshold, "sklearn_cosine"),
        evaluate_hits(y_true, y_pred_faiss, threshold, "faiss_global_range"),
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    comparison_path = os.path.join(args.output_dir, "script_a_pair_score_comparison.csv")
    # to_csv writes one line per input pair plus optional previews for manual review.
    add_previews(comparison_df, args.max_preview_length).to_csv(comparison_path, index=False)

    metrics_path = os.path.join(args.output_dir, "script_a_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # JSON summary captures top-line diagnostics for quick run-to-run tracking.
    summary = {
        "input_file": args.input_file,
        "embedding_model": args.embedding_model,
        "threshold": threshold,
        "num_pairs": int(len(df)),
        "num_unique_texts": int(len(texts)),
        "num_faiss_missing_scores": int(comparison_df["similarity_faiss_global_range"].isna().sum()),
        "num_hit_mismatches": int(comparison_df["hit_mismatch"].sum()),
    }
    summary_path = os.path.join(args.output_dir, "script_a_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Saved comparison table: {}".format(comparison_path))
    print("Saved metrics table: {}".format(metrics_path))
    print("Saved summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
