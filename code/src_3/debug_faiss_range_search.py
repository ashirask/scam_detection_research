import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# This script tests global FAISS range search without LSH.
# It helps isolate whether false positives come from embeddings/thresholds
# rather than candidate generation.
DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent / "sample_labeled_pairs.csv")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output" / "faiss_range_debug")
DEFAULT_MODEL = "distiluse-base-multilingual-cased-v2"


def parse_args() -> argparse.Namespace:
    """Define CLI options for global FAISS range-search debugging."""
    parser = argparse.ArgumentParser(
        description=(
            "Run global FAISS IndexFlatIP range_search over a small labeled sample "
            "to debug semantic-threshold behavior without LSH."
        )
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="CSV with text_a, text_b, label.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for debug outputs.")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer model used to produce embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument(
        "--radius-threshold",
        type=float,
        default=0.80,
        help="Range-search threshold in inner-product space (cosine when normalized).",
    )
    parser.add_argument(
        "--query-source",
        choices=["all", "text_a", "text_b"],
        default="all",
        help="Which texts are used as FAISS queries.",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include self-matches where query text equals neighbor text.",
    )
    parser.add_argument(
        "--max-neighbors-per-query",
        type=int,
        default=0,
        help="Cap neighbors written per query after sorting by score. 0 means no cap.",
    )
    parser.add_argument(
        "--max-preview-length",
        type=int,
        default=200,
        help="Maximum preview length in exported CSV files.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def load_labeled_pairs(path: str) -> pd.DataFrame:
    """Load and validate labeled-pair CSV used for controlled testing."""
    if not os.path.isfile(path):
        raise SystemExit(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required = {"text_a", "text_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {sorted(missing)}")

    working = df.copy()
    working["text_a"] = working["text_a"].astype(str).map(normalize_text)
    working["text_b"] = working["text_b"].astype(str).map(normalize_text)

    if "pair_id" not in working.columns:
        working["pair_id"] = [f"pair_{i}" for i in range(len(working))]
    if "note" not in working.columns:
        working["note"] = ""

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

    working = working[(working["text_a"] != "") & (working["text_b"] != "")].reset_index(drop=True)
    if working.empty:
        raise SystemExit("No usable labeled pairs found after cleaning.")

    invalid_labels = sorted(set(working["label"].unique()) - {0, 1})
    if invalid_labels:
        raise SystemExit(f"Labels must be 0 or 1. Invalid values found: {invalid_labels}")

    return working


def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """Generate normalized float32 embeddings so IP equals cosine similarity."""
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def build_text_space(df: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    """Build a de-duplicated text corpus and text->index lookup."""
    all_texts = pd.unique(pd.concat([df["text_a"], df["text_b"]], ignore_index=True)).tolist()
    texts = [str(text) for text in all_texts]
    text_to_idx = {text: idx for idx, text in enumerate(texts)}
    return texts, text_to_idx


def choose_query_indices(df: pd.DataFrame, text_to_idx: Dict[str, int], source: str) -> List[int]:
    """Select which texts are used as queries during range search."""
    if source == "text_a":
        query_texts = pd.unique(df["text_a"]).tolist()
    elif source == "text_b":
        query_texts = pd.unique(df["text_b"]).tolist()
    else:
        query_texts = list(text_to_idx.keys())

    return [text_to_idx[text] for text in query_texts]


def evaluate_labeled_pairs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    text_to_idx: Dict[str, int],
    threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Score only labeled pairs and compute metrics at one threshold."""
    rows: List[Dict[str, object]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for _, row in df.iterrows():
        idx_a = text_to_idx[row["text_a"]]
        idx_b = text_to_idx[row["text_b"]]
        score = float(np.dot(embeddings[idx_a], embeddings[idx_b]))
        pred = int(score >= threshold)
        label = int(row["label"])

        rows.append(
            {
                "pair_id": row["pair_id"],
                "text_idx_a": idx_a,
                "text_idx_b": idx_b,
                "label": label,
                "predicted_flag": pred,
                "similarity": score,
                "note": row["note"],
                "text_a": row["text_a"],
                "text_b": row["text_b"],
            }
        )
        y_true.append(label)
        y_pred.append(pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    summary = {
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
    pair_scores = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    return pair_scores, summary


def build_label_lookup_with_indices(
    df: pd.DataFrame,
    text_to_idx: Dict[str, int],
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Map unordered text-index pairs to provided labels and metadata."""
    out: Dict[Tuple[int, int], Dict[str, object]] = {}
    for _, row in df.iterrows():
        a = text_to_idx[row["text_a"]]
        b = text_to_idx[row["text_b"]]
        pair_key = (a, b) if a < b else (b, a)
        out[pair_key] = {
            "label": int(row["label"]),
            "pair_id": str(row["pair_id"]),
            "note": str(row.get("note", "")),
        }
    return out


def run_faiss_range_search(
    embeddings: np.ndarray,
    texts: List[str],
    query_indices: List[int],
    label_lookup: Dict[Tuple[int, int], Dict[str, object]],
    threshold: float,
    include_self: bool,
    max_neighbors_per_query: int,
) -> pd.DataFrame:
    """Run global FAISS range_search and annotate results with labeled info when known.

    Uses IndexFlatIP (exact inner product). Because embeddings are normalized,
    this is exact cosine range search.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(vectors)

    query_vectors = np.ascontiguousarray(vectors[query_indices], dtype=np.float32)
    lims, scores, neighbors = index.range_search(query_vectors, float(threshold))

    rows: List[Dict[str, object]] = []
    for local_qi, global_qi in enumerate(query_indices):
        start = int(lims[local_qi])
        end = int(lims[local_qi + 1])

        per_query_rows: List[Dict[str, object]] = []
        for pos in range(start, end):
            nbr = int(neighbors[pos])
            sim = float(scores[pos])
            if (not include_self) and (nbr == global_qi):
                continue

            a, b = (global_qi, nbr) if global_qi < nbr else (nbr, global_qi)
            label_info = label_lookup.get((a, b))

            per_query_rows.append(
                {
                    "query_idx": global_qi,
                    "neighbor_idx": nbr,
                    "similarity": sim,
                    "query_text": texts[global_qi],
                    "neighbor_text": texts[nbr],
                    "is_self": int(global_qi == nbr),
                    "is_labeled_pair": int(label_info is not None),
                    "label": (int(label_info["label"]) if label_info is not None else -1),
                    "pair_id": (str(label_info["pair_id"]) if label_info is not None else ""),
                    "note": (str(label_info["note"]) if label_info is not None else ""),
                }
            )

        per_query_rows.sort(key=lambda row: row["similarity"], reverse=True)
        if max_neighbors_per_query > 0:
            per_query_rows = per_query_rows[:max_neighbors_per_query]
        rows.extend(per_query_rows)

    if not rows:
        return pd.DataFrame(
            columns=[
                "query_idx",
                "neighbor_idx",
                "similarity",
                "query_text",
                "neighbor_text",
                "is_self",
                "is_labeled_pair",
                "label",
                "pair_id",
                "note",
            ]
        )

    return pd.DataFrame(rows).sort_values(["query_idx", "similarity"], ascending=[True, False]).reset_index(drop=True)


def add_previews(df: pd.DataFrame, max_preview_length: int) -> pd.DataFrame:
    out = df.copy()
    if "query_text" in out.columns:
        out["query_text_preview"] = out["query_text"].str.slice(0, max_preview_length)
    if "neighbor_text" in out.columns:
        out["neighbor_text_preview"] = out["neighbor_text"].str.slice(0, max_preview_length)
    if "text_a" in out.columns:
        out["text_a_preview"] = out["text_a"].str.slice(0, max_preview_length)
    if "text_b" in out.columns:
        out["text_b_preview"] = out["text_b"].str.slice(0, max_preview_length)
    return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading labeled pairs from: {args.input_file}")
    labeled_df = load_labeled_pairs(args.input_file)
    print(f"Labeled pairs: {len(labeled_df)}")
    print(f"Positive pairs: {int((labeled_df['label'] == 1).sum())}")
    print(f"Negative pairs: {int((labeled_df['label'] == 0).sum())}")

    texts, text_to_idx = build_text_space(labeled_df)
    print(f"Unique texts in corpus: {len(texts)}")

    print(f"Embedding texts with: {args.embedding_model}")
    embeddings = compute_embeddings(texts, args.embedding_model, args.batch_size)

    threshold = float(args.radius_threshold)
    pair_scores, pair_summary = evaluate_labeled_pairs(labeled_df, embeddings, text_to_idx, threshold)

    label_lookup = build_label_lookup_with_indices(labeled_df, text_to_idx)
    query_indices = choose_query_indices(labeled_df, text_to_idx, args.query_source)
    print(f"Running FAISS range_search on {len(query_indices)} queries at threshold={threshold:.3f}")

    neighbors = run_faiss_range_search(
        embeddings=embeddings,
        texts=texts,
        query_indices=query_indices,
        label_lookup=label_lookup,
        threshold=threshold,
        include_self=args.include_self,
        max_neighbors_per_query=args.max_neighbors_per_query,
    )

    query_summary = (
        neighbors.groupby("query_idx", as_index=False)
        .agg(
            total_neighbors=("neighbor_idx", "count"),
            labeled_neighbors=("is_labeled_pair", "sum"),
            labeled_positive=("label", lambda s: int((s == 1).sum())),
            labeled_negative=("label", lambda s: int((s == 0).sum())),
            max_similarity=("similarity", "max"),
            avg_similarity=("similarity", "mean"),
        )
        if not neighbors.empty
        else pd.DataFrame(
            columns=[
                "query_idx",
                "total_neighbors",
                "labeled_neighbors",
                "labeled_positive",
                "labeled_negative",
                "max_similarity",
                "avg_similarity",
            ]
        )
    )

    pair_scores_path = os.path.join(args.output_dir, "faiss_labeled_pair_scores.csv")
    add_previews(pair_scores, args.max_preview_length).to_csv(pair_scores_path, index=False)

    neighbors_path = os.path.join(args.output_dir, "faiss_range_neighbors.csv")
    add_previews(neighbors, args.max_preview_length).to_csv(neighbors_path, index=False)

    labeled_neighbors_path = os.path.join(args.output_dir, "faiss_range_neighbors_labeled_only.csv")
    labeled_neighbors = neighbors[neighbors["is_labeled_pair"] == 1].copy() if not neighbors.empty else neighbors.copy()
    add_previews(labeled_neighbors, args.max_preview_length).to_csv(labeled_neighbors_path, index=False)

    query_summary_path = os.path.join(args.output_dir, "faiss_query_summary.csv")
    query_summary.to_csv(query_summary_path, index=False)

    summary = {
        "input_file": args.input_file,
        "embedding_model": args.embedding_model,
        "radius_threshold": threshold,
        "query_source": args.query_source,
        "include_self": bool(args.include_self),
        "num_labeled_pairs": int(len(labeled_df)),
        "num_unique_texts": int(len(texts)),
        "num_queries": int(len(query_indices)),
        "num_neighbors_returned": int(len(neighbors)),
        "pair_metrics": pair_summary,
    }
    summary_path = os.path.join(args.output_dir, "faiss_range_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Labeled-pair metrics at current threshold:")
    print(
        f"  accuracy={pair_summary['accuracy']:.3f} precision={pair_summary['precision']:.3f} "
        f"recall={pair_summary['recall']:.3f} f1={pair_summary['f1']:.3f}"
    )
    print(
        f"  tn={int(pair_summary['tn'])} fp={int(pair_summary['fp'])} "
        f"fn={int(pair_summary['fn'])} tp={int(pair_summary['tp'])}"
    )

    print(f"Saved labeled pair scores: {pair_scores_path}")
    print(f"Saved all neighbors: {neighbors_path}")
    print(f"Saved labeled-only neighbors: {labeled_neighbors_path}")
    print(f"Saved query summary: {query_summary_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
