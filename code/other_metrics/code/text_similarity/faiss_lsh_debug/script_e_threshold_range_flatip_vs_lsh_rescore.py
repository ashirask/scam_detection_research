import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


"""
script_e_threshold_range_flatip_vs_lsh_rescore.py

Purpose
-------
This script compares two retrieval paths under the SAME cosine threshold:

1) Exact baseline: FAISS IndexFlatIP + range_search(radius=threshold)
2) Approximate path: FAISS IndexLSH search(k) to get candidates, then
   manual cosine re-scoring on original normalized embeddings, and finally
   threshold filtering.

Why this is useful
------------------
- It aligns with threshold-based suspicious-pair detection.
- It separates "candidate retrieval differences" from "score computation".
- It provides direct recall/precision against the exact threshold baseline.

Outputs
-------
- random_similarity_samples.csv
- random_similarity_kde.png
- per_query_threshold_comparison.csv
- overlap_score_differences.csv
- false_negative_examples.csv
- false_positive_examples.csv
- threshold_range_summary.json
"""


DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output" / "script_e_threshold_range")
SKIP_TEXT = {"[removed]", "[deleted]"}
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for threshold-based FlatIP-vs-LSH comparison."""
    parser = argparse.ArgumentParser(
        description="Compare IndexFlatIP range_search against IndexLSH candidates re-scored by cosine"
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--embedding-model", default="distiluse-base-multilingual-cased-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--nbits", type=int, default=32, help="IndexLSH bit length")
    parser.add_argument(
        "--lsh-candidate-k",
        type=int,
        default=200,
        help="Top-k candidates from IndexLSH before manual cosine re-scoring",
    )
    parser.add_argument(
        "--random-pairs",
        type=int,
        default=50000,
        help="Number of random pairs for threshold estimation",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Threshold quantile from random-pair similarity distribution",
    )
    parser.add_argument(
        "--max-error-samples",
        type=int,
        default=300,
        help="Max rows to keep in FN/FP example tables",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace and return a normalized string."""
    return " ".join((text or "").split())


def infer_post_type(record: Dict) -> str:
    """Infer whether record is comment or submission from body presence."""
    body = (record.get("body") or "").strip()
    return "comment" if body else "submission"


def build_text(record: Dict, post_type: str) -> str:
    """Build model input text from Reddit record fields."""
    if post_type == "comment":
        return normalize_text((record.get("body") or "").strip())
    title = normalize_text((record.get("title") or "").strip())
    selftext = normalize_text((record.get("selftext") or "").strip())
    return normalize_text(" ".join(part for part in [title, selftext] if part))


def load_records(path: str, min_words: int) -> pd.DataFrame:
    """Load, filter, and normalize records from JSONL into a DataFrame."""
    rows: List[Dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            author = (record.get("author") or "").strip()
            if not author or author in SKIP_AUTHORS:
                continue

            post_type = infer_post_type(record)
            text = build_text(record, post_type)
            if not text or text in SKIP_TEXT:
                continue
            if len(text.split()) < min_words:
                continue

            rows.append({"author": author, "text": text, "raw": record})

    return pd.DataFrame(rows)


def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """Encode texts into L2-normalized float32 embeddings."""
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def sample_random_pair_similarities(embeddings: np.ndarray, n_pairs: int, seed: int) -> np.ndarray:
    """Sample random pair cosine scores (dot products on normalized vectors)."""
    n = len(embeddings)
    if n < 2 or n_pairs <= 0:
        return np.asarray([], dtype=np.float32)

    rng = np.random.default_rng(seed)

    # Generate random index arrays for pair endpoints.
    left = rng.integers(0, n, size=n_pairs, dtype=np.int64)
    right = rng.integers(0, n, size=n_pairs, dtype=np.int64)

    # Ensure left != right so we do not sample trivial self-similarity (=1.0).
    same_mask = left == right
    while np.any(same_mask):
        right[same_mask] = rng.integers(0, n, size=int(np.sum(same_mask)), dtype=np.int64)
        same_mask = left == right

    sims = np.sum(embeddings[left] * embeddings[right], axis=1)
    return np.asarray(sims, dtype=np.float32)


def save_kde_plot(similarities: np.ndarray, threshold: float, out_path: str) -> Tuple[bool, str]:
    """Save KDE plot for random similarity distribution with threshold marker.

    Returns (saved, message). Plot creation is optional; if plotting deps are
    unavailable, function returns False and a message.
    """
    if similarities.size == 0:
        return False, "No similarity samples available for plotting"

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return False, f"matplotlib unavailable: {exc}"

    xs = np.linspace(float(np.min(similarities)), float(np.max(similarities)), 400)
    ys = None

    # Prefer gaussian_kde when scipy is available; fallback to interpolated histogram.
    try:
        from scipy.stats import gaussian_kde  # type: ignore

        kde = gaussian_kde(similarities)
        ys = kde(xs)
    except Exception:
        hist, bin_edges = np.histogram(similarities, bins=80, density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ys = np.interp(xs, centers, hist)

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, label="Random-pair density")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
    plt.title("Random Similarity Distribution")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True, "ok"


def run_flatip_range_search(index_ip: faiss.IndexFlatIP, embeddings: np.ndarray, threshold: float) -> List[Dict[int, float]]:
    """Run exact range_search and return per-query neighbor->score mapping."""
    lims, dists, inds = index_ip.range_search(np.ascontiguousarray(embeddings, dtype=np.float32), float(threshold))
    per_query: List[Dict[int, float]] = []

    # Iterate each query window [lims[q], lims[q+1]) and collect neighbors.
    for qi in range(len(embeddings)):
        start = int(lims[qi])
        end = int(lims[qi + 1])
        neighbor_scores: Dict[int, float] = {}

        for pos in range(start, end):
            j = int(inds[pos])
            if j < 0 or j == qi:
                continue
            score = float(dists[pos])
            # Keep max score if duplicate index appears.
            if j not in neighbor_scores or score > neighbor_scores[j]:
                neighbor_scores[j] = score

        per_query.append(neighbor_scores)

    return per_query


def run_lsh_rescored_threshold(
    index_lsh: faiss.IndexLSH,
    embeddings: np.ndarray,
    candidate_k: int,
    threshold: float,
) -> List[Dict[int, float]]:
    """Get IndexLSH candidates with search(k), then manual cosine re-score + threshold."""
    _, neighbors = index_lsh.search(np.ascontiguousarray(embeddings, dtype=np.float32), int(candidate_k))
    per_query: List[Dict[int, float]] = []

    # For each query, examine LSH candidates and keep only those above threshold.
    for qi in range(len(embeddings)):
        neighbor_scores: Dict[int, float] = {}

        for raw_j in neighbors[qi]:
            j = int(raw_j)
            if j < 0 or j == qi:
                continue

            # Manual cosine from normalized embeddings
            score = float(np.dot(embeddings[qi], embeddings[j]))
            if score < threshold:
                continue

            if j not in neighbor_scores or score > neighbor_scores[j]:
                neighbor_scores[j] = score

        per_query.append(neighbor_scores)

    return per_query


def build_pairs_full_comparison(
    texts: List[str],
    exact_neighbors: List[Dict[int, float]],
    lsh_neighbors: List[Dict[int, float]],
) -> pd.DataFrame:
    """Build a comprehensive DataFrame of all pairs from both methods with text and scores.

    For each pair that appears in either exact or LSH results, create a row with:
    - query_idx, neighbor_idx
    - query_text, neighbor_text (full text, not preview)
    - exact_score (or NaN if not in exact set)
    - lsh_manual_score (or NaN if not in LSH set)
    - abs_diff (difference in scores, or NaN if not in overlap)
    - source: one of ['overlap', 'false_negative', 'false_positive']
    """
    pairs: List[Dict] = []

    # Iterate each query and enumerate all pairs from either method.
    for qi in range(len(exact_neighbors)):
        exact_map = exact_neighbors[qi]
        lsh_map = lsh_neighbors[qi]

        exact_set = set(exact_map.keys())
        lsh_set = set(lsh_map.keys())

        # Union of both sets: cover all pairs that appear in either method.
        all_neighbor_indices = exact_set | lsh_set

        for j in sorted(all_neighbor_indices):
            in_exact = j in exact_set
            in_lsh = j in lsh_set

            exact_score = exact_map[j] if in_exact else np.nan
            lsh_score = lsh_map[j] if in_lsh else np.nan
            diff = abs(exact_score - lsh_score) if (in_exact and in_lsh) else np.nan

            # Label the source of this pair.
            if in_exact and in_lsh:
                source = "overlap"
            elif in_exact:
                source = "false_negative"
            else:
                source = "false_positive"

            pairs.append(
                {
                    "query_idx": qi,
                    "neighbor_idx": j,
                    "query_text": texts[qi],
                    "neighbor_text": texts[j],
                    "exact_score": exact_score,
                    "lsh_manual_score": lsh_score,
                    "abs_diff": diff,
                    "source": source,
                }
            )

    return pd.DataFrame(pairs)


def compare_threshold_sets(
    texts: List[str],
    exact_neighbors: List[Dict[int, float]],
    lsh_neighbors: List[Dict[int, float]],
    max_error_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare exact threshold neighbors vs LSH-rescored neighbors per query."""
    per_query_rows: List[Dict] = []
    overlap_score_rows: List[Dict] = []
    fn_rows: List[Dict] = []
    fp_rows: List[Dict] = []

    for qi in range(len(exact_neighbors)):
        exact_map = exact_neighbors[qi]
        lsh_map = lsh_neighbors[qi]

        exact_set = set(exact_map.keys())
        lsh_set = set(lsh_map.keys())

        overlap = exact_set & lsh_set
        missing = sorted(exact_set - lsh_set)
        extra = sorted(lsh_set - exact_set)

        exact_count = len(exact_set)
        lsh_count = len(lsh_set)
        overlap_count = len(overlap)

        recall = (overlap_count / exact_count) if exact_count > 0 else 1.0
        precision = (overlap_count / lsh_count) if lsh_count > 0 else (1.0 if exact_count == 0 else 0.0)

        # Score-diff check on overlaps: these should be near zero (same cosine pair score).
        diffs: List[float] = []
        for j in overlap:
            exact_score = float(exact_map[j])
            lsh_score = float(lsh_map[j])
            diff = abs(exact_score - lsh_score)
            diffs.append(diff)
            overlap_score_rows.append(
                {
                    "query_idx": qi,
                    "neighbor_idx": j,
                    "exact_score": exact_score,
                    "lsh_manual_score": lsh_score,
                    "abs_diff": diff,
                }
            )

        per_query_rows.append(
            {
                "query_idx": qi,
                "exact_count": exact_count,
                "lsh_count": lsh_count,
                "overlap_count": overlap_count,
                "recall_vs_exact": recall,
                "precision_vs_exact": precision,
                "false_negative_count": len(missing),
                "false_positive_count": len(extra),
                "mean_abs_score_diff_overlap": float(np.mean(diffs)) if diffs else 0.0,
                "query_text_preview": texts[qi][:160],
            }
        )

        # Collect bounded error examples for manual review.
        if len(fn_rows) < max_error_samples:
            remaining = max_error_samples - len(fn_rows)
            for j in missing[:remaining]:
                fn_rows.append(
                    {
                        "query_idx": qi,
                        "neighbor_idx": int(j),
                        "exact_score": float(exact_map[j]),
                        "query_text_preview": texts[qi][:160],
                        "neighbor_text_preview": texts[j][:160],
                    }
                )

        if len(fp_rows) < max_error_samples:
            remaining = max_error_samples - len(fp_rows)
            for j in extra[:remaining]:
                fp_rows.append(
                    {
                        "query_idx": qi,
                        "neighbor_idx": int(j),
                        "lsh_manual_score": float(lsh_map[j]),
                        "query_text_preview": texts[qi][:160],
                        "neighbor_text_preview": texts[j][:160],
                    }
                )

    return (
        pd.DataFrame(per_query_rows),
        pd.DataFrame(overlap_score_rows),
        pd.DataFrame(fn_rows),
        pd.DataFrame(fp_rows),
    )


def main() -> None:
    """Main pipeline for threshold-based FlatIP range vs LSH re-score comparison."""
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading records from: {args.input_file}")
    df = load_records(args.input_file, min_words=args.min_words)
    if df.empty:
        raise SystemExit("No usable records after filtering")

    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)

    print(f"Using {len(df)} records (authors={df['author'].nunique()})")
    texts = df["text"].tolist()

    print("Computing embeddings...")
    embeddings = compute_embeddings(texts, args.embedding_model, args.batch_size)
    dim = embeddings.shape[1]

    print(f"Estimating threshold from {args.random_pairs} random pairs at quantile={args.quantile}...")
    random_sims = sample_random_pair_similarities(embeddings, args.random_pairs, args.seed)
    if random_sims.size == 0:
        raise SystemExit("Unable to estimate threshold: random similarity sample is empty")
    threshold = float(np.quantile(random_sims, args.quantile))

    random_sims_path = os.path.join(args.output_dir, "random_similarity_samples.csv")
    pd.DataFrame({"similarity": random_sims}).to_csv(random_sims_path, index=False)

    kde_path = os.path.join(args.output_dir, "random_similarity_kde.png")
    kde_saved, kde_msg = save_kde_plot(random_sims, threshold, kde_path)
    if kde_saved:
        print(f"Saved KDE plot to: {kde_path}")
    else:
        print(f"KDE plot skipped: {kde_msg}")

    print("Building FAISS indexes...")
    index_build_start = time.perf_counter()
    index_ip = faiss.IndexFlatIP(dim)
    index_ip.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    index_lsh = faiss.IndexLSH(dim, int(args.nbits))
    index_lsh.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    index_build_time = float(time.perf_counter() - index_build_start)

    print("Running exact FlatIP range_search...")
    flatip_start = time.perf_counter()
    exact_neighbors = run_flatip_range_search(index_ip, embeddings, threshold)
    flatip_time = float(time.perf_counter() - flatip_start)

    print(f"Running LSH search(k={args.lsh_candidate_k}) + manual cosine re-score...")
    lsh_start = time.perf_counter()
    lsh_neighbors = run_lsh_rescored_threshold(index_lsh, embeddings, args.lsh_candidate_k, threshold)
    lsh_time = float(time.perf_counter() - lsh_start)

    print("Comparing threshold neighbor sets per query...")
    per_query_df, overlap_score_df, fn_df, fp_df = compare_threshold_sets(
        texts=texts,
        exact_neighbors=exact_neighbors,
        lsh_neighbors=lsh_neighbors,
        max_error_samples=int(args.max_error_samples),
    )

    print("Building full pairs comparison CSV...")
    pairs_full_df = build_pairs_full_comparison(
        texts=texts,
        exact_neighbors=exact_neighbors,
        lsh_neighbors=lsh_neighbors,
    )

    per_query_path = os.path.join(args.output_dir, "per_query_threshold_comparison.csv")
    overlap_score_path = os.path.join(args.output_dir, "overlap_score_differences.csv")
    fn_path = os.path.join(args.output_dir, "false_negative_examples.csv")
    fp_path = os.path.join(args.output_dir, "false_positive_examples.csv")
    pairs_full_path = os.path.join(args.output_dir, "pairs_full_comparison.csv")

    per_query_df.to_csv(per_query_path, index=False)
    overlap_score_df.to_csv(overlap_score_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    fp_df.to_csv(fp_path, index=False)
    pairs_full_df.to_csv(pairs_full_path, index=False)

    total_exact = int(per_query_df["exact_count"].sum()) if not per_query_df.empty else 0
    total_lsh = int(per_query_df["lsh_count"].sum()) if not per_query_df.empty else 0
    total_overlap = int(per_query_df["overlap_count"].sum()) if not per_query_df.empty else 0

    summary = {
        "input_file": args.input_file,
        "n_records": int(len(df)),
        "sample_size": int(len(df)),
        "nbits": int(args.nbits),
        "lsh_candidate_k": int(args.lsh_candidate_k),
        "quantile": float(args.quantile),
        "threshold": threshold,
        "random_pairs": int(args.random_pairs),
        "kde_plot_saved": bool(kde_saved),
        "kde_plot_message": kde_msg,
        "total_exact_threshold_neighbors": total_exact,
        "total_lsh_threshold_neighbors": total_lsh,
        "total_overlap": total_overlap,
        "micro_recall_vs_exact": (total_overlap / total_exact) if total_exact > 0 else 1.0,
        "micro_precision_vs_exact": (total_overlap / total_lsh) if total_lsh > 0 else 1.0,
        "mean_recall_vs_exact": float(per_query_df["recall_vs_exact"].mean()) if not per_query_df.empty else 1.0,
        "mean_precision_vs_exact": float(per_query_df["precision_vs_exact"].mean()) if not per_query_df.empty else 1.0,
        "mean_abs_score_diff_overlap": float(overlap_score_df["abs_diff"].mean()) if not overlap_score_df.empty else 0.0,
        "runtime_seconds": {
            "index_build_seconds": index_build_time,
            "flatip_range_search_seconds": flatip_time,
            "lsh_search_and_rescore_seconds": lsh_time,
            "flatip_vs_lsh_speedup": flatip_time / lsh_time if lsh_time > 0 else float("inf"),
        },
        "outputs": {
            "random_similarity_samples": random_sims_path,
            "random_similarity_kde": kde_path,
            "per_query_threshold_comparison": per_query_path,
            "overlap_score_differences": overlap_score_path,
            "false_negative_examples": fn_path,
            "false_positive_examples": fp_path,
            "pairs_full_comparison": pairs_full_path,
        },
    }

    summary_path = os.path.join(args.output_dir, "threshold_range_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Threshold (quantile={args.quantile:.4f}): {threshold:.6f}")
    print(f"\n=== Timing Summary ===")
    print(f"Index build: {index_build_time:.2f}s")
    print(f"FlatIP range_search: {flatip_time:.2f}s")
    print(f"LSH search + re-score: {lsh_time:.2f}s")
    print(f"Speedup (FlatIP / LSH): {flatip_time / lsh_time:.2f}x" if lsh_time > 0 else "N/A")
    print(f"\nSaved per-query comparison: {per_query_path}")
    print(f"Saved overlap score diffs: {overlap_score_path}")
    print(f"Saved false negatives: {fn_path}")
    print(f"Saved false positives: {fp_path}")
    print(f"Saved full pairs comparison: {pairs_full_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
