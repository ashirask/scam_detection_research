import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer

import itertools


"""
script_f_bucketed_threshold_exact_vs_topk.py

Purpose
-------
Compare two bucketed retrieval paths under the SAME cosine threshold:

1) Exact bucket baseline:
   - build random-hyperplane LSH buckets
   - within each bucket, run FAISS IndexFlatIP.range_search(radius=threshold)

2) Production-style bucketed candidate path:
   - build the same random-hyperplane LSH buckets
   - within each bucket, run FAISS IndexFlatIP.search(k)
   - manually re-score returned neighbors by cosine
   - keep only neighbors above the same threshold

Important note
--------------
This script does NOT use FAISS IndexLSH directly.
The explicit random-hyperplane bucketing is the LSH stage in the production pipeline.
So for this comparison, the correct question is:
"Within the same buckets, how different are range_search and top-k + rescore?"

Outputs
-------
- random_similarity_samples.csv
- random_similarity_kde.png
- bucket_pair_comparison.csv
- overlap_score_differences.csv
- per_query_bucket_comparison.csv
- false_negative_examples.csv
- false_positive_examples.csv
- bucket_summary.json
"""


DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output" / "script_f_bucketed_threshold")
SKIP_TEXT = {"[removed]", "[deleted]"}
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the bucketed threshold comparison."""
    parser = argparse.ArgumentParser(
        description="Compare bucketed IndexFlatIP.range_search against bucketed top-k + cosine rescore"
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--embedding-model", default="distiluse-base-multilingual-cased-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use all usable records")
    parser.add_argument("--random-pairs", type=int, default=50000)
    parser.add_argument("--quantile", type=float, default=0.999)
    parser.add_argument("--lsh-bits", type=int, default=16)
    parser.add_argument("--lsh-tables", type=int, default=4)
    parser.add_argument("--faiss-top-k", type=int, default=40)
    parser.add_argument("--use-indexlsh", action="store_true", help="Run IndexLSH + re-score single run")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep over nbits/ntables/k-values")
    parser.add_argument("--sweep-nbits", default="8,16,32", help="Comma-separated nbits values for sweep")
    parser.add_argument("--sweep-ntables", default="1,2,4", help="Comma-separated ntables values for sweep")
    parser.add_argument("--sweep-k-values", default="20,40,80", help="Comma-separated k values for sweep")
    parser.add_argument("--min-bucket-size", type=int, default=2)
    parser.add_argument("--max-error-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace to keep text stable across the pipeline."""
    return " ".join((text or "").split())


def infer_post_type(record: Dict) -> str:
    """Infer comment vs submission from the presence of a body field."""
    body = (record.get("body") or "").strip()
    return "comment" if body else "submission"


def build_text(record: Dict, post_type: str) -> str:
    """Construct the text that will be embedded for a record."""
    if post_type == "comment":
        return normalize_text((record.get("body") or "").strip())

    title = normalize_text((record.get("title") or "").strip())
    selftext = normalize_text((record.get("selftext") or "").strip())
    return normalize_text(" ".join(part for part in [title, selftext] if part))


def load_records(path: str, min_words: int) -> pd.DataFrame:
    """Load JSONL records, filter noisy rows, and return a modeling DataFrame."""
    rows: List[Dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
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

            word_count = len(text.split())
            if word_count < min_words:
                continue

            rows.append(
                {
                    "author": author,
                    "text": text,
                    "word_count": word_count,
                    "post_type": post_type,
                    "subreddit": record.get("subreddit", ""),
                    "created_utc": record.get("created_utc"),
                    "id": record.get("id") or record.get("name") or f"row_{line_num}",
                    "raw": record,
                }
            )

    return pd.DataFrame(rows)


def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """Embed texts into normalized float32 vectors."""
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
    """Estimate background cosine similarity from random, distinct pairs."""
    n_rows = embeddings.shape[0]
    if n_rows < 2 or n_pairs <= 0:
        return np.asarray([], dtype=np.float32)

    rng = np.random.default_rng(seed)
    sims = np.zeros(n_pairs, dtype=np.float32)

    # Draw random pairs without replacement within each pair, then compute cosine-equivalent dot products.
    for i in range(n_pairs):
        a, b = rng.choice(n_rows, size=2, replace=False)
        sims[i] = float(np.dot(embeddings[a], embeddings[b]))

    return sims


def save_kde_plot(similarities: np.ndarray, threshold: float, output_file: str) -> Tuple[bool, str]:
    """Save a density plot for the random-pair similarity distribution."""
    if similarities.size == 0:
        return False, "No similarity samples available"

    try:
        xs = np.linspace(float(np.min(similarities)), float(np.max(similarities)), 400)
        try:
            from scipy.stats import gaussian_kde  # type: ignore

            ys = gaussian_kde(similarities)(xs)
        except Exception:
            # Fallback if scipy is not available.
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
        plt.savefig(output_file, dpi=150)
        plt.close()
        return True, "ok"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def build_lsh_signature_matrix(embeddings: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Convert embeddings into integer LSH signatures for one random-hyperplane table."""
    bit_matrix = (embeddings @ projection_matrix) >= 0.0
    bit_weights = (1 << np.arange(bit_matrix.shape[1], dtype=np.uint64))
    return (bit_matrix.astype(np.uint64) * bit_weights).sum(axis=1)


def canonical_pair(a: int, b: int) -> Tuple[int, int]:
    """Return a canonical unordered pair key."""
    return (a, b) if a < b else (b, a)


def generate_bucket_plan(
    embeddings: np.ndarray,
    lsh_bits: int,
    lsh_tables: int,
    seed: int,
) -> List[Tuple[int, Dict[int, List[int]]]]:
    """Build the random-hyperplane buckets for each independent LSH table.

    Returns a list of (table_index, buckets) pairs, where each bucket map is
    signature -> list of global vector indices.
    """
    rng = np.random.default_rng(seed)
    dim = embeddings.shape[1]
    plans: List[Tuple[int, Dict[int, List[int]]]] = []

    for table_idx in range(lsh_tables):
        projection_matrix = rng.standard_normal(size=(dim, lsh_bits), dtype=np.float32)
        signatures = build_lsh_signature_matrix(embeddings, projection_matrix)
        buckets: DefaultDict[int, List[int]] = defaultdict(list)

        # Group global row indices by identical signature so each bucket can be searched independently.
        for idx, signature in enumerate(signatures):
            buckets[int(signature)].append(idx)

        plans.append((table_idx, buckets))

    return plans


def collect_bucketed_exact_pairs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    lsh_bits: int,
    lsh_tables: int,
    min_bucket_size: int,
    seed: int,
) -> Tuple[Dict[Tuple[int, int], float], Dict[str, float]]:
    """Collect exact threshold pairs using bucketed FAISS range_search."""
    if df.empty:
        return {}, {
            "num_tables": 0,
            "num_buckets": 0,
            "num_bucket_rows": 0,
            "num_neighbors_returned": 0,
            "num_pairs": 0,
            "bucket_size_min": 0,
            "bucket_size_max": 0,
            "bucket_size_mean": 0.0,
            "runtime_seconds": 0.0,
        }

    start_time = time.perf_counter()
    dim = embeddings.shape[1]
    plans = generate_bucket_plan(embeddings, lsh_bits, lsh_tables, seed)

    seen_pairs: Set[Tuple[int, int]] = set()
    pair_scores: Dict[Tuple[int, int], float] = {}
    num_buckets = 0
    num_bucket_rows = 0
    num_neighbors_returned = 0
    bucket_sizes: List[int] = []

    for table_idx, buckets in plans:
        # Each table uses a fresh random projection matrix; bucket membership changes per table.
        for idx_list in buckets.values():
            bucket_size = len(idx_list)
            if bucket_size < min_bucket_size:
                continue

            num_buckets += 1
            num_bucket_rows += bucket_size
            bucket_sizes.append(bucket_size)

            bucket_vectors = np.ascontiguousarray(embeddings[idx_list], dtype=np.float32)
            index = faiss.IndexFlatIP(dim)
            index.add(bucket_vectors)

            # Exact threshold retrieval inside the current bucket.
            lims, scores, neighbors = index.range_search(bucket_vectors, float(threshold))

            for local_i, global_i in enumerate(idx_list):
                start = int(lims[local_i])
                end = int(lims[local_i + 1])
                num_neighbors_returned += (end - start)

                # Walk neighbors returned for this query.
                for pos in range(start, end):
                    local_j = int(neighbors[pos])
                    if local_j == local_i:
                        continue

                    global_j = idx_list[local_j]
                    key = canonical_pair(global_i, global_j)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    # Keep only cross-account pairs. Same-account self-links are not used for suspiciousness.
                    if df.iloc[key[0]]["author"] == df.iloc[key[1]]["author"]:
                        continue

                    pair_scores[key] = max(pair_scores.get(key, -1.0), float(scores[pos]))

    runtime_seconds = float(time.perf_counter() - start_time)
    stats = {
        "num_tables": int(lsh_tables),
        "num_buckets": int(num_buckets),
        "num_bucket_rows": int(num_bucket_rows),
        "num_neighbors_returned": int(num_neighbors_returned),
        "num_pairs": int(len(pair_scores)),
        "bucket_size_min": int(min(bucket_sizes)) if bucket_sizes else 0,
        "bucket_size_max": int(max(bucket_sizes)) if bucket_sizes else 0,
        "bucket_size_mean": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
        "runtime_seconds": runtime_seconds,
    }
    return pair_scores, stats


def parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated list of integers into Python list."""
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception:
        return []


def collect_indexlsh_rescored_pairs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    nbits: int,
    ntables: int,
    faiss_top_k: int,
    seed: int,
) -> Tuple[Dict[Tuple[int, int], float], Dict[str, float]]:
    """Collect threshold pairs using FAISS IndexLSH + manual cosine re-scoring.

    This builds `ntables` independent `IndexLSH` instances (different tables)
    and unions candidate sets per-query before re-scoring with exact cosine.
    """
    if df.empty:
        return {}, {
            "nbits": nbits,
            "ntables": ntables,
            "num_pairs": 0,
            "num_neighbors_returned": 0,
            "runtime_seconds": 0.0,
        }

    start_time = time.perf_counter()
    n_rows, dim = embeddings.shape[0], embeddings.shape[1]

    # Build multiple independent IndexLSH instances to emulate multiple tables.
    lsh_indices: List[faiss.Index] = []
    for t in range(int(ntables)):
        index = faiss.IndexLSH(dim, int(nbits))
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        lsh_indices.append(index)

    seen_pairs: Set[Tuple[int, int]] = set()
    pair_scores: Dict[Tuple[int, int], float] = {}
    num_neighbors_returned = 0

    # For each query, union candidates across tables, rescore, and filter by threshold.
    for qi in range(n_rows):
        q_vec = np.ascontiguousarray(embeddings[qi : qi + 1], dtype=np.float32)
        candidates: Set[int] = set()
        for index in lsh_indices:
            sims, neigh = index.search(q_vec, int(min(faiss_top_k, n_rows)))
            for raw in neigh[0]:
                if int(raw) < 0:
                    continue
                candidates.add(int(raw))

        num_neighbors_returned += len(candidates)

        for cj in candidates:
            if cj == qi:
                continue
            score = float(np.dot(embeddings[qi], embeddings[cj]))
            if score < threshold:
                continue
            key = canonical_pair(qi, cj)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            if df.iloc[key[0]]["author"] == df.iloc[key[1]]["author"]:
                continue

            pair_scores[key] = max(pair_scores.get(key, -1.0), score)

    runtime_seconds = float(time.perf_counter() - start_time)
    stats = {
        "nbits": int(nbits),
        "ntables": int(ntables),
        "num_rows": int(n_rows),
        "num_neighbors_returned": int(num_neighbors_returned),
        "num_pairs": int(len(pair_scores)),
        "runtime_seconds": runtime_seconds,
    }
    return pair_scores, stats


def parameter_sweep(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    sweep_nbits: List[int],
    sweep_ntables: List[int],
    sweep_ks: List[int],
    seed: int,
    base_lsh_bits: int,
    base_lsh_tables: int,
    min_bucket_size: int,
) -> pd.DataFrame:
    """Run grid sweep over nbits, ntables, and k and return a summary DataFrame."""
    rows: List[Dict] = []
    # We'll need the exact_pairs baseline to compute recall/precision vs exact (bucketed exact)
    exact_pairs, _ = collect_bucketed_exact_pairs(
        df=df,
        embeddings=embeddings,
        threshold=threshold,
        lsh_bits=base_lsh_bits,
        lsh_tables=base_lsh_tables,
        min_bucket_size=min_bucket_size,
        seed=seed,
    )

    for nbits, ntables, k in itertools.product(sweep_nbits, sweep_ntables, sweep_ks):
        start = time.perf_counter()
        lsh_pairs, lsh_stats = collect_indexlsh_rescored_pairs(
            df=df,
            embeddings=embeddings,
            threshold=threshold,
            nbits=nbits,
            ntables=ntables,
            faiss_top_k=k,
            seed=seed,
        )

        # compute overlap stats vs exact_pairs
        exact_set = set(exact_pairs.keys())
        lsh_set = set(lsh_pairs.keys())
        overlap = exact_set & lsh_set

        total_exact = len(exact_set)
        total_lsh = len(lsh_set)
        total_overlap = len(overlap)

        runtime = float(time.perf_counter() - start)
        rows.append(
            {
                "nbits": int(nbits),
                "ntables": int(ntables),
                "k": int(k),
                "total_exact_pairs": int(total_exact),
                "total_lsh_pairs": int(total_lsh),
                "total_overlap": int(total_overlap),
                "micro_recall_vs_exact": (total_overlap / total_exact) if total_exact > 0 else 1.0,
                "micro_precision_vs_exact": (total_overlap / total_lsh) if total_lsh > 0 else 1.0,
                "lsh_runtime_seconds": runtime,
            }
        )

    return pd.DataFrame(rows)


def collect_bucketed_topk_pairs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    lsh_bits: int,
    lsh_tables: int,
    min_bucket_size: int,
    faiss_top_k: int,
    seed: int,
) -> Tuple[Dict[Tuple[int, int], float], Dict[str, float]]:
    """Collect threshold pairs using bucketed FAISS top-k + manual cosine re-scoring."""
    if df.empty:
        return {}, {
            "num_tables": 0,
            "num_buckets": 0,
            "num_bucket_rows": 0,
            "num_neighbors_returned": 0,
            "num_pairs": 0,
            "bucket_size_min": 0,
            "bucket_size_max": 0,
            "bucket_size_mean": 0.0,
            "runtime_seconds": 0.0,
        }

    start_time = time.perf_counter()
    dim = embeddings.shape[1]
    plans = generate_bucket_plan(embeddings, lsh_bits, lsh_tables, seed)

    seen_pairs: Set[Tuple[int, int]] = set()
    pair_scores: Dict[Tuple[int, int], float] = {}
    num_buckets = 0
    num_bucket_rows = 0
    num_neighbors_returned = 0
    bucket_sizes: List[int] = []

    for table_idx, buckets in plans:
        # Re-use the same bucket structure as the exact branch so only the retrieval mode differs.
        for idx_list in buckets.values():
            bucket_size = len(idx_list)
            if bucket_size < min_bucket_size:
                continue

            num_buckets += 1
            num_bucket_rows += bucket_size
            bucket_sizes.append(bucket_size)

            bucket_vectors = np.ascontiguousarray(embeddings[idx_list], dtype=np.float32)
            index = faiss.IndexFlatIP(dim)
            index.add(bucket_vectors)

            # top-k is capped by the bucket size so FAISS does not request more neighbors than exist.
            k = min(int(faiss_top_k), bucket_size)
            sims, neighbors = index.search(bucket_vectors, k)

            for local_i, global_i in enumerate(idx_list):
                # Count the raw neighbors returned by FAISS for this query.
                num_neighbors_returned += int(np.sum(neighbors[local_i] >= 0))

                for rank, raw_j in enumerate(neighbors[local_i]):
                    local_j = int(raw_j)
                    if local_j < 0 or local_j == local_i:
                        continue

                    # Manual cosine rescoring uses the original normalized embeddings.
                    score = float(np.dot(embeddings[global_i], embeddings[idx_list[local_j]]))
                    if score < threshold:
                        continue

                    global_j = idx_list[local_j]
                    key = canonical_pair(global_i, global_j)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    if df.iloc[key[0]]["author"] == df.iloc[key[1]]["author"]:
                        continue

                    pair_scores[key] = max(pair_scores.get(key, -1.0), score)

    runtime_seconds = float(time.perf_counter() - start_time)
    stats = {
        "num_tables": int(lsh_tables),
        "num_buckets": int(num_buckets),
        "num_bucket_rows": int(num_bucket_rows),
        "num_neighbors_returned": int(num_neighbors_returned),
        "num_pairs": int(len(pair_scores)),
        "bucket_size_min": int(min(bucket_sizes)) if bucket_sizes else 0,
        "bucket_size_max": int(max(bucket_sizes)) if bucket_sizes else 0,
        "bucket_size_mean": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
        "runtime_seconds": runtime_seconds,
    }
    return pair_scores, stats


def pair_scores_to_full_df(
    df: pd.DataFrame,
    exact_pairs: Dict[Tuple[int, int], float],
    topk_pairs: Dict[Tuple[int, int], float],
) -> pd.DataFrame:
    """Build a full comparison table with text and both scores for every observed pair."""
    rows: List[Dict[str, object]] = []
    all_keys = sorted(set(exact_pairs.keys()) | set(topk_pairs.keys()))

    # Full inspection CSV: every pair seen by either method gets one row.
    for a, b in all_keys:
        in_exact = (a, b) in exact_pairs
        in_topk = (a, b) in topk_pairs
        exact_score = exact_pairs[(a, b)] if in_exact else np.nan
        topk_score = topk_pairs[(a, b)] if in_topk else np.nan
        source = "overlap" if in_exact and in_topk else ("exact_only" if in_exact else "topk_only")
        rows.append(
            {
                "idx_a": a,
                "idx_b": b,
                "author_a": df.iloc[a]["author"],
                "author_b": df.iloc[b]["author"],
                "post_type_a": df.iloc[a]["post_type"],
                "post_type_b": df.iloc[b]["post_type"],
                "subreddit_a": df.iloc[a]["subreddit"],
                "subreddit_b": df.iloc[b]["subreddit"],
                "text_a": df.iloc[a]["text"],
                "text_b": df.iloc[b]["text"],
                "exact_score": exact_score,
                "topk_rescore_score": topk_score,
                "abs_diff": abs(exact_score - topk_score) if in_exact and in_topk else np.nan,
                "source": source,
            }
        )

    return pd.DataFrame(rows)


def compare_pair_sets(
    df: pd.DataFrame,
    exact_pairs: Dict[Tuple[int, int], float],
    topk_pairs: Dict[Tuple[int, int], float],
    max_error_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare exact threshold pairs against top-k rescored pairs per query."""
    query_to_exact: DefaultDict[int, List[Tuple[int, float]]] = defaultdict(list)
    query_to_topk: DefaultDict[int, List[Tuple[int, float]]] = defaultdict(list)

    # Build per-query lookup lists so we can compute recall and precision.
    for (a, b), score in exact_pairs.items():
        query_to_exact[a].append((b, score))
        query_to_exact[b].append((a, score))

    for (a, b), score in topk_pairs.items():
        query_to_topk[a].append((b, score))
        query_to_topk[b].append((a, score))

    per_query_rows: List[Dict] = []
    overlap_rows: List[Dict] = []
    fn_rows: List[Dict] = []
    fp_rows: List[Dict] = []

    for qi in range(len(df)):
        exact_map = {j: score for j, score in query_to_exact.get(qi, [])}
        topk_map = {j: score for j, score in query_to_topk.get(qi, [])}

        exact_set = set(exact_map.keys())
        topk_set = set(topk_map.keys())
        overlap = exact_set & topk_set
        missing = sorted(exact_set - topk_set)
        extra = sorted(topk_set - exact_set)

        exact_count = len(exact_set)
        topk_count = len(topk_set)
        overlap_count = len(overlap)
        recall = (overlap_count / exact_count) if exact_count > 0 else 1.0
        precision = (overlap_count / topk_count) if topk_count > 0 else (1.0 if exact_count == 0 else 0.0)

        # Score agreement should be essentially exact for overlap pairs.
        diffs: List[float] = []
        for j in overlap:
            exact_score = float(exact_map[j])
            topk_score = float(topk_map[j])
            diff = abs(exact_score - topk_score)
            diffs.append(diff)
            overlap_rows.append(
                {
                    "query_idx": qi,
                    "neighbor_idx": j,
                    "exact_score": exact_score,
                    "topk_rescore_score": topk_score,
                    "abs_diff": diff,
                    "query_text": df.iloc[qi]["text"],
                    "neighbor_text": df.iloc[j]["text"],
                }
            )

        per_query_rows.append(
            {
                "query_idx": qi,
                "exact_count": exact_count,
                "topk_count": topk_count,
                "overlap_count": overlap_count,
                "recall_vs_exact": recall,
                "precision_vs_exact": precision,
                "false_negative_count": len(missing),
                "false_positive_count": len(extra),
                "mean_abs_score_diff_overlap": float(np.mean(diffs)) if diffs else 0.0,
                "query_text": df.iloc[qi]["text"],
            }
        )

        # Collect bounded examples for manual inspection.
        if len(fn_rows) < max_error_samples:
            remaining = max_error_samples - len(fn_rows)
            for j in missing[:remaining]:
                fn_rows.append(
                    {
                        "query_idx": qi,
                        "neighbor_idx": j,
                        "exact_score": float(exact_map[j]),
                        "query_text": df.iloc[qi]["text"],
                        "neighbor_text": df.iloc[j]["text"],
                    }
                )

        if len(fp_rows) < max_error_samples:
            remaining = max_error_samples - len(fp_rows)
            for j in extra[:remaining]:
                fp_rows.append(
                    {
                        "query_idx": qi,
                        "neighbor_idx": j,
                        "topk_rescore_score": float(topk_map[j]),
                        "query_text": df.iloc[qi]["text"],
                        "neighbor_text": df.iloc[j]["text"],
                    }
                )

    return (
        pd.DataFrame(per_query_rows),
        pd.DataFrame(overlap_rows),
        pd.DataFrame(fn_rows),
        pd.DataFrame(fp_rows),
    )


def main() -> None:
    """Run the bucketed threshold comparison end-to-end."""
    args = parse_args()
    random.seed(args.seed)
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

    print("Collecting exact bucketed threshold pairs...")
    exact_pairs, exact_stats = collect_bucketed_exact_pairs(
        df=df,
        embeddings=embeddings,
        threshold=threshold,
        lsh_bits=args.lsh_bits,
        lsh_tables=args.lsh_tables,
        min_bucket_size=args.min_bucket_size,
        seed=args.seed,
    )

    print(f"Collecting bucketed top-k + rescore pairs (k={args.faiss_top_k})...")
    topk_pairs, topk_stats = collect_bucketed_topk_pairs(
        df=df,
        embeddings=embeddings,
        threshold=threshold,
        lsh_bits=args.lsh_bits,
        lsh_tables=args.lsh_tables,
        min_bucket_size=args.min_bucket_size,
        faiss_top_k=args.faiss_top_k,
        seed=args.seed,
    )

    print("Comparing pair sets...")
    per_query_df, overlap_df, fn_df, fp_df = compare_pair_sets(
        df=df,
        exact_pairs=exact_pairs,
        topk_pairs=topk_pairs,
        max_error_samples=int(args.max_error_samples),
    )

    print("Building full comparison table...")
    pairs_full_df = pair_scores_to_full_df(df=df, exact_pairs=exact_pairs, topk_pairs=topk_pairs)

    exact_stats_path = os.path.join(args.output_dir, "exact_bucket_stats.json")
    topk_stats_path = os.path.join(args.output_dir, "topk_bucket_stats.json")
    per_query_path = os.path.join(args.output_dir, "per_query_bucket_comparison.csv")
    overlap_path = os.path.join(args.output_dir, "overlap_score_differences.csv")
    fn_path = os.path.join(args.output_dir, "false_negative_examples.csv")
    fp_path = os.path.join(args.output_dir, "false_positive_examples.csv")
    pairs_full_path = os.path.join(args.output_dir, "bucket_pair_comparison.csv")

    per_query_df.to_csv(per_query_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    fp_df.to_csv(fp_path, index=False)
    pairs_full_df.to_csv(pairs_full_path, index=False)

    with open(exact_stats_path, "w", encoding="utf-8") as handle:
        json.dump(exact_stats, handle, indent=2)
    with open(topk_stats_path, "w", encoding="utf-8") as handle:
        json.dump(topk_stats, handle, indent=2)

    # Optional: run IndexLSH + re-score single run for direct comparison
    if args.use_indexlsh:
        print(f"Running IndexLSH + re-score (nbits={args.lsh_bits}, ntables={args.lsh_tables}, k={args.faiss_top_k})...")
        indexlsh_pairs, indexlsh_stats = collect_indexlsh_rescored_pairs(
            df=df,
            embeddings=embeddings,
            threshold=threshold,
            nbits=args.lsh_bits,
            ntables=args.lsh_tables,
            faiss_top_k=args.faiss_top_k,
            seed=args.seed,
        )

        # compare to exact baseline
        idxlsh_per_query_df, idxlsh_overlap_df, idxlsh_fn_df, idxlsh_fp_df = compare_pair_sets(
            df=df, exact_pairs=exact_pairs, topk_pairs=indexlsh_pairs, max_error_samples=int(args.max_error_samples)
        )

        indexlsh_pairs_full = pair_scores_to_full_df(df=df, exact_pairs=exact_pairs, topk_pairs=indexlsh_pairs)
        indexlsh_pairs_path = os.path.join(args.output_dir, "indexlsh_bucket_pair_comparison.csv")
        indexlsh_pairs_full.to_csv(indexlsh_pairs_path, index=False)

        indexlsh_stats_path = os.path.join(args.output_dir, "indexlsh_stats.json")
        with open(indexlsh_stats_path, "w", encoding="utf-8") as handle:
            json.dump(indexlsh_stats, handle, indent=2)

        idxlsh_per_query_path = os.path.join(args.output_dir, "indexlsh_per_query_comparison.csv")
        idxlsh_overlap_path = os.path.join(args.output_dir, "indexlsh_overlap_score_differences.csv")
        idxlsh_fn_path = os.path.join(args.output_dir, "indexlsh_false_negative_examples.csv")
        idxlsh_fp_path = os.path.join(args.output_dir, "indexlsh_false_positive_examples.csv")

        idxlsh_per_query_df.to_csv(idxlsh_per_query_path, index=False)
        idxlsh_overlap_df.to_csv(idxlsh_overlap_path, index=False)
        idxlsh_fn_df.to_csv(idxlsh_fn_path, index=False)
        idxlsh_fp_df.to_csv(idxlsh_fp_path, index=False)

        print(f"Saved IndexLSH comparison CSV: {indexlsh_pairs_path}")
        print(f"Saved IndexLSH stats: {indexlsh_stats_path}")

    # Optional: run parameter sweep over nbits / ntables / k
    if args.sweep:
        sweep_nbits = parse_int_list(args.sweep_nbits)
        sweep_ntables = parse_int_list(args.sweep_ntables)
        sweep_ks = parse_int_list(args.sweep_k_values)
        print(f"Running parameter sweep nbits={sweep_nbits} ntables={sweep_ntables} k={sweep_ks} ...")
        sweep_df = parameter_sweep(
            df=df,
            embeddings=embeddings,
            threshold=threshold,
            sweep_nbits=sweep_nbits,
            sweep_ntables=sweep_ntables,
            sweep_ks=sweep_ks,
            seed=args.seed,
            base_lsh_bits=args.lsh_bits,
            base_lsh_tables=args.lsh_tables,
            min_bucket_size=args.min_bucket_size,
        )
        sweep_path = os.path.join(args.output_dir, "indexlsh_parameter_sweep.csv")
        sweep_df.to_csv(sweep_path, index=False)
        print(f"Saved sweep results: {sweep_path}")

    total_exact = int(per_query_df["exact_count"].sum()) if not per_query_df.empty else 0
    total_topk = int(per_query_df["topk_count"].sum()) if not per_query_df.empty else 0
    total_overlap = int(per_query_df["overlap_count"].sum()) if not per_query_df.empty else 0

    summary = {
        "input_file": args.input_file,
        "n_records": int(len(df)),
        "sample_size": int(len(df)),
        "threshold": threshold,
        "random_pairs": int(args.random_pairs),
        "quantile": float(args.quantile),
        "lsh_bits": int(args.lsh_bits),
        "lsh_tables": int(args.lsh_tables),
        "faiss_top_k": int(args.faiss_top_k),
        "min_bucket_size": int(args.min_bucket_size),
        "kde_plot_saved": bool(kde_saved),
        "kde_plot_message": kde_msg,
        "exact_bucket_stats_file": exact_stats_path,
        "topk_bucket_stats_file": topk_stats_path,
        "total_exact_threshold_pairs": total_exact,
        "total_topk_threshold_pairs": total_topk,
        "total_overlap": total_overlap,
        "micro_recall_vs_exact": (total_overlap / total_exact) if total_exact > 0 else 1.0,
        "micro_precision_vs_exact": (total_overlap / total_topk) if total_topk > 0 else 1.0,
        "mean_recall_vs_exact": float(per_query_df["recall_vs_exact"].mean()) if not per_query_df.empty else 1.0,
        "mean_precision_vs_exact": float(per_query_df["precision_vs_exact"].mean()) if not per_query_df.empty else 1.0,
        "mean_abs_score_diff_overlap": float(overlap_df["abs_diff"].mean()) if not overlap_df.empty else 0.0,
        "outputs": {
            "random_similarity_samples": random_sims_path,
            "random_similarity_kde": kde_path,
            "bucket_pair_comparison": pairs_full_path,
            "per_query_bucket_comparison": per_query_path,
            "overlap_score_differences": overlap_path,
            "false_negative_examples": fn_path,
            "false_positive_examples": fp_path,
        },
    }

    summary_path = os.path.join(args.output_dir, "bucket_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Threshold (q={args.quantile:.4f}): {threshold:.6f}")
    print(f"Saved full comparison CSV: {pairs_full_path}")
    print(f"Saved per-query CSV: {per_query_path}")
    print(f"Saved overlap score CSV: {overlap_path}")
    print(f"Saved false negatives: {fn_path}")
    print(f"Saved false positives: {fp_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
