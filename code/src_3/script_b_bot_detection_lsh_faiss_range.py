import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer


DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output" / "script_b_lsh_faiss")
SKIP_TEXT = {"[removed]", "[deleted]"}
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Script B (large-scale JSONL pipeline).

    Script B keeps the production flow but uses LSH bucket pruning, then runs
    FAISS range_search inside each bucket to reduce candidate growth vs global all-pairs.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Script B: production-style detection using LSH buckets + FAISS range_search "
            "inside each bucket for runtime reduction on large JSONL data."
        )
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="Input sampled JSONL")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--min-words", type=int, default=10, help="Minimum words required per post")
    parser.add_argument(
        "--embedding-model",
        default="distiluse-base-multilingual-cased-v2",
        help="SentenceTransformer model",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--random-pairs", type=int, default=50000, help="Random pairs for quantile threshold")
    parser.add_argument("--quantile", type=float, default=0.99, help="Quantile for threshold")
    parser.add_argument(
        "--faiss-range-threshold",
        type=float,
        default=None,
        help="Optional override threshold for FAISS range_search",
    )
    parser.add_argument("--lsh-bits", type=int, default=16, help="Number of random-hyperplane bits")
    parser.add_argument("--lsh-tables", type=int, default=4, help="Number of independent LSH tables")
    parser.add_argument("--min-bucket-size", type=int, default=2, help="Minimum bucket size to process")
    parser.add_argument(
        "--min-cross-account-matches",
        type=int,
        default=1,
        help="Minimum suspicious links to flag an account",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize whitespace for stable downstream matching/embedding."""
    return " ".join(text.split())


def infer_post_type(record: Dict) -> str:
    """Infer Reddit record type from available fields.

    If body exists -> comment.
    Else -> submission.
    """
    body = (record.get("body") or "").strip()
    if body:
        return "comment"
    return "submission"


def build_text(record: Dict, post_type: str) -> str:
    """Build normalized text content based on inferred post type."""
    if post_type == "comment":
        return normalize_text((record.get("body") or "").strip())

    # Submissions combine title + selftext before final normalization.
    title = normalize_text((record.get("title") or "").strip())
    selftext = normalize_text((record.get("selftext") or "").strip())
    combined = " ".join(part for part in [title, selftext] if part)
    return normalize_text(combined)


def load_records(path: str, min_words: int) -> pd.DataFrame:
    """Load and filter sampled JSONL records into a modeling DataFrame.

    Filtering includes:
    - malformed JSON lines
    - deleted/automated authors
    - deleted/removed text
    - short posts below min_words
    """
    rows: List[Dict] = []

    # Stream line-by-line to avoid loading full file into memory.
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

            # Word-count guard reduces noisy short snippets.
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
                    "id": record.get("id") or record.get("name") or "row_{}".format(line_num),
                    "raw": record,
                }
            )

    return pd.DataFrame(rows)


def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    """Encode texts to normalized float32 embeddings.

    normalize_embeddings=True makes cosine similarity equal dot product.
    """
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # FAISS expects float32 arrays for efficient search.
    return np.asarray(emb, dtype=np.float32)


def sample_random_pair_similarities(embeddings: np.ndarray, n_pairs: int, seed: int) -> np.ndarray:
    """Estimate baseline similarity distribution from random embedding pairs."""
    n_rows = embeddings.shape[0]
    if n_rows < 2:
        return np.array([], dtype=np.float32)

    rng = np.random.default_rng(seed)
    sims = np.zeros(n_pairs, dtype=np.float32)
    # Draw n_pairs random index pairs and compute cosine-equivalent dot products.
    for i in range(n_pairs):
        a, b = rng.choice(n_rows, size=2, replace=False)
        sims[i] = float(np.dot(embeddings[a], embeddings[b]))
    return sims


def plot_similarity_distribution(
    similarities: np.ndarray,
    output_file: str,
    threshold: Optional[float] = None,
    quantile: Optional[float] = None,
) -> None:
    """Plot KDE of random-pair similarities and optional threshold marker."""
    plt.figure(figsize=(9, 6))
    sns.kdeplot(similarities, fill=True, color="#1f77b4")

    if threshold is not None:
        # Include quantile value in legend text for traceability.
        label = "q={:.3f} threshold={:.4f}".format(quantile, threshold) if quantile is not None else "threshold={:.4f}".format(threshold)
        plt.axvline(x=threshold, color="#d62728", linestyle="--", linewidth=2, label=label)

    plt.title("KDE of Random Pair Semantic Cosine Similarity")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    if threshold is not None:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def _build_lsh_signatures(embeddings: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Build integer LSH signatures from random-hyperplane sign bits.

    Steps:
    1) project vectors onto random hyperplanes
    2) convert signs to bits
    3) pack bits into one integer per vector
    """
    bit_matrix = (embeddings @ projection_matrix) >= 0.0
    bit_weights = (1 << np.arange(bit_matrix.shape[1], dtype=np.uint64))
    return (bit_matrix.astype(np.uint64) * bit_weights).sum(axis=1)


def find_high_similarity_pairs_lsh_faiss_range(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    lsh_bits: int,
    lsh_tables: int,
    min_bucket_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Find high-similarity cross-account pairs using LSH bucket pruning + FAISS.

    Core idea:
    - LSH narrows candidate sets into buckets
    - FAISS range_search runs only inside each bucket
    - duplicate pairs across tables/buckets are deduplicated globally
    """
    if df.empty:
        return pd.DataFrame(), {
            "num_tables": 0,
            "num_buckets": 0,
            "num_bucket_rows": 0,
            "num_neighbors_returned": 0,
            "num_cross_account_pairs": 0,
            "bucket_size_min": 0,
            "bucket_size_max": 0,
            "bucket_size_mean": 0.0,
            "runtime_seconds": 0.0,
        }

    start_time = time.perf_counter()

    rng = np.random.default_rng(seed)
    dim = embeddings.shape[1]
    seen_pairs: Set[Tuple[int, int]] = set()
    rows: List[Dict] = []

    num_buckets = 0
    num_bucket_rows = 0
    num_neighbors_returned = 0
    bucket_sizes: List[int] = []

    # Repeat with independent LSH tables to improve neighbor recall.
    for _ in range(lsh_tables):
        projection_matrix = rng.standard_normal(size=(dim, lsh_bits), dtype=np.float32)
        signatures = _build_lsh_signatures(embeddings, projection_matrix)

        buckets: DefaultDict[int, List[int]] = defaultdict(list)
        # Group vector indices by integer signature (same bucket = same candidate set).
        for idx, signature in enumerate(signatures):
            buckets[int(signature)].append(idx)

        # Process each bucket independently.
        for idx_list in buckets.values():
            bsize = len(idx_list)
            # Tiny buckets don't produce meaningful cross-item matches.
            if bsize < min_bucket_size:
                continue

            num_buckets += 1
            num_bucket_rows += bsize
            bucket_sizes.append(bsize)

            # Build FAISS index for this bucket only.
            bucket_vectors = np.ascontiguousarray(embeddings[idx_list], dtype=np.float32)
            index = faiss.IndexFlatIP(dim)
            index.add(bucket_vectors)

            # Return all bucket neighbors above threshold.
            lims, scores, neighbors = index.range_search(bucket_vectors, float(threshold))

            # Walk each query vector in the bucket.
            for local_i, global_i in enumerate(idx_list):
                start = int(lims[local_i])
                end = int(lims[local_i + 1])
                num_neighbors_returned += (end - start)

                # Walk neighbors for this query.
                for pos in range(start, end):
                    local_j = int(neighbors[pos])
                    sim = float(scores[pos])
                    # Skip self-match.
                    if local_j == local_i:
                        continue

                    global_j = idx_list[local_j]
                    # Canonical unordered key deduplicates mirrored pairs and cross-table repeats.
                    a, b = (global_i, global_j) if global_i < global_j else (global_j, global_i)
                    if (a, b) in seen_pairs:
                        continue
                    seen_pairs.add((a, b))

                    # Keep only cross-account pairs for coordinated-behavior detection.
                    author_a = df.iloc[a]["author"]
                    author_b = df.iloc[b]["author"]
                    if author_a == author_b:
                        continue

                    rows.append(
                        {
                            "idx_a": a,
                            "idx_b": b,
                            "author_a": author_a,
                            "author_b": author_b,
                            "post_type_a": df.iloc[a]["post_type"],
                            "post_type_b": df.iloc[b]["post_type"],
                            "similarity": round(sim, 6),
                            "subreddit_a": df.iloc[a]["subreddit"],
                            "subreddit_b": df.iloc[b]["subreddit"],
                            "text_a": df.iloc[a]["text"][:250],
                            "text_b": df.iloc[b]["text"][:250],
                        }
                    )

    runtime_seconds = float(time.perf_counter() - start_time)

    if not rows:
        return pd.DataFrame(), {
            "num_tables": int(lsh_tables),
            "num_buckets": int(num_buckets),
            "num_bucket_rows": int(num_bucket_rows),
            "num_neighbors_returned": int(num_neighbors_returned),
            "num_cross_account_pairs": 0,
            "bucket_size_min": int(min(bucket_sizes)) if bucket_sizes else 0,
            "bucket_size_max": int(max(bucket_sizes)) if bucket_sizes else 0,
            "bucket_size_mean": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
            "runtime_seconds": runtime_seconds,
        }

    out = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    return out, {
        "num_tables": int(lsh_tables),
        "num_buckets": int(num_buckets),
        "num_bucket_rows": int(num_bucket_rows),
        "num_neighbors_returned": int(num_neighbors_returned),
        "num_cross_account_pairs": int(len(out)),
        "bucket_size_min": int(min(bucket_sizes)) if bucket_sizes else 0,
        "bucket_size_max": int(max(bucket_sizes)) if bucket_sizes else 0,
        "bucket_size_mean": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
        "runtime_seconds": runtime_seconds,
    }


def build_suspicious_accounts(pairs: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Aggregate pair-level evidence into account-level suspiciousness metrics."""
    if pairs.empty:
        return pd.DataFrame(columns=["author", "cross_account_matches", "max_similarity", "avg_similarity"])

    stats: Dict[str, List[float]] = {}
    # Collect all pair similarity scores for each author appearance.
    for _, row in pairs.iterrows():
        stats.setdefault(row["author_a"], []).append(float(row["similarity"]))
        stats.setdefault(row["author_b"], []).append(float(row["similarity"]))

    out_rows: List[Dict[str, object]] = []
    # Summarize count/max/mean per author and apply minimum-link filter.
    for author, sims in stats.items():
        if len(sims) < min_matches:
            continue
        out_rows.append(
            {
                "author": author,
                "cross_account_matches": len(sims),
                "max_similarity": round(float(np.max(sims)), 6),
                "avg_similarity": round(float(np.mean(sims)), 6),
            }
        )

    if not out_rows:
        return pd.DataFrame(columns=["author", "cross_account_matches", "max_similarity", "avg_similarity"])

    return pd.DataFrame(out_rows).sort_values(["cross_account_matches", "avg_similarity"], ascending=[False, False])


def save_suspicious_posts(df: pd.DataFrame, suspicious_authors: Set[str], output_file: str) -> int:
    """Write original JSON rows for suspicious authors to JSONL output."""
    count = 0
    with open(output_file, "w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            # Skip non-flagged authors to keep handoff file focused.
            if row["author"] not in suspicious_authors:
                continue
            handle.write(json.dumps(row["raw"], ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    """Run Script B end-to-end and emit production-style artifacts + diagnostics."""
    args = parse_args()
    # Seed Python and NumPy RNGs for reproducibility in sampling and LSH generation.
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading sampled records from: {}".format(args.input_file))
    df = load_records(args.input_file, min_words=args.min_words)
    if df.empty:
        raise SystemExit("No usable posts found after filtering.")

    print("Usable posts: {}".format(len(df)))
    print("Unique authors: {}".format(df["author"].nunique()))

    print("Computing semantic embeddings with {}...".format(args.embedding_model))
    embeddings = compute_embeddings(df["text"].tolist(), args.embedding_model, args.batch_size)

    print("Sampling random pair similarities for quantile estimation...")
    sampled_sims = sample_random_pair_similarities(embeddings, n_pairs=args.random_pairs, seed=args.seed)
    if sampled_sims.size == 0:
        raise SystemExit("Not enough posts to compute random similarity quantiles.")

    # Report multiple quantiles so threshold calibration is transparent.
    quantiles_to_report = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
    quantile_values = np.quantile(sampled_sims, quantiles_to_report)
    print("Random-pair semantic cosine similarity quantiles:")
    for q, v in zip(quantiles_to_report, quantile_values):
        print("  q={:.3f} -> {:.6f}".format(q, v))

    # Default threshold is quantile-based; optionally overridden by explicit CLI value.
    similarity_threshold = float(np.quantile(sampled_sims, args.quantile))
    print("Using q={:.3f} threshold: cosine >= {:.6f}".format(args.quantile, similarity_threshold))

    if args.faiss_range_threshold is not None:
        similarity_threshold = float(args.faiss_range_threshold)
        print("Overriding threshold to: {:.6f}".format(similarity_threshold))

    kde_path = os.path.join(args.output_dir, "random_similarity_kde.png")
    plot_similarity_distribution(sampled_sims, kde_path, threshold=similarity_threshold, quantile=args.quantile)
    print("Saved KDE plot: {}".format(kde_path))

    print("Finding high-similarity cross-account pairs with LSH + FAISS range_search...")
    suspicious_pairs, lsh_stats = find_high_similarity_pairs_lsh_faiss_range(
        df=df,
        embeddings=embeddings,
        threshold=similarity_threshold,
        lsh_bits=args.lsh_bits,
        lsh_tables=args.lsh_tables,
        min_bucket_size=args.min_bucket_size,
        seed=args.seed,
    )

    pairs_path = os.path.join(args.output_dir, "suspicious_pairs_range_lsh.csv")
    suspicious_pairs.to_csv(pairs_path, index=False)
    print("Saved suspicious pairs: {} ({} rows)".format(pairs_path, len(suspicious_pairs)))

    # Aggregate from pair-level to author-level suspiciousness for downstream triage.
    suspicious_accounts = build_suspicious_accounts(suspicious_pairs, min_matches=args.min_cross_account_matches)
    accounts_path = os.path.join(args.output_dir, "suspicious_accounts_range_lsh.csv")
    suspicious_accounts.to_csv(accounts_path, index=False)
    print("Saved suspicious accounts: {} ({} accounts)".format(accounts_path, len(suspicious_accounts)))

    suspicious_authors = set(suspicious_accounts["author"].tolist()) if not suspicious_accounts.empty else set()
    suspicious_jsonl_path = os.path.join(args.output_dir, "suspicious_posts_range_lsh.jsonl")
    kept = save_suspicious_posts(df, suspicious_authors, suspicious_jsonl_path)
    print("Saved suspicious posts JSONL: {} ({} rows)".format(suspicious_jsonl_path, kept))

    summary = {
        "input_file": args.input_file,
        "embedding_model": args.embedding_model,
        "detection_method": "lsh_bucketed_faiss_range_search",
        "usable_posts": int(len(df)),
        "usable_authors": int(df["author"].nunique()),
        "random_pairs_sampled": int(sampled_sims.size),
        "quantile": float(args.quantile),
        "similarity_threshold": float(similarity_threshold),
        "lsh_bits": int(args.lsh_bits),
        "lsh_tables": int(args.lsh_tables),
        "min_bucket_size": int(args.min_bucket_size),
        "lsh_stats": lsh_stats,
        "suspicious_pairs": int(len(suspicious_pairs)),
        "suspicious_accounts": int(len(suspicious_accounts)),
    }
    summary_path = os.path.join(args.output_dir, "detection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Saved summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
