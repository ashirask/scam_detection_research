import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer


# Sampled JSONL is the expected starting point after your sampling stage.
DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")
# Placeholder content/authors that should not be treated as real authored posts.
SKIP_TEXT = {"[removed]", "[deleted]"}
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_args() -> argparse.Namespace:
    """Parse CLI options for semantic suspicious-account detection.

    This version replaces TF-IDF matching with multilingual semantic embeddings,
    then combines random-hyperplane LSH with FAISS nearest-neighbor search.

    Returns:
        argparse.Namespace containing all runtime options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Detect suspicious Reddit accounts from sampled JSONL using multilingual "
            "semantic embeddings + LSH + FAISS."
        )
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="Path to sampled JSONL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for detection outputs.")
    parser.add_argument("--min-words", type=int, default=10, help="Minimum words required in a post.")
    parser.add_argument(
        "--embedding-model",
        default="distiluse-base-multilingual-cased-v2",
        help="SentenceTransformer model name for semantic embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--random-pairs",
        type=int,
        default=50000,
        help="How many random post pairs to sample for quantile threshold estimation.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.995,
        help="Quantile used as cosine similarity cutoff for suspicious pairs.",
    )
    parser.add_argument(
        "--lsh-bits",
        type=int,
        default=16,
        help="Number of random-hyperplane bits per LSH table.",
    )
    parser.add_argument(
        "--lsh-tables",
        type=int,
        default=4,
        help="Number of independent LSH tables. More tables increases recall but costs more time.",
    )
    parser.add_argument(
        "--faiss-top-k",
        type=int,
        default=40,
        help="Top-k neighbors to retrieve per point within each LSH bucket.",
    )
    parser.add_argument(
        "--min-bucket-size",
        type=int,
        default=2,
        help="Minimum bucket size to run FAISS search in that bucket.",
    )
    parser.add_argument(
        "--min-cross-account-matches",
        type=int,
        default=1,
        help="Minimum number of cross-account high-similarity matches to flag an account.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace for stable text normalization before embedding."""
    return " ".join(text.split())


def infer_post_type(record: Dict) -> str:
    """Infer comment vs submission from available fields.

    Scenario:
    - If body exists, treat as comment.
    - Otherwise treat as submission (title/selftext-based).
    """
    body = (record.get("body") or "").strip()
    if body:
        return "comment"
    return "submission"


def build_text(record: Dict, post_type: str) -> str:
    """Build normalized text for a record based on inferred post type.

    Scenario:
    - comment: uses body only
    - submission: combines title + selftext
    """
    if post_type == "comment":
        return normalize_text((record.get("body") or "").strip())
    title = normalize_text((record.get("title") or "").strip())
    selftext = normalize_text((record.get("selftext") or "").strip())
    combined = " ".join(part for part in [title, selftext] if part)
    return normalize_text(combined)


def load_records(path: str, min_words: int) -> pd.DataFrame:
    """Load and clean sampled JSONL into a DataFrame.

    Important filtering rules:
    - drops malformed JSON
    - drops deleted/automated authors
    - keeps only posts with at least min_words words

    Args:
        path: Input JSONL path.
        min_words: Minimum tokenized word count required.

    Returns:
        DataFrame containing normalized rows for downstream matching.
    """
    rows: List[Dict] = []

    with open(path, encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip broken lines instead of failing the entire run.
                continue

            author = (record.get("author") or "").strip()
            if not author or author in SKIP_AUTHORS:
                continue

            post_type = infer_post_type(record)
            text = build_text(record, post_type)
            if not text or text in SKIP_TEXT:
                continue

            # Word count filters low-information short posts that create noisy matches.
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
    """Generate normalized semantic embeddings for all texts.

    Args:
        texts: Input list of normalized texts.
        model_name: SentenceTransformer model id.
        batch_size: Encoding batch size.

    Returns:
        Float32 matrix of L2-normalized embeddings with shape [n_texts, dim].
    """
    model = SentenceTransformer(model_name)

    # normalize_embeddings=True gives unit vectors, so dot product == cosine similarity.
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # FAISS expects float32 contiguous arrays for best performance.
    return np.asarray(emb, dtype=np.float32)


def sample_random_pair_similarities(
    embeddings: np.ndarray,
    n_pairs: int,
    seed: int,
) -> np.ndarray:
    """Estimate background cosine similarity by random post-pair sampling.

    We sample random pairs from normalized embedding vectors and compute dot products.
    The chosen quantile acts as a dynamic threshold that adapts to each corpus.

    Args:
        embeddings: Normalized embedding matrix [n, d].
        n_pairs: Number of random pairs to sample.
        seed: RNG seed for reproducibility.

    Returns:
        1D numpy array of sampled cosine similarities.
    """
    n_rows = embeddings.shape[0]
    if n_rows < 2:
        return np.array([], dtype=np.float32)

    rng = np.random.default_rng(seed)
    sims = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        a, b = rng.choice(n_rows, size=2, replace=False)
        sims[i] = float(np.dot(embeddings[a], embeddings[b]))

    return sims


def plot_similarity_distribution(
    similarities: np.ndarray,
    output_file: str,
    threshold: float | None = None,
    quantile: float | None = None,
) -> None:
    """Save a KDE plot and optionally draw the chosen similarity cutoff."""
    plt.figure(figsize=(9, 6))
    sns.kdeplot(similarities, fill=True, color="#1f77b4")

    if threshold is not None:
        label = f"q={quantile:.3f} threshold={threshold:.4f}" if quantile is not None else f"threshold={threshold:.4f}"
        plt.axvline(
            x=threshold,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label=label,
        )

    plt.title("KDE of Random Pair Semantic Cosine Similarity")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    if threshold is not None:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def _build_lsh_signatures(
    embeddings: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    """Generate integer signatures for one random-hyperplane LSH table.

    Args:
        embeddings: Normalized embedding matrix [n, d].
        projection_matrix: Random hyperplanes [d, bits].

    Returns:
        Integer signature array of shape [n], where equal values fall in same bucket.
    """
    # Each projected sign bit indicates which side of a hyperplane the point falls on.
    bit_matrix = (embeddings @ projection_matrix) >= 0.0

    # Pack boolean bits to a single integer signature for efficient bucket grouping.
    bit_weights = (1 << np.arange(bit_matrix.shape[1], dtype=np.uint64))
    return (bit_matrix.astype(np.uint64) * bit_weights).sum(axis=1)


def find_high_similarity_pairs_lsh_faiss(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    lsh_bits: int,
    lsh_tables: int,
    faiss_top_k: int,
    min_bucket_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Find cross-account high-similarity pairs using LSH-pruned FAISS search.

    Why this design:
    - LSH reduces global O(n^2) pair checks by hashing semantically similar vectors
      into shared candidate buckets.
    - FAISS performs fast nearest-neighbor retrieval inside each bucket.
    - Multi-table LSH improves recall by allowing multiple independent bucketings.

    Args:
        df: Cleaned posts dataframe.
        embeddings: Normalized embedding matrix [n, d].
        threshold: Cosine similarity threshold for suspicious pair selection.
        lsh_bits: Number of random hyperplane bits per table.
        lsh_tables: Number of independent LSH tables.
        faiss_top_k: Neighbor count per point inside each bucket.
        min_bucket_size: Minimum bucket size to evaluate.
        seed: RNG seed.

    Returns:
        Tuple:
        - DataFrame of suspicious cross-account pairs sorted by similarity desc.
        - Dictionary with candidate search diagnostics.
    """
    if df.empty:
        return pd.DataFrame(), {"num_buckets": 0, "num_pairs": 0, "num_candidates": 0}

    rng = np.random.default_rng(seed)
    dim = embeddings.shape[1]

    seen_pairs: set[Tuple[int, int]] = set()
    rows: List[Dict] = []

    num_buckets = 0
    num_candidates = 0

    for _ in range(lsh_tables):
        # Independent random hyperplanes per table increase chance of co-bucketing true neighbors.
        projection_matrix = rng.standard_normal(size=(dim, lsh_bits), dtype=np.float32)
        signatures = _build_lsh_signatures(embeddings, projection_matrix)

        buckets: DefaultDict[int, List[int]] = defaultdict(list)
        for idx, signature in enumerate(signatures):
            buckets[int(signature)].append(idx)

        for idx_list in buckets.values():
            if len(idx_list) < min_bucket_size:
                continue
            num_buckets += 1

            # FAISS IndexFlatIP is exact inner-product search.
            # Because vectors are normalized, this equals cosine similarity search.
            bucket_vectors = np.ascontiguousarray(embeddings[idx_list], dtype=np.float32)
            index = faiss.IndexFlatIP(dim)
            index.add(bucket_vectors)

            k = min(faiss_top_k + 1, len(idx_list))
            sims, neighbors = index.search(bucket_vectors, k)

            for local_i, global_i in enumerate(idx_list):
                for local_j, sim in zip(neighbors[local_i], sims[local_i]):
                    if local_j < 0:
                        continue
                    global_j = idx_list[int(local_j)]
                    if global_i == global_j:
                        continue
                    num_candidates += 1

                    if float(sim) < threshold:
                        continue

                    a, b = (global_i, global_j) if global_i < global_j else (global_j, global_i)
                    if (a, b) in seen_pairs:
                        continue
                    seen_pairs.add((a, b))

                    # Only cross-account similarity is useful for coordinated-behavior detection.
                    if df.iloc[a]["author"] == df.iloc[b]["author"]:
                        continue

                    rows.append(
                        {
                            "idx_a": a,
                            "idx_b": b,
                            "author_a": df.iloc[a]["author"],
                            "author_b": df.iloc[b]["author"],
                            "post_type_a": df.iloc[a]["post_type"],
                            "post_type_b": df.iloc[b]["post_type"],
                            "similarity": round(float(sim), 6),
                            "subreddit_a": df.iloc[a]["subreddit"],
                            "subreddit_b": df.iloc[b]["subreddit"],
                            "text_a": df.iloc[a]["text"][:250],
                            "text_b": df.iloc[b]["text"][:250],
                        }
                    )

    if not rows:
        return pd.DataFrame(), {
            "num_buckets": num_buckets,
            "num_pairs": 0,
            "num_candidates": num_candidates,
        }

    out = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    return out, {
        "num_buckets": num_buckets,
        "num_pairs": int(len(out)),
        "num_candidates": num_candidates,
    }


def build_suspicious_accounts(pairs: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Aggregate pair-level signals to account-level suspiciousness."""
    if pairs.empty:
        return pd.DataFrame(columns=["author", "cross_account_matches", "max_similarity", "avg_similarity"])

    stats: Dict[str, List[float]] = {}

    for _, row in pairs.iterrows():
        stats.setdefault(row["author_a"], []).append(float(row["similarity"]))
        stats.setdefault(row["author_b"], []).append(float(row["similarity"]))

    out_rows = []
    for author, sims in stats.items():
        # Require at least N suspicious links before flagging an account.
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

    return pd.DataFrame(out_rows).sort_values(
        ["cross_account_matches", "avg_similarity"], ascending=[False, False]
    )


def save_suspicious_posts(df: pd.DataFrame, suspicious_authors: set[str], output_file: str) -> int:
    """Write only suspicious authors' original JSON rows to a JSONL file.

    This file is the handoff into clustering/topic steps.
    """
    count = 0
    with open(output_file, "w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            if row["author"] not in suspicious_authors:
                continue
            handle.write(json.dumps(row["raw"], ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    """Run full suspicious-account detection pipeline from sampled JSONL.

    Pipeline summary:
    1) load + clean posts
    2) compute multilingual semantic embeddings
    3) estimate random-pair semantic quantiles
    4) LSH-pruned FAISS search for high-similarity cross-account pairs
    5) aggregate suspicious accounts and export artifacts
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading sampled records from: {args.input_file}")
    df = load_records(args.input_file, min_words=args.min_words)
    if df.empty:
        raise SystemExit("No usable posts found after filtering.")

    print(f"Usable posts: {len(df)}")
    print(f"Unique authors: {df['author'].nunique()}")

    print(f"Computing semantic embeddings with {args.embedding_model}...")
    embeddings = compute_embeddings(df["text"].tolist(), args.embedding_model, args.batch_size)

    print("Sampling random pair similarities for quantile estimation...")
    sampled_sims = sample_random_pair_similarities(embeddings, n_pairs=args.random_pairs, seed=args.seed)
    if sampled_sims.size == 0:
        raise SystemExit("Not enough posts to compute random similarity quantiles.")

    # Report reference quantiles to make threshold choice transparent.
    quantiles_to_report = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
    quantile_values = np.quantile(sampled_sims, quantiles_to_report)

    print("Random-pair semantic cosine similarity quantiles:")
    for q, v in zip(quantiles_to_report, quantile_values):
        print(f"  q={q:.3f} -> {v:.6f}")

    # Core decision boundary: pairs above this are unusually similar vs random baseline.
    similarity_threshold = float(np.quantile(sampled_sims, args.quantile))
    print(f"Using q={args.quantile:.3f} threshold: cosine >= {similarity_threshold:.6f}")

    kde_path = os.path.join(args.output_dir, "random_similarity_kde.png")
    plot_similarity_distribution(
        sampled_sims,
        kde_path,
        threshold=similarity_threshold,
        quantile=args.quantile,
    )
    print(f"Saved KDE plot: {kde_path}")

    print("Finding high-similarity cross-account pairs with LSH + FAISS...")
    suspicious_pairs, lsh_stats = find_high_similarity_pairs_lsh_faiss(
        df=df,
        embeddings=embeddings,
        threshold=similarity_threshold,
        lsh_bits=args.lsh_bits,
        lsh_tables=args.lsh_tables,
        faiss_top_k=args.faiss_top_k,
        min_bucket_size=args.min_bucket_size,
        seed=args.seed,
    )

    pairs_path = os.path.join(args.output_dir, "suspicious_pairs.csv")
    suspicious_pairs.to_csv(pairs_path, index=False)
    print(f"Saved suspicious pairs: {pairs_path} ({len(suspicious_pairs)} rows)")

    suspicious_accounts = build_suspicious_accounts(
        suspicious_pairs, min_matches=args.min_cross_account_matches
    )
    accounts_path = os.path.join(args.output_dir, "suspicious_accounts.csv")
    suspicious_accounts.to_csv(accounts_path, index=False)
    print(f"Saved suspicious accounts: {accounts_path} ({len(suspicious_accounts)} accounts)")

    suspicious_authors = set(suspicious_accounts["author"].tolist()) if not suspicious_accounts.empty else set()
    suspicious_jsonl_path = os.path.join(args.output_dir, "suspicious_posts.jsonl")
    kept = save_suspicious_posts(df, suspicious_authors, suspicious_jsonl_path)
    print(f"Saved suspicious posts JSONL: {suspicious_jsonl_path} ({kept} rows)")

    # JSON summary is useful for tracking model/matching settings over time.
    summary = {
        "input_file": args.input_file,
        "embedding_model": args.embedding_model,
        "usable_posts": int(len(df)),
        "usable_authors": int(df["author"].nunique()),
        "random_pairs_sampled": int(sampled_sims.size),
        "quantile": float(args.quantile),
        "similarity_threshold": similarity_threshold,
        "lsh_bits": int(args.lsh_bits),
        "lsh_tables": int(args.lsh_tables),
        "faiss_top_k": int(args.faiss_top_k),
        "lsh_stats": lsh_stats,
        "suspicious_pairs": int(len(suspicious_pairs)),
        "suspicious_accounts": int(len(suspicious_accounts)),
    }
    summary_path = os.path.join(args.output_dir, "detection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
