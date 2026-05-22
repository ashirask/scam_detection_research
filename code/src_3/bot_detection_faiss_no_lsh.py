"""

***SELECTED DRAFT***

Bot detection using global FAISS range search (no LSH bucketing). 

This script replaces the LSH-based candidate generation in src_2 with a simpler,
exact-match approach: build one global FAISS IndexFlatIP over all embeddings,
then use range_search to find all neighbors above the threshold.

Why this design:
- Eliminates LSH approximation errors that can miss true neighbors or create false positives
- Uses exact cosine similarity (dot product on normalized vectors)
- Cleaner code path for debugging and understanding
- Still efficient with FAISS's optimized nearest-neighbor search

Pipeline:
1. Load and clean sampled posts (JSONL)
2. Compute multilingual semantic embeddings (distiluse-base-multilingual-cased-v2)
3. Sample random pairs to estimate baseline similarity distribution
4. Calculate 99% percentile as dynamic threshold (adapts to each corpus)
5. Build global FAISS index over all normalized embeddings
6. Use FAISS range_search to find all cross-account pairs above threshold
7. Aggregate suspicious accounts and export artifacts
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer


# Default paths for sampled input and output directory
DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")

# Reddit artifacts to skip (not real user content)
SKIP_TEXT = {"[removed]", "[deleted]"}
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for FAISS-based bot detection.

    Example CLI invocation:
        python bot_detection_faiss_no_lsh.py \\
            --input-file sampled_data/sample.jsonl \\
            --output-dir code/src_3/output \\
            --quantile 0.99 \\
            --embedding-model distiluse-base-multilingual-cased-v2

    Returns:
        argparse.Namespace containing:
        - input_file: path to sampled JSONL
        - output_dir: directory for outputs (suspicious_pairs.csv, suspicious_accounts.csv, etc.)
        - min_words: minimum words required per post (filters short noise)
        - embedding_model: SentenceTransformer model name
        - batch_size: embedding generation batch size
        - random_pairs: how many random pairs to sample for quantile estimation
        - quantile: quantile level for threshold (e.g., 0.99 = 99th percentile)
        - faiss_range_threshold: candidate threshold for FAISS range_search
        - min_cross_account_matches: minimum suspicious links to flag an account
        - seed: RNG seed for reproducibility
    """
    parser = argparse.ArgumentParser(
        description=(
            "Detect suspicious Reddit accounts from sampled JSONL using multilingual "
            "semantic embeddings + global FAISS range search (no LSH)."
        )
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Path to sampled JSONL file containing Reddit posts/comments.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where detection outputs (CSV, JSONL, JSON) are written.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum tokenized words required to keep a post (filters short noise).",
    )
    parser.add_argument(
        "--embedding-model",
        default="distiluse-base-multilingual-cased-v2",
        help="SentenceTransformer model name for semantic embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for SentenceTransformer.encode() during embedding.",
    )
    parser.add_argument(
        "--random-pairs",
        type=int,
        default=50000,
        help="Number of random post pairs to sample for quantile threshold estimation.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help=(
            "Quantile level used to set cosine similarity threshold. "
            "E.g., 0.99 means threshold = 99th percentile of random-pair similarities."
        ),
    )
    parser.add_argument(
        "--faiss-range-threshold",
        type=float,
        default=None,
        help=(
            "Optional override for FAISS range_search threshold. "
            "If not set, uses the quantile-based threshold from random pairs."
        ),
    )
    parser.add_argument(
        "--min-cross-account-matches",
        type=int,
        default=1,
        help="Minimum number of cross-account high-similarity matches to flag an account as suspicious.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility in sampling and RNG operations.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace in text for stable pre-embedding normalization.

    Input example:
        "Hello  world   \n  this  is   a   test"

    Computation:
        Calls str.split() (splits on any whitespace, removes empty strings)
        then " ".join() to rebuild with single spaces

    Output example:
        "Hello world this is a test"
    """
    return " ".join(text.split())


def infer_post_type(record: Dict) -> str:
    """Infer whether a Reddit record is a comment or submission.

    Input example:
        record = {
            "body": "This is a comment",
            "title": "Post title",
            "selftext": "Post body"
        }

    Computation:
        Checks if "body" field is non-empty (indicator of comment)
        vs missing/empty (indicator of submission with title/selftext)

    Output example:
        "comment" if body exists, else "submission"
    """
    body = (record.get("body") or "").strip()
    if body:
        return "comment"
    return "submission"


def build_text(record: Dict, post_type: str) -> str:
    """Assemble normalized text content for embedding based on post type.

    Input example:
        record = {
            "title": "My Question",
            "selftext": "Can anyone help?",
            "body": ""
        },
        post_type = "submission"

    Computation:
        For submission: concatenate title + selftext
        For comment: use body only
        Normalize each part independently, then join

    Output example:
        "My Question Can anyone help?"
    """
    if post_type == "comment":
        return normalize_text((record.get("body") or "").strip())

    title = normalize_text((record.get("title") or "").strip())
    selftext = normalize_text((record.get("selftext") or "").strip())
    combined = " ".join(part for part in [title, selftext] if part)
    return normalize_text(combined)


def load_records(path: str, min_words: int) -> pd.DataFrame:
    """Load, clean, and filter JSONL records for downstream processing.

    Input:
        path = "sampled_data/sample.jsonl" (one JSON object per line)
        min_words = 10

    File format example:
        Line 1: {"id": "abc123", "author": "user1", "text": "...", ...}
        Line 2: {"id": "def456", "author": "user2", "body": "...", ...}

    Computation (applied line-by-line):
        1. Parse JSON (skip malformed lines)
        2. Extract author, infer post type (comment vs submission)
        3. Build normalized text content
        4. Count words via split()
        5. Filter: no deleted authors, no skip-text patterns, >= min_words words
        6. Retain original raw JSON and metadata

    Output:
        DataFrame with columns:
        - author: post author username
        - text: normalized text content
        - word_count: tokenized word count
        - post_type: "comment" or "submission"
        - subreddit: community name
        - created_utc: UNIX timestamp
        - id: unique post identifier
        - raw: original JSON record (for downstream export)

        Example row:
        {
            "author": "user123",
            "text": "This is a suspicious post...",
            "word_count": 45,
            "post_type": "submission",
            "subreddit": "AskReddit",
            "created_utc": 1234567890,
            "id": "xyz789",
            "raw": {...full JSON...}
        }
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
                # Skip broken lines instead of failing entire run
                continue

            author = (record.get("author") or "").strip()
            if not author or author in SKIP_AUTHORS:
                continue

            post_type = infer_post_type(record)
            text = build_text(record, post_type)
            if not text or text in SKIP_TEXT:
                continue

            # Word count filters low-information short posts that create noisy matches
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
    """Compute multilingual semantic embeddings for all posts using SentenceTransformer.

    Input:
        texts = ["Hello world this is a test", "Another sample text", ...]
                (typically 5000-50000 posts)
        model_name = "distiluse-base-multilingual-cased-v2"
        batch_size = 64

    Computation:
        1. Load SentenceTransformer model
        2. Encode all texts in batches
        3. Normalize embeddings to unit vectors (L2 norm)
           - This makes dot product equal cosine similarity
           - Important for FAISS IndexFlatIP range_search
        4. Convert to float32 for FAISS compatibility

    Output:
        Numpy array of shape [num_texts, embedding_dim]
        - dtype: float32
        - Each row is an L2-normalized embedding vector
        - Each vector has norm ~1.0 (unit vector)

        Example shape:
        (45123, 768)  # 45k texts, 768-dim embeddings
    """
    model = SentenceTransformer(model_name)

    # normalize_embeddings=True gives unit vectors, so dot product == cosine similarity
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # FAISS expects float32 contiguous arrays for best performance
    return np.asarray(emb, dtype=np.float32)


def sample_random_pair_similarities(
    embeddings: np.ndarray,
    n_pairs: int,
    seed: int,
) -> np.ndarray:
    """Estimate background cosine similarity by randomly sampling post pairs.

    Why this step:
        The 99% percentile of random-pair similarities is used as the threshold.
        This adapts the threshold to each corpus's distribution (not a fixed value).
        Pairs above this percentile are statistically unusual vs random baseline.

    Input:
        embeddings = [n_posts, 768] matrix of unit vectors
        n_pairs = 50000 (how many random pairs to sample)
        seed = 42 (RNG seed for reproducibility)

    Computation:
        1. Initialize RNG with seed
        2. Loop n_pairs times:
           - Randomly sample two different post indices (a, b)
           - Compute cosine similarity: dot(embedding[a], embedding[b])
             (already cosine because embeddings are normalized)
           - Append to similarities array

    Output:
        1D numpy array of shape [n_pairs] containing sampled similarities

        Example:
        array([0.123, 0.245, 0.089, 0.567, ...], dtype=float32)
        (50000 samples of random-pair dot products)
    """
    n_rows = embeddings.shape[0]
    if n_rows < 2:
        return np.array([], dtype=np.float32)

    rng = np.random.default_rng(seed)
    sims = np.zeros(n_pairs, dtype=np.float32)

    # Sample random pairs and compute similarities
    for i in range(n_pairs):
        # Choose 2 different indices without replacement
        a, b = rng.choice(n_rows, size=2, replace=False)
        # Dot product on normalized vectors = cosine similarity
        sims[i] = float(np.dot(embeddings[a], embeddings[b]))

    return sims


def plot_similarity_distribution(
    similarities: np.ndarray,
    output_file: str,
    # threshold: float | None = None,
    threshold: Optional[float] = None,
    #quantile: float | None = None,
    quantile: Optional[float] = None,
) -> None:
    """Save a KDE plot of random-pair similarity distribution with threshold overlay.

    Input:
        similarities = [50000] array of random-pair similarities
        output_file = "output/random_similarity_kde.png"
        threshold = 0.6234 (99th percentile value)
        quantile = 0.99 (which percentile was used)

    Computation:
        1. Create KDE plot of similarities using seaborn
        2. If threshold provided, overlay a vertical line at that value
        3. Label the line with quantile and threshold value

    Output:
        PNG file written to disk (e.g., random_similarity_kde.png)
        Visual shows distribution shape and threshold cutoff
    """
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


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a global exact inner-product search index over all embeddings.

    Why IndexFlatIP:
        - "Flat" means no compression (exact search, not approximate)
        - "IP" means inner product (which equals cosine similarity on normalized vectors)
        - Allows FAISS range_search to find all neighbors within a radius

    Input:
        embeddings = [n_posts, 768] float32 matrix, unit normalized
                     shape example: (45000, 768)

    Computation:
        1. Extract embedding dimension
        2. Create IndexFlatIP(dim)
        3. Add all embeddings to index
        4. Index is ready for range_search queries

    Output:
        faiss.IndexFlatIP object containing all vectors

        Example usage:
        index = build_faiss_index(embeddings)  # shape [45000, 768]
        # Later: index.range_search(query_vector, radius=0.6)
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Convert to contiguous float32 for FAISS compatibility
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(vectors)

    return index


def find_high_similarity_pairs_faiss(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Find all cross-account high-similarity pairs using global FAISS range search.

    Why this design (vs LSH):
        - No approximation errors: we find ALL pairs above threshold
        - Simpler logic: one index, one query pass
        - Exact cosine similarity matching
        - Better for debugging and reproducibility

    Input:
        df = DataFrame with columns: author, text, post_type, subreddit, ...
             shape: (45000,)
        embeddings = [45000, 768] unit-normalized float32 vectors
        index = FAISS IndexFlatIP object containing all 45000 vectors
        threshold = 0.6234 (99th percentile similarity cutoff)

    Computation:
        1. For each post's embedding vector:
           - Call index.range_search(vector, radius=threshold)
           - Get back all neighbors with similarity >= threshold
        2. For each pair found:
           - Skip self-matches (same index)
           - Skip same-author matches (not useful for coordinated behavior)
           - Collect pairs with (idx_a < idx_b) to avoid duplicates
        3. Sort results by similarity descending

    Output:
        Tuple:
        - DataFrame with columns:
          * idx_a, idx_b: row indices in df
          * author_a, author_b: authors of the two posts
          * similarity: cosine similarity value
          * post_type_a, post_type_b: "comment" or "submission"
          * subreddit_a, subreddit_b: community names
          * text_a, text_b: first 250 chars of each post

        - Dictionary with diagnostics:
          Example: {"num_queries": 45000, "num_neighbors_found": 150234, "num_cross_account_pairs": 4562}
    """
    n_posts = embeddings.shape[0]
    seen_pairs: set[Tuple[int, int]] = set()
    rows: List[Dict] = []

    num_queries = 0
    num_neighbors_found = 0

    # Prepare query vectors
    query_vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Run range_search for all vectors at once
    # Returns: lims (limits array), scores (similarity values), neighbors (neighbor indices)
    lims, scores, neighbors = index.range_search(query_vectors, float(threshold))

    # Process results from range_search
    for query_idx in range(n_posts):
        start = int(lims[query_idx])
        end = int(lims[query_idx + 1])

        num_queries += 1
        num_neighbors_found += (end - start)

        # Iterate through all neighbors found for this query
        for pos in range(start, end):
            neighbor_idx = int(neighbors[pos])
            similarity = float(scores[pos])

            # Skip self-matches
            if neighbor_idx == query_idx:
                continue

            # Normalize pair to (smaller_idx, larger_idx) to avoid duplicates
            a, b = (query_idx, neighbor_idx) if query_idx < neighbor_idx else (neighbor_idx, query_idx)

            # Skip if we've already seen this pair
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))

            # Only cross-account similarity is useful for coordinated-behavior detection
            author_a = df.iloc[a]["author"]
            author_b = df.iloc[b]["author"]
            if author_a == author_b:
                continue

            # Collect the pair
            rows.append(
                {
                    "idx_a": a,
                    "idx_b": b,
                    "author_a": author_a,
                    "author_b": author_b,
                    "post_type_a": df.iloc[a]["post_type"],
                    "post_type_b": df.iloc[b]["post_type"],
                    "similarity": round(similarity, 6),
                    "subreddit_a": df.iloc[a]["subreddit"],
                    "subreddit_b": df.iloc[b]["subreddit"],
                    "text_a": df.iloc[a]["text"][:250],
                    "text_b": df.iloc[b]["text"][:250],
                }
            )

    if not rows:
        return pd.DataFrame(), {
            "num_queries": num_queries,
            "num_neighbors_found": num_neighbors_found,
            "num_cross_account_pairs": 0,
        }

    out = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    return out, {
        "num_queries": num_queries,
        "num_neighbors_found": num_neighbors_found,
        "num_cross_account_pairs": int(len(out)),
    }


def build_suspicious_accounts(pairs: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Aggregate pair-level similarity signals to account-level suspiciousness scores.

    Input:
        pairs = DataFrame with columns: author_a, author_b, similarity
                shape: (4562, 11)  # 4562 suspicious pairs
        min_matches = 1 (minimum cross-account matches to flag an account)

    Computation (high level):
        1. For each author, collect all similarity scores from their suspicious pairs
        2. Calculate aggregate statistics: count, max, mean
        3. Filter: keep only authors with >= min_matches suspicious links
        4. Sort by: count (descending), then mean similarity (descending)

    Detailed loop example:
        For author "user_X":
        - Collect similarities: [0.823, 0.756, 0.812]
        - Count: 3 matches
        - Max: 0.823
        - Mean: 0.797
        - Include if count >= min_matches

    Output:
        DataFrame with columns:
        - author: username
        - cross_account_matches: how many suspicious pairs this author was in
        - max_similarity: highest similarity to any co-author
        - avg_similarity: mean similarity across all matches

        Example:
        {
            "author": "user_X",
            "cross_account_matches": 3,
            "max_similarity": 0.823,
            "avg_similarity": 0.797
        }

        Sorted by cross_account_matches (desc) then avg_similarity (desc)
    """
    if pairs.empty:
        return pd.DataFrame(columns=["author", "cross_account_matches", "max_similarity", "avg_similarity"])

    stats: Dict[str, List[float]] = {}

    # Collect all similarity values for each author
    for _, row in pairs.iterrows():
        author_a = row["author_a"]
        author_b = row["author_b"]
        sim = float(row["similarity"])

        stats.setdefault(author_a, []).append(sim)
        stats.setdefault(author_b, []).append(sim)

    out_rows = []
    # Build aggregate statistics per author
    for author, sims in stats.items():
        # Require at least N suspicious links before flagging an account
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
    """Write posts from suspicious authors to JSONL file for downstream clustering.

    Why this output:
        The clustering stage needs to operate on the original JSON records,
        not just the text. This preserves all metadata for topic analysis.

    Input:
        df = DataFrame with columns: author, raw (original JSON)
             shape: (45000,)
        suspicious_authors = {"user_X", "user_Y", ...}
        output_file = "output/suspicious_posts.jsonl"

    Computation (simple loop):
        For each row in df:
        - Check if author is in suspicious_authors set
        - If yes, write df.raw (original JSON) as one line to JSONL
        - Count how many rows written

    Output:
        1. JSONL file with one JSON object per line (original Reddit records)
        2. Return count of rows written

        Example JSONL lines:
        {"id": "abc123", "author": "user_X", "subreddit": "AskReddit", ...}
        {"id": "def456", "author": "user_Y", "subreddit": "worldnews", ...}
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
    """Orchestrate the full bot detection pipeline using global FAISS range search.

    High-level flow:
    1. Parse CLI arguments
    2. Load and clean posts from sampled JSONL
    3. Compute semantic embeddings
    4. Estimate threshold as 99% percentile of random-pair similarities
    5. Build global FAISS index
    6. Find all cross-account pairs above threshold using FAISS range_search
    7. Aggregate suspicious accounts
    8. Export artifacts for downstream clustering

    Outputs written to output_dir:
    - suspicious_pairs.csv: all cross-account pairs above threshold
    - suspicious_accounts.csv: aggregated account-level suspiciousness
    - suspicious_posts.jsonl: original Reddit JSON for flagged authors
    - detection_summary.json: metadata and diagnostic stats
    - random_similarity_kde.png: visualization of threshold choice
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ========== STEP 1: Load and clean posts ==========
    print(f"Loading sampled records from: {args.input_file}")
    df = load_records(args.input_file, min_words=args.min_words)
    if df.empty:
        raise SystemExit("No usable posts found after filtering.")

    print(f"Usable posts: {len(df)}")
    print(f"Unique authors: {df['author'].nunique()}")

    # ========== STEP 2: Compute semantic embeddings ==========
    print(f"Computing semantic embeddings with {args.embedding_model}...")
    embeddings = compute_embeddings(df["text"].tolist(), args.embedding_model, args.batch_size)

    # ========== STEP 3: Estimate random-pair quantiles and set threshold ==========
    print("Sampling random pair similarities for quantile estimation...")
    sampled_sims = sample_random_pair_similarities(embeddings, n_pairs=args.random_pairs, seed=args.seed)
    if sampled_sims.size == 0:
        raise SystemExit("Not enough posts to compute random similarity quantiles.")

    # Report reference quantiles to make threshold choice transparent
    quantiles_to_report = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
    quantile_values = np.quantile(sampled_sims, quantiles_to_report)

    print("Random-pair semantic cosine similarity quantiles:")
    for q, v in zip(quantiles_to_report, quantile_values):
        print(f"  q={q:.3f} -> {v:.6f}")

    # ========== STEP 4: Calculate 99% percentile threshold ==========
    # This is the key decision: pairs above this are unusually similar vs random baseline
    similarity_threshold = float(np.quantile(sampled_sims, args.quantile))
    print(f"\nUsing q={args.quantile:.3f} threshold: cosine >= {similarity_threshold:.6f}")

    # Optional override for testing
    if args.faiss_range_threshold is not None:
        similarity_threshold = float(args.faiss_range_threshold)
        print(f"Overriding threshold to: {similarity_threshold:.6f}")

    # Plot for visual inspection
    kde_path = os.path.join(args.output_dir, "random_similarity_kde.png")
    plot_similarity_distribution(
        sampled_sims,
        kde_path,
        threshold=similarity_threshold,
        quantile=args.quantile,
    )
    print(f"Saved KDE plot: {kde_path}")

    # ========== STEP 5: Build global FAISS index ==========
    print("Building global FAISS IndexFlatIP...")
    index = build_faiss_index(embeddings)
    print(f"FAISS index built with {embeddings.shape[0]} vectors of dim {embeddings.shape[1]}")

    # ========== STEP 6: Find high-similarity pairs using FAISS range_search ==========
    print(f"Finding high-similarity cross-account pairs with FAISS range_search...")
    suspicious_pairs, faiss_stats = find_high_similarity_pairs_faiss(
        df=df,
        embeddings=embeddings,
        index=index,
        threshold=similarity_threshold,
    )

    pairs_path = os.path.join(args.output_dir, "suspicious_pairs.csv")
    suspicious_pairs.to_csv(pairs_path, index=False)
    print(f"Saved suspicious pairs: {pairs_path} ({len(suspicious_pairs)} rows)")

    # ========== STEP 7: Aggregate suspicious accounts ==========
    suspicious_accounts = build_suspicious_accounts(
        suspicious_pairs, min_matches=args.min_cross_account_matches
    )
    accounts_path = os.path.join(args.output_dir, "suspicious_accounts.csv")
    suspicious_accounts.to_csv(accounts_path, index=False)
    print(f"Saved suspicious accounts: {accounts_path} ({len(suspicious_accounts)} accounts)")

    # ========== STEP 8: Export suspicious posts for downstream clustering ==========
    suspicious_authors = set(suspicious_accounts["author"].tolist()) if not suspicious_accounts.empty else set()
    suspicious_jsonl_path = os.path.join(args.output_dir, "suspicious_posts.jsonl")
    kept = save_suspicious_posts(df, suspicious_authors, suspicious_jsonl_path)
    print(f"Saved suspicious posts JSONL: {suspicious_jsonl_path} ({kept} rows)")

    # ========== STEP 9: Save diagnostic summary ==========
    summary = {
        "input_file": args.input_file,
        "embedding_model": args.embedding_model,
        "detection_method": "faiss_global_range_search_no_lsh",
        "usable_posts": int(len(df)),
        "usable_authors": int(df["author"].nunique()),
        "random_pairs_sampled": int(sampled_sims.size),
        "quantile": float(args.quantile),
        "similarity_threshold": similarity_threshold,
        "faiss_stats": faiss_stats,
        "suspicious_pairs": int(len(suspicious_pairs)),
        "suspicious_accounts": int(len(suspicious_accounts)),
    }
    summary_path = os.path.join(args.output_dir, "detection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\n" + "=" * 60)
    print("Bot detection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
