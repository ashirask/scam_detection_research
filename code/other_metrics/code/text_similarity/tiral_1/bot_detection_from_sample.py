import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# Sampled JSONL is the expected starting point after your sampling stage.
DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")
# Placeholder content/authors that should not be treated as real authored posts.
SKIP_TEXT = {"[removed]", "[deleted]"}
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_args() -> argparse.Namespace:
    """Define CLI options for quantile-based suspicious-account detection.

    Example:
    python bot_detection_from_sample.py --input-file sampled_data/sample_20000.jsonl --quantile 0.995
    """
    parser = argparse.ArgumentParser(
        description="Detect suspicious Reddit accounts from sampled JSONL using cosine similarity quantiles."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="Path to sampled JSONL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for detection outputs.")
    parser.add_argument("--min-words", type=int, default=10, help="Minimum words required in a post.")
    parser.add_argument(
        "--random-pairs",
        type=int,
        default=50000,
        help="How many random post pairs to sample for quantile estimation.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Quantile used as cosine similarity cutoff for suspicious pairs.",
    )
    parser.add_argument(
        "--min-cross-account-matches",
        type=int,
        default=1,
        help="Minimum number of cross-account high-similarity matches to mark an account suspicious.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="Maximum TF-IDF features used for similarity scoring.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace for stable vectorization and comparisons."""
    return " ".join(text.split())


def infer_post_type(record: Dict) -> str:
    """Infer post type from available fields.

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

            # Word count is used to avoid short/noisy posts like "ok" or "lol".
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


def sample_random_pair_similarities(matrix, n_pairs: int, seed: int) -> np.ndarray:
    """Estimate background cosine similarity by random post-pair sampling.

    Scenario:
    If most random pairs are low similarity but a tiny tail is high,
    the quantile threshold captures only unusually similar pairs.
    """
    n_rows = matrix.shape[0]
    if n_rows < 2:
        return np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    sims: List[float] = []

    for _ in range(n_pairs):
        i, j = rng.choice(n_rows, size=2, replace=False)
        # TF-IDF vectors are sparse; dot product gives cosine similarity here.
        sim = float(matrix[i].dot(matrix[j].T).toarray()[0, 0])
        sims.append(sim)

    return np.asarray(sims, dtype=float)


def plot_similarity_distribution(
    similarities: np.ndarray,
    output_file: str,
    threshold: float | None = None,
    quantile: float | None = None,
) -> None:
    """Save a KDE plot and optionally draw the chosen similarity cutoff.

    If threshold is provided, a vertical dashed line is drawn so you can see
    exactly where the suspicious-pair decision boundary sits on the baseline
    similarity distribution.
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

    plt.title("KDE of Random Pair Cosine Similarity")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    if threshold is not None:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def find_high_similarity_pairs(df: pd.DataFrame, matrix, threshold: float) -> pd.DataFrame:
    """Find cross-account post pairs above a cosine threshold.

    Important line:
    - radius = 1 - threshold converts cosine similarity cutoff into a
      cosine-distance radius used by NearestNeighbors.
    """
    if df.empty:
        return pd.DataFrame()

    # cosine_distance = 1 - cosine_similarity
    radius = 1.0 - threshold
    nbrs = NearestNeighbors(metric="cosine", algorithm="brute")
    nbrs.fit(matrix)
    distances, indices = nbrs.radius_neighbors(matrix, radius=radius, return_distance=True)

    seen_pairs: set[Tuple[int, int]] = set()
    rows: List[Dict] = []

    for i, (dist_list, idx_list) in enumerate(zip(distances, indices)):
        for dist, j in zip(dist_list, idx_list):
            if i == j:
                continue

            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))

            # Only cross-account similarity matters for bot-ring detection.
            if df.iloc[a]["author"] == df.iloc[b]["author"]:
                continue

            similarity = 1.0 - float(dist)
            rows.append(
                {
                    "idx_a": a,
                    "idx_b": b,
                    "author_a": df.iloc[a]["author"],
                    "author_b": df.iloc[b]["author"],
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
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


def build_suspicious_accounts(pairs: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Aggregate pair-level signals to account-level suspiciousness.

    Scenario:
    - author X appears in many high-similarity cross-account pairs
    - author X gets flagged with high cross_account_matches
    """
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
    2) compute TF-IDF vectors
    3) estimate random-pair quantiles
    4) flag high-similarity cross-account pairs
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

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
        max_features=args.max_features,
    )
    matrix = vectorizer.fit_transform(df["text"])

    print("Sampling random pair similarities for quantile estimation...")
    sampled_sims = sample_random_pair_similarities(matrix, n_pairs=args.random_pairs, seed=args.seed)
    if sampled_sims.size == 0:
        raise SystemExit("Not enough posts to compute random similarity quantiles.")

    # Report reference quantiles to make threshold choice transparent.
    quantiles_to_report = [0.5, 0.9, 0.95, 0.99, 0.999]
    quantile_values = np.quantile(sampled_sims, quantiles_to_report)

    print("Random-pair cosine similarity quantiles:")
    for q, v in zip(quantiles_to_report, quantile_values):
        print(f"  q={q:.3f} -> {v:.6f}")

    # Core decision boundary: pairs above this are "unusually similar" vs random baseline.
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

    print("Finding high-similarity cross-account pairs...")
    suspicious_pairs = find_high_similarity_pairs(df, matrix, similarity_threshold)

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

    # JSON summary is useful for tracking runs and thresholds over time.
    summary = {
        "input_file": args.input_file,
        "usable_posts": int(len(df)),
        "usable_authors": int(df["author"].nunique()),
        "random_pairs_sampled": int(sampled_sims.size),
        "quantile": float(args.quantile),
        "similarity_threshold": similarity_threshold,
        "suspicious_pairs": int(len(suspicious_pairs)),
        "suspicious_accounts": int(len(suspicious_accounts)),
    }
    summary_path = os.path.join(args.output_dir, "detection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
