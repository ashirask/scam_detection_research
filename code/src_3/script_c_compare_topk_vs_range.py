import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Set, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

"""
script_c_compare_topk_vs_range.py

Diagnostic comparing two pipeline retrieval modes run on the SAME
embeddings and LSH tables:

- top-k retrieval inside each LSH bucket (original `bot_detection_semantic.py`)
- range_search inside each LSH bucket (Script B: `script_b_bot_detection_lsh_faiss_range.py`)

Outputs (written to `--output-dir`):
- `pair_presence_comparison.csv`: which pairs each method found
- `score_differences_both_methods.csv`: score diffs for pairs both methods found
- `pairs_only_in_topk_sample.csv` / `pairs_only_in_range_sample.csv`: inspection samples
- `comparison_summary.json`: aggregate counts and stats

Use this to determine whether differences are due to (a) which pairs
are selected by each retrieval method or (b) different similarity
calculations for the same pairs.
"""


DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output" / "script_c_compare")
SKIP_TEXT = {"[removed]", "[deleted]"}
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare top-k vs range_search behavior on same embeddings")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--embedding-model", default="distiluse-base-multilingual-cased-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-pairs", type=int, default=50000)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--lsh-bits", type=int, default=16)
    parser.add_argument("--lsh-tables", type=int, default=4)
    parser.add_argument("--faiss-top-k", type=int, default=40)
    parser.add_argument("--min-bucket-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def infer_post_type(record: Dict) -> str:
    body = (record.get("body") or "").strip()
    return "comment" if body else "submission"


def build_text(record: Dict, post_type: str) -> str:
    if post_type == "comment":
        return normalize_text((record.get("body") or "").strip())
    title = normalize_text((record.get("title") or "").strip())
    selftext = normalize_text((record.get("selftext") or "").strip())
    return normalize_text(" ".join(part for part in [title, selftext] if part))


def load_records(path: str, min_words: int) -> pd.DataFrame:
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
            rows.append({
                "author": author,
                "text": text,
                "word_count": word_count,
                "post_type": post_type,
                "subreddit": record.get("subreddit", ""),
                "created_utc": record.get("created_utc"),
                "id": record.get("id") or record.get("name") or f"row_{line_num}",
                "raw": record,
            })
    return pd.DataFrame(rows)


def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
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
    n_rows = embeddings.shape[0]
    if n_rows < 2:
        return np.array([], dtype=np.float32)
    rng = np.random.default_rng(seed)
    sims = np.zeros(n_pairs, dtype=np.float32)
    for i in range(n_pairs):
        a, b = rng.choice(n_rows, size=2, replace=False)
        sims[i] = float(np.dot(embeddings[a], embeddings[b]))
    return sims


def _build_lsh_signatures(embeddings: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    bit_matrix = (embeddings @ projection_matrix) >= 0.0
    bit_weights = (1 << np.arange(bit_matrix.shape[1], dtype=np.uint64))
    return (bit_matrix.astype(np.uint64) * bit_weights).sum(axis=1)


def method_topk(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    lsh_bits: int,
    lsh_tables: int,
    faiss_top_k: int,
    min_bucket_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    """Run the original LSH + FAISS top-k pipeline.

    Behavior:
    - For each LSH table, build buckets via random hyperplanes
    - For each bucket, build an IndexFlatIP on bucket vectors
    - Run `index.search(k)` and then filter returned neighbors by `threshold`
    - Deduplicate mirrored pairs and skip same-author matches

    Returns a DataFrame of cross-account pairs found by the top-k approach
    and simple diagnostics.
    """
    if df.empty:
        return pd.DataFrame(), {}
    rng = np.random.default_rng(seed)
    dim = embeddings.shape[1]
    seen_pairs: Set[Tuple[int, int]] = set()
    rows: List[Dict] = []
    num_buckets = 0
    for table_idx in range(lsh_tables):
        projection_matrix = rng.standard_normal(size=(dim, lsh_bits), dtype=np.float32)
        signatures = _build_lsh_signatures(embeddings, projection_matrix)
        buckets: DefaultDict[int, List[int]] = defaultdict(list)
        for idx, signature in enumerate(signatures):
            buckets[int(signature)].append(idx)
        for idx_list in buckets.values():
            if len(idx_list) < min_bucket_size:
                continue
            num_buckets += 1
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
                    if float(sim) < threshold:
                        continue
                    a, b = (global_i, global_j) if global_i < global_j else (global_j, global_i)
                    if (a, b) in seen_pairs:
                        continue
                    seen_pairs.add((a, b))
                    if df.iloc[a]["author"] == df.iloc[b]["author"]:
                        continue
                    rows.append({
                        "idx_a": a,
                        "idx_b": b,
                        "author_a": df.iloc[a]["author"],
                        "author_b": df.iloc[b]["author"],
                        "similarity": round(float(sim), 6),
                        "method": "topk",
                        "table_idx": table_idx,
                    })
    return pd.DataFrame(rows), {"num_buckets": num_buckets, "num_pairs": len(rows), "method": "topk"}


def method_range_search(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    lsh_bits: int,
    lsh_tables: int,
    min_bucket_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    """Run the LSH + FAISS range_search pipeline (Script B behavior).

    Behavior:
    - For each LSH table, build the same random-hyperplane buckets
    - For each bucket, build an IndexFlatIP and call `index.range_search(query, radius)`
    - Return all neighbors >= threshold, deduplicate, and filter same-author pairs

    This is the strict threshold-based retrieval to compare against top-k.
    """
    if df.empty:
        return pd.DataFrame(), {}
    rng = np.random.default_rng(seed)
    dim = embeddings.shape[1]
    seen_pairs: Set[Tuple[int, int]] = set()
    rows: List[Dict] = []
    num_buckets = 0
    for table_idx in range(lsh_tables):
        projection_matrix = rng.standard_normal(size=(dim, lsh_bits), dtype=np.float32)
        signatures = _build_lsh_signatures(embeddings, projection_matrix)
        buckets: DefaultDict[int, List[int]] = defaultdict(list)
        for idx, signature in enumerate(signatures):
            buckets[int(signature)].append(idx)
        for idx_list in buckets.values():
            if len(idx_list) < min_bucket_size:
                continue
            num_buckets += 1
            bucket_vectors = np.ascontiguousarray(embeddings[idx_list], dtype=np.float32)
            index = faiss.IndexFlatIP(dim)
            index.add(bucket_vectors)
            lims, scores, neighbors = index.range_search(bucket_vectors, float(threshold))
            for local_i, global_i in enumerate(idx_list):
                start = int(lims[local_i])
                end = int(lims[local_i + 1])
                for pos in range(start, end):
                    local_j = int(neighbors[pos])
                    sim = float(scores[pos])
                    if local_j == local_i:
                        continue
                    global_j = idx_list[local_j]
                    a, b = (global_i, global_j) if global_i < global_j else (global_j, global_i)
                    if (a, b) in seen_pairs:
                        continue
                    seen_pairs.add((a, b))
                    if df.iloc[a]["author"] == df.iloc[b]["author"]:
                        continue
                    rows.append({
                        "idx_a": a,
                        "idx_b": b,
                        "author_a": df.iloc[a]["author"],
                        "author_b": df.iloc[b]["author"],
                        "similarity": round(sim, 6),
                        "method": "range_search",
                        "table_idx": table_idx,
                    })
    return pd.DataFrame(rows), {"num_buckets": num_buckets, "num_pairs": len(rows), "method": "range_search"}


def compare_results(topk_pairs: pd.DataFrame, range_pairs: pd.DataFrame, output_dir: str) -> Dict:
    """Compare outputs from top-k and range_search.

    Produces CSVs in `output_dir` enumerating pairs unique to each method,
    pairs found by both, and score diffs for pairs present in both.
    Returns a small summary dictionary.
    """
    topk_pairs = topk_pairs.copy()
    range_pairs = range_pairs.copy()
    topk_pairs["pair_key"] = topk_pairs.apply(lambda r: f"{r['idx_a']}_{r['idx_b']}", axis=1)
    range_pairs["pair_key"] = range_pairs.apply(lambda r: f"{r['idx_a']}_{r['idx_b']}", axis=1)
    topk_keys = set(topk_pairs["pair_key"].tolist())
    range_keys = set(range_pairs["pair_key"].tolist())
    only_in_topk = topk_keys - range_keys
    only_in_range = range_keys - topk_keys
    in_both = topk_keys & range_keys
    score_diffs = []
    for pair_key in in_both:
        topk_score = float(topk_pairs[topk_pairs["pair_key"] == pair_key]["similarity"].values[0])
        range_score = float(range_pairs[range_pairs["pair_key"] == pair_key]["similarity"].values[0])
        score_diffs.append({"pair_key": pair_key, "topk_score": topk_score, "range_score": range_score, "diff": abs(topk_score - range_score)})
    score_diff_df = pd.DataFrame(score_diffs)
    max_diff = float(score_diff_df["diff"].max()) if not score_diff_df.empty else 0.0
    mean_diff = float(score_diff_df["diff"].mean()) if not score_diff_df.empty else 0.0
    os.makedirs(output_dir, exist_ok=True)
    comp_df = pd.DataFrame([{"pair_key": pk, "in_topk": pk in topk_keys, "in_range": pk in range_keys} for pk in topk_keys | range_keys])
    comp_df.to_csv(os.path.join(output_dir, "pair_presence_comparison.csv"), index=False)
    score_diff_df.to_csv(os.path.join(output_dir, "score_differences_both_methods.csv"), index=False)
    if only_in_topk:
        topk_pairs[topk_pairs["pair_key"].isin(list(only_in_topk)[:100])].to_csv(os.path.join(output_dir, "pairs_only_in_topk_sample.csv"), index=False)
    if only_in_range:
        range_pairs[range_pairs["pair_key"].isin(list(only_in_range)[:100])].to_csv(os.path.join(output_dir, "pairs_only_in_range_sample.csv"), index=False)
    return {"only_in_topk": len(only_in_topk), "only_in_range": len(only_in_range), "in_both": len(in_both), "max_score_diff": max_diff, "mean_score_diff": mean_diff}


def main() -> None:
    """Orchestrate the diagnostic run.

    Steps:
    1. Load and filter records (identical text normalization as production)
    2. Compute normalized SentenceTransformer embeddings (one pass)
    3. Estimate a threshold from random pairs (quantile)
    4. Run both retrieval methods and compare outputs
    5. Save CSV/JSON diagnostics to `--output-dir`
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading sampled records from: {args.input_file}")
    df = load_records(args.input_file, min_words=args.min_words)
    if df.empty:
        raise SystemExit("No usable posts found after filtering.")
    print(f"Usable posts: {len(df)} Unique authors: {df['author'].nunique()}")
    embeddings = compute_embeddings(df["text"].tolist(), args.embedding_model, args.batch_size)
    sampled_sims = sample_random_pair_similarities(embeddings, n_pairs=args.random_pairs, seed=args.seed)
    if sampled_sims.size == 0:
        raise SystemExit("Not enough posts to compute random similarity quantiles.")
    similarity_threshold = float(np.quantile(sampled_sims, args.quantile))
    print(f"Using threshold (q={args.quantile}): {similarity_threshold:.6f}")
    print("Running top-k method...")
    topk_pairs, _ = method_topk(df, embeddings, similarity_threshold, args.lsh_bits, args.lsh_tables, args.faiss_top_k, args.min_bucket_size, args.seed)
    print(f"Top-k pairs found: {len(topk_pairs)}")
    print("Running range_search method...")
    range_pairs, _ = method_range_search(df, embeddings, similarity_threshold, args.lsh_bits, args.lsh_tables, args.min_bucket_size, args.seed)
    print(f"Range-search pairs found: {len(range_pairs)}")
    comparison = compare_results(topk_pairs, range_pairs, args.output_dir)
    summary = {"input_file": args.input_file, "threshold": similarity_threshold, "quantile": args.quantile, "lsh_bits": args.lsh_bits, "lsh_tables": args.lsh_tables, "faiss_top_k": args.faiss_top_k, "topk_pairs": len(topk_pairs), "range_pairs": len(range_pairs), "comparison": comparison}
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved comparison summary to {os.path.join(args.output_dir, 'comparison_summary.json')}")


if __name__ == "__main__":
    main()
