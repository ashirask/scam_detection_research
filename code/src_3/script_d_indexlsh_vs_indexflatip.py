import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

"""
script_d_indexlsh_vs_indexflatip.py

Diagnostic comparing FAISS IndexLSH neighbor sets to IndexFlatIP exact neighbors.

What it does:
- Builds `IndexFlatIP` (exact inner-product) and `IndexLSH` (LSH bit-codes) on the same
    embedding sample.
- For each query (self-queries), requests top-k neighbors from both indexes.
- Computes overlap fraction, average scores, and produces mismatch samples for manual review.

Outputs (in `--output-dir`):
- `per_query_summary.csv`: per-query overlap and average scores
- `mismatch_samples.csv`: sample queries where overlap is small with neighbor previews
- `lsh_vs_flatip_summary.json`: run metadata and file paths

Notes:
- `IndexLSH` returns neighbor indices based on LSH bit-code similarity (Hamming-like),
    not raw cosine. We compute manual cosine scores for LSH neighbors for direct comparison.
"""


DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output" / "script_d_lsh_vs_flatip")
SKIP_TEXT = {"[removed]", "[deleted]"}
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare FAISS IndexLSH vs IndexFlatIP neighbor sets and scores")
    p.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--min-words", type=int, default=10)
    p.add_argument("--embedding-model", default="distiluse-base-multilingual-cased-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sample-size", type=int, default=1000, help="Number of rows to sample for the diagnostic")
    p.add_argument("--nbits", type=int, default=32, help="Number of bits for IndexLSH")
    p.add_argument("--k", type=int, default=10, help="Top-k neighbors to request")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


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
            if len(text.split()) < min_words:
                continue
            rows.append({"author": author, "text": text, "raw": record})
    return pd.DataFrame(rows)


def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def main() -> None:
    """Run the LSH vs FlatIP diagnostic.

    Steps:
    1. Load and optionally sample records
    2. Embed texts once (normalized vectors)
    3. Build both FAISS indexes and run top-k queries
    4. Compute overlap stats and save per-query/mismatch outputs
    """
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
    print(f"Using {len(df)} records for diagnostic (authors={df['author'].nunique()})")

    texts = df["text"].tolist()
    embeddings = compute_embeddings(texts, args.embedding_model, args.batch_size)
    dim = embeddings.shape[1]

    # Build IndexFlatIP (exact) and IndexLSH (approximate on bit-codes)
    print("Building IndexFlatIP (exact inner-product)...")
    index_ip = faiss.IndexFlatIP(dim)
    index_ip.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    print(f"Building IndexLSH (nbits={args.nbits})...")
    # IndexLSH stores binary hash codes derived from random projections and
    # retrieves neighbors by Hamming-similarity on those codes. It does not
    # return cosine scores; we'll compute dot products manually for LSH neighbors
    # to compare against the exact IndexFlatIP scores.
    index_lsh = faiss.IndexLSH(dim, args.nbits)
    index_lsh.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    # Query with all vectors (self-queries) to compare neighbor sets
    print(f"Running searches for k={args.k} neighbors per query...")
    # IndexFlatIP returns (scores, indices)
    ip_sims, ip_neighbors = index_ip.search(np.ascontiguousarray(embeddings, dtype=np.float32), args.k)
    # IndexLSH returns (hamming distances?, indices) - we'll ignore returned scores
    lsh_dists, lsh_neighbors = index_lsh.search(np.ascontiguousarray(embeddings, dtype=np.float32), args.k)

    # Evaluate overlap and score differences
    per_query_rows: List[Dict] = []
    for qi in range(len(embeddings)):
        ip_nb = [int(x) for x in ip_neighbors[qi] if int(x) >= 0]
        lsh_nb = [int(x) for x in lsh_neighbors[qi] if int(x) >= 0]
        set_ip = set(ip_nb)
        set_lsh = set(lsh_nb)
        overlap = set_ip & set_lsh
        overlap_frac = len(overlap) / float(args.k) if args.k > 0 else 0.0

        # compute average cosine for IP top-k and for LSH neighbors (manual dot)
        ip_scores = [float(ip_sims[qi][i]) for i in range(len(ip_nb))]
        ip_avg = float(np.mean(ip_scores)) if ip_scores else 0.0
        # lsh manual scores
        lsh_scores = [float(np.dot(embeddings[qi], embeddings[j])) for j in lsh_nb] if lsh_nb else []
        lsh_avg = float(np.mean(lsh_scores)) if lsh_scores else 0.0

        # for neighbors in both, compute mean abs diff between ip score and manual dot
        common_diffs = []
        for j in overlap:
            # find ip score for neighbor j
            try:
                idx = ip_nb.index(j)
                ip_score = float(ip_sims[qi][idx])
            except ValueError:
                ip_score = float(np.dot(embeddings[qi], embeddings[j]))
            lsh_score = float(np.dot(embeddings[qi], embeddings[j]))
            common_diffs.append(abs(ip_score - lsh_score))

        per_query_rows.append({
            "query_idx": qi,
            "num_ip_found": len(ip_nb),
            "num_lsh_found": len(lsh_nb),
            "overlap_count": len(overlap),
            "overlap_frac": overlap_frac,
            "ip_topk_avg": ip_avg,
            "lsh_neighbors_avg": lsh_avg,
            "mean_common_score_diff": float(np.mean(common_diffs)) if common_diffs else 0.0,
        })

    per_query_df = pd.DataFrame(per_query_rows)
    per_query_path = os.path.join(args.output_dir, "per_query_summary.csv")
    per_query_df.to_csv(per_query_path, index=False)

    # Save some detailed mismatch samples for manual inspection
    mismatch_rows: List[Dict] = []
    for qi in range(min(200, len(embeddings))):
        ip_nb = [int(x) for x in ip_neighbors[qi] if int(x) >= 0]
        lsh_nb = [int(x) for x in lsh_neighbors[qi] if int(x) >= 0]
        overlap = set(ip_nb) & set(lsh_nb)
        if len(overlap) < max(1, int(0.5 * args.k)):
            mismatch_rows.append({
                "query_idx": qi,
                "query_text_preview": texts[qi][:200],
                "ip_neighbors": ";".join([f"{j}:{np.dot(embeddings[qi], embeddings[j]):.6f}" for j in ip_nb]),
                "lsh_neighbors": ";".join([f"{j}:{np.dot(embeddings[qi], embeddings[j]):.6f}" for j in lsh_nb]),
                "overlap_count": len(overlap),
            })

    mismatch_path = os.path.join(args.output_dir, "mismatch_samples.csv")
    pd.DataFrame(mismatch_rows).to_csv(mismatch_path, index=False)

    summary = {
        "input_file": args.input_file,
        "n_records": int(len(df)),
        "sample_size": int(len(df)),
        "nbits": int(args.nbits),
        "k": int(args.k),
        "per_query_summary": per_query_path,
        "mismatch_sample": mismatch_path,
    }
    with open(os.path.join(args.output_dir, "lsh_vs_flatip_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved per-query summary to: {per_query_path}")
    print(f"Saved mismatch samples to: {mismatch_path}")
    print(f"Saved summary JSON to: {os.path.join(args.output_dir, 'lsh_vs_flatip_summary.json')}")


if __name__ == "__main__":
    main()
