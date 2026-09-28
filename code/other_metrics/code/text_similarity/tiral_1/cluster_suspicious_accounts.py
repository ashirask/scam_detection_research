import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


# Default source is sampled JSONL, but this stage only keeps rows from suspicious authors.
DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_SUSPICIOUS_ACCOUNTS = str(Path(__file__).resolve().parent / "output" / "suspicious_accounts.csv")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")
# Placeholder content should not participate in embedding/clustering.
SKIP_TEXT = {"[removed]", "[deleted]"}


def parse_args() -> argparse.Namespace:
    """Define CLI options for suspicious-account clustering and topic modeling.

    Example:
    python cluster_suspicious_accounts.py --eps 0.28 --min-samples 4
    """
    parser = argparse.ArgumentParser(
        description="Cluster posts from suspicious Reddit accounts with Qwen embeddings and BERTopic."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="Path to sampled JSONL.")
    parser.add_argument(
        "--suspicious-accounts-file",
        default=DEFAULT_SUSPICIOUS_ACCOUNTS,
        help="CSV with suspicious accounts (must include 'author').",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for clustering outputs.")
    parser.add_argument("--min-words", type=int, default=10, help="Minimum words required in a post.")
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument("--eps", type=float, default=0.30, help="DBSCAN epsilon (cosine distance).")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min samples.")
    parser.add_argument("--batch-size", type=int, default=1, help="SentenceTransformer encode batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max sequence length.")
    parser.add_argument(
        "--disable-fp16",
        action="store_true",
        help="Disable fp16 inference on CUDA.",
    )
    parser.add_argument(
        "--disable-flash-attention",
        action="store_true",
        help="Disable CUDA flash attention backend.",
    )
    parser.add_argument("--bertopic-min-topic-size", type=int, default=10, help="BERTopic minimum topic size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-preview-length",
        type=int,
        default=180,
        help="Maximum preview length in exported cluster text files.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace so embedding input is consistently normalized."""
    return " ".join(text.split())


def infer_post_type(record: Dict) -> str:
    """Infer comment vs submission from available fields.

    Scenario:
    - body exists -> comment
    - otherwise -> submission (title/selftext)
    """
    body = (record.get("body") or "").strip()
    if body:
        return "comment"
    return "submission"


def build_text(record: Dict, post_type: str) -> str:
    """Build normalized model-ready text based on post type."""
    if post_type == "comment":
        return normalize_text((record.get("body") or "").strip())
    title = normalize_text((record.get("title") or "").strip())
    selftext = normalize_text((record.get("selftext") or "").strip())
    combined = " ".join(part for part in [title, selftext] if part)
    return normalize_text(combined)


def load_suspicious_authors(path: str) -> set[str]:
    """Load suspicious authors list produced by detection stage.

    Expected input: CSV that contains an 'author' column.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"Suspicious accounts file not found: {path}")
    df = pd.read_csv(path)
    if "author" not in df.columns:
        raise SystemExit("Suspicious accounts CSV must include an 'author' column.")
    return set(df["author"].astype(str).tolist())


def load_filtered_posts(path: str, suspicious_authors: set[str], min_words: int) -> pd.DataFrame:
    """Load JSONL and keep only posts from suspicious authors.

    Important filtering rules:
    - author must be in suspicious_authors
    - text must be non-placeholder
    - text must satisfy minimum word length
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
                # Ignore malformed lines rather than failing full job.
                continue

            author = (record.get("author") or "").strip()
            # This stage intentionally narrows the dataset to inauthentic candidates.
            if author not in suspicious_authors:
                continue

            post_type = infer_post_type(record)
            text = build_text(record, post_type)
            if not text or text in SKIP_TEXT:
                continue

            # Keeps low-information short texts from dominating tiny clusters.
            word_count = len(text.split())
            if word_count < min_words:
                continue

            rows.append(
                {
                    "author": author,
                    "post_type": post_type,
                    "text": text,
                    "word_count": word_count,
                    "subreddit": record.get("subreddit", ""),
                    "created_utc": record.get("created_utc"),
                    "id": record.get("id") or record.get("name") or f"row_{line_num}",
                }
            )

    return pd.DataFrame(rows)


def plot_clusters_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_file: str,
    title: str,
    seed: int,
) -> None:
    """Project embeddings to 2D with UMAP and save cluster scatter plot.

    Note:
    This plot is for visualization only; DBSCAN clustering is already done
    in the original embedding space.
    """
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=seed)
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    unique_labels = sorted(set(labels.tolist()))
    if not unique_labels:
        unique_labels = [-1]

    if len(unique_labels) == 1:
        color_map = {unique_labels[0]: "#1f77b4"}
    else:
        denom = max(1, len(unique_labels) - 1)
        color_map = {
            label: plt.cm.tab20(i / denom)
            for i, label in enumerate(unique_labels)
        }

    for label in unique_labels:
        mask = labels == label
        pts = reduced[mask]
        if pts.size == 0:
            continue
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=18,
            alpha=0.8,
            c=[color_map[label]],
            label=label_name,
        )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best", fontsize="small", markerscale=1.5)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def save_cluster_preview(
    df_slice: pd.DataFrame,
    labels: np.ndarray,
    output_file: str,
    max_preview_length: int,
) -> None:
    """Write readable per-cluster previews for manual inspection.

    Scenario:
    Analysts can quickly inspect top rows of each cluster before deep dive.
    """
    working = df_slice.copy()
    working["cluster"] = labels

    with open(output_file, "w", encoding="utf-8") as handle:
        grouped = working.groupby("cluster", dropna=False)
        for cluster_id, chunk in sorted(grouped, key=lambda item: (item[0] == -1, item[0])):
            handle.write(f"\n=== Cluster {cluster_id} ({len(chunk)} items) ===\n")
            for _, row in chunk.head(10).iterrows():
                preview = row["text"][:max_preview_length]
                handle.write(f"[{row['author']}] {preview}\n")
                handle.write(
                    f"  post_type={row['post_type']} subreddit={row['subreddit']} created_utc={row['created_utc']}\n"
                )


def run_bertopic(
    texts: List[str],
    embeddings: np.ndarray,
    output_prefix: str,
    min_topic_size: int,
    seed: int,
) -> None:
    """Run BERTopic on precomputed embeddings and export topic artifacts.

    Output files:
    - *_bertopic_topics.csv: topic-level summary
    - *_bertopic_top_words.txt: top words per topic
    - *_bertopic_doc_topics.csv: document-to-topic assignments
    """
    if len(texts) < 5:
        # Too few documents for stable topic extraction.
        return

    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
        nr_topics="auto",
        # Embeddings are already computed with Qwen model outside BERTopic.
        embedding_model=None,
        umap_model=umap.UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        ),
    )

    topics, _ = topic_model.fit_transform(texts, embeddings)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(f"{output_prefix}_bertopic_topics.csv", index=False)

    with open(f"{output_prefix}_bertopic_top_words.txt", "w", encoding="utf-8") as handle:
        for topic_id in topic_info["Topic"].tolist():
            if int(topic_id) == -1:
                continue
            words = topic_model.get_topic(int(topic_id)) or []
            top_words = ", ".join(word for word, _ in words[:10])
            handle.write(f"Topic {topic_id}: {top_words}\n")

    doc_topics = pd.DataFrame({"text": texts, "topic": topics})
    doc_topics.to_csv(f"{output_prefix}_bertopic_doc_topics.csv", index=False)


def main() -> None:
    """Run suspicious-subset clustering and topic modeling pipeline.

    Pipeline summary:
    1) load suspicious authors
    2) filter sampled JSONL to those authors
    3) embed separately for comments and submissions
    4) cluster with DBSCAN + visualize with UMAP
    5) run BERTopic per post type
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    suspicious_authors = load_suspicious_authors(args.suspicious_accounts_file)
    print(f"Loaded suspicious authors: {len(suspicious_authors)}")

    df = load_filtered_posts(args.input_file, suspicious_authors, min_words=args.min_words)
    if df.empty:
        raise SystemExit("No posts available for suspicious accounts after filtering.")

    print(f"Posts retained for clustering: {len(df)}")

    if args.disable_flash_attention and torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
        # Force non-flash SDP kernels when troubleshooting CUDA memory errors.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    # Requested embedding backbone for clustering/topic workflow.
    model = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_length
    if torch.cuda.is_available() and not args.disable_fp16:
        # fp16 significantly lowers activation memory during embedding inference.
        model.half()

    all_assignments: List[pd.DataFrame] = []

    for post_type in ["comment", "submission"]:
        # Separate analyses keep comment and submission narratives disentangled.
        df_slice = df[df["post_type"] == post_type].reset_index(drop=True)
        if df_slice.empty:
            print(f"Skipping {post_type}: no posts.")
            continue

        print(f"Embedding {len(df_slice)} {post_type} posts using {args.embedding_model}...")
        embeddings = model.encode(
            df_slice["text"].tolist(),
            batch_size=args.batch_size,
            show_progress_bar=True,
            # Normalized vectors align with cosine-distance based clustering.
            normalize_embeddings=True,
        )

        # DBSCAN finds dense groups and marks outliers as -1 (noise).
        clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine")
        labels = clustering.fit_predict(embeddings)

        assignments = df_slice.copy()
        assignments["cluster"] = labels
        all_assignments.append(assignments)

        assignments_path = os.path.join(args.output_dir, f"{post_type}_cluster_assignments.csv")
        assignments.to_csv(assignments_path, index=False)
        print(f"Saved assignments: {assignments_path}")

        plot_path = os.path.join(args.output_dir, f"{post_type}_clusters_umap.png")
        plot_clusters_umap(
            embeddings=embeddings,
            labels=labels,
            output_file=plot_path,
            title=f"Suspicious {post_type.title()} Clusters (UMAP + DBSCAN)",
            seed=args.seed,
        )
        print(f"Saved plot: {plot_path}")

        preview_path = os.path.join(args.output_dir, f"{post_type}_clusters_preview.txt")
        save_cluster_preview(df_slice, labels, preview_path, args.max_preview_length)
        print(f"Saved preview: {preview_path}")

        topic_prefix = os.path.join(args.output_dir, post_type)
        run_bertopic(
            texts=df_slice["text"].tolist(),
            embeddings=embeddings,
            output_prefix=topic_prefix,
            min_topic_size=args.bertopic_min_topic_size,
            seed=args.seed,
        )
        print(f"Saved BERTopic outputs for: {post_type}")

    if not all_assignments:
        raise SystemExit("No comments or submissions available for clustering.")

    merged = pd.concat(all_assignments, ignore_index=True)
    merged_path = os.path.join(args.output_dir, "suspicious_cluster_assignments_all.csv")
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged assignments: {merged_path}")


if __name__ == "__main__":
    main()
