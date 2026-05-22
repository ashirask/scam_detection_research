import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from transformers import AutoTokenizer


# Default source is sampled JSONL, but this stage only keeps rows from suspicious authors.
DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
DEFAULT_SUSPICIOUS_ACCOUNTS = str(Path(__file__).resolve().parent / "output" / "suspicious_accounts.csv")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")
# Placeholder content should not participate in embedding/clustering.
SKIP_TEXT = {"[removed]", "[deleted]"}


def parse_args() -> argparse.Namespace:
    """Define CLI options for suspicious-account clustering and topic modeling.

    Methodology updates in this version:
    1) Uses all-mpnet-base-v2 instead of Qwen embedding backbone.
    2) Uses chunked-token embedding for long posts (> model max length).
    3) Aggregates chunk embeddings by mean + renormalization.
    4) Exports cluster-to-topic breakdown files for interpretability.

    Returns:
        argparse.Namespace with runtime options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cluster posts from suspicious Reddit accounts using mpnet embeddings "
            "with long-text chunk averaging and BERTopic."
        )
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
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="SentenceTransformer encode batch size.")
    parser.add_argument(
        "--chunk-size-tokens",
        type=int,
        default=512,
        help="Token chunk length used for long-post embedding chunks.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.30,
        help="DBSCAN epsilon in cosine distance space.",
    )
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min samples.")
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

    Expected input: CSV containing an 'author' column.
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


def split_text_into_token_chunks(text: str, tokenizer: AutoTokenizer, chunk_size_tokens: int) -> List[str]:
    """Split text into tokenizer-aware chunks so long posts can exceed model length.

    Why this is needed:
    all-mpnet-base-v2 effectively handles up to ~512 tokens per forward pass.
    For longer posts, we split into 512-token windows and later average embeddings.

    Args:
        text: Full normalized post text.
        tokenizer: Hugging Face tokenizer matching embedding model.
        chunk_size_tokens: Max tokens per chunk (typically 512).

    Returns:
        List of decoded text chunks. Returns at least one chunk for non-empty text.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks: List[str] = []
    for i in range(0, len(token_ids), chunk_size_tokens):
        chunk_ids = token_ids[i : i + chunk_size_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def encode_texts_with_chunk_averaging(
    texts: List[str],
    model: SentenceTransformer,
    tokenizer: AutoTokenizer,
    chunk_size_tokens: int,
    batch_size: int,
) -> np.ndarray:
    """Encode long texts by averaging chunk-level semantic embeddings.

    Process:
    1) Split each text into token chunks of fixed size.
    2) Encode all chunks in batches with normalized embeddings.
    3) Aggregate chunk vectors per original text using mean.
    4) Re-normalize final mean vectors to unit length.

    Args:
        texts: Original post texts.
        model: SentenceTransformer embedding model.
        tokenizer: Matching tokenizer used for chunking.
        chunk_size_tokens: Max tokens per chunk.
        batch_size: Encoding batch size for chunk texts.

    Returns:
        Normalized float32 embedding matrix [len(texts), dim].
    """
    parent_index_for_chunk: List[int] = []
    chunk_texts: List[str] = []

    for parent_idx, text in enumerate(texts):
        chunks = split_text_into_token_chunks(text, tokenizer, chunk_size_tokens)

        # Ensure every post contributes at least one embedding, even if tokenizer output is empty.
        if not chunks:
            chunks = [text]

        for chunk in chunks:
            parent_index_for_chunk.append(parent_idx)
            chunk_texts.append(chunk)

    # normalize_embeddings=True ensures each chunk vector is unit norm.
    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)

    n_posts = len(texts)
    dim = chunk_embeddings.shape[1]

    # Sum chunk vectors per original post, then divide by chunk counts for mean pooling.
    sums = np.zeros((n_posts, dim), dtype=np.float32)
    counts = np.zeros(n_posts, dtype=np.int32)

    for chunk_idx, parent_idx in enumerate(parent_index_for_chunk):
        sums[parent_idx] += chunk_embeddings[chunk_idx]
        counts[parent_idx] += 1

    counts[counts == 0] = 1
    means = sums / counts[:, None]

    # Re-normalize so cosine distance in clustering behaves correctly.
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (means / norms).astype(np.float32)


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


def plot_clusters_umap_with_topics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_topic_breakdown: pd.DataFrame,
    output_file: str,
    title: str,
    seed: int,
) -> None:
    """Save a UMAP cluster plot annotated with each cluster's dominant BERTopic label.

    This view preserves cluster geometry/colors and overlays a short dominant-topic
    tag at each cluster centroid so cluster semantics are visible in one figure.
    """
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=seed)
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 9))

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

    dominant_rows = (
        cluster_topic_breakdown.sort_values(["cluster", "n_docs"], ascending=[True, False])
        .groupby("cluster", as_index=False)
        .first()
    )
    cluster_to_topic_text = {
        int(row["cluster"]): f"T{int(row['topic'])}: {str(row['topic_label'])}"
        for _, row in dominant_rows.iterrows()
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
            alpha=0.82,
            c=[color_map[label]],
            label=label_name,
        )

        if label == -1:
            continue

        # Annotate each cluster at centroid with dominant BERTopic signal.
        center_x = float(np.mean(pts[:, 0]))
        center_y = float(np.mean(pts[:, 1]))
        topic_text = cluster_to_topic_text.get(int(label), "Topic N/A")
        short_text = topic_text[:60] + ("..." if len(topic_text) > 60 else "")
        plt.text(
            center_x,
            center_y,
            f"C{int(label)} | {short_text}",
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
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
    """Write readable per-cluster previews for manual inspection."""
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
) -> Tuple[np.ndarray | None, pd.DataFrame | None]:
    """Run BERTopic on precomputed embeddings and export topic artifacts.

    Output files:
    - *_bertopic_topics.csv: topic-level summary
    - *_bertopic_top_words.txt: top words per topic
    - *_bertopic_doc_topics.csv: document-to-topic assignments

    Returns:
        Tuple (topics_array, topic_info_df). If too few docs, returns (None, None).
    """
    if len(texts) < 5:
        # Too few documents for stable topic extraction.
        return None, None

    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
        nr_topics="auto",
        # Embeddings are already computed with mpnet outside BERTopic.
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

    return np.asarray(topics), topic_info


def build_cluster_topic_breakdown(
    labels: np.ndarray,
    topics: np.ndarray,
    topic_info: pd.DataFrame,
) -> pd.DataFrame:
    """Build an interpretable mapping of which topics appear in each cluster.

    Args:
        labels: DBSCAN cluster labels per document.
        topics: BERTopic topic assignment per document.
        topic_info: BERTopic topic metadata table.

    Returns:
        DataFrame with per-(cluster, topic) counts and within-cluster percentages.
    """
    label_topic = pd.DataFrame({"cluster": labels.astype(int), "topic": topics.astype(int)})

    # Prefer BERTopic's Name when present, otherwise fallback to generic label.
    topic_label_map: Dict[int, str] = {}
    if "Topic" in topic_info.columns:
        for _, row in topic_info.iterrows():
            topic_id = int(row["Topic"])
            if "Name" in topic_info.columns and pd.notna(row["Name"]):
                topic_label_map[topic_id] = str(row["Name"])
            else:
                topic_label_map[topic_id] = f"Topic {topic_id}"

    group = (
        label_topic.groupby(["cluster", "topic"], dropna=False)
        .size()
        .reset_index(name="n_docs")
        .sort_values(["cluster", "n_docs"], ascending=[True, False])
    )
    cluster_sizes = label_topic.groupby("cluster").size().rename("cluster_size")

    group["cluster_size"] = group["cluster"].map(cluster_sizes)
    group["pct_in_cluster"] = group["n_docs"] / group["cluster_size"]
    group["topic_label"] = group["topic"].map(lambda t: topic_label_map.get(int(t), f"Topic {int(t)}"))

    return group[
        ["cluster", "cluster_size", "topic", "topic_label", "n_docs", "pct_in_cluster"]
    ].reset_index(drop=True)


def main() -> None:
    """Run suspicious-subset clustering and topic modeling pipeline.

    Pipeline summary:
    1) load suspicious authors
    2) filter sampled JSONL to those authors
    3) embed comments/submissions with mpnet + chunk averaging
    4) cluster with DBSCAN + visualize with UMAP
    5) run BERTopic per post type
    6) export cluster-to-topic inclusion tables
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    suspicious_authors = load_suspicious_authors(args.suspicious_accounts_file)
    print(f"Loaded suspicious authors: {len(suspicious_authors)}")

    df = load_filtered_posts(args.input_file, suspicious_authors, min_words=args.min_words)
    if df.empty:
        raise SystemExit("No posts available for suspicious accounts after filtering.")

    print(f"Posts retained for clustering: {len(df)}")

    model = SentenceTransformer(args.embedding_model)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    all_assignments: List[pd.DataFrame] = []

    for post_type in ["comment", "submission"]:
        # Separate analyses keep comment and submission narratives disentangled.
        df_slice = df[df["post_type"] == post_type].reset_index(drop=True)
        if df_slice.empty:
            print(f"Skipping {post_type}: no posts.")
            continue

        print(
            f"Embedding {len(df_slice)} {post_type} posts using {args.embedding_model} "
            f"with chunk-size={args.chunk_size_tokens}..."
        )
        embeddings = encode_texts_with_chunk_averaging(
            texts=df_slice["text"].tolist(),
            model=model,
            tokenizer=tokenizer,
            chunk_size_tokens=args.chunk_size_tokens,
            batch_size=args.batch_size,
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
        topics, topic_info = run_bertopic(
            texts=df_slice["text"].tolist(),
            embeddings=embeddings,
            output_prefix=topic_prefix,
            min_topic_size=args.bertopic_min_topic_size,
            seed=args.seed,
        )
        if topics is not None and topic_info is not None:
            print(f"Saved BERTopic outputs for: {post_type}")

            cluster_topic_breakdown = build_cluster_topic_breakdown(labels, topics, topic_info)
            breakdown_path = os.path.join(args.output_dir, f"{post_type}_cluster_topic_breakdown.csv")
            cluster_topic_breakdown.to_csv(breakdown_path, index=False)
            print(f"Saved cluster-topic breakdown: {breakdown_path}")

            # Dominant topic view makes quick review easier in spreadsheets.
            dominant = (
                cluster_topic_breakdown.sort_values(["cluster", "n_docs"], ascending=[True, False])
                .groupby("cluster", as_index=False)
                .first()
            )
            dominant_path = os.path.join(args.output_dir, f"{post_type}_cluster_dominant_topic.csv")
            dominant.to_csv(dominant_path, index=False)
            print(f"Saved dominant-topic summary: {dominant_path}")

            cluster_topic_plot_path = os.path.join(args.output_dir, f"{post_type}_clusters_umap_with_topics.png")
            plot_clusters_umap_with_topics(
                embeddings=embeddings,
                labels=labels,
                cluster_topic_breakdown=cluster_topic_breakdown,
                output_file=cluster_topic_plot_path,
                title=f"Suspicious {post_type.title()} Clusters with Dominant BERTopic Labels",
                seed=args.seed,
            )
            print(f"Saved cluster-topic UMAP: {cluster_topic_plot_path}")

    if not all_assignments:
        raise SystemExit("No comments or submissions available for clustering.")

    merged = pd.concat(all_assignments, ignore_index=True)
    merged_path = os.path.join(args.output_dir, "suspicious_cluster_assignments_all.csv")
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged assignments: {merged_path}")


if __name__ == "__main__":
    main()
