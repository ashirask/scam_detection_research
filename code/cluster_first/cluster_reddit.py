import argparse
import json
from collections import defaultdict
from pathlib import Path
import os

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import umap


# Default to the small sample so the script is fast to test and easy to override.
DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
# Skip placeholder bodies that do not represent real authored content.
SKIP_BODIES = {"[removed]", "[deleted]"}


def normalize_text(text):
    """Collapse repeated whitespace so semantic clustering starts from clean text."""
    return " ".join(text.split())


def load_records(input_file):
    """Yield cleaned records that have usable text and a real author."""
    with open(input_file, encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)

            # Normalize the comment text before embedding it.
            text = normalize_text(data.get("body", "").strip())
            author = data.get("author")

            # Drop empty comments and removed placeholders.
            if not text or text in SKIP_BODIES:
                continue

            # Drop deleted authors so we only compare real accounts.
            if not author or author == "[deleted]":
                continue

            yield data, text, author


def main():
    """Cluster semantically similar comments and print the strongest groups."""
    parser = argparse.ArgumentParser(description="Cluster Reddit comments by semantic similarity.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="Path to a JSONL file of Reddit comments.")
    parser.add_argument("--eps", type=float, default=0.35, help="DBSCAN epsilon for cosine distance.")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN minimum samples per cluster.")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum cluster size to print.")
    parser.add_argument("--max-preview-length", type=int, default=150, help="Maximum text length to print for each item.")
    args = parser.parse_args()

    # Load the cleaned records once so we can reuse the text, author, and metadata.
    records = list(load_records(args.input_file))
    texts = [text for _, text, _ in records]
    authors = [author for _, _, author in records]
    metadata = [data for data, _, _ in records]

    print(f"Loaded {len(texts)} texts")

    # Create sentence embeddings for the filtered comment text.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Group comments that are close in embedding space.
    clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine")
    labels = clustering.fit_predict(embeddings)

    # Collect members by cluster label and ignore noise points.
    clusters = defaultdict(list)

    for index, label in enumerate(labels):
        if label == -1:
            continue
        clusters[label].append((texts[index], authors[index], metadata[index]))

    # Print only the larger clusters so the output stays readable.
    for label, items in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(items) < args.min_cluster_size:
            continue

        print(f"\n=== Cluster {label} ({len(items)} items) ===")
        for text, author, data in items[:5]:
            print(f"[{author}] {text[: args.max_preview_length]}")
            print(f"  subreddit={data.get('subreddit')} created_utc={data.get('created_utc')} link_id={data.get('link_id')}")

    # Save clustering results to a file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    cluster_output_file = os.path.join(output_dir, "clusters.txt")
    save_clusters_to_file(clusters, cluster_output_file)
    print(f"Clustering results saved to {cluster_output_file}")

    # Save the UMAP plot to a file
    plot_output_file = os.path.join(output_dir, "clusters_plot.png")
    plot_clusters(embeddings, labels, output_file=plot_output_file)


def save_clusters_to_file(clusters, output_file):
    """Save clustering results to a text file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for label, items in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0])):
            f.write(f"\n=== Cluster {label} ({len(items)} items) ===\n")
            for text, author, data in items:
                f.write(f"[{author}] {text}\n")
                f.write(f"  subreddit={data.get('subreddit')} created_utc={data.get('created_utc')} link_id={data.get('link_id')}\n")


def plot_clusters(embeddings, labels, output_file=None):
    """Visualize clusters using UMAP for dimensionality reduction and optionally save the plot."""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # Noise points
            color = "gray"
            label_name = "Noise"
        else:
            color = plt.cm.tab20(label / max(unique_labels))
            label_name = f"Cluster {label}"

        cluster_points = reduced_embeddings[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=label_name, s=10, alpha=0.7, c=[color])

    plt.title("UMAP Projection of Clusters")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best", markerscale=2, fontsize="small")

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    main()