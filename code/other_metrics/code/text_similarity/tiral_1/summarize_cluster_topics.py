import argparse
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


POST_TYPES = ("comment", "submission")


def extract_tail_int_values(file_path: str) -> List[int]:
    """Extract the last integer before trailing commas from each data row.

    This tolerates malformed CSV rows where inner quotes/commas can shift columns.
    Expected row shape still ends with ",<int>,,,,,".
    """
    values: List[int] = []
    tail_int_pattern = re.compile(r",\s*(-?\d+)\s*,*\s*$")

    with open(file_path, encoding="utf-8") as handle:
        # Skip header line.
        _ = handle.readline()
        for line in handle:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            match = tail_int_pattern.search(raw)
            if not match:
                continue
            values.append(int(match.group(1)))

    return values


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cluster-to-topic summarization.

    Returns:
        argparse.Namespace containing:
        - output_dir: Folder with clustering/BERTopic CSV outputs.
        - top_k: Number of top topics to include per cluster summary row.
    """
    parser = argparse.ArgumentParser(
        description="Build cluster-to-topic summaries from existing clustering and BERTopic outputs."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing *_cluster_assignments.csv and *_bertopic_doc_topics.csv files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many top BERTopic topics to keep per cluster.",
    )
    return parser.parse_args()


def file_paths(base_dir: str, post_type: str) -> Tuple[str, str, str]:
    """Build expected input file paths for a given post type.

    Args:
        base_dir: Output directory produced by the clustering pipeline.
        post_type: One of supported post types (e.g., comment/submission).

    Returns:
        Tuple of:
        - cluster assignments CSV path
        - BERTopic document-topic CSV path
        - BERTopic topic-info CSV path
    """
    cluster_path = os.path.join(base_dir, f"{post_type}_cluster_assignments.csv")
    doc_topic_path = os.path.join(base_dir, f"{post_type}_bertopic_doc_topics.csv")
    topic_info_path = os.path.join(base_dir, f"{post_type}_bertopic_topics.csv")
    return cluster_path, doc_topic_path, topic_info_path


def load_topic_labels(topic_info_path: str) -> Dict[int, str]:
    """Load human-readable topic labels from BERTopic topic info output.

    The BERTopic topic-info file can vary across versions and sometimes contains
    malformed rows. This loader is intentionally defensive:
    - returns an empty map if file/column is missing,
    - skips non-numeric topic ids,
    - prefers Name, then Representation, then a fallback label.

    Args:
        topic_info_path: Path to *_bertopic_topics.csv.

    Returns:
        Mapping from numeric topic id to display label.
    """
    if not os.path.isfile(topic_info_path):
        return {}

    topic_info = pd.read_csv(topic_info_path)
    if "Topic" not in topic_info.columns:
        return {}

    labels: Dict[int, str] = {}
    for _, row in topic_info.iterrows():
        topic_id_raw = pd.to_numeric(row["Topic"], errors="coerce")
        if pd.isna(topic_id_raw):
            continue
        topic_id = int(topic_id_raw)
        if "Name" in topic_info.columns and pd.notna(row["Name"]):
            labels[topic_id] = str(row["Name"])
        elif "Representation" in topic_info.columns and pd.notna(row["Representation"]):
            labels[topic_id] = str(row["Representation"])
        else:
            labels[topic_id] = f"Topic {topic_id}"
    return labels


def summarize_one_post_type(base_dir: str, post_type: str, top_k: int) -> pd.DataFrame:
    """Create a cluster-to-topic summary table for one post type.

    Workflow:
    1) locate expected cluster/topic files,
    2) parse robust integer cluster/topic sequences from line endings,
    3) align rows by position (truncate to shared minimum length),
    4) aggregate cluster-topic counts,
    5) emit one summary row per cluster with dominant/top-k topic statistics.

    Args:
        base_dir: Output directory containing model artifacts.
        post_type: Target subset, e.g. comment or submission.
        top_k: Number of top topics to keep per cluster.

    Returns:
        DataFrame containing per-cluster summary metrics.
        Returns empty DataFrame when required files for this post_type are missing.
    """
    cluster_path, doc_topic_path, topic_info_path = file_paths(base_dir, post_type)

    if not os.path.isfile(cluster_path):
        print(f"Skipping {post_type}: missing {cluster_path}")
        return pd.DataFrame()

    if not os.path.isfile(doc_topic_path):
        print(f"Skipping {post_type}: missing {doc_topic_path}")
        return pd.DataFrame()

    cluster_values = extract_tail_int_values(cluster_path)
    topic_values = extract_tail_int_values(doc_topic_path)

    if not cluster_values:
        raise SystemExit(f"Could not parse any cluster values from {cluster_path}")
    if not topic_values:
        raise SystemExit(f"Could not parse any topic values from {doc_topic_path}")

    # Keep row-wise alignment deterministic by trimming both arrays to common length.
    min_len = min(len(cluster_values), len(topic_values))
    if len(cluster_values) != len(topic_values):
        print(
            f"Warning: row count mismatch for {post_type} "
            f"(clusters={len(cluster_values)}, topics={len(topic_values)}). Using first {min_len} rows by order."
        )

    merged = pd.DataFrame(
        {
            "cluster": cluster_values[:min_len],
            "topic": topic_values[:min_len],
        }
    )

    topic_labels = load_topic_labels(topic_info_path)

    # Count documents per (cluster, topic) pair to compute topic distribution per cluster.
    group = (
        merged.groupby(["cluster", "topic"], dropna=False)
        .size()
        .reset_index(name="n_docs")
        .sort_values(["cluster", "n_docs"], ascending=[True, False])
    )

    cluster_totals = merged.groupby("cluster", dropna=False).size().rename("cluster_size")

    rows: List[Dict] = []
    for cluster_id, chunk in group.groupby("cluster", dropna=False):
        cluster_size = int(cluster_totals.loc[cluster_id])
        # Take top-k topics in this cluster and convert counts into in-cluster percentages.
        top = chunk.head(top_k).copy()
        top["pct"] = top["n_docs"] / cluster_size

        dominant_topic = int(top.iloc[0]["topic"])
        dominant_n = int(top.iloc[0]["n_docs"])
        dominant_pct = float(top.iloc[0]["pct"])

        # Compact text representation for quick human inspection in CSV viewers.
        top_topics_compact = []
        for _, r in top.iterrows():
            topic_id = int(r["topic"])
            label = topic_labels.get(topic_id, f"Topic {topic_id}")
            top_topics_compact.append(
                f"{topic_id} ({label}): {int(r['n_docs'])}/{cluster_size} ({float(r['pct']):.1%})"
            )

        rows.append(
            {
                "post_type": post_type,
                "cluster": int(cluster_id),
                "cluster_size": cluster_size,
                "dominant_topic": dominant_topic,
                "dominant_topic_label": topic_labels.get(dominant_topic, f"Topic {dominant_topic}"),
                "dominant_topic_docs": dominant_n,
                "dominant_topic_pct": round(dominant_pct, 6),
                "top_topics": " | ".join(top_topics_compact),
            }
        )

    return pd.DataFrame(rows).sort_values(["post_type", "cluster"])


def main() -> None:
    """Entry point: build and write cluster-topic summaries for all post types.

    For each configured post type, this function writes:
    - <post_type>_cluster_topic_summary.csv

    It also writes a combined file across all generated summaries:
    - cluster_topic_summary_all.csv

    Exits with a clear message if output directory is missing or no summaries
    can be generated.
    """
    args = parse_args()

    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"Output directory not found: {args.output_dir}")

    summaries: List[pd.DataFrame] = []
    for post_type in POST_TYPES:
        # Continue processing remaining post types even if one is missing.
        summary = summarize_one_post_type(args.output_dir, post_type, args.top_k)
        if not summary.empty:
            out_path = os.path.join(args.output_dir, f"{post_type}_cluster_topic_summary.csv")
            summary.to_csv(out_path, index=False)
            print(f"Saved: {out_path} ({len(summary)} clusters)")
            summaries.append(summary)

    if not summaries:
        raise SystemExit("No summaries were generated. Check that required files exist in --output-dir.")

    combined = pd.concat(summaries, ignore_index=True)
    combined_path = os.path.join(args.output_dir, "cluster_topic_summary_all.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Saved: {combined_path} ({len(combined)} clusters total)")


if __name__ == "__main__":
    main()
