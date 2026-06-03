#!/usr/bin/env python3
"""Direct co-submission detection with TF-IDF.

This script groups comments by the submission they were posted on (link_id),
builds one document per user from submission tokens, and flags user pairs whose
cosine similarity exceeds an observed percentile threshold.

The input is comments JSONL, including the output produced by
extract_comments_for_posts.py.
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_df_value(value):
    """Parse a TF-IDF DF argument as int or float."""
    text = str(value).strip()
    if "." in text:
        return float(text)
    return int(text)


def normalize_fullname(value: Optional[str]) -> str:
    """Strip Reddit fullname prefixes like t1_, t2_, or t3_."""
    if not value:
        return ""
    text = str(value).strip()
    if text.startswith(("t1_", "t2_", "t3_")):
        return text[3:]
    return text


def truncate_text(text, limit=500):
    """Create a compact preview string for report output."""
    if not text:
        return ""
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def load_comments_data(filepath, verbose=False, preview_limit=12):
    """Load submission tokens from comment JSONL rows."""
    author_tokens = defaultdict(list)
    author_metadata = defaultdict(list)
    author_sources = defaultdict(list)
    processed_count = 0
    debug_events = 0

    print(f"Loading comments data from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                comment = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                if verbose and debug_events < preview_limit:
                    print(f"[comments] line {line_no}: invalid JSON, skipped")
                    debug_events += 1
                continue

            author = (comment.get("author") or "").strip()
            if not author or author in SKIP_AUTHORS:
                if verbose and debug_events < preview_limit:
                    print(f"[comments] line {line_no}: missing/deleted author, skipped")
                    debug_events += 1
                continue

            submission_id = normalize_fullname(comment.get("link_id"))
            if not submission_id:
                if verbose and debug_events < preview_limit:
                    print(f"[comments] line {line_no}: missing link_id, skipped")
                    debug_events += 1
                continue

            submission_token = submission_id.lower()
            subreddit = (comment.get("subreddit") or "").strip()
            body = (comment.get("body") or "").strip()

            author_tokens[author].append(submission_token)
            author_metadata[author].append(
                {
                    "feature": submission_token,
                    "feature_type": "submission",
                    "submission_id": submission_token,
                    "subreddit": subreddit,
                    "body": body,
                    "comment_id": normalize_fullname(comment.get("id") or comment.get("name")),
                    "permalink": comment.get("permalink", ""),
                    "created_utc": comment.get("created_utc"),
                }
            )
            author_sources[author].append(
                {
                    "source_type": "comment",
                    "submission_id": submission_token,
                    "subreddit": subreddit,
                    "body": body,
                    "comment_id": normalize_fullname(comment.get("id") or comment.get("name")),
                    "permalink": comment.get("permalink", ""),
                    "created_utc": comment.get("created_utc"),
                }
            )

            if verbose and debug_events < preview_limit:
                body_preview = truncate_text(body, 200)
                print(
                    f"[comments] line {line_no}: author={author} submission={submission_token} "
                    f"subreddit={subreddit} body={body_preview!r}"
                )
                debug_events += 1

            processed_count = line_no
            if processed_count % 10000 == 0:
                print(f"  Processed {processed_count} comments...")

    print(f"  Total comments processed: {processed_count}")
    return author_tokens, author_metadata, author_sources


def filter_authors(author_tokens, min_comments=5, count_mode="total"):
    """Keep users with enough submission tokens from their comments."""
    if count_mode not in ("unique", "total"):
        raise ValueError("count_mode must be 'unique' or 'total'")

    descriptor = "unique" if count_mode == "unique" else "total"
    print(f"\nFiltering users with < {min_comments} {descriptor} submission tokens...")

    initial_count = len(author_tokens)
    filtered = {}
    for user, tokens in author_tokens.items():
        if count_mode == "unique":
            count = len(set(tokens))
            kept_tokens = sorted(set(tokens))
        else:
            count = len(tokens)
            kept_tokens = list(tokens)

        if count >= min_comments:
            filtered[user] = kept_tokens

    removed = initial_count - len(filtered)
    print(f"  Initial users: {initial_count}")
    print(f"  Users with >= {min_comments} ({descriptor}) submission tokens: {len(filtered)}")
    print(f"  Users removed: {removed}")

    return filtered


def build_user_matrix(user_tokens_list, min_df=2, max_df=0.9):
    """Build a TF-IDF matrix with one user per document and submission IDs as features."""
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        norm="l2",
    )
    # in context for suspicious pairs in the report this method will be called with user_tokens_list containing only the users that passed the initial filtering, so we can build the TF-IDF matrix directly without worrying about empty documents at this stage
    documents = [" ".join(tokens) for tokens in user_tokens_list]
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


def filter_authors_by_tfidf_features(author_list, tfidf_matrix, min_features=2):
    """Remove authors that have fewer than min_features non-zero TF-IDF features."""
    feature_counts = np.diff(tfidf_matrix.indptr)
    keep_mask = feature_counts >= min_features

    kept_authors = [author for author, keep in zip(author_list, keep_mask) if keep]
    filtered_matrix = tfidf_matrix[keep_mask]

    removed = len(author_list) - len(kept_authors)
    print(f"\nFiltering authors with < {min_features} non-zero TF-IDF features...")
    print(f"  Authors before TF-IDF pruning filter: {len(author_list)}")
    print(f"  Authors kept after TF-IDF pruning filter: {len(kept_authors)}")
    print(f"  Authors removed after TF-IDF pruning filter: {removed}")

    return kept_authors, filtered_matrix


def compute_threshold(similarities, percentile=99, method="observed", sample_size=10000, seed=None):
    """Compute the percentile threshold from observed user-user similarities."""
    if similarities.shape[0] < 2:
        raise ValueError("Need at least two users to compute similarities.")

    observed_sims = similarities[np.triu_indices_from(similarities, k=1)]
    if observed_sims.size == 0:
        return observed_sims, 0.0

    if method == "observed":
        threshold = np.percentile(observed_sims, percentile)
        return observed_sims, threshold

    if method == "sampled_pairs":
        rng = np.random.default_rng(seed)
        k = min(sample_size, observed_sims.size)
        sampled = (
            observed_sims
            if k == observed_sims.size
            else observed_sims[rng.choice(observed_sims.size, size=k, replace=False)]
        )
        threshold = np.percentile(sampled, percentile)
        return observed_sims, threshold

    raise ValueError(f"Unknown threshold method: {method}")


def detect_suspicious_pairs(user_metadata_dict, similarities, user_list, threshold, baseline_sims):
    """Return user pairs whose cosine similarity is above the threshold."""
    suspicious_pairs = []

    for i in range(len(user_list)):
        for j in range(i + 1, len(user_list)):
            sim = similarities[i, j]
            if sim > threshold:
                user_a = user_list[i]
                user_b = user_list[j]

                feats_a = set(item["feature"] for item in user_metadata_dict[user_a])
                feats_b = set(item["feature"] for item in user_metadata_dict[user_b])
                shared = feats_a.intersection(feats_b)

                meta_a = user_metadata_dict[user_a]
                meta_b = user_metadata_dict[user_b]
                subreddits_a = ", ".join(sorted(set(item["subreddit"] for item in meta_a if item.get("subreddit"))))
                subreddits_b = ", ".join(sorted(set(item["subreddit"] for item in meta_b if item.get("subreddit"))))
                submissions_a = ", ".join(sorted(feats_a)[:10])
                submissions_b = ", ".join(sorted(feats_b)[:10])

                suspicious_pairs.append(
                    {
                        "user_1": user_a,
                        "user_2": user_b,
                        "cosine_similarity": sim,
                        "percentile_rank": stats.percentileofscore(baseline_sims, sim) if baseline_sims.size else 0.0,
                        "user_1_subreddits": subreddits_a,
                        "user_2_subreddits": subreddits_b,
                        "user_1_submissions": submissions_a,
                        "user_2_submissions": submissions_b,
                        "shared_submissions": "; ".join(sorted(shared)[:10]),
                        "shared_submission_count": len(shared),
                    }
                )

    return sorted(suspicious_pairs, key=lambda item: item["cosine_similarity"], reverse=True)


def build_user_preview(user_sources, limit=5):
    """Create a compact preview of a user's comment text only."""
    preview_items = []
    for item in user_sources[:limit]:
        body = truncate_text(item.get("body", ""), 250)
        if body:
            preview_items.append(body)
    return " || ".join(preview_items)


def write_user_report(report_path, suspicious_pairs, user_sources_dict):
    """Write a Markdown report that lists each user once with their submission context."""
    lines = []
    lines.append("# Co-Submission User Context Report")
    lines.append("")

    if not suspicious_pairs:
        lines.append("No suspicious user pairs were found above the selected threshold.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    seen = set()
    users = []
    for pair in suspicious_pairs:
        for user in (pair["user_1"], pair["user_2"]):
            if user not in seen:
                seen.add(user)
                users.append(user)

    for user in users:
        sources = user_sources_dict.get(user, [])
        unique_submissions = sorted({item.get("submission_id", "") for item in sources if item.get("submission_id")})

        lines.append(f"## {user}")
        lines.append("")
        lines.append(f"- Total comments: {len(sources)}")
        lines.append(f"- Unique submissions commented on: {len(unique_submissions)}")
        if unique_submissions:
            lines.append(f"- Submission IDs: {', '.join(unique_submissions[:10])}")
        lines.append("")

        if not sources:
            lines.append("_No submission context captured for this user._")
            lines.append("")
            continue

        for idx, item in enumerate(sources, start=1):
            lines.append(f"### Source {idx}")
            lines.append("")
            lines.append(f"- Type: {item.get('source_type', 'source')}")
            lines.append(f"- Submission ID: {item.get('submission_id', 'N/A')}")
            lines.append(f"- Subreddit: r/{item.get('subreddit', 'N/A')}")
            if item.get("comment_id"):
                lines.append(f"- Comment ID: {item.get('comment_id')}")
            if item.get("permalink"):
                lines.append(f"- Permalink: {item.get('permalink')}")
            lines.append("")
            if item.get("body"):
                lines.append("- Comment text:")
                lines.append("```text")
                lines.append(item.get("body", "").strip() or "[empty text]")
                lines.append("```")
                lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_output_stem(args):
    """Create a descriptive filename stem from CLI arguments."""
    parts = [
        f"min-comments={args.min_comments}",
        f"min-df={args.min_df}",
        f"max-df={args.max_df}",
    ]
    if args.run_id:
        parts.append(f"run-id={args.run_id}")
    return "co_submission__" + "_".join(str(part).replace(" ", "-") for part in parts)


def main():
    parser = argparse.ArgumentParser(description="Detect suspicious users by shared submission-commenting behavior")
    parser.add_argument("--input", nargs="+", required=True, help="Path(s) to input comment JSONL file(s)")
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--min-comments", type=int, default=5, help="Minimum total comment tokens required to keep a user")
    parser.add_argument("--count-mode", choices=["unique", "total"], default="total", help="Count submission tokens by unique or total occurrences")
    parser.add_argument("--min-df", type=parse_df_value, default=2, help="Minimum TF-IDF document frequency for submission tokens")
    parser.add_argument("--max-df", type=parse_df_value, default=0.9, help="Maximum TF-IDF document frequency for submission tokens")
    parser.add_argument("--null-method", choices=["observed", "sampled_pairs"], default="observed", help="How to derive the threshold from observed similarities")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of observed pairs to sample when using sampled_pairs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampled pair selection")
    parser.add_argument("--percentile", type=int, default=99, help="Percentile threshold for suspicious pairs")
    parser.add_argument("--verbose", action="store_true", help="Print detailed loading and vectorization traces")
    parser.add_argument("--preview-limit", type=int, default=12, help="Maximum number of verbose debug events per loader")
    parser.add_argument("--run-id", default="", help="Optional label to include in output filenames")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_user_tokens = defaultdict(list)
    all_user_metadata = defaultdict(list)
    all_user_sources = defaultdict(list)

    for input_file in args.input:
        #the load_comments_data function returns a dictionary of users and their associated data, we need to aggregate this data across multiple input files if provided, so we extend the lists for each user in the all_user_* dictionaries
        user_tokens, user_metadata, user_sources = load_comments_data(
            input_file,
            verbose=args.verbose,
            preview_limit=args.preview_limit,
        )

        for user, tokens in user_tokens.items():
            all_user_tokens[user].extend(tokens)
        for user, metadata in user_metadata.items():
            all_user_metadata[user].extend(metadata)
        for user, sources in user_sources.items():
            all_user_sources[user].extend(sources)

    print(f"\nLoaded users before filtering: {len(all_user_tokens)}")
    filtered_users = filter_authors(all_user_tokens, args.min_comments, count_mode=args.count_mode)
    if len(filtered_users) < 2:
        print("\nError: Need at least 2 users with enough submission tokens")
        return

    #user_list includes all the users that passed the initial filtering based on min_comments, and user_tokens_list is a list of their corresponding submission token lists, which we will use to build the TF-IDF matrix where each row corresponds to a user and each column corresponds to a submission token feature
    user_list = list(filtered_users.keys())
    user_tokens_list = [filtered_users[user] for user in user_list]

    #IDF is calculated based on the number of comments that mention each submission token across all users, so the min_df and max_df parameters will filter out tokens that are too rare or too common across the user base, which helps to focus on more distinctive submission tokens when computing cosine similarity between users
    print(f"\nComputing TF-IDF vectors for {len(user_list)} users...")
    print(f"  min_df={args.min_df}, max_df={args.max_df}")
    try:
        _, tfidf_matrix = build_user_matrix(
            user_tokens_list,
            min_df=args.min_df,
            max_df=args.max_df,
        )
    except ValueError as exc:
        print(f"\nError while building TF-IDF matrix: {exc}")
        return

    author_list = list(filtered_users.keys())
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Non-zero entries: {tfidf_matrix.nnz}")
    author_list, tfidf_matrix = filter_authors_by_tfidf_features(author_list, tfidf_matrix, min_features=2)
    if len(author_list) < 2:
        print('\nError: Need at least 2 authors with sufficient TF-IDF features after pruning')
        return

    print(f"  TF-IDF matrix shape after row pruning: {tfidf_matrix.shape}")
    print(f"  Non-zero entries after row pruning: {tfidf_matrix.nnz}")
    if tfidf_matrix.shape[1] == 0:
        print("\nError: No submission features remained after TF-IDF pruning")
        return

    filtered_author_metadata = {author: all_user_metadata.get(author, []) for author in author_list}

    similarities = cosine_similarity(tfidf_matrix)
    observed_sims, threshold = compute_threshold(
        similarities,
        percentile=args.percentile,
        method=args.null_method,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"  {args.percentile}th percentile threshold from {args.null_method}: {threshold:.4f}")

    print("\nDetecting suspicious user pairs...")
    suspicious_pairs = detect_suspicious_pairs(filtered_author_metadata, similarities, author_list, threshold, observed_sims)
    print(f"Found {len(suspicious_pairs)} suspicious user pairs")

    file_stem = build_output_stem(args)
    csv_path = output_dir / f"{file_stem}.csv"
    report_path = output_dir / f"{file_stem}.md"

    if suspicious_pairs:
        for pair in suspicious_pairs:
            pair["user_1_preview"] = build_user_preview(all_user_sources.get(pair["user_1"], []))
            pair["user_2_preview"] = build_user_preview(all_user_sources.get(pair["user_2"], []))
            pair["user_1_source_count"] = len(all_user_sources.get(pair["user_1"], []))
            pair["user_2_source_count"] = len(all_user_sources.get(pair["user_2"], []))

        results_df = pd.DataFrame(suspicious_pairs)
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        print("\nTop 10 suspicious pairs:")
        print(results_df.head(10).to_string())
    else:
        print("No suspicious pairs found above threshold")

    write_user_report(report_path, suspicious_pairs, all_user_sources)
    print(f"User context report saved to: {report_path}")

    print("\nGenerating observed similarity plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(observed_sims, bins=50, density=True, alpha=0.6, label="Observed Similarities", color="teal")

    if len(observed_sims) > 1 and float(np.max(observed_sims)) > float(np.min(observed_sims)):
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(observed_sims)
        x_min = float(np.min(observed_sims))
        x_max = float(np.max(observed_sims))
        x_range = np.linspace(x_min, x_max if x_max > x_min else x_min + 1e-6, 200)
        ax.plot(x_range, kde(x_range), "b-", linewidth=2, label="KDE (Observed)")

    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"{args.percentile}th Percentile: {threshold:.4f}")
    ax.scatter(observed_sims, np.zeros_like(observed_sims), alpha=0.3, s=20, color="green", label="Observed Pairs")
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Direct Co-Submission Sharing: Observed Similarity Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plot_path = output_dir / f"{file_stem}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total users analyzed: {len(user_list)}")
    print(f"Total unique submission tokens: {len(set(token for tokens in user_tokens_list for token in tokens))}")
    print(f"Threshold ({args.percentile}th percentile of observed similarities): {threshold:.4f}")
    print(f"Suspicious pairs found: {len(suspicious_pairs)}")
    print(f"Actual similarity range: [{observed_sims.min():.4f}, {observed_sims.max():.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
