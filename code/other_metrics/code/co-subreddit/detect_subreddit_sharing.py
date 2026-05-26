#!/usr/bin/env python3
"""Direct co-subreddit detection with TF-IDF.

This script mirrors the URL-based workflow, but the token for each post/comment
is the subreddit it was posted in. For each user, we build a document from the
subreddits they posted in, compute TF-IDF, derive a 99th-percentile baseline
from the observed pairwise similarity distribution, and then flag user pairs
above that threshold.
"""

import argparse
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')


def parse_df_value(value):
    """Parse a TF-IDF DF argument as int or float."""
    text = str(value).strip()
    if '.' in text:
        return float(text)
    return int(text)


def normalize_subreddit(subreddit):
    """Normalize a subreddit name so comparisons are consistent."""
    if not subreddit or not isinstance(subreddit, str):
        return ''
    text = subreddit.strip()
    if text.lower().startswith('r/'):
        text = text[2:]
    return text.lower()


def truncate_text(text, limit=500):
    """Create a compact preview string for report output."""
    if not text:
        return ''
    compact = ' '.join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3] + '...'


def load_posts_data(filepath, verbose=False, preview_limit=12):
    """Load subreddit tokens from posts JSONL rows.

    The sample files store subreddit in the plain `subreddit` field.
    """
    author_tokens = defaultdict(list)
    author_metadata = defaultdict(list)
    author_sources = defaultdict(list)
    processed_count = 0
    debug_events = 0

    print(f'Loading posts data from {filepath}...')
    with open(filepath, 'r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                post = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                if verbose and debug_events < preview_limit:
                    print(f'[posts] line {line_no}: invalid JSON, skipped')
                    debug_events += 1
                continue

            author = post.get('author')
            subreddit = normalize_subreddit(post.get('subreddit', ''))
            if not author or author == '[deleted]':
                if verbose and debug_events < preview_limit:
                    print(f'[posts] line {line_no}: missing/deleted author, skipped')
                    debug_events += 1
                continue
            if not subreddit:
                if verbose and debug_events < preview_limit:
                    print(f'[posts] line {line_no}: missing subreddit, skipped')
                    debug_events += 1
                continue

            author_tokens[author].append(subreddit)
            author_metadata[author].append({
                'feature': subreddit,
                'feature_type': 'subreddit',
                'subreddit': subreddit,
                'post_hint': post.get('post_hint', 'N/A'),
                'title': post.get('title', ''),
                'selftext': post.get('selftext', ''),
                'permalink': post.get('permalink', ''),
            })
            author_sources[author].append({
                'source_type': 'post',
                'subreddit': subreddit,
                'title': post.get('title', ''),
                'selftext': post.get('selftext', ''),
                'permalink': post.get('permalink', ''),
                'post_hint': post.get('post_hint', 'N/A'),
            })

            if verbose and debug_events < preview_limit:
                title_preview = truncate_text(post.get('title', ''), 200)
                selftext_preview = truncate_text(post.get('selftext', ''), 200)
                print(f'[posts] line {line_no}: author={author} subreddit={subreddit} title={title_preview!r} selftext={selftext_preview!r}')
                debug_events += 1

            processed_count = line_no
            if processed_count % 10000 == 0:
                print(f'  Processed {processed_count} posts...')

    print(f'  Total posts processed: {processed_count}')
    return author_tokens, author_metadata, author_sources


def load_comments_data(filepath, verbose=False, preview_limit=12):
    """Load subreddit tokens from comment JSONL rows."""
    author_tokens = defaultdict(list)
    author_metadata = defaultdict(list)
    author_sources = defaultdict(list)
    processed_count = 0
    debug_events = 0

    print(f'Loading comments data from {filepath}...')
    with open(filepath, 'r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                comment = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                if verbose and debug_events < preview_limit:
                    print(f'[comments] line {line_no}: invalid JSON, skipped')
                    debug_events += 1
                continue

            author = comment.get('author')
            subreddit = normalize_subreddit(comment.get('subreddit', ''))
            if not author or author == '[deleted]':
                if verbose and debug_events < preview_limit:
                    print(f'[comments] line {line_no}: missing/deleted author, skipped')
                    debug_events += 1
                continue
            if not subreddit:
                if verbose and debug_events < preview_limit:
                    print(f'[comments] line {line_no}: missing subreddit, skipped')
                    debug_events += 1
                continue

            author_tokens[author].append(subreddit)
            author_metadata[author].append({
                'feature': subreddit,
                'feature_type': 'subreddit',
                'subreddit': subreddit,
                'post_hint': 'comment',
                'body': comment.get('body', ''),
                'permalink': comment.get('permalink', ''),
            })
            author_sources[author].append({
                'source_type': 'comment',
                'subreddit': subreddit,
                'body': comment.get('body', ''),
                'permalink': comment.get('permalink', ''),
                'post_hint': 'comment',
            })

            if verbose and debug_events < preview_limit:
                body_preview = truncate_text(comment.get('body', ''), 200)
                print(f'[comments] line {line_no}: author={author} subreddit={subreddit} body={body_preview!r}')
                debug_events += 1

            processed_count = line_no
            if processed_count % 10000 == 0:
                print(f'  Processed {processed_count} comments...')

    print(f'  Total comments processed: {processed_count}')
    return author_tokens, author_metadata, author_sources


def filter_authors(author_tokens, min_posts=5, count_mode='total'):
    """Keep users with enough subreddit-bearing posts/comments.

    The default is total counts, since repeated posting in subreddits is part of
    the behavior we want to measure.
    """
    if count_mode not in ('unique', 'total'):
        raise ValueError("count_mode must be 'unique' or 'total'")

    descriptor = 'unique' if count_mode == 'unique' else 'total'
    print(f'\nFiltering users with < {min_posts} {descriptor} subreddit tokens...')

    initial_count = len(author_tokens)
    filtered = {}
    for user, tokens in author_tokens.items():
        if count_mode == 'unique':
            count = len(set(tokens))
            kept_tokens = sorted(set(tokens))
        else:
            count = len(tokens)
            kept_tokens = list(tokens)

        if count >= min_posts:
            filtered[user] = kept_tokens

    removed = initial_count - len(filtered)
    print(f'  Initial users: {initial_count}')
    print(f'  Users with >= {min_posts} ({descriptor}) subreddit tokens: {len(filtered)}')
    print(f'  Users removed: {removed}')

    return filtered


def build_user_matrix(user_tokens_list, min_df=2, max_df=0.9):
    """Build a TF-IDF matrix with one user per document and subreddit names as features."""
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        norm='l2'
    )
    documents = [' '.join(tokens) for tokens in user_tokens_list]
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


def compute_threshold(similarities, percentile=99, method='observed', sample_size=10000, seed=None):
    """Compute the percentile threshold from observed user-user similarities."""
    if similarities.shape[0] < 2:
        raise ValueError('Need at least two users to compute similarities.')

    observed_sims = similarities[np.triu_indices_from(similarities, k=1)]
    if observed_sims.size == 0:
        return observed_sims, 0.0

    if method == 'observed':
        threshold = np.percentile(observed_sims, percentile)
        return observed_sims, threshold

    if method == 'sampled_pairs':
        rng = np.random.default_rng(seed)
        k = min(sample_size, observed_sims.size)
        sampled = observed_sims if k == observed_sims.size else observed_sims[rng.choice(observed_sims.size, size=k, replace=False)]
        threshold = np.percentile(sampled, percentile)
        return observed_sims, threshold

    raise ValueError(f'Unknown threshold method: {method}')


def detect_suspicious_pairs(user_metadata_dict, similarities, user_list, threshold):
    """Return user pairs whose cosine similarity is above the threshold."""
    suspicious_pairs = []

    for i in range(len(user_list)):
        for j in range(i + 1, len(user_list)):
            sim = similarities[i, j]
            if sim > threshold:
                user_a = user_list[i]
                user_b = user_list[j]

                feats_a = set(item['feature'] for item in user_metadata_dict[user_a])
                feats_b = set(item['feature'] for item in user_metadata_dict[user_b])
                shared = feats_a.intersection(feats_b)

                meta_a = user_metadata_dict[user_a]
                meta_b = user_metadata_dict[user_b]
                subreddits_a = ', '.join(sorted(set(item['subreddit'] for item in meta_a)))
                subreddits_b = ', '.join(sorted(set(item['subreddit'] for item in meta_b)))

                suspicious_pairs.append({
                    'user_1': user_a,
                    'user_2': user_b,
                    'cosine_similarity': sim,
                    'percentile_rank': stats.percentileofscore(similarities.flatten(), sim),
                    'user_1_subreddits': subreddits_a,
                    'user_2_subreddits': subreddits_b,
                    'shared_subreddits': '; '.join(sorted(shared)[:10]),
                    'shared_subreddit_count': len(shared),
                })

    return sorted(suspicious_pairs, key=lambda item: item['cosine_similarity'], reverse=True)


def build_user_preview(user_sources, limit=2):
    """Create a compact preview of the subreddits a user has interacted with."""
    preview_items = []
    for item in user_sources[:limit]:
        source_subreddit = item.get('subreddit', '')
        source_type = item.get('source_type', 'source')
        title = truncate_text(item.get('title', ''), 120)
        body = truncate_text(item.get('body', ''), 120)
        text_bits = [bit for bit in [title, body] if bit]
        text_part = ' | '.join(text_bits)
        if text_part:
            preview_items.append(f'[{source_type}] r/{source_subreddit} {text_part}')
        else:
            preview_items.append(f'[{source_type}] r/{source_subreddit}')
    return ' || '.join(preview_items)


def write_user_report(report_path, suspicious_pairs, user_sources_dict):
    """Write a Markdown report that lists each user once with their subreddit context."""
    lines = []
    lines.append('# Co-Subreddit User Context Report')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append('')

    if not suspicious_pairs:
        lines.append('No suspicious user pairs were found above the selected threshold.')
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        return

    seen = set()
    users = []
    for pair in suspicious_pairs:
        for user in (pair['user_1'], pair['user_2']):
            if user not in seen:
                seen.add(user)
                users.append(user)

    for user in users:
        lines.append(f'## {user}')
        lines.append('')
        sources = user_sources_dict.get(user, [])
        if not sources:
            lines.append('_No subreddit context captured for this user._')
            lines.append('')
            continue

        for idx, item in enumerate(sources, start=1):
            lines.append(f'### Source {idx}')
            lines.append('')
            lines.append(f'- Type: {item.get("source_type", "source")}')
            lines.append(f'- Subreddit: r/{item.get("subreddit", "N/A")}')
            if item.get('post_hint'):
                lines.append(f'- Post hint: {item.get("post_hint")}')
            if item.get('permalink'):
                lines.append(f'- Permalink: {item.get("permalink")}')
            lines.append('')
            if item.get('title'):
                lines.append(f'- Title: {item.get("title")}')
            if item.get('selftext'):
                lines.append('- Selftext:')
                lines.append('```text')
                lines.append(item.get('selftext', '').strip() or '[empty text]')
                lines.append('```')
                lines.append('')
            if item.get('body'):
                lines.append('- Body:')
                lines.append('```text')
                lines.append(item.get('body', '').strip() or '[empty text]')
                lines.append('```')
                lines.append('')

    report_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Detect suspicious users by shared subreddit behavior')
    parser.add_argument('--input', nargs='+', required=True, help='Path(s) to input JSONL file(s)')
    parser.add_argument('--type', nargs='+', required=True, help='Data type(s): posts or comments')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--min-posts', type=int, default=5, help='Minimum total subreddit tokens required to keep a user')
    parser.add_argument('--count-mode', choices=['unique', 'total'], default='total', help='Count subreddit tokens by unique or total occurrences')
    parser.add_argument('--min-df', type=parse_df_value, default=2, help='Minimum TF-IDF document frequency for subreddit tokens')
    parser.add_argument('--max-df', type=parse_df_value, default=0.9, help='Maximum TF-IDF document frequency for subreddit tokens')
    parser.add_argument('--null-method', choices=['observed', 'sampled_pairs'], default='observed', help='How to derive the threshold from observed similarities')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of observed pairs to sample when using sampled_pairs')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampled pair selection')
    parser.add_argument('--percentile', type=int, default=99, help='Percentile threshold for suspicious pairs')
    parser.add_argument('--verbose', action='store_true', help='Print detailed loading and vectorization traces')
    parser.add_argument('--preview-limit', type=int, default=12, help='Maximum number of verbose debug events per loader')

    args = parser.parse_args()

    if len(args.input) != len(args.type):
        print('Error: Number of --input files must match number of --type specifications')
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_user_tokens = defaultdict(list)
    all_user_metadata = defaultdict(list)
    all_user_sources = defaultdict(list)

    for input_file, data_type in zip(args.input, args.type):
        data_type_lower = data_type.lower()
        if data_type_lower == 'posts':
            user_tokens, user_metadata, user_sources = load_posts_data(
                input_file,
                verbose=args.verbose,
                preview_limit=args.preview_limit,
            )
        elif data_type_lower == 'comments':
            user_tokens, user_metadata, user_sources = load_comments_data(
                input_file,
                verbose=args.verbose,
                preview_limit=args.preview_limit,
            )
        else:
            print(f'Unknown data type: {data_type}')
            continue

        for user, tokens in user_tokens.items():
            all_user_tokens[user].extend(tokens)
        for user, metadata in user_metadata.items():
            all_user_metadata[user].extend(metadata)
        for user, sources in user_sources.items():
            all_user_sources[user].extend(sources)

    print(f'\nLoaded users before filtering: {len(all_user_tokens)}')
    filtered_users = filter_authors(all_user_tokens, args.min_posts, count_mode=args.count_mode)
    if len(filtered_users) < 2:
        print('\nError: Need at least 2 users with enough subreddit tokens')
        return

    user_list = list(filtered_users.keys())
    user_tokens_list = [filtered_users[user] for user in user_list]

    print(f'\nComputing TF-IDF vectors for {len(user_list)} users...')
    print(f'  min_df={args.min_df}, max_df={args.max_df}')
    vectorizer, tfidf_matrix = build_user_matrix(
        user_tokens_list,
        min_df=args.min_df,
        max_df=args.max_df,
    )
    print(f'  TF-IDF matrix shape: {tfidf_matrix.shape}')
    print(f'  Non-zero entries: {tfidf_matrix.nnz}')

    similarities = cosine_similarity(tfidf_matrix)
    observed_sims, threshold = compute_threshold(
        similarities,
        percentile=args.percentile,
        method=args.null_method,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f'  {args.percentile}th percentile threshold from {args.null_method}: {threshold:.4f}')

    print('\nDetecting suspicious user pairs...')
    suspicious_pairs = detect_suspicious_pairs(all_user_metadata, similarities, user_list, threshold)
    print(f'Found {len(suspicious_pairs)} suspicious user pairs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'co_subreddit_suspicious_pairs_{timestamp}.csv'
    report_path = output_dir / f'co_subreddit_user_context_{timestamp}.md'

    if suspicious_pairs:
        for pair in suspicious_pairs:
            pair['user_1_preview'] = build_user_preview(all_user_sources.get(pair['user_1'], []))
            pair['user_2_preview'] = build_user_preview(all_user_sources.get(pair['user_2'], []))
            pair['user_1_source_count'] = len(all_user_sources.get(pair['user_1'], []))
            pair['user_2_source_count'] = len(all_user_sources.get(pair['user_2'], []))

        results_df = pd.DataFrame(suspicious_pairs)
        results_df.to_csv(csv_path, index=False)
        print(f'\nResults saved to: {csv_path}')
        print('\nTop 10 suspicious pairs:')
        print(results_df.head(10).to_string())
    else:
        print('No suspicious pairs found above threshold')

    write_user_report(report_path, suspicious_pairs, all_user_sources)
    print(f'User context report saved to: {report_path}')

    print('\nGenerating observed similarity plot...')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(observed_sims, bins=50, density=True, alpha=0.6, label='Observed Similarities', color='teal')

    if len(observed_sims) > 1:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(observed_sims)
        x_min = float(np.min(observed_sims))
        x_max = float(np.max(observed_sims))
        x_range = np.linspace(x_min, x_max if x_max > x_min else x_min + 1e-6, 200)
        ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE (Observed)')

    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'{args.percentile}th Percentile: {threshold:.4f}')
    ax.scatter(observed_sims, np.zeros_like(observed_sims), alpha=0.3, s=20, color='green', label='Observed Pairs')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Direct Co-Subreddit Sharing: Observed Similarity Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plot_path = output_dir / f'co_subreddit_similarity_distribution_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {plot_path}')

    print('\n' + '=' * 60)
    print('SUMMARY STATISTICS')
    print('=' * 60)
    print(f'Total users analyzed: {len(user_list)}')
    print(f'Total unique subreddit tokens: {len(set(token for tokens in user_tokens_list for token in tokens))}')
    print(f'Threshold ({args.percentile}th percentile of observed similarities): {threshold:.4f}')
    print(f'Suspicious pairs found: {len(suspicious_pairs)}')
    print(f'Actual similarity range: [{observed_sims.min():.4f}, {observed_sims.max():.4f}]')
    print('=' * 60)


if __name__ == '__main__':
    main()