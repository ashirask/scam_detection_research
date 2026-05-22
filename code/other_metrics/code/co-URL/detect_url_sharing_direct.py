#!/usr/bin/env python3
"""
Direct Co-URL Detection Script

This version removes the null distribution shuffle loop and instead uses the
observed author-author similarity matrix to derive the baseline threshold.
It also supports URL/domain feature selection for posts and TF-IDF filtering
via min_df and max_df.
"""

import argparse
import json
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')


def parse_df_value(value):
    """Parse min_df/max_df values as int or float."""
    text = str(value).strip()
    if '.' in text:
        return float(text)
    return int(text)


def extract_urls_from_text(text):
    """Extract URLs from text using regex."""
    if not text or not isinstance(text, str):
        return []
    url_pattern = r'https?://[^\s\)"\']+'
    return re.findall(url_pattern, text)


def normalize_url(url):
    """Normalize a URL: lowercase host, strip www, remove query/fragment, strip trailing slash."""
    try:
        p = urlparse(url)
        scheme = (p.scheme or 'http').lower()
        netloc = (p.netloc or '').lower()
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        path = (p.path or '').rstrip('/')
        cleaned = urlunparse((scheme, netloc, path, '', '', ''))
        return cleaned
    except Exception:
        return url


def build_post_text(post):
    """Build readable post text from title and selftext."""
    title = (post.get('title') or '').strip()
    selftext = (post.get('selftext') or '').strip()
    if title and selftext:
        return f'{title}\n\n{selftext}'
    return title or selftext


def truncate_text(text, limit=300):
    """Create a compact preview for CSV output."""
    if not text:
        return ''
    compact = ' '.join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3] + '...'


def load_posts_data(filepath, post_mode='domain'):
    """Load post-derived features and metadata."""
    author_tokens = defaultdict(list)
    author_metadata = defaultdict(list)
    author_sources = defaultdict(list)

    print(f"Loading posts data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                post = json.loads(line)

                if not post.get('author') or post.get('author') == '[deleted]':
                    continue

                if post.get('is_self', True):
                    continue

                token = None
                if post_mode == 'domain':
                    token = post.get('domain')
                elif post_mode == 'full_url':
                    token = post.get('url_overridden_by_dest') or post.get('url')
                else:
                    raise ValueError(f"Unknown post_mode: {post_mode}")

                if token and isinstance(token, str) and token.strip():
                    author = post['author']
                    # normalize token depending on mode
                    if post_mode == 'domain':
                        tok = token.strip().lower()
                        if tok.startswith('www.'):
                            tok = tok[4:]
                    else:
                        tok = normalize_url(token.strip())

                    source_url = post.get('url_overridden_by_dest') or post.get('url') or ''
                    source_text = build_post_text(post)

                    author_tokens[author].append(tok)
                    author_metadata[author].append({
                        'feature': tok,
                        'feature_type': post_mode,
                        'subreddit': post.get('subreddit', 'N/A'),
                        'domain': post.get('domain', 'N/A'),
                        'post_hint': post.get('post_hint', 'N/A'),
                        'url_overridden_by_dest': post.get('url_overridden_by_dest', 'N/A')
                    })
                    author_sources[author].append({
                        'source_type': 'post',
                        'text': source_text,
                        'url': source_url,
                        'subreddit': post.get('subreddit', 'N/A'),
                        'domain': post.get('domain', 'N/A'),
                        'post_hint': post.get('post_hint', 'N/A')
                    })

                    # also extract URLs from title/selftext and add normalized versions
                    post_text_urls = extract_urls_from_text(source_text)
                    for u in post_text_urls:
                        try:
                            norm = normalize_url(u.strip())
                        except Exception:
                            norm = u.strip()
                        # skip if this normalized URL equals the main token for this post
                        if not norm or norm == tok:
                            continue
                        author_tokens[author].append(norm)
                        author_metadata[author].append({
                            'feature': norm,
                            'feature_type': 'full_url_in_text',
                            'subreddit': post.get('subreddit', 'N/A'),
                            'domain': post.get('domain', 'N/A'),
                            'post_hint': post.get('post_hint', 'N/A'),
                            'url_overridden_by_dest': norm
                        })

                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1} posts...")

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    print(f"  Total posts processed: {i + 1}")
    return author_tokens, author_metadata, author_sources


def load_comments_data(filepath):
    """Load URLs extracted from comment body text."""
    author_tokens = defaultdict(list)
    author_metadata = defaultdict(list)
    author_sources = defaultdict(list)

    print(f"Loading comments data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                comment = json.loads(line)

                if not comment.get('author') or comment.get('author') == '[deleted]':
                    continue

                body = comment.get('body', '')
                urls = extract_urls_from_text(body)

                if urls:
                    author = comment['author']
                    body = comment.get('body', '')
                    for url in urls:
                        token = normalize_url(url.strip())
                        author_tokens[author].append(token)
                        author_metadata[author].append({
                            'feature': token,
                            'feature_type': 'full_url',
                            'subreddit': comment.get('subreddit', 'N/A'),
                            'domain': 'comment',
                            'post_hint': 'comment_text',
                            'url_overridden_by_dest': token
                        })
                    author_sources[author].append({
                        'source_type': 'comment',
                        'text': body,
                        'url': '',
                        'subreddit': comment.get('subreddit', 'N/A'),
                        'domain': 'comment',
                        'post_hint': 'comment_text'
                    })

                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1} comments...")

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    print(f"  Total comments processed: {i + 1}")
    return author_tokens, author_metadata, author_sources


def filter_authors(author_tokens, min_tokens=2, count_mode='unique'):
    """Keep authors with more than min_tokens tokens.

    count_mode: 'unique' counts distinct tokens per author (original behavior).
                'total' counts all token occurrences (includes duplicates).
    """
    if count_mode not in ('unique', 'total'):
        raise ValueError("count_mode must be 'unique' or 'total'")

    descriptor = 'unique' if count_mode == 'unique' else 'total'
    print(f"\nFiltering authors with <= {min_tokens} {descriptor} tokens...")

    initial_count = len(author_tokens)
    filtered = {}
    for author, tokens in author_tokens.items():
        if count_mode == 'unique':
            count = len(set(tokens))
            kept_tokens = sorted(set(tokens))
        else:
            count = len(tokens)
            kept_tokens = list(tokens)

        if count > min_tokens:
            filtered[author] = kept_tokens

    removed = initial_count - len(filtered)
    print(f"  Initial authors: {initial_count}")
    print(f"  Authors with >{min_tokens} ({descriptor}) tokens: {len(filtered)}")
    print(f"  Authors removed: {removed}")

    return filtered


def build_author_matrix(author_tokens_list, min_df=2, max_df=0.9):
    """Build TF-IDF matrix where each author is one document and each token is a feature."""
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        norm='l2'
    )
    documents = [' '.join(tokens) for tokens in author_tokens_list]
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


def compute_observed_threshold(similarities, percentile=99):
    """Derive the threshold directly from observed author-author similarities."""
    if similarities.shape[0] < 2:
        raise ValueError('Need at least two authors to compute similarities.')

    observed_sims = similarities[np.triu_indices_from(similarities, k=1)]
    threshold = np.percentile(observed_sims, percentile)
    return observed_sims, threshold


def compute_threshold(similarities, method='observed', percentile=99, sample_size=10000, seed=None):
    """Compute threshold using either all observed similarities or a sampled subset.

    method: 'observed' uses all upper-triangle similarities.
            'sampled_pairs' randomly samples `sample_size` observed pairs.
    Returns (observed_sims_full, threshold)
    """
    if similarities.shape[0] < 2:
        raise ValueError('Need at least two authors to compute similarities.')

    observed_sims = similarities[np.triu_indices_from(similarities, k=1)]

    if method == 'observed':
        threshold = np.percentile(observed_sims, percentile)
        return observed_sims, threshold

    if method == 'sampled_pairs':
        total = observed_sims.size
        if total == 0:
            return observed_sims, 0.0
        rng = np.random.default_rng(seed)
        k = min(sample_size, total)
        if k == total:
            sampled = observed_sims
        else:
            idx = rng.choice(total, size=k, replace=False)
            sampled = observed_sims[idx]
        threshold = np.percentile(sampled, percentile)
        return observed_sims, threshold

    raise ValueError(f'Unknown method: {method}')


def detect_suspicious_pairs(author_metadata_dict, similarities, author_list, threshold):
    """Extract suspicious author pairs with metadata."""
    suspicious_pairs = []

    for i in range(len(author_list)):
        for j in range(i + 1, len(author_list)):
            sim = similarities[i, j]
            if sim > threshold:
                author1 = author_list[i]
                author2 = author_list[j]

                features1 = set(m['feature'] for m in author_metadata_dict[author1])
                features2 = set(m['feature'] for m in author_metadata_dict[author2])
                shared_features = features1.intersection(features2)

                meta1 = author_metadata_dict[author1]
                meta2 = author_metadata_dict[author2]

                subreddits1 = ', '.join(sorted(set(m['subreddit'] for m in meta1)))
                subreddits2 = ', '.join(sorted(set(m['subreddit'] for m in meta2)))
                domains1 = ', '.join(sorted(set(m['domain'] for m in meta1)))
                domains2 = ', '.join(sorted(set(m['domain'] for m in meta2)))
                post_hints1 = ', '.join(sorted(set(m['post_hint'] for m in meta1)))
                post_hints2 = ', '.join(sorted(set(m['post_hint'] for m in meta2)))

                suspicious_pairs.append({
                    'author_1': author1,
                    'author_2': author2,
                    'cosine_similarity': sim,
                    'percentile_rank': stats.percentileofscore(similarities.flatten(), sim),
                    'author_1_subreddits': subreddits1,
                    'author_2_subreddits': subreddits2,
                    'author_1_domains': domains1,
                    'author_2_domains': domains2,
                    'author_1_post_hints': post_hints1,
                    'author_2_post_hints': post_hints2,
                    'shared_features': '; '.join(sorted(shared_features)[:10]),
                    'shared_feature_count': len(shared_features)
                })

    return sorted(suspicious_pairs, key=lambda x: x['cosine_similarity'], reverse=True)


def print_top_pair_diagnostics(suspicious_pairs, author_list, tfidf_matrix, vectorizer, max_features=30):
    """Print detailed TF-IDF diagnostics for the top suspicious pair."""
    if not suspicious_pairs:
        print('\nNo suspicious pairs available for diagnostics.')
        return

    top_pair = suspicious_pairs[0]
    author1 = top_pair['author_1']
    author2 = top_pair['author_2']

    i = author_list.index(author1)
    j = author_list.index(author2)

    row1 = tfidf_matrix.getrow(i)
    row2 = tfidf_matrix.getrow(j)

    feature_names = vectorizer.get_feature_names_out()
    idx1 = row1.indices
    idx2 = row2.indices
    val1 = row1.data
    val2 = row2.data

    pairs1 = sorted(zip(idx1, val1), key=lambda x: x[1], reverse=True)
    pairs2 = sorted(zip(idx2, val2), key=lambda x: x[1], reverse=True)

    set1 = set(idx1)
    set2 = set(idx2)
    shared_idx = sorted(set1.intersection(set2))

    print('\n' + '=' * 60)
    print('TOP PAIR TF-IDF DIAGNOSTICS')
    print('=' * 60)
    print(f"Top pair: {author1} vs {author2}")
    print(f"Cosine similarity: {top_pair['cosine_similarity']:.6f}")
    print(f'Feature space size after DF filtering: {len(feature_names)}')
    print(f'{author1} non-zero features: {len(idx1)}')
    print(f'{author2} non-zero features: {len(idx2)}')
    print(f'Shared non-zero features: {len(shared_idx)}')

    print(f'\nTop {max_features} features for {author1}:')
    for idx, weight in pairs1[:max_features]:
        print(f'  {feature_names[idx]} -> {weight:.6f}')

    print(f'\nTop {max_features} features for {author2}:')
    for idx, weight in pairs2[:max_features]:
        print(f'  {feature_names[idx]} -> {weight:.6f}')

    print(f'\nShared features (up to {max_features}):')
    for idx in shared_idx[:max_features]:
        w1 = row1[0, idx]
        w2 = row2[0, idx]
        print(f'  {feature_names[idx]} -> {author1}: {w1:.6f}, {author2}: {w2:.6f}')
    print('=' * 60)


def build_author_preview(author_sources, limit=2):
    """Create a short preview string from an author's source entries."""
    preview_items = []
    for item in author_sources[:limit]:
        source_text = truncate_text(item.get('text', ''), 220)
        source_url = item.get('url', '')
        if source_url:
            preview_items.append(f"[{item.get('source_type', 'source')}] {source_text} | URL: {truncate_text(source_url, 140)}")
        else:
            preview_items.append(f"[{item.get('source_type', 'source')}] {source_text}")
    return ' || '.join(preview_items)


def write_pair_report(report_path, suspicious_pairs, author_sources_dict):
    """Write a Markdown report showing each author's posted/commented content.
    
    This serves as a text reference to understand what authors shared.
    Pair similarity details are already in the CSV output.
    """

    lines = []
    lines.append('# Author Content Reference')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append('')

    if not suspicious_pairs:
        lines.append('No suspicious pairs were found above the selected threshold.')
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        return

    # Collect unique authors that appear in the suspicious pairs, preserving order
    seen = set()
    authors = []
    for pair in suspicious_pairs:
        for a in (pair['author_1'], pair['author_2']):
            if a not in seen:
                seen.add(a)
                authors.append(a)

    for a in authors:
        lines.append(f'## {a}')
        lines.append('')
        sources = author_sources_dict.get(a, [])
        if not sources:
            lines.append('_No source text captured for this author._')
            lines.append('')
            continue

        for item_idx, item in enumerate(sources, start=1):
            lines.append(f'### Source {item_idx}')
            lines.append('')
            lines.append(f'- Type: {item.get("source_type", "source")}')
            lines.append(f'- Subreddit: {item.get("subreddit", "N/A")}')
            lines.append(f'- Domain: {item.get("domain", "N/A")}')
            if item.get('url'):
                lines.append(f'- URL: {item.get("url")}')
            lines.append('')
            lines.append('```text')
            lines.append(item.get('text', '').strip() or '[empty text]')
            lines.append('```')
            lines.append('')

    report_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='Detect suspicious authors using direct author-author similarity thresholds'
    )
    parser.add_argument('--input', nargs='+', required=True, help='Path(s) to input JSONL file(s)')
    parser.add_argument('--type', nargs='+', required=True, help='Data type(s): posts or comments')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--post-mode', choices=['domain', 'full_url'], default='domain',
                        help='For posts, compare either domains or full URLs')
    parser.add_argument('--min-urls', type=int, default=2,
                        help='Minimum unique tokens per author required to keep the author')
    parser.add_argument('--count-mode', choices=['unique', 'total'], default='total',
                        help="Count tokens by 'unique' values or by 'total' occurrences (default: total)")
    parser.add_argument('--min-df', type=parse_df_value, default=2,
                        help='Minimum document frequency for TF-IDF tokens (int or float)')
    parser.add_argument('--max-df', type=parse_df_value, default=0.9,
                        help='Maximum document frequency for TF-IDF tokens (int or float)')
    parser.add_argument('--null-method', choices=['observed', 'sampled_pairs'], default='sampled_pairs',
                        help="How to derive threshold: 'observed' uses all pairs; 'sampled_pairs' samples observed pairs")
    parser.add_argument('--sample-size', type=int, default=500,
                        help='Number of observed pairs to sample when --null-method is sampled_pairs')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for sampled_pairs selection')
    parser.add_argument('--percentile', type=int, default=99,
                        help='Percentile threshold computed from observed similarities')

    args = parser.parse_args()

    if len(args.input) != len(args.type):
        print('Error: Number of --input files must match number of --type specifications')
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_author_tokens = defaultdict(list)
    all_author_metadata = defaultdict(list)
    all_author_sources = defaultdict(list)

    for input_file, data_type in zip(args.input, args.type):
        if data_type.lower() == 'posts':
            author_tokens, author_metadata, author_sources = load_posts_data(input_file, args.post_mode)
        elif data_type.lower() == 'comments':
            author_tokens, author_metadata, author_sources = load_comments_data(input_file)
        else:
            print(f'Unknown data type: {data_type}')
            continue

        for author, tokens in author_tokens.items():
            all_author_tokens[author].extend(tokens)

        for author, metadata in author_metadata.items():
            all_author_metadata[author].extend(metadata)

        for author, sources in author_sources.items():
            all_author_sources[author].extend(sources)

    filtered_authors = filter_authors(all_author_tokens, args.min_urls, count_mode=args.count_mode)
    if len(filtered_authors) < 2:
        print('\nError: Need at least 2 authors with sufficient tokens')
        return

    author_list = list(filtered_authors.keys())
    author_tokens_list = [filtered_authors[a] for a in author_list]

    print(f"\nComputing TF-IDF vectors for {len(author_list)} authors...")
    print(f"  min_df={args.min_df}, max_df={args.max_df}, post_mode={args.post_mode}")
    vectorizer, tfidf_matrix = build_author_matrix(
        author_tokens_list,
        min_df=args.min_df,
        max_df=args.max_df
    )
    similarities = cosine_similarity(tfidf_matrix)

    observed_sims, threshold = compute_threshold(
        similarities,
        method=args.null_method,
        percentile=args.percentile,
        sample_size=args.sample_size,
        seed=args.seed
    )
    print(f"  {args.percentile}th percentile threshold from {args.null_method}: {threshold:.4f}")

    print('\nDetecting suspicious pairs...')
    suspicious_pairs = detect_suspicious_pairs(
        all_author_metadata,
        similarities,
        author_list,
        threshold
    )

    print_top_pair_diagnostics(
        suspicious_pairs,
        author_list,
        tfidf_matrix,
        vectorizer
    )

    print(f'Found {len(suspicious_pairs)} suspicious author pairs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'co_url_suspicious_pairs_direct_{timestamp}.csv'

    if suspicious_pairs:
        for pair in suspicious_pairs:
            pair['author_1_text_preview'] = build_author_preview(all_author_sources.get(pair['author_1'], []))
            pair['author_2_text_preview'] = build_author_preview(all_author_sources.get(pair['author_2'], []))
            pair['author_1_source_count'] = len(all_author_sources.get(pair['author_1'], []))
            pair['author_2_source_count'] = len(all_author_sources.get(pair['author_2'], []))

        df_results = pd.DataFrame(suspicious_pairs)
        df_results.to_csv(csv_path, index=False)
        print(f'\nResults saved to: {csv_path}')
        print('\nTop 10 suspicious pairs:')
        print(df_results.head(10).to_string())

        report_path = output_dir / f'co_url_pair_report_direct_{timestamp}.md'
        write_pair_report(report_path, suspicious_pairs, all_author_sources)
        print(f'Pair report saved to: {report_path}')
    else:
        print('No suspicious pairs found above threshold')
        report_path = output_dir / f'co_url_pair_report_direct_{timestamp}.md'
        write_pair_report(report_path, suspicious_pairs, all_author_sources)
        print(f'Pair report saved to: {report_path}')

    print('\nGenerating observed similarity plot...')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(observed_sims, bins=50, density=True, alpha=0.6, label='Observed Similarities', color='blue')

    if len(observed_sims) > 1:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(observed_sims)
        x_min = float(np.min(observed_sims))
        x_max = float(np.max(observed_sims))
        x_range = np.linspace(x_min, x_max if x_max > x_min else x_min + 1e-6, 200)
        ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE (Observed)')

    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'{args.percentile}th Percentile: {threshold:.4f}')

    ax.scatter(observed_sims, np.zeros_like(observed_sims), alpha=0.3, s=20, color='green', label='Observed Pairs')

    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Direct Co-URL Sharing: Observed Similarity Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plot_path = output_dir / f'co_url_similarity_distribution_direct_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {plot_path}')

    print('\n' + '=' * 60)
    print('SUMMARY STATISTICS')
    print('=' * 60)
    print(f'Total authors analyzed: {len(author_list)}')
    print(f'Total unique tokens: {len(set(token for tokens in author_tokens_list for token in tokens))}')
    print(f'Threshold ({args.percentile}th percentile of observed similarities): {threshold:.4f}')
    print(f'Suspicious pairs found: {len(suspicious_pairs)}')
    print(f'Actual similarity range: [{observed_sims.min():.4f}, {observed_sims.max():.4f}]')
    print('=' * 60)


if __name__ == '__main__':
    main()