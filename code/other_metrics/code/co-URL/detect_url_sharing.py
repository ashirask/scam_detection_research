#!/usr/bin/env python3
"""
Co-URL Detection Script
Detects suspicious authors sharing unusually large numbers of URLs
using TF-IDF vectorization and cosine similarity analysis.
"""

import json
import argparse
import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def extract_urls_from_text(text):
    """Extract URLs from text using regex."""
    if not text or not isinstance(text, str):
        return []
    url_pattern = r'https?://[^\s\)"\']+'
    return re.findall(url_pattern, text)


def load_posts_data(filepath, url_field='domain'):
    """Load URLs from posts JSON file with metadata."""
    author_urls = defaultdict(list)
    author_metadata = defaultdict(list)
    
    print(f"Loading posts data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                post = json.loads(line)
                
                # Skip if author is None or deleted
                if not post.get('author') or post.get('author') == '[deleted]':
                    continue
                
                # Only external links (is_self == False)
                if post.get('is_self', True):
                    continue
                
                # Extract URL based on specified field
                url = None
                if url_field == 'domain':
                    url = post.get('domain')
                elif url_field == 'url_overridden_by_dest':
                    url = post.get('url_overridden_by_dest')
                
                # Only add non-empty URLs
                if url and isinstance(url, str) and url.strip():
                    author = post['author']
                    author_urls[author].append(url)
                    
                    # Store metadata
                    author_metadata[author].append({
                        'url': url,
                        'subreddit': post.get('subreddit', 'N/A'),
                        'domain': post.get('domain', 'N/A'),
                        'post_hint': post.get('post_hint', 'N/A'),
                        'url_overridden_by_dest': post.get('url_overridden_by_dest', 'N/A')
                    })
                
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1} posts...")
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    
    print(f"  Total posts processed: {i + 1}")
    return author_urls, author_metadata


def load_comments_data(filepath):
    """Load URLs from comments JSON file (extract from body text) with metadata."""
    author_urls = defaultdict(list)
    author_metadata = defaultdict(list)
    
    print(f"Loading comments data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                comment = json.loads(line)
                
                # Skip if author is None or deleted
                if not comment.get('author') or comment.get('author') == '[deleted]':
                    continue
                
                # Extract URLs from comment body
                body = comment.get('body', '')
                urls = extract_urls_from_text(body)
                
                if urls:
                    author = comment['author']
                    for url in urls:
                        author_urls[author].append(url)
                        
                        # Store metadata for comments
                        author_metadata[author].append({
                            'url': url,
                            'subreddit': comment.get('subreddit', 'N/A'),
                            'domain': 'comment',
                            'post_hint': 'comment_text',
                            'url_overridden_by_dest': url
                        })
                
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1} comments...")
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    
    print(f"  Total comments processed: {i + 1}")
    return author_urls, author_metadata


def filter_authors(author_urls, min_urls=2):
    """Filter authors with insufficient URLs."""
    print(f"\nFiltering authors with ≤ {min_urls} unique URLs...")
    
    initial_count = len(author_urls)
    
    # Get unique URLs per author
    author_unique_urls = {}
    for author, urls in author_urls.items():
        unique_urls = list(set(urls))
        if len(unique_urls) > min_urls:
            author_unique_urls[author] = unique_urls
    
    removed = initial_count - len(author_unique_urls)
    print(f"  Initial authors: {initial_count}")
    print(f"  Authors with >{min_urls} URLs: {len(author_unique_urls)}")
    print(f"  Authors removed: {removed}")
    
    return author_unique_urls


def compute_null_distribution(author_urls_list, sample_size=1000, percentile=99):
    """
    Compute null distribution via random permutation sampling.
    Randomly shuffles author-URL assignments and computes similarities.
    """
    print(f"\nComputing null distribution ({sample_size} permutations)...")
    
    all_urls = []
    for urls in author_urls_list:
        all_urls.extend(urls)
    
    null_similarities = []
    
    for perm in range(sample_size):
        # Create random permutation of authors and URLs
        authors_shuffled = list(range(len(author_urls_list)))
        np.random.shuffle(authors_shuffled)
        
        # Create shuffled URL assignments
        shuffled_urls = all_urls.copy()
        np.random.shuffle(shuffled_urls)
        
        shuffled_author_urls = {}
        idx = 0
        for author_idx in range(len(author_urls_list)):
            original_count = len(author_urls_list[author_idx])
            shuffled_author_urls[author_idx] = shuffled_urls[idx:idx + original_count]
            idx += original_count
        
        # Compute similarities for shuffled data
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
        authors_list = list(shuffled_author_urls.keys())
        url_strings = [' '.join(shuffled_author_urls[a]) for a in authors_list]
        
        try:
            tfidf_matrix = vectorizer.fit_transform(url_strings)
            similarities = cosine_similarity(tfidf_matrix)
            
            # Extract upper triangle (unique pairs)
            for i in range(len(authors_list)):
                for j in range(i + 1, len(authors_list)):
                    null_similarities.append(similarities[i, j])
        except:
            pass
        
        if (perm + 1) % 100 == 0:
            print(f"  Completed {perm + 1} permutations...")
    
    threshold = np.percentile(null_similarities, percentile)
    print(f"  {percentile}th percentile threshold: {threshold:.4f}")
    
    return null_similarities, threshold


def detect_suspicious_pairs(author_metadata_dict, similarities, author_list, threshold):
    """Extract suspicious author pairs with metadata."""
    suspicious_pairs = []
    
    for i in range(len(author_list)):
        for j in range(i + 1, len(author_list)):
            sim = similarities[i, j]
            if sim > threshold:
                author1 = author_list[i]
                author2 = author_list[j]
                
                # Get shared URLs
                urls1 = set(m['url'] for m in author_metadata_dict[author1])
                urls2 = set(m['url'] for m in author_metadata_dict[author2])
                shared_urls = urls1.intersection(urls2)
                
                # Aggregate metadata for each author
                meta1 = author_metadata_dict[author1]
                meta2 = author_metadata_dict[author2]
                
                subreddits1 = ', '.join(set(m['subreddit'] for m in meta1))
                subreddits2 = ', '.join(set(m['subreddit'] for m in meta2))
                domains1 = ', '.join(set(m['domain'] for m in meta1))
                domains2 = ', '.join(set(m['domain'] for m in meta2))
                post_hints1 = ', '.join(set(m['post_hint'] for m in meta1))
                post_hints2 = ', '.join(set(m['post_hint'] for m in meta2))
                
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
                    'shared_urls': '; '.join(list(shared_urls)[:10]),
                    'shared_url_count': len(shared_urls)
                })
    
    return sorted(suspicious_pairs, key=lambda x: x['cosine_similarity'], reverse=True)


def main():
    parser = argparse.ArgumentParser(
        description='Detect suspicious authors sharing unusually large numbers of URLs'
    )
    parser.add_argument('--input', nargs='+', required=True, help='Path(s) to input JSONL file(s)')
    parser.add_argument('--type', nargs='+', required=True, help='Data type(s): posts or comments')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--url-field', default='domain', help='URL field for posts: domain or url_overridden_by_dest')
    parser.add_argument('--min-urls', type=int, default=2, help='Minimum unique URLs per author')
    parser.add_argument('--percentile', type=int, default=99, help='Percentile threshold')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of random permutations')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.input) != len(args.type):
        print("Error: Number of --input files must match number of --type specifications")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    all_author_urls = defaultdict(list)
    all_author_metadata = defaultdict(list)
    
    for input_file, data_type in zip(args.input, args.type):
        if data_type.lower() == 'posts':
            author_urls, author_metadata = load_posts_data(input_file, args.url_field)
        elif data_type.lower() == 'comments':
            author_urls, author_metadata = load_comments_data(input_file)
        else:
            print(f"Unknown data type: {data_type}")
            continue
        
        # Merge into combined dictionaries
        for author, urls in author_urls.items():
            all_author_urls[author].extend(urls)
        
        for author, metadata in author_metadata.items():
            all_author_metadata[author].extend(metadata)
    
    # Filter authors
    filtered_authors = filter_authors(all_author_urls, args.min_urls)
    
    if len(filtered_authors) < 2:
        print("\nError: Need at least 2 authors with sufficient URLs")
        return
    
    # Prepare data for TF-IDF
    author_list = list(filtered_authors.keys())
    author_urls_list = [filtered_authors[a] for a in author_list]
    
    # Compute null distribution
    null_similarities, threshold = compute_null_distribution(
        author_urls_list, 
        args.sample_size, 
        args.percentile
    )
    
    # Compute actual TF-IDF similarities
    print(f"\nComputing TF-IDF vectors for {len(author_list)} authors...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
    url_strings = [' '.join(urls) for urls in author_urls_list]
    
    tfidf_matrix = vectorizer.fit_transform(url_strings)
    similarities = cosine_similarity(tfidf_matrix)
    
    # Find suspicious pairs
    print("\nDetecting suspicious pairs...")
    suspicious_pairs = detect_suspicious_pairs(
        all_author_metadata, 
        similarities, 
        author_list, 
        threshold
    )
    
    print(f"Found {len(suspicious_pairs)} suspicious author pairs")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"co_url_suspicious_pairs_{timestamp}.csv"
    
    if suspicious_pairs:
        df_results = pd.DataFrame(suspicious_pairs)
        df_results.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        print("\nTop 10 suspicious pairs:")
        print(df_results.head(10).to_string())
    else:
        print("No suspicious pairs found above threshold")
    
    # Create KDE plot
    print("\nGenerating KDE plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram and KDE of null distribution
    ax.hist(null_similarities, bins=50, density=True, alpha=0.6, label='Null Distribution', color='blue')
    
    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(null_similarities)
    x_range = np.linspace(0, 1, 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE (Null)')
    
    # Add threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'{args.percentile}th Percentile: {threshold:.4f}')
    
    # Add actual similarities
    actual_sims = similarities[np.triu_indices_from(similarities, k=1)]
    ax.scatter(actual_sims, np.zeros_like(actual_sims), alpha=0.3, s=20, color='green', label='Actual Pairs')
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Co-URL Sharing: Cosine Similarity Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plot_path = output_dir / f"co_url_similarity_distribution_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total authors analyzed: {len(author_list)}")
    print(f"Total unique URLs: {len(set(url for urls in author_urls_list for url in urls))}")
    print(f"Similarity threshold ({args.percentile}th percentile): {threshold:.4f}")
    print(f"Suspicious pairs found: {len(suspicious_pairs)}")
    print(f"Actual similarity range: [{actual_sims.min():.4f}, {actual_sims.max():.4f}]")
    print(f"Null distribution range: [{min(null_similarities):.4f}, {max(null_similarities):.4f}]")
    print("="*60)


if __name__ == '__main__':
    main()
