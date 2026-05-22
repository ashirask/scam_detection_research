#!/usr/bin/env python3
"""Compute account similarity from URLs in title/selftext only.

This keeps the same global URL filtering idea as the reference workflow, but
outputs a simple CSV instead of building a graph.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from urllib.parse import urlparse, urlunparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


URL_RE = re.compile(r"https?://[^\s'\)\]\>]+", re.IGNORECASE)


def extract_urls_from_text(text):
    """Extract URL substrings from a block of text."""
    if not text:
        return []
    return URL_RE.findall(text)


def normalize_url(url):
    """Normalize a URL for comparison."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower() or 'http'
        netloc = parsed.netloc.lower()
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return urlunparse((scheme, netloc, parsed.path.rstrip('/'), '', '', ''))
    except Exception:
        return url


def extract_title_selftext_urls(record):
    """Extract URLs only from title and selftext fields."""
    urls = []
    for field_name in ('title', 'selftext'):
        value = record.get(field_name)
        if value:
            urls.extend(extract_urls_from_text(value))
    return [normalize_url(url) for url in urls if isinstance(url, str)]


def load_authors_urls(input_path, exclude_domains=None):
    """Read JSONL and return author -> set(normalized URLs)."""
    authors = defaultdict(set)
    exclude_domains = set(d.lower() for d in (exclude_domains or []))

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            author = rec.get('author') or rec.get('user') or rec.get('author_fullname')
            if not author:
                continue

            for url in extract_title_selftext_urls(rec):
                try:
                    dom = urlparse(url).netloc.lower()
                except Exception:
                    dom = ''
                if dom.startswith('www.'):
                    dom = dom[4:]
                if dom and dom in exclude_domains:
                    continue
                authors[author].add(url)

    return authors


def build_tfidf_matrix(author_urls, use_domain_tokens=False):
    """Build a TF-IDF matrix where each author is a document of URL tokens."""
    authors = list(author_urls.keys())
    docs = []
    token_lists = []

    for author in authors:
        tokens = []
        for url in sorted(author_urls[author]):
            token = urlparse(url).netloc.lower() if use_domain_tokens else url
            if token.startswith('www.'):
                token = token[4:]
            tokens.append(token)
        token_lists.append(tokens)
        docs.append(' '.join(tokens))

    if not docs:
        return authors, None, None

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x, token_pattern=None)
    X = vectorizer.fit_transform(docs)
    return authors, X, token_lists


def pairwise_flag(authors, X, token_lists, min_shared, sim_thresh, output_path):
    """Compute pairwise similarities and write flagged rows to CSV."""
    sims = cosine_similarity(X)
    rows = []

    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            sim = float(sims[i, j])
            shared = set(token_lists[i]) & set(token_lists[j])
            shared_count = len(shared)
            if sim >= sim_thresh and shared_count >= min_shared:
                rows.append((authors[i], authors[j], sim, shared_count, ';'.join(sorted(shared))))

    with open(output_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['author_a', 'author_b', 'cosine_sim', 'shared_count', 'shared_tokens'])
        writer.writerows(rows)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file (one JSON per line)')
    parser.add_argument('--min-urls', type=int, default=2, help='Minimum unique URLs per account to include')
    parser.add_argument('--min-shared', type=int, default=2, help='Minimum shared URL tokens to flag a pair')
    parser.add_argument('--similarity', type=float, default=0.6, help='Cosine similarity threshold')
    parser.add_argument('--exclude', action='append', default=['i.redd.it'], help='Domain to exclude (can be repeated)')
    parser.add_argument('--domain-tokens', action='store_true', help='Use domains as tokens instead of full URLs')
    parser.add_argument('--output', '-o', default='flagged_pairs.csv', help='Output CSV path')
    args = parser.parse_args()

    authors_urls = load_authors_urls(args.input, exclude_domains=args.exclude)
    filtered = {author: urls for author, urls in authors_urls.items() if len(urls) >= args.min_urls}

    print(f'Loaded {len(authors_urls)} authors; {len(filtered)} meet min-urls={args.min_urls}')

    authors, X, token_lists = build_tfidf_matrix(filtered, use_domain_tokens=args.domain_tokens)
    if X is None or X.shape[0] <= 1:
        print('Not enough accounts after filtering to compare. Exiting.')
        return

    rows = pairwise_flag(authors, X, token_lists, args.min_shared, args.similarity, args.output)
    print(f'Wrote {len(rows)} flagged pairs to {args.output}')


if __name__ == '__main__':
    main()
