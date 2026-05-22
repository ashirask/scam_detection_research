#!/usr/bin/env python3
"""Detect account pairs that share external links using TF-IDF + shared-count.

Usage:
  python code/link_similarity.py --input sample.jsonl --min-urls 2 --min-shared 2 \
    --similarity 0.6 --exclude i.redd.it --output flagged_pairs.csv

The script expects a JSONL file where each line is a Reddit post or comment JSON.
It extracts URLs from selected fields, normalizes them, groups by `author`, drops
accounts with fewer than `min_urls`, computes TF-IDF on URL tokens (per-account),
and flags account pairs meeting both cosine-similarity and shared-URL-count thresholds.
"""
import argparse
import json
import re
from collections import defaultdict
from urllib.parse import urlparse, urlunparse
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


URL_RE = re.compile(r"https?://[^\s'\)\]\>]+", re.IGNORECASE)


def extract_urls_from_text(text):
    """Extract all URLs from a text string using a regex.

    Returns a list of URL strings (may be empty). Handles None/empty input.
    """
    # quick guard for empty values
    if not text:
        return []
    # findall returns every substring matching the URL pattern
    return URL_RE.findall(text)


def normalize_url(url):
    """Normalize a URL for comparison.

    - Lowercases scheme and netloc
    - Strips leading "www." from the host
    - Removes query and fragment components (tracking params)
    - Removes trailing slash from path

    Returns the cleaned URL string; if parsing fails returns original input.
    """
    try:
        p = urlparse(url)
        scheme = p.scheme.lower() or 'http'
        netloc = p.netloc.lower()
        # strip www. to avoid trivial host mismatches
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        # remove query and fragment to reduce tracking noise
        cleaned = urlunparse((scheme, netloc, p.path.rstrip('/'), '', '', ''))
        return cleaned
    except Exception:
        # if parsing fails, return the original value so we don't lose data
        return url


def extract_urls_from_record(rec):
    """Extract and normalize URLs from a Reddit post/comment record.

    This function inspects multiple fields that commonly contain URLs:
    - explicit link fields (`url`, `url_overridden_by_dest`)
    - textual fields (`selftext`, `title`, `body`) where URLs may appear
    - preview/image fields, media/oembed, gallery `media_metadata`
    - crossposted parents (recursively)

    It returns a list of normalized URL strings (may contain duplicates before
    set deduplication by callers).
    """
    urls = []

    # 1) explicit link fields present on link posts
    for f in ('url', 'url_overridden_by_dest'):
        v = rec.get(f)
        if v:
            urls.append(v)

    # domain is available but not a full URL; keep it for potential filtering
    domain = rec.get('domain')
    if domain:
        # we don't append domain as a URL here, but the field is noted
        pass

    # 2) text fields that may contain inline URLs
    for f in ('selftext', 'selftext_html', 'title', 'body', 'body_html'):
        v = rec.get(f)
        if v:
            # extract_urls_from_text returns a list of matches from the text
            urls.extend(extract_urls_from_text(v))

    # 3) preview images often include a source URL (can be internal or external)
    preview = rec.get('preview')
    if isinstance(preview, dict):
        imgs = preview.get('images') or []
        for img in imgs:
            src = img.get('source') or {}
            if isinstance(src, dict):
                u = src.get('url')
                if u:
                    urls.append(u)

    # 4) media fields (oEmbed or thumbnails)
    for media_field in ('media', 'secure_media'):
        m = rec.get(media_field)
        if isinstance(m, dict):
            oembed = m.get('oembed')
            if isinstance(oembed, dict):
                # prefer a canonical URL, fall back to thumbnail URL
                u = oembed.get('url') or oembed.get('thumbnail_url')
                if u:
                    urls.append(u)

    # 5) media_metadata (gallery items) can contain multiple image URLs
    mm = rec.get('media_metadata')
    if isinstance(mm, dict):
        for v in mm.values():
            s = v.get('s') or v.get('p') or {}
            if isinstance(s, dict):
                u = s.get('u') or s.get('url')
                if u:
                    urls.append(u)

    # 6) handle crossposted parent posts (they themselves are records)
    cp = rec.get('crosspost_parent_list')
    if isinstance(cp, list):
        for item in cp:
            # recurse into the parent record to extract URLs there too
            urls.extend(extract_urls_from_record(item))

    # finally, normalize each URL and filter out non-strings
    normed = [normalize_url(u) for u in urls if isinstance(u, str)]
    return normed


def load_authors_urls(input_path, exclude_domains=None):
    """Read a JSONL file and return a mapping author -> set(normalized URLs).

    - `exclude_domains` is an iterable of domain strings to ignore (e.g. image/CDN hosts).
    - Uses `extract_urls_from_record` to get candidate URLs from each JSON record.
    """
    authors = defaultdict(set)
    exclude_domains = set(d.lower() for d in (exclude_domains or []))

    # stream over the JSONL file to avoid loading everything into memory
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # if a line fails to parse, skip it
                continue

            # identify the author field (varies between dumps)
            author = rec.get('author') or rec.get('user') or rec.get('author_fullname')
            if not author:
                continue

            # extract and normalize URLs from the record
            urls = extract_urls_from_record(rec)

            # filter out excluded domains and add each unique URL to the author's set
            for u in urls:
                try:
                    dom = urlparse(u).netloc.lower()
                except Exception:
                    dom = ''
                if dom.startswith('www.'):
                    dom = dom[4:]
                if dom and dom in exclude_domains:
                    # skip known uninformative hosts (images, reddit media, etc.)
                    continue
                authors[author].add(u)

    return authors


def build_tfidf_matrix(author_urls, use_domain_tokens=False):
    # author_urls: dict author -> set(urls)
    authors = list(author_urls.keys())
    docs = []
    token_lists = []

    # Construct a token list per author. Token choice depends on `use_domain_tokens`:
    # - if True: each token is the domain (host) of the URL
    # - if False: each token is the full normalized URL
    for a in authors:
        tokens = []
        # sort to make order deterministic (not required, but helpful for tests)
        for u in sorted(author_urls[a]):
            token = urlparse(u).netloc.lower() if use_domain_tokens else u
            if token.startswith('www.'):
                token = token[4:]
            tokens.append(token)

        token_lists.append(tokens)
        # TfidfVectorizer expects a text document; join tokens with spaces
        docs.append(' '.join(tokens))

    if not docs:
        return authors, None, None

    # Create a TF-IDF matrix where each "document" is an author and tokens are URL tokens
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x, token_pattern=None)
    X = vectorizer.fit_transform(docs)
    return authors, X, token_lists


def pairwise_flag(authors, X, token_lists, min_shared, sim_thresh, output_path):
    """Compute pairwise cosine similarities and return flagged pairs.

    For each pair (i,j) of authors we compute cosine similarity using the TF-IDF
    matrix `X` and also compute the raw shared-token count from `token_lists`.
    We flag pairs that meet both `sim_thresh` and `min_shared` and write them to CSV.
    """
    n = X.shape[0]

    # dense pairwise cosine similarity matrix (n x n)
    sims = cosine_similarity(X)
    rows = []

    # iterate only upper-triangle pairs (i < j) to avoid duplicates
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sims[i, j])
            # compute raw overlap between token lists (set intersection)
            shared = set(token_lists[i]) & set(token_lists[j])
            shared_count = len(shared)

            # require both a similarity threshold and a raw shared-token minimum
            if sim >= sim_thresh and shared_count >= min_shared:
                rows.append((authors[i], authors[j], sim, shared_count, ';'.join(sorted(shared))))

    # write results to CSV for downstream analysis
    import csv
    with open(output_path, 'w', newline='', encoding='utf-8') as csvf:
        w = csv.writer(csvf)
        w.writerow(['author_a', 'author_b', 'cosine_sim', 'shared_count', 'shared_tokens'])
        for r in rows:
            w.writerow(r)

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input JSONL file (one JSON per line)')
    p.add_argument('--min-urls', type=int, default=2, help='Minimum unique URLs per account to include')
    p.add_argument('--min-shared', type=int, default=2, help='Minimum shared URL tokens to flag a pair')
    p.add_argument('--similarity', type=float, default=0.6, help='Cosine similarity threshold')
    p.add_argument('--exclude', action='append', default=['i.redd.it'], help='Domain to exclude (can be repeated)')
    p.add_argument('--domain-tokens', action='store_true', help='Use domains as tokens instead of full URLs')
    p.add_argument('--output', '-o', default='flagged_pairs.csv', help='Output CSV path')
    args = p.parse_args()

    # 1) load and extract URLs grouped by author
    authors_urls = load_authors_urls(args.input, exclude_domains=args.exclude)

    # 2) drop accounts with fewer than the required number of unique URLs
    filtered = {a: urls for a, urls in authors_urls.items() if len(urls) >= args.min_urls}

    print(f'Loaded {len(authors_urls)} authors; {len(filtered)} meet min-urls={args.min_urls}')

    # 3) build TF-IDF matrix (authors x tokens)
    authors, X, token_lists = build_tfidf_matrix(filtered, use_domain_tokens=args.domain_tokens)
    if X is None or X.shape[0] <= 1:
        print('Not enough accounts after filtering to compare. Exiting.')
        return

    # 4) compute pairwise similarities and write flagged pairs
    rows = pairwise_flag(authors, X, token_lists, args.min_shared, args.similarity, args.output)
    print(f'Wrote {len(rows)} flagged pairs to {args.output}')


if __name__ == '__main__':
    main()
