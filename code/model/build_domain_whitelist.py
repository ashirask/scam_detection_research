#!/usr/bin/env python3
"""
build_domain_whitelist.py

Extracts all URLs from bot and human comment/submission files,
counts domain frequencies, and writes the top-N domains to a text file.
This whitelist is used by build_features.py to identify suspicious URLs.

Usage:
  python build_domain_whitelist.py \
    --comments-bot      user_comments_bots.jsonl \
    --comments-human    user_comments_humans.jsonl \
    --submissions-bot   user_submissions_bots.jsonl \
    --submissions-human user_submissions_humans.jsonl \
    --top-n             500 \
    --output            domain_whitelist.txt
"""

import re
import argparse
from collections import Counter
from urllib.parse import urlparse
import orjson


# Regex pattern to match HTTP/HTTPS URLs in text
# Matches strings starting with http:// or https:// followed by non-whitespace characters
# Excludes common URL-terminating characters like ), ], ", '
URL_PATTERN = re.compile(r'https?://[^\s\)\]\"\']+')


def extract_urls(text):
    """
    Extract all URLs from a given text string.
    
    Args:
        text: String to search for URLs (can be None or empty)
    
    Returns:
        List of URL strings found in the text
    """
    return URL_PATTERN.findall(text or "")


def extract_domain(url):
    """
    Extract domain from a URL, stripping 'www.' prefix.
    
    Args:
        url: URL string to parse
    
    Returns:
        Domain name in lowercase without 'www.' prefix, or None if parsing fails
    """
    try:
        domain = urlparse(url).netloc.lower()
        return domain.lstrip("www.")
    except Exception:
        return None


def stream_jsonl(path):
    """
    Generator to read JSONL file line by line.
    
    Args:
        path: Path to JSONL file
    
    Yields:
        Parsed JSON objects (dicts) from each line
    """
    with open(path, "rb") as f:
        for line in f:
            yield orjson.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description="Build domain whitelist from Reddit comment and submission data"
    )
    parser.add_argument("--comments-bot", required=True, help="Path to bot comments JSONL")
    parser.add_argument("--comments-human", required=True, help="Path to human comments JSONL")
    parser.add_argument("--submissions-bot", required=True, help="Path to bot submissions JSONL")
    parser.add_argument("--submissions-human", required=True, help="Path to human submissions JSONL")
    parser.add_argument("--top-n", type=int, default=500, help="Number of top domains to include")
    parser.add_argument("--output", required=True, help="Output text file path")
    
    args = parser.parse_args()
    
    # Counter to track domain frequencies across all files
    domain_counter = Counter()
    
    # Process bot comments
    print(f"Processing bot comments: {args.comments_bot}")
    for record in stream_jsonl(args.comments_bot):
        comments = record.get("comments", [])
        for comment in comments:
            body = comment.get("body", "")
            urls = extract_urls(body)
            for url in urls:
                domain = extract_domain(url)
                if domain:
                    domain_counter[domain] += 1
    
    # Process human comments
    print(f"Processing human comments: {args.comments_human}")
    for record in stream_jsonl(args.comments_human):
        comments = record.get("comments", [])
        for comment in comments:
            body = comment.get("body", "")
            urls = extract_urls(body)
            for url in urls:
                domain = extract_domain(url)
                if domain:
                    domain_counter[domain] += 1
    
    # Process bot submissions
    print(f"Processing bot submissions: {args.submissions_bot}")
    for record in stream_jsonl(args.submissions_bot):
        submissions = record.get("submissions", [])
        for submission in submissions:
            # Extract URLs from selftext (submission body text)
            selftext = submission.get("selftext", "")
            urls = extract_urls(selftext)
            # Also extract from the submission URL field (the link itself)
            link_url = submission.get("url", "")
            if link_url and link_url.startswith("http"):
                urls.append(link_url)
            for url in urls:
                domain = extract_domain(url)
                if domain:
                    domain_counter[domain] += 1
    
    # Process human submissions
    print(f"Processing human submissions: {args.submissions_human}")
    for record in stream_jsonl(args.submissions_human):
        submissions = record.get("submissions", [])
        for submission in submissions:
            selftext = submission.get("selftext", "")
            urls = extract_urls(selftext)
            link_url = submission.get("url", "")
            if link_url and link_url.startswith("http"):
                urls.append(link_url)
            for url in urls:
                domain = extract_domain(url)
                if domain:
                    domain_counter[domain] += 1
    
    # Get top-N domains by frequency
    top_domains = [domain for domain, _ in domain_counter.most_common(args.top_n)]
    
    # Write to output file (one domain per line)
    print(f"Writing {len(top_domains)} domains to {args.output}")
    with open(args.output, "w") as f:
        for domain in top_domains:
            f.write(domain + "\n")
    
    print(f"Total unique domains found: {len(domain_counter)}")
    print(f"Top domain: {top_domains[0] if top_domains else 'N/A'} (count: {domain_counter[top_domains[0]] if top_domains else 0})")


if __name__ == "__main__":
    main()
