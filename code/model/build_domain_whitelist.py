#!/usr/bin/env python3
"""
build_domain_whitelist.py

Extracts all URLs from bot and human comment/submission files,
counts domain frequencies, and writes the top-N domains to a text file.
This whitelist is used by build_features.py to identify suspicious URLs.

New features:
- Calculates cumulative coverage to find optimal N for target coverage
- Plots coverage curve when --plot-coverage flag is used
- Automatically uses minimum N needed for target coverage if smaller than --top-n

Usage (basic):
  python build_domain_whitelist.py \
    --comments-bot      user_comments_bots.jsonl \
    --comments-human    user_comments_humans.jsonl \
    --submissions-bot   user_submissions_bots.jsonl \
    --submissions-human user_submissions_humans.jsonl \
    --top-n             500 \
    --output            domain_whitelist.txt

Usage (with coverage analysis and plot):
  python build_domain_whitelist.py \
    --comments-bot      user_comments_bots.jsonl \
    --comments-human    user_comments_humans.jsonl \
    --submissions-bot   user_submissions_bots.jsonl \
    --submissions-human user_submissions_humans.jsonl \
    --top-n             500 \
    --coverage-target   0.95 \
    --plot-coverage \
    --output            domain_whitelist.txt
"""

import re
import argparse
from collections import Counter
from urllib.parse import urlparse
import orjson
import matplotlib.pyplot as plt


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
    parser.add_argument("--plot-coverage", action="store_true", help="Plot cumulative coverage and save as PNG")
    parser.add_argument("--coverage-target", type=float, default=0.90, help="Target coverage (0.0-1.0) to find minimum N")
    
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
    
    # Calculate total URL occurrences for coverage analysis
    total_urls = sum(domain_counter.values())
    print(f"\nTotal URL occurrences: {total_urls:,}")
    print(f"Total unique domains: {len(domain_counter):,}")
    
    # Sort domains by frequency (descending)
    sorted_domains = domain_counter.most_common()
    
    # Calculate cumulative coverage for each rank
    cumulative_counts = []
    running_total = 0
    for domain, count in sorted_domains:
        running_total += count
        cumulative_counts.append(running_total)
    
    # Convert to cumulative proportions
    cumulative_proportions = [c / total_urls for c in cumulative_counts]
    
    # Find minimum N to achieve target coverage
    target_n = None
    for i, prop in enumerate(cumulative_proportions):
        if prop >= args.coverage_target:
            target_n = i + 1  # +1 because index is 0-based
            break
    
    if target_n:
        print(f"\nMinimum domains for {args.coverage_target*100:.0f}% coverage: {target_n}")
        print(f"  Coverage achieved: {cumulative_proportions[target_n-1]*100:.2f}%")
    else:
        print(f"\nWarning: Could not achieve {args.coverage_target*100:.0f}% coverage with all {len(sorted_domains)} domains")
        print(f"  Maximum coverage: {cumulative_proportions[-1]*100:.2f}%")
    
    # Print coverage at various N values for inspection
    print(f"\nCoverage at different N values:")
    for n in [100, 500, 1000, 2000, 5000]:
        if n <= len(sorted_domains):
            coverage = cumulative_proportions[n-1] * 100
            print(f"  Top {n:5d} domains: {coverage:6.2f}% coverage")
    
    # Plot cumulative coverage if requested
    if args.plot_coverage:
        print(f"\nGenerating coverage plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_proportions) + 1), 
                 [p * 100 for p in cumulative_proportions], 
                 linewidth=2)
        plt.axhline(y=args.coverage_target * 100, color='r', linestyle='--', 
                   label=f'{args.coverage_target*100:.0f}% target')
        if target_n:
            plt.axvline(x=target_n, color='g', linestyle='--', 
                       label=f'N={target_n} for target')
        plt.xlabel('Number of top domains (N)', fontsize=12)
        plt.ylabel('Cumulative coverage (%)', fontsize=12)
        plt.title('Domain Whitelist Coverage Analysis', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = args.output.replace('.txt', '_coverage.png')
        plt.savefig(plot_path, dpi=150)
        print(f"  Plot saved to: {plot_path}")
        plt.close()
    
    # Get top-N domains by frequency (use target_n if it's smaller than --top-n)
    effective_n = min(args.top_n, target_n) if target_n and target_n < args.top_n else args.top_n
    top_domains = [domain for domain, _ in sorted_domains[:effective_n]]
    
    # Write to output file (one domain per line)
    print(f"Writing {len(top_domains)} domains to {args.output}")
    with open(args.output, "w") as f:
        for domain in top_domains:
            f.write(domain + "\n")
    
    print(f"Total unique domains found: {len(domain_counter)}")
    print(f"Top domain: {top_domains[0] if top_domains else 'N/A'} (count: {domain_counter[top_domains[0]] if top_domains else 0})")


if __name__ == "__main__":
    main()
