"""
fetch_botrank.py
----------------
Scrapes https://botrank.pastimes.eu/ and saves results to CSV.

Usage examples:
  # Top 500 by rank (default)
  python fetch_botrank.py

  # Top 1000 sorted by bad votes, all columns
  python fetch_botrank.py --sort bad-votes --top-n 1000

  # Top 500 by score, only name + score columns
  python fetch_botrank.py --sort score --columns "bot_name,score" --output botrank_score.csv

Sort options:
  rank         (default) — overall BotRank rank
  score        — bot score (0–1)
  bad-votes    — bad bot votes (used by BotBusters paper)
  good-votes   — good bot votes
  comment-karma
  link-karma

Column options (comma-separated, no spaces):
  rank, bot_name, score, good_votes, bad_votes, comment_karma, link_karma
"""

import argparse
import time
import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://botrank.pastimes.eu/"

SORT_OPTIONS = [
    "rank",
    "score",
    "bad-votes",
    "good-votes",
    "comment-karma",
    "link-karma",
]

# Maps user-facing column names to the raw scraped header text
COLUMN_MAP = {
    "rank":           "Rank",
    "bot_name":       "Bot Name",
    "score":          "Score",
    "good_votes":     "Good Bot Votes",
    "bad_votes":      "Bad Bot Votes",
    "comment_karma":  "Comment Karma",
    "link_karma":     "Link Karma",
}

ALL_COLUMNS = list(COLUMN_MAP.keys())


def scrape_page(session: requests.Session, sort: str, page: int) -> list[dict]:
    """Fetch one page and return a list of row dicts."""
    params = {"sort": sort, "page": page}
    try:
        resp = session.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Page {page} failed: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []  # no table = past last page

    # Parse header row to get column order dynamically
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    if not headers:
        return []

    rows = []
    for tr in table.find_all("tr")[1:]:  # skip header
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) == len(headers):
            rows.append(dict(zip(headers, cols)))

    return rows


def scrape_botrank(
    sort: str,
    top_n: int,
    delay: float,
    columns: list[str],
    output: str,
) -> pd.DataFrame:

    session = requests.Session()
    session.headers.update({
        "User-Agent": "academic-bot-research/1.0"
    })

    all_rows = []
    page = 1
    print(f"Scraping BotRank | sort={sort} | target={top_n} rows | delay={delay}s/page")

    while len(all_rows) < top_n:
        print(f"  Page {page} — collected {len(all_rows)} so far...", end="\r")
        rows = scrape_page(session, sort, page)

        if not rows:
            print(f"\n  No more data after page {page - 1}. Stopping.")
            break

        all_rows.extend(rows)
        page += 1
        time.sleep(delay)

    print(f"\n  Done. Total rows scraped: {len(all_rows)}")

    df = pd.DataFrame(all_rows)

    # Rename columns to snake_case using COLUMN_MAP (inverted)
    inverse_map = {v: k for k, v in COLUMN_MAP.items()}
    df.rename(columns=inverse_map, inplace=True)

    # Trim to top_n
    df = df.head(top_n)

    # Select only requested columns (keep any that exist)
    available = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"  [WARN] Columns not found in scraped data: {missing}", file=sys.stderr)
    df = df[available]

    # Type coercion
    for col in ["rank", "good_votes", "bad_votes", "comment_karma", "link_karma"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Save
    df.to_csv(output, index=False)
    print(f"  Saved {len(df)} rows to '{output}'")
    print(f"  Columns: {list(df.columns)}")

    if "score" in df.columns:
        print(f"  Score range: {df['score'].min():.4f} – {df['score'].max():.4f}")
    if "bad_votes" in df.columns:
        print(f"  Bad votes range: {int(df['bad_votes'].min())} – {int(df['bad_votes'].max())}")

    return df


def parse_columns(col_str: str) -> list[str]:
    cols = [c.strip() for c in col_str.split(",")]
    invalid = [c for c in cols if c not in COLUMN_MAP]
    if invalid:
        print(f"[ERROR] Unknown column(s): {invalid}", file=sys.stderr)
        print(f"Valid columns: {ALL_COLUMNS}", file=sys.stderr)
        sys.exit(1)
    return cols


def main():
    parser = argparse.ArgumentParser(
        description="Scrape BotRank (https://botrank.pastimes.eu/) to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sort",
        default="rank",
        choices=SORT_OPTIONS,
        help="Sort order for results (default: rank)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="Number of rows to collect (default: 500)",
    )
    parser.add_argument(
        "--columns",
        default=",".join(ALL_COLUMNS),
        help=(
            "Comma-separated columns to include. "
            f"Options: {', '.join(ALL_COLUMNS)}. "
            "Default: all columns."
        ),
    )
    parser.add_argument(
        "--output",
        default="botrank.csv",
        help="Output CSV filename (default: botrank.csv)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to wait between page requests (default: 1.5)",
    )

    args = parser.parse_args()
    columns = parse_columns(args.columns)

    scrape_botrank(
        sort=args.sort,
        top_n=args.top_n,
        delay=args.delay,
        columns=columns,
        output=args.output,
    )


if __name__ == "__main__":
    main()