#!/usr/bin/env python3
"""
merge_per_author.py
====================
Consolidates the per-author .jsonl.gz files produced by download_arctic_shift.py
into the 4 JSONL files build_features.py expects:

    user_comments_bot.jsonl      (one row per author: {"author":..., "comments":[...]})
    user_comments_human.jsonl
    user_submissions_bot.jsonl
    user_submissions_human.jsonl

Unlike the original month-partitioned merge (merge_bot_extractions.py), no
SQLite disk-spill is needed here: the download stage already isolates each
author into their own small file, so grouping is free -- we just iterate
authors and stream each one straight through. RAM stays flat because we only
ever hold one author's records at a time.

Applies --min-posts as a combined (comments + submissions) threshold, same
semantics as the original script.

USAGE
-----
python merge_per_author.py \
    --raw-dir    raw_arctic_shift/ \
    --output-dir merged/ \
    --min-posts  3
"""

import argparse
import gzip
import json
import logging
from pathlib import Path


def count_lines_gz(path):
    if not path.exists():
        return 0
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def read_records_gz(path):
    if not path.exists():
        return []
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)  # keep raw text, splice into output like the original script
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="Output dir passed to download_arctic_shift.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-posts", type=int, default=3, help="Minimum comments+submissions combined to keep an author")
    ap.add_argument("--require-both", action="store_true", help="Only include authors with both comments AND submissions")
    ap.add_argument("--log-file", default=None, help="Optional file to write log messages to")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(args.log_file)] if args.log_file else []),
    )

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"bot": {"authors_seen": 0, "authors_kept": 0, "filtered_missing": 0}, 
             "human": {"authors_seen": 0, "authors_kept": 0, "filtered_missing": 0}}

    for label in ("bot", "human"):
        comments_dir = raw_dir / "comments" / label
        submissions_dir = raw_dir / "submissions" / label

        # Union of authors that have *any* output file for this label
        authors = set()
        if comments_dir.exists():
            authors |= {p.name[: -len(".jsonl.gz")] for p in comments_dir.glob("*.jsonl.gz")}
        if submissions_dir.exists():
            authors |= {p.name[: -len(".jsonl.gz")] for p in submissions_dir.glob("*.jsonl.gz")}

        comments_out = output_dir / f"user_comments_{label}.jsonl"
        submissions_out = output_dir / f"user_submissions_{label}.jsonl"

        with open(comments_out, "w", encoding="utf-8") as f_c, open(submissions_out, "w", encoding="utf-8") as f_s:
            for idx, author in enumerate(sorted(authors)):
                stats[label]["authors_seen"] += 1

                # Check if both files exist if --require-both is set
                comments_path = comments_dir / f"{author}.jsonl.gz"
                submissions_path = submissions_dir / f"{author}.jsonl.gz"
                
                if args.require_both:
                    if not comments_path.exists():
                        stats[label]["filtered_missing"] += 1
                        logging.debug(f"[{label}] Skipping {author}: missing comments file")
                        continue
                    if not submissions_path.exists():
                        stats[label]["filtered_missing"] += 1
                        logging.debug(f"[{label}] Skipping {author}: missing submissions file")
                        continue

                comment_lines = read_records_gz(comments_path)
                submission_lines = read_records_gz(submissions_path)

                total = len(comment_lines) + len(submission_lines)
                if total < args.min_posts:
                    continue

                stats[label]["authors_kept"] += 1
                author_json = json.dumps(author)
                f_c.write('{"author":' + author_json + ',"comments":[' + ",".join(comment_lines) + "]}\n")
                f_s.write('{"author":' + author_json + ',"submissions":[' + ",".join(submission_lines) + "]}\n")
                
                # Log progress every 100 authors
                if (idx + 1) % 100 == 0:
                    logging.info(f"[{label}] Processed {idx + 1}/{len(authors)} authors, kept {stats[label]['authors_kept']} so far")

        logging.info(f"[{label}] seen={stats[label]['authors_seen']:,} kept={stats[label]['authors_kept']:,} "
                    f"filtered_missing={stats[label]['filtered_missing']:,} "
                    f"-> {comments_out.name}, {submissions_out.name}")

    logging.info("\nDone. Pass these 4 files to build_features.py:")
    logging.info(f"  --comments-bot      {output_dir / 'user_comments_bot.jsonl'}")
    logging.info(f"  --comments-human    {output_dir / 'user_comments_human.jsonl'}")
    logging.info(f"  --submissions-bot   {output_dir / 'user_submissions_bot.jsonl'}")
    logging.info(f"  --submissions-human {output_dir / 'user_submissions_human.jsonl'}")


if __name__ == "__main__":
    main()
