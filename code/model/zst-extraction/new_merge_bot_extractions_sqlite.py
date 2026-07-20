# =============================================================================
# merge_bot_extractions.py  (memory-efficient rewrite)
# =============================================================================
# Consolidates Pass 2 compressed output files into per-author aggregated records.
#
# MEMORY STRATEGY
# ---------------
# The original script held all records for every author in two in-memory dicts
# simultaneously, which can reach hundreds of GB for large bot populations.
#
# This rewrite uses SQLite as a disk-backed intermediate store:
#   1. Stream every record → INSERT one row into SQLite (constant RAM)
#   2. SELECT ordered by author → write one author at a time to output (constant RAM)
#
# Peak RAM is now roughly:  one author's full record list  +  SQLite page cache
# (configurable via --cache-mb; default 512 MB, safe even on 4 GB nodes)
#
# Usage:
#   python merge_bot_extractions.py \
#     --input-dir    results/ \
#     --authors-file bot_authors_global.txt \
#     --output-dir   merged/ \
#     --min-posts    3 \
#     [--cache-mb    512] \
#     [--tmp-dir     /scratch/tmp]
# =============================================================================

import argparse
import glob
import gzip
import json
import os
import sqlite3
import tempfile
from itertools import groupby
from pathlib import Path

try:
    import orjson
    _loads = orjson.loads
    _dumps = lambda obj: orjson.dumps(obj).decode("utf-8")
except ImportError:
    orjson = None
    _loads = json.loads
    _dumps = json.dumps


# =============================================================================
# CONSTANTS
# =============================================================================

SKIP_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}


# =============================================================================
# HELPERS
# =============================================================================

def stream_jsonl_gz(path):
    """Stream JSON records from a gzip-compressed JSONL file."""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield _loads(line)
            except Exception:
                continue


def make_db(tmp_dir, cache_mb):
    """
    Create an on-disk SQLite database in tmp_dir.

    Using on-disk (not :memory:) is the whole point — SQLite pages spill to
    disk so RAM stays flat regardless of total data size.
    """
    db_path = os.path.join(tmp_dir, "merge_bot_tmp.sqlite")
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Tune for bulk-insert performance
    cur.executescript(f"""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous  = NORMAL;
        PRAGMA cache_size   = -{cache_mb * 1024};  -- negative = kilobytes
        PRAGMA temp_store   = FILE;
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS records (
            author  TEXT    NOT NULL,
            kind    TEXT    NOT NULL,   -- 'comment' or 'submission'
            payload TEXT    NOT NULL    -- raw JSON string
        )
    """)
    # Index created AFTER bulk insert (much faster that way)
    con.commit()
    return con, db_path


def bulk_insert(con, rows, batch_size=50_000):
    """Insert rows in batches to avoid huge transactions."""
    cur = con.cursor()
    for i in range(0, len(rows), batch_size):
        cur.executemany(
            "INSERT INTO records (author, kind, payload) VALUES (?, ?, ?)",
            rows[i : i + batch_size],
        )
        con.commit()


def ingest_files(con, pattern, kind, skip_authors, stats_key, stats):
    """
    Stream all files matching pattern, insert records into SQLite.
    Accumulates rows in a buffer and flushes every FLUSH_EVERY records
    to keep memory flat during ingestion too.
    """
    FLUSH_EVERY = 100_000
    buffer = []
    file_count = 0
    record_count = 0

    for filepath in sorted(glob.glob(pattern)):
        if "_summary" in filepath:
            print(f"  Skipping {filepath} (summary file)")
            continue

        print(f"  Reading {filepath}...")
        file_count += 1

        for record in stream_jsonl_gz(filepath):
            author = record.get("author", "")
            if author in skip_authors:
                continue
            buffer.append((author, kind, _dumps(record)))
            record_count += 1

            if len(buffer) >= FLUSH_EVERY:
                bulk_insert(con, buffer)
                buffer.clear()

    if buffer:
        bulk_insert(con, buffer)
        buffer.clear()

    stats[f"{stats_key}_files"] = file_count
    stats[f"total_{stats_key}s"] = record_count
    print(f"  Processed {file_count} files, {record_count:,} records")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge Pass 2 bot extractions into per-author JSONL files (memory-efficient).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-dir",    required=True)
    parser.add_argument("--authors-file", required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--min-posts",    type=int, default=3)
    parser.add_argument(
        "--cache-mb",
        type=int,
        default=512,
        help="SQLite page-cache size in MB (default: 512). Raise to speed up I/O if RAM allows.",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for the temporary SQLite file (default: system tmp). "
             "Use a fast local scratch dir on the cluster (e.g. /scratch/$USER).",
    )

    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "comment_files": 0,
        "submission_files": 0,
        "total_comments": 0,
        "total_submissions": 0,
        "unique_authors": 0,
        "authors_dropped": 0,
        "authors_kept": 0,
    }

    print("=== Merging Pass 2 Bot Extractions (memory-efficient) ===")

    # -------------------------------------------------------------------------
    # Step 1 & 2: Stream all files into SQLite (disk-backed, flat RAM)
    # -------------------------------------------------------------------------
    tmp_ctx = tempfile.TemporaryDirectory(dir=args.tmp_dir)
    with tmp_ctx as tmp_dir:
        print(f"\nTemporary SQLite DB: {tmp_dir}/merge_bot_tmp.sqlite")
        con, db_path = make_db(tmp_dir, args.cache_mb)

        print("\nIngesting comment files...")
        ingest_files(
            con,
            str(input_dir / "comments_bots_*.jsonl.gz"),
            "comment",
            SKIP_AUTHORS,
            "comment",
            stats,
        )

        print("\nIngesting submission files...")
        ingest_files(
            con,
            str(input_dir / "submissions_bots_*.jsonl.gz"),
            "submission",
            SKIP_AUTHORS,
            "submission",
            stats,
        )

        # Build index now that all data is inserted
        print("\nBuilding index on (author, kind)...")
        con.execute("CREATE INDEX idx_author_kind ON records (author, kind)")
        con.commit()

        # -------------------------------------------------------------------------
        # Step 3: Compute per-author totals and apply min-posts filter
        # -------------------------------------------------------------------------
        print(f"\nApplying minimum post threshold ({args.min_posts})...")

        cur = con.execute("""
            SELECT author, COUNT(*) as total
            FROM records
            GROUP BY author
            HAVING total >= ?
        """, (args.min_posts,))
        kept_authors = {row[0] for row in cur}

        cur2 = con.execute("SELECT COUNT(DISTINCT author) FROM records")
        stats["unique_authors"] = cur2.fetchone()[0]
        stats["authors_kept"]   = len(kept_authors)
        stats["authors_dropped"] = stats["unique_authors"] - stats["authors_kept"]

        print(f"  Unique authors found          : {stats['unique_authors']:,}")
        print(f"  Dropped (< {args.min_posts} posts)         : {stats['authors_dropped']:,}")
        print(f"  Final bot authors kept        : {stats['authors_kept']:,}")

        # -------------------------------------------------------------------------
        # Step 4: Write output — one author at a time (constant RAM)
        # -------------------------------------------------------------------------
        print("\nWriting output files...")

        comments_out    = output_dir / "user_comments_bots.jsonl"
        submissions_out = output_dir / "user_submissions_bots.jsonl"

        # Single pass over the table, ordered so we can group without extra RAM.
        # We write both output files simultaneously to avoid a second DB pass.
        with open(comments_out, 'w', encoding='utf-8') as f_comments, \
             open(submissions_out, 'w', encoding='utf-8') as f_submissions:

            cur = con.execute("""
                SELECT author, kind, payload
                FROM records
                WHERE author IN (
                    SELECT author FROM records
                    GROUP BY author
                    HAVING COUNT(*) >= ?
                )
                ORDER BY author, kind
            """, (args.min_posts,))

            # groupby works because rows are ORDER BY author.
            # We keep payloads as raw strings and stitch them into JSON manually
            # — this avoids deserializing every record into a Python dict and
            # re-serializing it, which was the source of the RAM spike.
            for author, author_rows in groupby(cur, key=lambda r: r[0]):
                comment_payloads    = []
                submission_payloads = []
                for _, kind, payload in author_rows:
                    if kind == "comment":
                        comment_payloads.append(payload)
                    else:
                        submission_payloads.append(payload)

                # Build JSON lines by hand: {"author": "...", "comments": [<raw>, <raw>, ...]}
                # json.dumps only touches the author string; payloads are spliced in as-is.
                author_json = json.dumps(author)
                f_comments.write(
                    '{"author":' + author_json +
                    ',"comments":[' + ','.join(comment_payloads) + ']}\n'
                )
                f_submissions.write(
                    '{"author":' + author_json +
                    ',"submissions":[' + ','.join(submission_payloads) + ']}\n'
                )

        print(f"  Wrote {stats['authors_kept']:,} authors to {comments_out}")
        print(f"  Wrote {stats['authors_kept']:,} authors to {submissions_out}")

        # Explicitly close DB before tmp dir cleanup so SQLite WAL/SHM
        # journal files are flushed and removed — prevents OSError on cleanup
        con.close()

    # tmp_ctx cleans up the SQLite file here

    # -------------------------------------------------------------------------
    # Step 5: Sanity check against global authors list
    # -------------------------------------------------------------------------
    print("\nSanity check against global authors list...")

    with open(args.authors_file, 'r', encoding='utf-8') as f:
        expected_authors = set(line.strip() for line in f if line.strip())

    unexpected = kept_authors - expected_authors
    missing    = expected_authors - kept_authors

    if unexpected:
        print(f"  [WARN] {len(unexpected)} authors in output not in global list")
    if missing:
        print(f"  [INFO] {len(missing)} expected authors have no records in Pass 2 output")
        print(f"         (likely posted only in corrupted months or below min-posts threshold)")

    # -------------------------------------------------------------------------
    # Step 6: Summary
    # -------------------------------------------------------------------------
    summary_lines = [
        "=== merge_bot_extractions Summary ===",
        "",
        "Input",
        f"  Comment files processed   : {stats['comment_files']}",
        f"  Submission files processed : {stats['submission_files']}",
        f"  Total comment records     : {stats['total_comments']:,}",
        f"  Total submission records  : {stats['total_submissions']:,}",
        "",
        "Authors",
        f"  Unique authors found      : {stats['unique_authors']:,}",
        f"  Dropped (< {args.min_posts} posts)       : {stats['authors_dropped']:,}",
        f"  Final bot authors kept    : {stats['authors_kept']:,}",
        "",
        "Output",
        f"  user_comments_bots.jsonl    : {stats['authors_kept']:,} authors",
        f"  user_submissions_bots.jsonl : {stats['authors_kept']:,} authors",
        "",
        "Note: label assignment (y=1) happens in build_features.py, not here.",
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_file = output_dir / "merge_bot_extractions_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')
    print(f"\nSummary written to: {summary_file}")


if __name__ == "__main__":
    main()
