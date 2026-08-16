"""
Extract all Reddit posts/comments from specified subreddits.

This script streams through zst/jsonl files and extracts all records that match
the specified subreddit names (case-insensitive). It filters on both "subreddit"
and "subreddit_id" fields.

Creates separate output files per subreddit, split by comments/submissions based
on the input directory path.

Usage:
    python extract_all_from_subreddit.py \
        --subreddit biohackers scams \
        --input-files /path/to/comments/RC_2024-01.zst /path/to/submissions/RS_2024-01.zst \
        --output-dir sampled_data \
        --fields id author body subreddit created_utc

Output structure:
    sampled_data/
        comments/
            biohackers.jsonl
            scams.jsonl
        submissions/
            biohackers.jsonl
            scams.jsonl
        extraction_summary.json
"""

import argparse
import io
import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Set

try:
    import zstandard as zstd  # type: ignore
except ImportError:  # pragma: no cover
    zstd = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all Reddit posts/comments from specified subreddits."
    )
    parser.add_argument(
        "--subreddit",
        nargs="+",
        required=True,
        help="One or more subreddit names (e.g., biohackers, scams). 'r/' prefix is optional.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="One or more input files (.jsonl or .zst).",
    )
    parser.add_argument(
        "--output-dir",
        default="sampled_data",
        help="Directory for output files (default: sampled_data).",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Suffix to append to output filenames (e.g., month for parallel jobs).",
    )
    parser.add_argument(
        "--max-window-size",
        type=int,
        default=2147483648,
        help="Max zstd decode window size for .zst inputs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print progress every N seen records.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help="Specific fields to extract from matched records. If not specified, extracts all fields.",
    )
    return parser.parse_args()


def normalize_subreddit(subreddit: str) -> str:
    """Normalize subreddit name by stripping 'r/' prefix and lowercasing."""
    subreddit = str(subreddit).strip()
    if subreddit.startswith("r/"):
        subreddit = subreddit[2:]
    return subreddit.lower()


def normalize_record_subreddit(value: object) -> str:
    """Normalize subreddit value from a record for comparison."""
    if not value:
        return ""
    return normalize_subreddit(str(value))


def detect_file_type(input_path: Path) -> str:
    """Detect file type (comments or submissions) based on directory path."""
    path_str = str(input_path).lower()
    if "comments" in path_str or "/rc_" in path_str or "\\rc_" in path_str:
        return "comments"
    elif "submissions" in path_str or "/rs_" in path_str or "\\rs_" in path_str:
        return "submissions"
    else:
        # Default to based on filename prefix
        name = input_path.name.lower()
        if name.startswith("rc_"):
            return "comments"
        elif name.startswith("rs_"):
            return "submissions"
        return "unknown"


@contextmanager
def open_text_stream(path: Path, max_window_size: int) -> Iterator[io.TextIOBase]:
    """Context manager to open both .zst and .jsonl files as text streams."""
    if path.suffix.lower() == ".zst":
        if zstd is None:
            raise RuntimeError("zstandard is required to read .zst files")
        with path.open("rb") as raw:
            dctx = zstd.ZstdDecompressor(max_window_size=max_window_size)
            with dctx.stream_reader(raw) as reader:
                yield io.TextIOWrapper(reader, encoding="utf-8")
        return

    with path.open("r", encoding="utf-8") as handle:
        yield handle


def extract_from_subreddit(
    input_paths: List[Path],
    target_subreddits: Set[str],
    output_dir: Path,
    max_window_size: int,
    progress_every: int,
    fields: List[str] | None = None,
    output_suffix: str | None = None,
) -> tuple[int, int, Dict[str, int], Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Stream through input files and extract records matching target subreddits.

    Creates separate output files per subreddit, split by comments/submissions.

    Returns:
        (total_seen, total_matched, matched_by_subreddit, matched_by_type_subreddit, file_stats)
    """
    total_seen = 0
    total_matched = 0
    bad_json = 0
    matched_by_subreddit: Counter[str] = Counter()
    matched_by_type_subreddit: Dict[str, Dict[str, int]] = {}  # {type: {subreddit: count}}
    seen_by_file: Dict[str, int] = {}
    matched_by_file: Dict[str, int] = {}
    bad_json_by_file: Dict[str, int] = {}

    # Open output files for each subreddit x type combination
    output_handles: Dict[str, Dict[str, io.TextIOBase]] = {}  # {type: {subreddit: handle}}

    for input_path in input_paths:
        if not input_path.is_file():
            print(f"Warning: {input_path} not found, skipping")
            continue

        file_type = detect_file_type(input_path)
        file_seen = 0
        file_matched = 0
        file_bad = 0

        print(f"Processing: {input_path} (type: {file_type})")

        # Ensure output directory exists for this file type
        type_dir = output_dir / file_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # Open output files for this type if not already open
        if file_type not in output_handles:
            output_handles[file_type] = {}
            for subreddit in target_subreddits:
                suffix = f"_{output_suffix}" if output_suffix else ""
                output_file = type_dir / f"{subreddit}{suffix}.jsonl"
                output_handles[file_type][subreddit] = output_file.open("w", encoding="utf-8")

        # Initialize matched_by_type_subreddit for this type
        if file_type not in matched_by_type_subreddit:
            matched_by_type_subreddit[file_type] = {s: 0 for s in target_subreddits}

        try:
            with open_text_stream(input_path, max_window_size=max_window_size) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue

                    total_seen += 1
                    file_seen += 1

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        bad_json += 1
                        file_bad += 1
                        continue

                    # Check subreddit field
                    subreddit = normalize_record_subreddit(record.get("subreddit"))
                    if subreddit in target_subreddits:
                        output_record = {k: record[k] for k in fields if k in record} if fields else record
                        output_handles[file_type][subreddit].write(json.dumps(output_record, ensure_ascii=False) + "\n")
                        total_matched += 1
                        file_matched += 1
                        matched_by_subreddit[subreddit] += 1
                        matched_by_type_subreddit[file_type][subreddit] += 1
                        continue

                    # Check subreddit_id field (also normalize)
                    subreddit_id = normalize_record_subreddit(record.get("subreddit_id"))
                    if subreddit_id in target_subreddits:
                        output_record = {k: record[k] for k in fields if k in record} if fields else record
                        output_handles[file_type][subreddit_id].write(json.dumps(output_record, ensure_ascii=False) + "\n")
                        total_matched += 1
                        file_matched += 1
                        matched_by_subreddit[subreddit_id] += 1
                        matched_by_type_subreddit[file_type][subreddit_id] += 1
                        continue

                    if progress_every > 0 and file_seen % progress_every == 0:
                        print(
                            f"  Progress: seen={file_seen}, matched={file_matched}, "
                            f"bad_json={file_bad}"
                        )
        except zstd.ZstdError as e:
            # File-level corruption - cannot continue past this point
            print(f"[ERROR] Zstd decompression error in {input_path}: {e}", file=sys.stderr)
            print(f"[ERROR] File may be corrupted or truncated", file=sys.stderr)
            print(f"[INFO] Partial results saved: {file_matched} records extracted before error", file=sys.stderr)
        except Exception as e:
            # Unexpected error - treat as corruption but preserve partial results
            print(f"[ERROR] Unexpected error in {input_path}: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"[INFO] Partial results saved: {file_matched} records extracted before error", file=sys.stderr)

        seen_by_file[input_path.name] = file_seen
        matched_by_file[input_path.name] = file_matched
        bad_json_by_file[input_path.name] = file_bad

        print(
            f"Completed {input_path.name}: seen={file_seen}, matched={file_matched}, "
            f"bad_json={file_bad}"
        )

    # Close all output handles
    for file_type, handles in output_handles.items():
        for subreddit, handle in handles.items():
            handle.close()

    return total_seen, total_matched, dict(matched_by_subreddit), matched_by_type_subreddit, {
        "seen_by_file": seen_by_file,
        "matched_by_file": matched_by_file,
        "bad_json_by_file": bad_json_by_file,
    }


def main() -> None:
    args = parse_args()

    # Normalize target subreddits
    target_subreddits = {normalize_subreddit(s) for s in args.subreddit}
    print(f"Target subreddits: {sorted(target_subreddits)}")

    # Validate input files
    input_paths = [Path(f) for f in args.input_files]
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"Input file not found: {path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Extracting records from subreddits")
    print("=" * 60)

    total_seen, total_matched, matched_by_subreddit, matched_by_type_subreddit, file_stats = extract_from_subreddit(
        input_paths=input_paths,
        target_subreddits=target_subreddits,
        output_dir=output_dir,
        max_window_size=args.max_window_size,
        progress_every=args.progress_every,
        fields=args.fields,
        output_suffix=args.output_suffix,
    )

    print("\n" + "=" * 60)
    print("Extraction complete")
    print("=" * 60)
    print(f"Total records seen: {total_seen}")
    print(f"Records extracted: {total_matched}")
    print(f"Malformed JSON lines skipped: {sum(file_stats['bad_json_by_file'].values())}")
    print(f"Output directory: {output_dir}")

    print("\nRecords extracted by subreddit:")
    for subreddit, count in sorted(matched_by_subreddit.items()):
        print(f"  {subreddit}: {count}")

    print("\nRecords extracted by type and subreddit:")
    for file_type, subreddit_counts in sorted(matched_by_type_subreddit.items()):
        print(f"  {file_type}:")
        for subreddit, count in sorted(subreddit_counts.items()):
            print(f"    {subreddit}: {count}")

    # Generate summary
    summary = {
        "target_subreddits": sorted(target_subreddits),
        "input_files": [str(path) for path in input_paths],
        "output_dir": str(output_dir),
        "fields_extracted": args.fields if args.fields else "all",
        "total_records_seen": total_seen,
        "total_records_extracted": total_matched,
        "records_extracted_by_subreddit": dict(matched_by_subreddit),
        "records_extracted_by_type_subreddit": matched_by_type_subreddit,
        "malformed_json_count": sum(file_stats["bad_json_by_file"].values()),
        "file_stats": file_stats,
    }

    summary_path = output_dir / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
