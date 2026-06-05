"""
Extract all Reddit posts/comments from specified subreddits.

This script streams through zst/jsonl files and extracts all records that match
the specified subreddit names (case-insensitive). It filters on both "subreddit"
and "subreddit_id" fields.

Usage:
    python extract_all_from_subreddit.py \
        --subreddit biohackers scams \
        --input-file RC_2024-01.zst RC_2024-02.zst \
        --output-dir sampled_data
"""

import argparse
import io
import json
import os
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
        "--output-file",
        default="extracted_from_subreddit.jsonl",
        help="Output JSONL filename (default: extracted_from_subreddit.jsonl).",
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
    output_path: Path,
    max_window_size: int,
    progress_every: int,
) -> tuple[int, int, Dict[str, int], Dict[str, int]]:
    """
    Stream through input files and extract records matching target subreddits.

    Returns:
        (total_seen, total_matched, bad_json_count, matched_by_subreddit)
    """
    total_seen = 0
    total_matched = 0
    bad_json = 0
    matched_by_subreddit: Counter[str] = Counter()
    seen_by_file: Dict[str, int] = {}
    matched_by_file: Dict[str, int] = {}
    bad_json_by_file: Dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as out:
        for input_path in input_paths:
            if not input_path.is_file():
                print(f"Warning: {input_path} not found, skipping")
                continue

            file_seen = 0
            file_matched = 0
            file_bad = 0

            print(f"Processing: {input_path}")

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
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_matched += 1
                        file_matched += 1
                        matched_by_subreddit[subreddit] += 1
                        continue

                    # Check subreddit_id field (also normalize)
                    subreddit_id = normalize_record_subreddit(record.get("subreddit_id"))
                    if subreddit_id in target_subreddits:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_matched += 1
                        file_matched += 1
                        matched_by_subreddit[subreddit_id] += 1
                        continue

                    if progress_every > 0 and file_seen % progress_every == 0:
                        print(
                            f"  Progress: seen={file_seen}, matched={file_matched}, "
                            f"bad_json={file_bad}"
                        )

            seen_by_file[input_path.name] = file_seen
            matched_by_file[input_path.name] = file_matched
            bad_json_by_file[input_path.name] = file_bad

            print(
                f"Completed {input_path.name}: seen={file_seen}, matched={file_matched}, "
                f"bad_json={file_bad}"
            )

    return total_seen, total_matched, dict(matched_by_subreddit), {
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
    output_path = output_dir / args.output_file

    print("=" * 60)
    print("Extracting records from subreddits")
    print("=" * 60)

    total_seen, total_matched, matched_by_subreddit, file_stats = extract_from_subreddit(
        input_paths=input_paths,
        target_subreddits=target_subreddits,
        output_path=output_path,
        max_window_size=args.max_window_size,
        progress_every=args.progress_every,
    )

    print("\n" + "=" * 60)
    print("Extraction complete")
    print("=" * 60)
    print(f"Total records seen: {total_seen}")
    print(f"Records extracted: {total_matched}")
    print(f"Malformed JSON lines skipped: {file_stats['bad_json_by_file']}")
    print(f"Output saved to: {output_path}")

    print("\nRecords extracted by subreddit:")
    for subreddit, count in sorted(matched_by_subreddit.items()):
        print(f"  {subreddit}: {count}")

    # Generate summary
    summary = {
        "target_subreddits": sorted(target_subreddits),
        "input_files": [str(path) for path in input_paths],
        "output_file": str(output_path),
        "total_records_seen": total_seen,
        "total_records_extracted": total_matched,
        "records_extracted_by_subreddit": dict(matched_by_subreddit),
        "malformed_json_count": sum(file_stats["bad_json_by_file"].values()),
        "file_stats": file_stats,
    }

    summary_path = output_dir / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
