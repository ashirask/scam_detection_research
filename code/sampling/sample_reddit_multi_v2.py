import argparse
import glob
import io
import json
import os
import random
from typing import List, Optional, Tuple

"""
This script samples JSON records equally from multiple .zst Reddit dumps. It supports filtering input files by year parsed from the filename, 
and adds source metadata to each sampled record for traceability. The output is written as a JSONL file containing the sampled records.

FINAL DRAFT for sampling**
"""

import zstandard as zstd  # type: ignore
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample JSON records equally from multiple .zst Reddit dumps."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Optional explicit list of .zst files",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing .zst files",
    )
    parser.add_argument(
        "--input-glob",
        default="*.zst",
        help="Glob pattern inside --input-dir (default: *.zst)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively inside --input-dir",
    )
    parser.add_argument(
        "--output-dir",
        default="zst_sample",
        help="Directory to write output file",
    )
    parser.add_argument(
        "--output-file",
        default="sample.jsonl",
        help="Output JSONL file name",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.001,
        help="Sampling probability (default: 0.001)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum total sampled rows across all files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    parser.add_argument(
        "--max-window-size",
        type=int,
        default=2147483648,
        help="Maximum zstd decode window size; increase if frame window is too large",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print progress every N seen rows (default: 100000)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional start year (inclusive) to filter input files by year in filename",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional end year (inclusive) to filter input files by year in filename",
    )
    return parser.parse_args()

def discover_input_files(inputs, input_dir, input_glob, recursive) -> List[str]:
    files: List[str] = []

    if inputs:
        files.extend(inputs)

    if input_dir:
        pattern = os.path.join(input_dir, "**", input_glob) if recursive else os.path.join(input_dir, input_glob)
        files.extend(glob.glob(pattern, recursive=recursive))

    unique_files = sorted({os.path.abspath(path) for path in files if path.lower().endswith(".zst")})
    return [path for path in unique_files if os.path.isfile(path)]


def _extract_year_from_path(path: str) -> Optional[int]:
    """Return the first 4-digit year found in the filename, else None."""
    base = os.path.basename(path)
    m = re.search(r"(19\d{2}|20\d{2})", base)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _extract_period_from_path(path: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (year, month) parsed from a filename if possible.

    Supports filenames like:
    - RC_2024-03.zst
    - comments_2024_03.zst
    - sample_2024.zst

    If only a year is found, month is returned as None.
    """
    base = os.path.basename(path)

    year_month = re.search(r"(19\d{2}|20\d{2})[-_](0[1-9]|1[0-2])", base)
    if year_month:
        return int(year_month.group(1)), int(year_month.group(2))

    year_only = _extract_year_from_path(path)
    return year_only, None

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not args.inputs and not args.input_dir:
        raise SystemExit("Provide at least one of --inputs or --input-dir")

    input_files = discover_input_files(args.inputs, args.input_dir, args.input_glob, args.recursive)
    if not input_files:
        raise SystemExit("No .zst input files found")

    # If year range provided, filter files by year parsed from filename
    if args.start_year is not None or args.end_year is not None:
        start = args.start_year if args.start_year is not None else -10_000
        end = args.end_year if args.end_year is not None else 10_000
        filtered = []
        skipped = []
        for p in input_files:
            y, _ = _extract_period_from_path(p)
            if y is None:
                # conservatively skip files without a clearly parseable year
                skipped.append(p)
                continue
            if start <= y <= end:
                filtered.append(p)
        if not filtered:
            raise SystemExit(f"No input files found in year range {start}-{end}")
        print(f"Filtered {len(input_files)} -> {len(filtered)} files by year range {start}-{end} (skipped {len(skipped)} files without year)")
        input_files = filtered

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    # Calculate per-file sample limit
    num_files = len(input_files)
    samples_per_file = max(1, args.max_samples // num_files)

    sampled_count = 0
    total_seen = 0
    bad_json = 0

    print(f"Found {num_files} input files")
    print(f"Writing sampled output to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as out:
        for file_index, input_path in enumerate(input_files, start=1):
            file_seen = 0
            file_sampled = 0
            file_bad = 0

            print(f"[{file_index}/{num_files}] Processing: {input_path}")

            with open(input_path, "rb") as f:
                dctx = zstd.ZstdDecompressor(max_window_size=args.max_window_size)
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

                    source_year, source_month = _extract_period_from_path(input_path)
                    source_period = None
                    if source_year is not None:
                        if source_month is not None:
                            source_period = f"{source_year:04d}-{source_month:02d}"
                        else:
                            source_period = f"{source_year:04d}"

                    for line in text_stream:
                        total_seen += 1
                        file_seen += 1

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            bad_json += 1
                            file_bad += 1
                            continue

                        if random.random() < args.p:
                            # Add source metadata so downstream analysis can trace each record
                            # back to its originating month/year file without changing the payload.
                            data = dict(data)
                            data["source_file"] = os.path.basename(input_path)
                            data["source_year"] = source_year
                            data["source_month"] = source_month
                            data["source_period"] = source_period
                            out.write(json.dumps(data, ensure_ascii=False) + "\n")
                            sampled_count += 1
                            file_sampled += 1

                        if file_sampled >= samples_per_file:
                            break

                        if args.progress_every > 0 and total_seen % args.progress_every == 0:
                            print(
                                f"Seen: {total_seen}, Sampled: {sampled_count}, "
                                f"Bad JSON: {bad_json}, Current file seen: {file_seen}"
                            )

            print(
                f"Completed file: seen={file_seen}, sampled={file_sampled}, bad_json={file_bad}"
            )

    print("Done!")
    print(f"Total sampled lines: {sampled_count}")
    print(f"Total malformed JSON lines skipped: {bad_json}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()