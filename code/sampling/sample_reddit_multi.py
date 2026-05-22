import argparse
import glob
import io
import json
import os
import random
from typing import List

import zstandard as zstd  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample JSON records from multiple .zst Reddit dumps."
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


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not args.inputs and not args.input_dir:
        raise SystemExit("Provide at least one of --inputs or --input-dir")

    input_files = discover_input_files(args.inputs, args.input_dir, args.input_glob, args.recursive)
    if not input_files:
        raise SystemExit("No .zst input files found")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    sampled_count = 0
    total_seen = 0
    bad_json = 0

    print(f"Found {len(input_files)} input files")
    print(f"Writing sampled output to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as out:
        for file_index, input_path in enumerate(input_files, start=1):
            if sampled_count >= args.max_samples:
                break

            file_seen = 0
            file_sampled = 0
            file_bad = 0

            print(f"[{file_index}/{len(input_files)}] Processing: {input_path}")

            with open(input_path, "rb") as f:
                dctx = zstd.ZstdDecompressor(max_window_size=args.max_window_size)
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

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
                            out.write(json.dumps(data) + "\n")
                            sampled_count += 1
                            file_sampled += 1

                        if sampled_count >= args.max_samples:
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