import zstandard as zstd # type: ignore
import json
import random
import os
import argparse
import io

# Utility to sample Reddit JSON records from a compressed .zst dump.

def parse_args():
    # Parse command-line options for the input file, output path, and sampling settings.
    parser = argparse.ArgumentParser(description="Sample JSON records from a .zst Reddit dump.")
    parser.add_argument("--input", required=True, help="Path to input .zst file")
    parser.add_argument("--output-dir", default="zst_sample", help="Directory to write output file")
    parser.add_argument("--output-file", default="sample.jsonl", help="Output JSONL file name")
    parser.add_argument("--p", type=float, default=0.001, help="Sampling probability (default: 0.001)")
    parser.add_argument("--max-samples", type=int, default=5000, help="Maximum number of sampled rows")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument(
        "--max-window-size",
        type=int,
        default=2147483648,
        help="Maximum zstd decode window size; increase if frame window is too large",
    )
    return parser.parse_args()


def main():
    # Read CLI arguments and initialize the random generator for reproducible sampling.
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Create the output directory if it does not already exist.
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    count = 0
    total_seen = 0
    bad_json = 0

    # Open the compressed input stream and decode it incrementally.
    with open(args.input, "rb") as f:
        dctx = zstd.ZstdDecompressor(max_window_size=args.max_window_size)
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            # Write sampled JSON objects to a JSONL output file.
            with open(output_path, "w", encoding="utf-8") as out:
                for line in text_stream:
                    total_seen += 1

                    # Skip malformed lines and count them for reporting.
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        bad_json += 1
                        continue

                    # Keep the record with probability p.
                    if random.random() < args.p:
                        out.write(json.dumps(data) + "\n")
                        count += 1

                    # Stop once the requested number of samples has been collected.
                    if count >= args.max_samples:
                        break

                    # Print progress periodically for large input files.
                    if total_seen % 100000 == 0:
                        print(f"Seen: {total_seen}, Sampled: {count}, Bad JSON: {bad_json}")

    print(f"Done! Sampled {count} lines.")
    print(f"Skipped {bad_json} malformed JSON lines.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()