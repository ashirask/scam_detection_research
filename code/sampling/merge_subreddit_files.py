"""
Merge monthly subreddit extraction files into single files per subreddit.

Combines all files matching {subreddit}_{suffix}.jsonl into {subreddit}.jsonl
for both comments and submissions directories.

Usage:
    python merge_subreddit_files.py \
        --input-dir sampled_data \
        --output-dir merged_data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge monthly subreddit extraction files into single files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the split files (e.g., sampled_data).",
    )
    parser.add_argument(
        "--output-dir",
        default="merged_data",
        help="Directory for merged output files (default: merged_data).",
    )
    return parser.parse_args()


def find_split_files(input_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Find all split files organized by type and subreddit.

    Returns:
        {type: {subreddit: [file_paths]}}
    """
    result: Dict[str, Dict[str, List[Path]]] = {}

    for file_type in ["comments", "submissions"]:
        type_dir = input_dir / file_type
        if not type_dir.exists():
            print(f"Warning: {type_dir} not found, skipping")
            continue

        result[file_type] = {}
        for file_path in type_dir.glob("*.jsonl"):
            # Extract subreddit name (remove suffix like _2025-11)
            stem = file_path.stem
            if "_" in stem:
                subreddit = stem.rsplit("_", 1)[0]
            else:
                subreddit = stem

            if subreddit not in result[file_type]:
                result[file_type][subreddit] = []
            result[file_type][subreddit].append(file_path)

    return result


def merge_files(
    input_files: List[Path],
    output_path: Path,
) -> int:
    """Merge multiple JSONL files into one."""
    total_lines = 0

    with output_path.open("w", encoding="utf-8") as out:
        for input_file in sorted(input_files):
            print(f"  Merging: {input_file.name}")
            with input_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        total_lines += 1

    return total_lines


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Merging subreddit files")
    print("=" * 60)

    split_files = find_split_files(input_dir)

    total_merged = 0
    merge_stats: Dict[str, Dict[str, int]] = {}

    for file_type, subreddit_files in split_files.items():
        print(f"\nProcessing {file_type}:")
        type_dir = output_dir / file_type
        type_dir.mkdir(parents=True, exist_ok=True)

        merge_stats[file_type] = {}

        for subreddit, files in sorted(subreddit_files.items()):
            if not files:
                continue

            output_path = type_dir / f"{subreddit}.jsonl"
            line_count = merge_files(files, output_path)
            merge_stats[file_type][subreddit] = line_count
            total_merged += line_count
            print(f"    → {output_path.name}: {line_count} lines")

    print("\n" + "=" * 60)
    print("Merge complete")
    print("=" * 60)
    print(f"Total lines merged: {total_merged}")
    print(f"Output directory: {output_dir}")

    print("\nMerge statistics:")
    for file_type, subreddit_counts in sorted(merge_stats.items()):
        print(f"  {file_type}:")
        for subreddit, count in sorted(subreddit_counts.items()):
            print(f"    {subreddit}: {count} lines")

    # Generate summary
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_lines_merged": total_merged,
        "merge_stats": merge_stats,
    }

    summary_path = output_dir / "merge_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
