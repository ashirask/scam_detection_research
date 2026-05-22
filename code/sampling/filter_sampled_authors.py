import argparse
import json
from pathlib import Path
from typing import Iterable, List

#
# filter_sampled_authors.py
#
# This script filters a Reddit JSONL sample, removing any records whose author username
# contains a given substring (default: 'bot').
#
# Example usage:
#   python filter_sampled_authors.py --input-file sample.jsonl --output-file filtered.jsonl
#   # Removes all records where author contains 'bot' (case-insensitive) or is [deleted]
#
# Input JSONL:
#   {"author": "user123", "body": "..."}
#   {"author": "scambot", "body": "..."}
#   {"author": "[deleted]", "body": "..."}
#
# Output JSONL (default):
#   {"author": "user123", "body": "..."}
#
# To keep [deleted] authors:
#   python filter_sampled_authors.py --input-file sample.jsonl --output-file filtered.jsonl --keep-deleted


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the filter script.

    Returns:
        argparse.Namespace with:
        - input_file: path to input JSONL
        - output_file: path to output JSONL
        - author_tokens: list of substrings to filter (default: ['bot'])
        - keep_deleted: if True, keep [deleted] authors
    """
    parser = argparse.ArgumentParser(
        description=(
            "Filter a sampled Reddit JSONL file by removing records whose author "
            "name contains one of the provided tokens (default: bot)."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--author-tokens",
        nargs="*",
        default=["bot"],
        help="Substrings to match in author username (case-insensitive). Default: bot",
    )
    parser.add_argument(
        "--keep-deleted",
        action="store_true",
        help="Keep [deleted] author rows. By default they are removed.",
    )
    return parser.parse_args()


def should_drop_author(author: str, tokens: Iterable[str], keep_deleted: bool) -> bool:
    """
    Decide whether to drop a record based on author username.

    Args:
        author: Username string from record (may be empty)
        tokens: Iterable of substrings (lowercase) to filter
        keep_deleted: If False, drop [deleted] authors

    Returns:
        True if record should be dropped, False if kept

    Example:
        should_drop_author('scambot', ['bot'], False) -> True
        should_drop_author('user123', ['bot'], False) -> False
        should_drop_author('[deleted]', ['bot'], False) -> True
        should_drop_author('[deleted]', ['bot'], True) -> False
    """
    username = (author or "").strip()
    if not username:
        return True

    if (not keep_deleted) and username == "[deleted]":
        return True

    username_lower = username.lower()
    return any(token in username_lower for token in tokens)


def main() -> None:
    """
    Main entry point: filter input JSONL and write output JSONL, printing stats.
    """
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokens: List[str] = [token.lower() for token in args.author_tokens if token.strip()]

    total = 0
    kept = 0
    dropped = 0
    bad_json = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            author = str(record.get("author", ""))
            if should_drop_author(author, tokens, args.keep_deleted):
                dropped += 1
                continue

            # Write the record to output if not dropped

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Author tokens used: {tokens}")
    print(f"Total rows seen: {total}")
    print(f"Rows kept: {kept}")
    print(f"Rows dropped by author filter: {dropped}")
    print(f"Malformed JSON rows skipped: {bad_json}")

# End of script


if __name__ == "__main__":
    main()
