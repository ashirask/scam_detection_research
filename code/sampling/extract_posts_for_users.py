"""
Extract posts for a random sample of Reddit users.

Workflow:
1. Load one or more source JSONL/ZST files and collect post authors
2. Randomly sample N unique users from those authors
3. Stream one or more target JSONL/ZST files and keep all posts by those users
4. Write matching posts to JSONL and save a small run summary

Typical usage:
    python extract_posts_for_users.py \
        --source-jsonl sampled_data/sample_posts_2024.jsonl \
        --target-jsonl sampled_data/sample_posts_2022-2025.jsonl \
        --num-users 100 \
        --seed 42 \
        --output-dir sampled_data/user_posts

If you want to extract from the same file you sampled users from, omit --target-jsonl.
"""

import argparse
import glob
import io
import json
import os
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Set, Tuple

try:
    import zstandard as zstd  # type: ignore
except ImportError:  # pragma: no cover - dependency is available in the repo env
    zstd = None  # type: ignore[assignment]


SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample Reddit users from one JSONL/ZST source and extract all of their posts "
            "from one or more JSONL/ZST target files."
        )
    )
    parser.add_argument(
        "--source-jsonl",
        nargs="*",
        default=None,
        help="One or more JSONL or ZST files used to sample users.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Directory to discover source files from.",
    )
    parser.add_argument(
        "--source-glob",
        default="*",
        help="Glob pattern used with --source-dir (default: *).",
    )
    parser.add_argument(
        "--source-recursive",
        action="store_true",
        help="Search recursively under --source-dir.",
    )
    parser.add_argument(
        "--target-jsonl",
        nargs="+",
        default=None,
        help=(
            "One or more JSONL or ZST files to scan for matching posts. "
            "Defaults to --source-jsonl if omitted."
        ),
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Directory to discover target files from.",
    )
    parser.add_argument(
        "--target-glob",
        default="*",
        help="Glob pattern used with --target-dir (default: *).",
    )
    parser.add_argument(
        "--target-recursive",
        action="store_true",
        help="Search recursively under --target-dir.",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=100,
        help="Number of unique users to sample from the source files.",
    )
    parser.add_argument(
        "--min-source-posts",
        type=int,
        default=1,
        help="Only sample users who appear at least this many times in the source files.",
    )
    parser.add_argument(
        "--exclude-author-tokens",
        nargs="*",
        default=["bot"],
        help="Case-insensitive substrings that exclude usernames when sampling.",
    )
    parser.add_argument(
        "--keep-deleted",
        action="store_true",
        help="Keep [deleted] authors instead of excluding them by default.",
    )
    parser.add_argument(
        "--output-dir",
        default="sampled_data",
        help="Directory for output JSONL and summary files.",
    )
    parser.add_argument(
        "--output-file",
        default="extracted_posts_for_sample_users.jsonl",
        help="Output JSONL filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible user sampling.",
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
        help="Print progress every N seen target records.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate post IDs if they appear in multiple target files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra per-file diagnostics while loading and extracting.",
    )
    return parser.parse_args()


def discover_input_files(
    inputs: Sequence[str] | None,
    input_dir: str | None,
    input_glob: str,
    recursive: bool,
) -> List[Path]:
    files: List[str] = []

    if inputs:
        files.extend(inputs)

    if input_dir:
        pattern = os.path.join(input_dir, "**", input_glob) if recursive else os.path.join(input_dir, input_glob)
        files.extend(glob.glob(pattern, recursive=recursive))

    unique_files = sorted({os.path.abspath(path) for path in files if path.lower().endswith((".jsonl", ".zst"))})
    return [Path(path) for path in unique_files if Path(path).is_file()]


def normalize_author(author: object) -> str:
    return str(author or "").strip()


def should_skip_author(author: str, exclude_tokens: Sequence[str], keep_deleted: bool) -> bool:
    if not author:
        return True

    if not keep_deleted and author in SKIP_AUTHORS:
        return True

    author_lower = author.lower()
    return any(token.strip().lower() and token.strip().lower() in author_lower for token in exclude_tokens)


@contextmanager
def open_text_stream(path: Path, max_window_size: int) -> Iterator[io.TextIOBase]:
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


def load_source_authors(
    source_paths: Sequence[Path],
    min_source_posts: int,
    exclude_tokens: Sequence[str],
    keep_deleted: bool,
    max_window_size: int,
    verbose: bool,
) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, int]]:
    author_counts: Counter[str] = Counter()
    file_stats: Dict[str, int] = {}
    total_seen = 0
    bad_json = 0
    source_bad_by_file: Dict[str, int] = {}

    for source_path in source_paths:
        file_seen = 0
        file_bad = 0

        if verbose:
            print(f"[source] reading {source_path}")

        with open_text_stream(source_path, max_window_size=max_window_size) as handle:
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

                author = normalize_author(record.get("author"))
                if should_skip_author(author, exclude_tokens, keep_deleted):
                    continue

                author_counts[author] += 1

        file_stats[source_path.name] = file_seen
        source_bad_by_file[source_path.name] = file_bad

        if verbose:
            print(f"[source] {source_path.name}: seen={file_seen}, bad_json={file_bad}")

    candidates = [author for author, count in author_counts.items() if count >= min_source_posts]
    candidates.sort()

    if not candidates:
        raise SystemExit(
            "No eligible users found in the source files after filtering; "
            "try lowering --min-source-posts or adjusting the author filters."
        )

    source_stats = {"total_seen": total_seen, "bad_json": bad_json}
    return candidates, dict(author_counts), source_stats, {"bad_json_by_file": source_bad_by_file, "seen_by_file": file_stats}


def sample_users(
    candidates: Sequence[str],
    author_counts: Dict[str, int],
    num_users: int,
    seed: int,
) -> List[str]:
    if num_users <= 0:
        raise SystemExit("--num-users must be greater than zero")

    eligible = list(candidates)
    random.seed(seed)
    if len(eligible) <= num_users:
        sampled = eligible
    else:
        sampled = random.sample(eligible, k=num_users)

    sampled.sort(key=lambda author: (-author_counts.get(author, 0), author))
    return sampled


def extract_posts_for_users(
    target_paths: Sequence[Path],
    selected_users: Set[str],
    output_path: Path,
    max_window_size: int,
    progress_every: int,
    allow_duplicates: bool,
    verbose: bool,
) -> Tuple[int, int, int, Dict[str, int]]:
    total_seen = 0
    total_matched = 0
    bad_json = 0
    duplicate_ids = 0
    matched_by_author: Counter[str] = Counter()
    seen_ids: Set[str] = set()

    with output_path.open("w", encoding="utf-8") as out:
        for target_path in target_paths:
            file_seen = 0
            file_matched = 0
            file_bad = 0

            if verbose:
                print(f"[target] reading {target_path}")

            with open_text_stream(target_path, max_window_size=max_window_size) as handle:
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

                    author = normalize_author(record.get("author"))
                    if author not in selected_users:
                        continue

                    post_id = normalize_author(record.get("id"))
                    if not allow_duplicates and post_id:
                        if post_id in seen_ids:
                            duplicate_ids += 1
                            continue
                        seen_ids.add(post_id)

                    output_record = dict(record)
                    output_record["source_file"] = target_path.name
                    out.write(json.dumps(output_record, ensure_ascii=False) + "\n")

                    total_matched += 1
                    file_matched += 1
                    matched_by_author[author] += 1

                    if progress_every > 0 and file_seen % progress_every == 0:
                        print(
                            f"  progress: seen={file_seen}, matched={file_matched}, "
                            f"bad_json={file_bad}, duplicates_skipped={duplicate_ids}"
                        )

            print(
                f"Completed {target_path.name}: seen={file_seen}, matched={file_matched}, "
                f"bad_json={file_bad}"
            )

    return total_seen, total_matched, bad_json, dict(matched_by_author)


def write_sampled_users(output_dir: Path, sampled_users: Sequence[str]) -> Path:
    sampled_users_path = output_dir / "sampled_users.txt"
    with sampled_users_path.open("w", encoding="utf-8") as handle:
        for user in sampled_users:
            handle.write(f"{user}\n")
    return sampled_users_path


def main() -> None:
    args = parse_args()

    source_paths = discover_input_files(
        inputs=args.source_jsonl,
        input_dir=args.source_dir,
        input_glob=args.source_glob,
        recursive=args.source_recursive,
    )
    if not source_paths:
        raise SystemExit("Provide at least one source file via --source-jsonl or --source-dir")

    if args.target_jsonl or args.target_dir:
        target_paths = discover_input_files(
            inputs=args.target_jsonl,
            input_dir=args.target_dir,
            input_glob=args.target_glob,
            recursive=args.target_recursive,
        )
        if not target_paths:
            raise SystemExit("No target files found via --target-jsonl or --target-dir")
    else:
        target_paths = list(source_paths)

    for path in source_paths + target_paths:
        if not path.is_file():
            raise SystemExit(f"Input file not found: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    print("=" * 60)
    print("STEP 1: Load source users")
    print("=" * 60)
    candidates, author_counts, source_stats, source_file_stats = load_source_authors(
        source_paths=source_paths,
        min_source_posts=args.min_source_posts,
        exclude_tokens=args.exclude_author_tokens,
        keep_deleted=args.keep_deleted,
        max_window_size=args.max_window_size,
        verbose=args.verbose,
    )
    sampled_users = sample_users(candidates, author_counts, args.num_users, args.seed)
    sampled_users_path = write_sampled_users(output_dir, sampled_users)

    print(f"Source authors considered: {len(candidates)}")
    print(f"Sampled users: {len(sampled_users)}")
    print(f"Sampled users file: {sampled_users_path}")

    print("\n" + "=" * 60)
    print("STEP 2: Extract posts for sampled users")
    print("=" * 60)
    total_seen, total_matched, bad_json, matched_by_author = extract_posts_for_users(
        target_paths=target_paths,
        selected_users=set(sampled_users),
        output_path=output_path,
        max_window_size=args.max_window_size,
        progress_every=args.progress_every,
        allow_duplicates=args.allow_duplicates,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("Extraction complete")
    print("=" * 60)
    print(f"Target records seen: {total_seen}")
    print(f"Posts extracted: {total_matched}")
    print(f"Malformed JSON lines skipped in target files: {bad_json}")
    print(f"Output saved to: {output_path}")

    summary = {
        "source_files": [str(path) for path in source_paths],
        "target_files": [str(path) for path in target_paths],
        "sampled_users_count": len(sampled_users),
        "sampled_users_file": str(sampled_users_path),
        "sampled_users": sampled_users,
        "author_counts": author_counts,
        "source_stats": source_stats,
        "source_file_stats": source_file_stats,
        "target_records_seen": total_seen,
        "posts_extracted": total_matched,
        "target_bad_json": bad_json,
        "posts_extracted_by_author": matched_by_author,
        "allow_duplicates": args.allow_duplicates,
        "min_source_posts": args.min_source_posts,
        "exclude_author_tokens": list(args.exclude_author_tokens),
        "keep_deleted": args.keep_deleted,
    }

    summary_path = output_dir / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()