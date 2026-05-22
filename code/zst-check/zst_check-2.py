#!/usr/bin/env python3
"""Quick integrity screening for .zst files.

This script follows a lightweight strategy:
1. Confirm the file has the expected Zstandard magic bytes.
2. Decompress only enough to read the first two lines.

It is intentionally faster than full-file validation and avoids loading large
decompressed content into memory.
"""

import argparse
import os
import subprocess
from typing import List, Tuple

# Standard 4-byte magic sequence for Zstandard frames.
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def find_zst_files(directory: str) -> List[str]:
    """Recursively collect all files ending in .zst under a directory."""
    out = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith(".zst"):
                out.append(os.path.join(root, name))
    return out


def has_zstd_magic(path: str) -> bool:
    """Check whether file begins with the expected Zstandard magic bytes.

    This quickly filters out files that are clearly not valid .zst streams.
    """
    try:
        with open(path, "rb") as f:
            return f.read(4) == ZSTD_MAGIC
    except OSError:
        # If the file cannot be opened/read, treat as invalid for screening.
        return False


def check_first_two_lines(path: str, long_window: int) -> Tuple[str, str]:
    """
    Returns:
    status in {"VALID", "SHORT_OK", "CORRUPTED"}
      error message (if any)

        Behavior:
        - Spawns `zstd -dc --long=<n>` and streams output.
        - Reads only the first two decompressed lines.
        - Stops early once two lines are successfully read.

        Why this helps:
        - It avoids full decompression and large memory usage.
        - It still catches obvious early corruption and decode failures.
    """
        # `-d` = decompress, `-c` = write to stdout.
        # `--long` allows decoding frames created with large window sizes.
    cmd = ["zstd", "-dc", f"--long={long_window}", path]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        bufsize=1,
    )

    lines_read = 0
    try:
        # Read exactly up to 2 lines from the decompressed stream.
        for _ in range(2):
            line = proc.stdout.readline()
            if line == "":
                # Empty read means EOF or process ended before producing more.
                break
            lines_read += 1

        if lines_read >= 2:
            # Reaching here means we successfully decoded enough content to
            # get two lines. For this quick-screen mode, that is considered
            # a likely healthy file.
            proc.terminate()
            try:
                proc.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                # Ensure subprocess is cleaned up if terminate is not enough.
                proc.kill()
                proc.communicate()
            return "VALID", ""

        # Did not get 2 lines. Wait for the process to finish so we can
        # distinguish a short-valid file from an actual decode failure.
        _, err = proc.communicate(timeout=15)
        if proc.returncode == 0:
            # Decode succeeded, but decompressed content had fewer than 2 lines.
            return "SHORT_OK", ""
        return "CORRUPTED", (err or "").strip()

    except Exception as e:
        # Any unexpected runtime issue in this probe is recorded as decode error.
        try:
            proc.kill()
            proc.communicate()
        except Exception:
            pass
        return "CORRUPTED", str(e)


def main():
    # Command-line options for directory scope and output report location.
    parser = argparse.ArgumentParser(
        description="Quick .zst screening: verify header and read first 2 decompressed lines."
    )
    parser.add_argument("--directory", required=True, help="Directory to scan recursively")
    parser.add_argument("--output", default="zst_quick_report.txt", help="Report path")
    parser.add_argument("--long-window", type=int, default=31, help="zstd --long value (default: 31)")
    args = parser.parse_args()

    # Discover candidate files once, then classify each into report buckets.
    files = find_zst_files(args.directory)

    valid_files = []
    short_ok_files = []
    bad_header = []
    corrupted_files = []

    total = len(files)
    print(f"Found {total} .zst files")

    for i, path in enumerate(files, 1):
        print(f"[{i}/{total}] {path}")

        # Header check is a fast guard before invoking zstd.
        if not has_zstd_magic(path):
            bad_header.append(path)
            continue

        # Probe decompression just enough to validate early readability.
        status, err = check_first_two_lines(path, args.long_window)
        if status == "VALID":
            valid_files.append(path)
        elif status == "SHORT_OK":
            short_ok_files.append(path)
        else:
            corrupted_files.append((path, err))

    # Persist a human-readable report with grouped outcomes.
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("=== VALID (header + first 2 lines readable) ===\n")
        for p in valid_files:
            f.write(p + "\n")

        f.write("\n=== SHORT_OK (valid decode, <2 lines) ===\n")
        for p in short_ok_files:
            f.write(p + "\n")

        f.write("\n=== BAD_HEADER (not zstd magic 28b52ffd) ===\n")
        for p in bad_header:
            f.write(p + "\n")

        f.write("\n=== CORRUPTED ===\n")
        for p, err in corrupted_files:
            f.write(p + "\n")
            if err:
                f.write("  " + err + "\n")

    # Console summary for quick visibility in logs.
    print(f"Done. Report saved to {args.output}")
    print(
        "Note: This is a quick screen, not full integrity verification. "
        "For strict validation, run full zstd -t separately."
    )


if __name__ == "__main__":
    main()