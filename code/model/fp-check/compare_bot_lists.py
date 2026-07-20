#!/usr/bin/env python3
"""
compare_bot_lists.py
====================
Compare old and new bot author lists to identify eliminated and added authors.

This script is used after re-running extract_bots_from_zst.py with improved
bot detection rules. It identifies which authors were eliminated by the new
rules and which were added.

USAGE:
python compare_bot_lists.py \
    --old-list old_bot_authors.txt \
    --new-list new_bot_authors.txt \
    --output comparison_report.txt
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--old-list",
        required=True,
        help="Path to old bot author list (before rule improvements)"
    )
    parser.add_argument(
        "--new-list",
        required=True,
        help="Path to new bot author list (after rule improvements)"
    )
    parser.add_argument(
        "--output",
        default="comparison_report.txt",
        help="Output file for the comparison report"
    )
    parser.add_argument(
        "--eliminated-output",
        default="eliminated_authors.txt",
        help="Output file for list of eliminated authors (one per line)"
    )
    
    args = parser.parse_args()
    
    old_path = Path(args.old_list)
    new_path = Path(args.new_list)
    output_path = Path(args.output)
    eliminated_path = Path(args.eliminated_output)
    
    # Load old list
    print(f"Loading old bot list from {old_path}...")
    with open(old_path, 'r', encoding='utf-8') as f:
        old_authors = set(line.strip() for line in f if line.strip())
    print(f"  Loaded {len(old_authors)} authors")
    
    # Load new list
    print(f"Loading new bot list from {new_path}...")
    with open(new_path, 'r', encoding='utf-8') as f:
        new_authors = set(line.strip() for line in f if line.strip())
    print(f"  Loaded {len(new_authors)} authors")
    
    # Calculate differences
    eliminated = old_authors - new_authors  # In old but not in new
    added = new_authors - old_authors      # In new but not in old
    kept = old_authors & new_authors       # In both
    
    # Write comparison report
    print(f"\nWriting comparison report to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BOT AUTHOR LIST COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Old list: {args.old_list}\n")
        f.write(f"New list: {args.new_list}\n\n")
        
        f.write(f"Total authors in old list: {len(old_authors)}\n")
        f.write(f"Total authors in new list: {len(new_authors)}\n")
        f.write(f"Authors kept (in both): {len(kept)}\n")
        f.write(f"Authors eliminated (removed by new rules): {len(eliminated)}\n")
        f.write(f"Authors added (added by new rules): {len(added)}\n")
        f.write(f"Net change: {len(new_authors) - len(old_authors):+d}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("ELIMINATED AUTHORS (removed by improved rules)\n")
        f.write("=" * 80 + "\n")
        f.write("These authors were in the old bot list but are not in the new list.\n")
        f.write("They should be removed from the merged JSONL files.\n\n")
        
        for author in sorted(eliminated):
            f.write(f"{author}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("ADDED AUTHORS (added by improved rules)\n")
        f.write("=" * 80 + "\n")
        f.write("These authors were not in the old bot list but are in the new list.\n\n")
        
        for author in sorted(added):
            f.write(f"{author}\n")
    
    # Write eliminated authors list (for filtering script)
    print(f"Writing eliminated authors to {eliminated_path}...")
    with open(eliminated_path, 'w', encoding='utf-8') as f:
        for author in sorted(eliminated):
            f.write(author + '\n')
    
    print(f"\nDone!")
    print(f"\nSummary:")
    print(f"  Old list: {len(old_authors)} authors")
    print(f"  New list: {len(new_authors)} authors")
    print(f"  Eliminated: {len(eliminated)} authors")
    print(f"  Added: {len(added)} authors")
    print(f"  Kept: {len(kept)} authors")
    print(f"\nNext steps:")
    print(f"  1. Review the comparison report: {output_path}")
    print(f"  2. Use filter_jsonl_by_authors.py to remove eliminated authors from JSONL files")
    print(f"     using: --remove-list {eliminated_path}")


if __name__ == "__main__":
    main()
