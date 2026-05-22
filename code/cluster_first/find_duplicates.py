import argparse
import json
from collections import defaultdict
from pathlib import Path
import os


# Default to the small sample so the script is fast to test and easy to override.
DEFAULT_INPUT_FILE = str(Path(__file__).resolve().parent.parent / "sampled_data" / "sample.jsonl")
# Skip placeholder bodies that do not represent real authored content.
SKIP_BODIES = {"[removed]", "[deleted]"}


def normalize_text(text):
    """Collapse repeated whitespace so exact duplicate matching is more stable."""
    return " ".join(text.split())


def load_records(input_file):
    """Yield cleaned records that have usable text and a real author."""
    with open(input_file, encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)

            # Normalize the comment text before using it as a duplicate key.
            text = normalize_text(data.get("body", "").strip())
            author = data.get("author")

            # Drop empty comments and removed placeholders.
            if not text or text in SKIP_BODIES:
                continue

            # Drop deleted authors so we only compare real accounts.
            if not author or author == "[deleted]":
                continue

            yield data, text, author


def main():
    """Find repeated comment text posted by different authors."""
    parser = argparse.ArgumentParser(description="Find exact duplicate Reddit comment text across different authors.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="Path to a JSONL file of Reddit comments.")
    parser.add_argument("--min-authors", type=int, default=2, help="Minimum distinct authors required to print a text.")
    parser.add_argument("--max-preview-length", type=int, default=300, help="Maximum characters to print for each duplicate text.")
    args = parser.parse_args()

    # Map each normalized comment to the set of authors who used it.
    text_authors = defaultdict(set)
    # Keep one example row per text so we can print useful metadata later.
    text_examples = {}

    for data, text, author in load_records(args.input_file):
        text_authors[text].add(author)
        text_examples.setdefault(text, data)

    # Print the strongest duplicate candidates first.
    for text, authors in sorted(text_authors.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(authors) < args.min_authors:
            continue

        example = text_examples[text]
        # Emit metadata that helps judge whether this looks like coordinated spam.
        print("----")
        print(f"Authors: {len(authors)}")
        print(f"Subreddit: {example.get('subreddit')}")
        print(f"Created UTC: {example.get('created_utc')}")
        print(f"Author created UTC: {example.get('author_created_utc')}")
        print(f"Link ID: {example.get('link_id')}")
        print(f"Permalink: {example.get('permalink')}")
        print(text[: args.max_preview_length])

    # Save duplicate detection results to a file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    duplicates_output_file = os.path.join(output_dir, "duplicates.txt")
    save_duplicates_to_file(text_authors, text_examples, duplicates_output_file)
    print(f"Duplicate detection results saved to {duplicates_output_file}")


def save_duplicates_to_file(text_authors, text_examples, output_file):
    """Save duplicate detection results to a text file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for text, authors in sorted(text_authors.items(), key=lambda item: (-len(item[1]), item[0])):
            if len(authors) < 2:  # Skip texts with fewer than 2 authors
                continue

            example = text_examples[text]
            f.write("----\n")
            f.write(f"Authors: {len(authors)}\n")
            f.write(f"Subreddit: {example.get('subreddit')}\n")
            f.write(f"Created UTC: {example.get('created_utc')}\n")
            f.write(f"Author created UTC: {example.get('author_created_utc')}\n")
            f.write(f"Link ID: {example.get('link_id')}\n")
            f.write(f"Permalink: {example.get('permalink')}\n")
            f.write(f"Text: {text}\n\n")


if __name__ == "__main__":
    main()