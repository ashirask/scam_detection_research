#!/usr/bin/env python3
"""
embed_posts.py

Compute one frozen mpnet embedding per valid post (comment or submission),
for every author across all three splits, in a single pass. This is the one
genuinely GPU-bound, expensive step in the whole extension.

Output: One row per valid post with columns:
- author, post_id, post_type, created_utc, subreddit, embedding (768-dim), has_any_valid_post

Authors with zero valid posts get a marker row with has_any_valid_post=False
and zero-vector embedding to distinguish "processed, has nothing" from "never processed."

This script is HPC-cluster compatible:
- GPU-bound with configurable batch size
- Resumability via progress file (every 500 authors)
- Uses the same logging pattern as build_author_offset_index.py
- Includes comprehensive error handling and manifest generation

Usage:
  python embed_posts.py \
    --comments-bot path/to/comments_bot.jsonl \
    --submissions-bot path/to/submissions_bot.jsonl \
    --comments-human path/to/comments_human_1.jsonl path/to/comments_human_2.jsonl \
    --submissions-human path/to/submissions_human_1.jsonl path/to/submissions_human_2.jsonl \
    --combined-index-dir graph_pipeline/raw_index/ \
    --splits-dir splits/ \
    --output-dir graph_pipeline/embeddings/post_embeddings/ \
    --project-root . \
    --batch-size 64

HPC Cluster Example:
  sbatch embed_posts.sbatch  # Or your cluster's job submission system
"""

import os
import sys
import argparse
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time

import torch
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Logging setup for HPC cluster compatibility
# ---------------------------------------------------------------------------

def setup_logging():
    """
    Configure logging for HPC cluster output to .out and .err files.
    
    INFO and DEBUG messages go to stdout (.out file)
    WARNING and ERROR messages go to stderr (.err file)
    All messages include timestamps for debugging.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # stdout handler for INFO/DEBUG (goes to .out file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                          datefmt='%Y-%m-%d %H:%M:%S')
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)
    
    # stderr handler for WARNING/ERROR (goes to .err file)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                          datefmt='%Y-%m-%d %H:%M:%S')
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)
    
    return logger


# Initialize logging
logger = setup_logging()


# ---------------------------------------------------------------------------
# Utility classes
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy/pandas types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Key functions
# ---------------------------------------------------------------------------

def validate_submission_ids(jsonl_paths: list) -> dict:
    """
    Validate that submissions have a stable id field by sampling a few records.
    
    Args:
        jsonl_paths: List of submission JSONL file paths to check
    
    Returns:
        Dict with validation result and details
    """
    logger.info("Validating submission ID fields...")
    
    validation_result = {
        "checked": True,
        "has_id_field": True,
        "sample_size": 0,
        "missing_id_count": 0,
        "files_checked": []
    }
    
    for jsonl_path in jsonl_paths:
        if not os.path.exists(jsonl_path):
            logger.warning(f"Submission file not found: {jsonl_path}")
            continue
        
        logger.info(f"  Checking {jsonl_path}")
        sample_count = 0
        missing_id = 0
        
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Sample first 10 lines per file
                        break
                    
                    try:
                        record = json.loads(line)
                        submissions = record.get("submissions", [])
                        
                        for sub in submissions:
                            sample_count += 1
                            if "id" not in sub or not sub["id"]:
                                missing_id += 1
                    
                    except json.JSONDecodeError:
                        continue
            
            validation_result["sample_size"] += sample_count
            validation_result["missing_id_count"] += missing_id
            validation_result["files_checked"].append(jsonl_path)
            
            logger.info(f"    Sampled {sample_count} submissions, {missing_id} missing ID")
            
        except Exception as e:
            logger.warning(f"    Error reading {jsonl_path}: {e}")
    
    if validation_result["missing_id_count"] > 0:
        validation_result["has_id_field"] = False
        logger.warning(f"Found {validation_result['missing_id_count']} submissions missing ID field")
        logger.warning("Will use fallback ID derivation: f\"{author}_sub_{created_utc}_{i}\"")
    else:
        logger.info(f"All {validation_result['sample_size']} sampled submissions have ID field")
    
    return validation_result


def extract_valid_posts(comments: list, submissions: list, author: str) -> list:
    """
    Extract valid posts from comments and submissions using get_texts() logic,
    but keep post-level granularity.
    
    Args:
        comments: List of comment dictionaries
        submissions: List of submission dictionaries
        author: Author name for fallback ID derivation if needed
    
    Returns:
        List of post dicts with fields: post_id, post_type, created_utc, subreddit, text
    """
    posts = []
    
    # Extract valid comments
    for c in comments:
        body = c.get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            posts.append({
                "post_id": c.get("id"),  # Should always exist for comments
                "post_type": "comment",
                "created_utc": c.get("created_utc"),
                "subreddit": c.get("subreddit"),
                "text": body
            })
    
    # Extract valid submissions
    for i, s in enumerate(submissions):
        title = s.get("title", "")
        selftext = s.get("selftext", "")
        
        # Skip placeholder text
        if selftext in ("[deleted]", "[removed]", ""):
            selftext = ""
        
        combined = (title + " " + selftext).strip()
        if combined:
            # Use id field if present, otherwise fallback
            post_id = s.get("id")
            if not post_id:
                # Fallback derivation (should not happen based on validation)
                created_utc = s.get("created_utc", 0)
                post_id = f"{author}_sub_{created_utc}_{i}"
            
            posts.append({
                "post_id": post_id,
                "post_type": "submission",
                "created_utc": s.get("created_utc"),
                "subreddit": s.get("subreddit"),
                "text": combined
            })
    
    return posts


def lookup_author_in_index(author: str, index_df: pd.DataFrame, jsonl_path: str) -> dict:
    """
    Look up author in combined index and retrieve their record from JSONL.
    
    Args:
        author: Author name to look up
        index_df: Combined index DataFrame
        jsonl_path: Path to the raw JSONL file
    
    Returns:
        Dict with comments and submissions lists, or None if author not found
    """
    # Find author in index
    author_row = index_df[index_df["author"] == author]
    
    if author_row.empty:
        return None
    
    # Get offset and length
    row = author_row.iloc[0]
    byte_offset = row["byte_offset"]
    byte_length = row["byte_length"]
    
    # Seek and read from JSONL
    try:
        with open(jsonl_path, "rb") as f:
            f.seek(byte_offset)
            line = f.read(byte_length).decode("utf-8")
            record = json.loads(line)
            return record
    except Exception as e:
        logger.error(f"Error reading author {author} from {jsonl_path}: {e}")
        return None


def embed_posts_batch(texts: list[str], model: SentenceTransformer, batch_size: int) -> list[list[float]]:
    """
    Embed a batch of texts using mpnet model.
    
    Args:
        texts: List of text strings to embed
        model: SentenceTransformer model
        batch_size: Batch size for encoding
    
    Returns:
        List of embedding vectors (each as list of floats)
    """
    with torch.no_grad():
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                   convert_to_numpy=True)
    return embeddings.tolist()


def flush_split_outputs_to_disk(split_outputs: dict, output_dir: str, splits: list = None) -> None:
    """
    Flush in-memory split outputs to parquet files and clear memory.
    
    Args:
        split_outputs: Dictionary with split names as keys and lists of output rows as values
        output_dir: Base output directory
        splits: List of split names to flush (default: all keys in split_outputs)
    """
    if splits is None:
        splits = list(split_outputs.keys())
    
    for split in splits:
        if not split_outputs[split]:
            continue
        
        output_path = os.path.join(output_dir, split, "post_embeddings.parquet")
        new_df = pd.DataFrame(split_outputs[split])
        
        if os.path.exists(output_path):
            # Append to existing file with deduplication
            existing_df = pd.read_parquet(output_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates by (author, post_id), keeping the latest version
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["author", "post_id"], keep="last")
            after_dedup = len(combined_df)
            if before_dedup != after_dedup:
                logger.info(f"Removed {before_dedup - after_dedup} duplicate rows from {output_path}")
            combined_df.to_parquet(output_path, index=False)
            logger.info(f"Flushed {len(new_df)} rows to {output_path} (total: {len(combined_df)})")
        else:
            # Create new file
            new_df.to_parquet(output_path, index=False)
            logger.info(f"Flushed {len(new_df)} rows to {output_path}")
        
        # Clear memory
        split_outputs[split] = []


def get_final_output_counts(output_dir: str, splits: list) -> dict:
    """
    Get final row counts from output parquet files.
    
    Args:
        output_dir: Base output directory
        splits: List of split names to check
    
    Returns:
        Dictionary with split names as keys and row counts as values
    """
    counts = {}
    for split in splits:
        output_path = os.path.join(output_dir, split, "post_embeddings.parquet")
        try:
            if os.path.exists(output_path):
                df = pd.read_parquet(output_path)
                counts[split] = len(df)
            else:
                counts[split] = 0
        except Exception as e:
            logger.warning(f"Could not read output file for {split}: {e}")
            counts[split] = 0
    return counts


def main():
    """
    Main execution function for embed_posts.py.
    
    This function orchestrates the post embedding process:
    1. Parse command-line arguments
    2. Validate submission ID fields
    3. Load combined indexes and split files
    4. Load mpnet model
    5. Process each author with resumability support
    6. Generate and write manifest
    
    Returns:
        None (saves embeddings and manifest to disk)
    
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If data validation fails
    """
    parser = argparse.ArgumentParser(
        description="Compute mpnet embeddings for all valid posts across train/val/test splits"
    )
    parser.add_argument("--comments-bot", required=True,
                        help="Path to bot comments JSONL file")
    parser.add_argument("--submissions-bot", required=True,
                        help="Path to bot submissions JSONL file")
    parser.add_argument("--comments-human", required=True, nargs="+",
                        help="Path(s) to human comments JSONL file(s)")
    parser.add_argument("--submissions-human", required=True, nargs="+",
                        help="Path(s) to human submissions JSONL file(s)")
    parser.add_argument("--combined-index-dir",
                        help="Directory containing combined index files (default: <project-root>/graph_pipeline/raw_index/)")
    parser.add_argument("--splits-dir",
                        help="Directory containing split files (default: <project-root>/splits/)")
    parser.add_argument("--output-dir",
                        help="Output directory for embeddings (default: <project-root>/graph_pipeline/embeddings/post_embeddings/)")
    parser.add_argument("--project-root", default=".",
                        help="Project root directory (default: current directory)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for mpnet encoding (default: 64)")
    parser.add_argument("--model-name", default="all-mpnet-base-v2",
                        help="Sentence-transformers model name (default: all-mpnet-base-v2)")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU usage even if CUDA is available (useful for testing or when GPU has issues)")
    
    args = parser.parse_args()
    
    # Set default paths
    if args.combined_index_dir is None:
        args.combined_index_dir = os.path.join(args.project_root, "graph_pipeline", "raw_index")
    if args.splits_dir is None:
        args.splits_dir = os.path.join(args.project_root, "splits")
    if args.output_dir is None:
        args.output_dir = os.path.join(args.project_root, "graph_pipeline", "embeddings", "post_embeddings")
    
    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
    
    logger.info("="*60)
    logger.info("Embed Posts")
    logger.info("="*60)
    logger.info(f"Project root: {args.project_root}")
    logger.info(f"Combined index dir: {args.combined_index_dir}")
    logger.info(f"Splits dir: {args.splits_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Validate submission IDs
    logger.info("\n" + "="*60)
    logger.info("Validating Submission IDs")
    logger.info("="*60)
    submission_validation = validate_submission_ids(
        [args.submissions_bot] + args.submissions_human
    )
    
    # Load combined indexes
    logger.info("\n" + "="*60)
    logger.info("Loading Combined Indexes")
    logger.info("="*60)
    
    bot_comments_index = pd.read_parquet(
        os.path.join(args.combined_index_dir, "combined_comments_bot_index.parquet")
    )
    logger.info(f"Bot comments index: {len(bot_comments_index)} authors")
    
    bot_submissions_index = pd.read_parquet(
        os.path.join(args.combined_index_dir, "combined_submissions_bot_index.parquet")
    )
    logger.info(f"Bot submissions index: {len(bot_submissions_index)} authors")
    
    human_comments_index = pd.read_parquet(
        os.path.join(args.combined_index_dir, "combined_comments_human_index.parquet")
    )
    logger.info(f"Human comments index: {len(human_comments_index)} authors")
    
    human_submissions_index = pd.read_parquet(
        os.path.join(args.combined_index_dir, "combined_submissions_human_index.parquet")
    )
    logger.info(f"Human submissions index: {len(human_submissions_index)} authors")
    
    # Load split files
    logger.info("\n" + "="*60)
    logger.info("Loading Split Files")
    logger.info("="*60)
    
    train_df = pd.read_parquet(os.path.join(args.splits_dir, "train_raw.parquet"))
    val_df = pd.read_parquet(os.path.join(args.splits_dir, "val_raw.parquet"))
    test_df = pd.read_parquet(os.path.join(args.splits_dir, "test_raw.parquet"))
    
    logger.info(f"Train split: {len(train_df)} authors")
    logger.info(f"Val split: {len(val_df)} authors")
    logger.info(f"Test split: {len(test_df)} authors")
    
    # Get union of all authors
    all_authors = set(train_df["author"].tolist() + val_df["author"].tolist() + test_df["author"].tolist())
    logger.info(f"Total unique authors across splits: {len(all_authors)}")
    
    # Create author -> split mapping
    author_to_split = {}
    for author in train_df["author"]:
        author_to_split[author] = "train"
    for author in val_df["author"]:
        author_to_split[author] = "val"
    for author in test_df["author"]:
        author_to_split[author] = "test"
    
    # Create author -> population mapping (y=1 for bot, y=0 for human)
    author_to_population = {}
    for _, row in train_df.iterrows():
        author_to_population[row["author"]] = "bot" if row["y"] == 1 else "human"
    for _, row in val_df.iterrows():
        author_to_population[row["author"]] = "bot" if row["y"] == 1 else "human"
    for _, row in test_df.iterrows():
        author_to_population[row["author"]] = "bot" if row["y"] == 1 else "human"
    
    # Load mpnet model
    logger.info("\n" + "="*60)
    logger.info("Loading mpnet Model")
    logger.info("="*60)
    
    # Determine device
    if args.force_cpu:
        device = "cpu"
        logger.info("Force CPU mode requested via --force-cpu flag")
    else:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {e}")
            device = "cpu"
        
        if device == "cpu":
            logger.warning("CUDA is not available - falling back to CPU")
            logger.warning("This will be significantly slower than GPU processing")
            logger.warning("Consider checking your CUDA installation or GPU driver")
        else:
            logger.info("CUDA is available - using GPU acceleration")
    
    logger.info(f"Using device: {device}")
    
    if device == "cpu":
        logger.warning("CPU mode will be significantly slower than GPU")
        logger.warning("Estimated processing time: 10-50 posts/second (vs 100-500 on GPU)")
        logger.warning("For large datasets, consider resolving CUDA/GPU issues for faster processing")
    
    model = SentenceTransformer(args.model_name, device=device)
    try:
        embedding_dim = model.get_embedding_dimension()  # Fixed method name
    except AttributeError:
        # Fallback for older sentence-transformers versions
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.warning("Using deprecated get_sentence_embedding_dimension() method")
    
    logger.info(f"Model loaded: {args.model_name}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Setup resumability
    progress_file = os.path.join(args.output_dir, "embeddings_progress.txt")
    processed_authors = set()
    
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            processed_authors = set(line.strip() for line in f if line.strip())
        logger.info(f"Resuming from {len(processed_authors)} already-processed authors")
    
    # Initialize output storage per split
    split_outputs = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # Statistics
    stats = {
        "train": {"authors_processed": 0, "zero_post_authors": 0, "total_posts": 0},
        "val": {"authors_processed": 0, "zero_post_authors": 0, "total_posts": 0},
        "test": {"authors_processed": 0, "zero_post_authors": 0, "total_posts": 0}
    }
    
    start_time = time.time()
    
    # Process authors
    logger.info("\n" + "="*60)
    logger.info("Processing Authors")
    logger.info("="*60)
    
    for i, author in enumerate(all_authors):
        if author in processed_authors:
            continue
        
        # Determine split and population
        split = author_to_split.get(author)
        population = author_to_population.get(author)
        
        if not split or not population:
            logger.warning(f"Author {author} not found in split or population mapping, skipping")
            continue
        
        # Select appropriate indexes and JSONL paths
        if population == "bot":
            comments_index = bot_comments_index
            submissions_index = bot_submissions_index
            comments_jsonl = args.comments_bot
            submissions_jsonl = args.submissions_bot
        else:
            comments_index = human_comments_index
            submissions_index = human_submissions_index
            # For human, get source_file from the index row for this specific author
            author_comments_row = comments_index[comments_index["author"] == author]
            author_submissions_row = submissions_index[submissions_index["author"] == author]
            
            if not author_comments_row.empty:
                comments_jsonl = author_comments_row.iloc[0]["source_file"]
            else:
                comments_jsonl = args.comments_human[0]  # Fallback to first path
            
            if not author_submissions_row.empty:
                submissions_jsonl = author_submissions_row.iloc[0]["source_file"]
            else:
                submissions_jsonl = args.submissions_human[0]  # Fallback to first path
        
        # Look up author in indexes
        comments_record = lookup_author_in_index(author, comments_index, comments_jsonl)
        submissions_record = lookup_author_in_index(author, submissions_index, submissions_jsonl)
        
        comments = comments_record.get("comments", []) if comments_record else []
        submissions = submissions_record.get("submissions", []) if submissions_record else []
        
        # Extract valid posts
        valid_posts = extract_valid_posts(comments, submissions, author)
        
        # Process posts
        if not valid_posts:
            # Zero valid posts - create marker row
            stats[split]["zero_post_authors"] += 1
            marker_row = {
                "author": author,
                "post_id": f"{author}__no_valid_posts__",
                "post_type": "none",
                "created_utc": None,
                "subreddit": None,
                "embedding": [0.0] * embedding_dim,
                "has_any_valid_post": False
            }
            split_outputs[split].append(marker_row)
        else:
            # Valid posts exist - embed them
            stats[split]["total_posts"] += len(valid_posts)
            texts = [post["text"] for post in valid_posts]
            embeddings = embed_posts_batch(texts, model, args.batch_size)
            
            for post, embedding in zip(valid_posts, embeddings):
                row = {
                    "author": author,
                    "post_id": post["post_id"],
                    "post_type": post["post_type"],
                    "created_utc": post["created_utc"],
                    "subreddit": post["subreddit"],
                    "embedding": embedding,
                    "has_any_valid_post": True
                }
                split_outputs[split].append(row)
        
        stats[split]["authors_processed"] += 1
        processed_authors.add(author)
        
        # Progress logging and checkpointing
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(all_authors)} authors...")
        
        if len(processed_authors) % 500 == 0:
            with open(progress_file, "w") as f:
                for auth in processed_authors:
                    f.write(auth + "\n")
            logger.info(f"Checkpointed {len(processed_authors)} authors to progress file")
            
            # Flush split outputs to disk to prevent memory growth and data loss
            flush_split_outputs_to_disk(split_outputs, args.output_dir)
        
        # Log GPU memory if available
        if device == "cuda" and (i + 1) % 100 == 0:
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
            except Exception as e:
                logger.warning(f"Could not log GPU memory: {e}")
    
    # Final progress save
    with open(progress_file, "w") as f:
        for auth in processed_authors:
            f.write(auth + "\n")
    
    wall_time = time.time() - start_time
    
    # Final flush of any remaining data in memory
    logger.info("\n" + "="*60)
    logger.info("Final Flush to Disk")
    logger.info("="*60)
    
    # Check if there's any remaining data to flush
    remaining_data = sum(len(split_outputs[split]) for split in split_outputs)
    if remaining_data > 0:
        logger.info(f"Flushing {remaining_data} remaining rows to disk")
        flush_split_outputs_to_disk(split_outputs, args.output_dir)
    else:
        logger.info("No remaining data to flush (all data already checkpointed)")
    
    # Get final output counts from disk
    final_counts = get_final_output_counts(args.output_dir, ["train", "val", "test"])
    
    # Write manifest
    logger.info("\n" + "="*60)
    logger.info("Writing Manifest")
    logger.info("="*60)
    
    manifest = {
        "submission_id_validation": submission_validation,
        "model_name": args.model_name,
        "embedding_dimensionality": embedding_dim,
        "batch_size": args.batch_size,
        "wall_time_seconds": wall_time,
        "stats": stats,
        "final_output_counts": final_counts,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    manifest_path = os.path.join(args.output_dir, "embed_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Wrote manifest to {manifest_path}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Embed Posts Complete")
    logger.info("="*60)
    logger.info(f"Total authors processed: {len(processed_authors)}")
    logger.info(f"Wall time: {wall_time:.2f} seconds")
    for split in ["train", "val", "test"]:
        logger.info(f"{split}: {stats[split]['authors_processed']} authors, "
                   f"{stats[split]['zero_post_authors']} zero-post, "
                   f"{stats[split]['total_posts']} posts embedded, "
                   f"{final_counts[split]} total rows in output file")


if __name__ == "__main__":
    main()
