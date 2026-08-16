#!/usr/bin/env python3
"""
analyze_entropy_ratio_columns.py

Analyzes columns with 'entropy' or 'ratio' in their names from raw and transformed datasets.
Shows distribution plots (raw vs transformed, split by label) and statistical summaries.

Usage:
  python analyze_entropy_ratio_columns.py \
    --raw-dataset path/to/raw_train.parquet \
    --transformed-dataset path/to/transformed_train.parquet \
    --output-dir output_directory
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(raw_path, transformed_path):
    """Load raw and transformed datasets."""
    print(f"Loading raw dataset: {raw_path}")
    df_raw = pd.read_parquet(raw_path)
    print(f"  Shape: {df_raw.shape}")
    
    print(f"Loading transformed dataset: {transformed_path}")
    df_transformed = pd.read_parquet(transformed_path)
    print(f"  Shape: {df_transformed.shape}")
    
    return df_raw, df_transformed


def identify_target_columns(df):
    """Identify columns with 'entropy' or 'ratio' in their names."""
    target_columns = [col for col in df.columns if 'entropy' in col.lower() or 'ratio' in col.lower()]
    print(f"\nFound {len(target_columns)} columns with 'entropy' or 'ratio':")
    for col in target_columns:
        print(f"  - {col}")
    return target_columns


def calculate_statistics(series, name):
    """Calculate and return statistics for a series."""
    stats = {
        'name': name,
        'count': len(series),
        'non_null': series.notna().sum(),
        'null_count': series.isna().sum(),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'median': series.median(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
    }
    return stats


def print_statistics_comparison(raw_stats, transformed_stats):
    """Print comparison of statistics between raw and transformed data."""
    print(f"\n{'='*60}")
    print(f"Statistics: {raw_stats['name']}")
    print(f"{'='*60}")
    
    print(f"{'Metric':<15} {'Raw':<15} {'Transformed':<15}")
    print("-" * 45)
    
    for metric in ['count', 'non_null', 'null_count', 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75']:
        raw_val = raw_stats[metric]
        trans_val = transformed_stats[metric]
        
        # Format values appropriately
        if metric in ['count', 'non_null', 'null_count']:
            raw_str = f"{raw_val:.0f}"
            trans_str = f"{trans_val:.0f}"
        else:
            raw_str = f"{raw_val:.4f}" if pd.notna(raw_val) else "NaN"
            trans_str = f"{trans_val:.4f}" if pd.notna(trans_val) else "NaN"
        
        print(f"{metric:<15} {raw_str:<15} {trans_str:<15}")


def plot_distribution_comparison(df_raw, df_transformed, column, output_dir):
    """Create side-by-side distribution plots comparing raw vs transformed data."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data and labels
    if 'y' in df_raw.columns:
        y_raw = df_raw['y']
        y_transformed = df_transformed['y']
        bots_raw = y_raw == 1
        humans_raw = y_raw == 0
        bots_transformed = y_transformed == 1
        humans_transformed = y_transformed == 0
    else:
        print(f"Warning: 'y' column not found, plotting without label separation")
        bots_raw = humans_raw = pd.Series([True] * len(df_raw), index=df_raw.index)
        bots_transformed = humans_transformed = pd.Series([True] * len(df_transformed), index=df_transformed.index)
    
    raw_data = df_raw[column]
    transformed_data = df_transformed[column]
    
    # Raw data histogram (split by label)
    ax1 = axes[0, 0]
    ax1.hist(raw_data[humans_raw].dropna(), bins=40, alpha=0.5, label="human", density=True, color='blue')
    ax1.hist(raw_data[bots_raw].dropna(), bins=40, alpha=0.5, label="bot", density=True, color='red')
    ax1.set_title(f"{column} (Raw)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)
    
    # Transformed data histogram (split by label)
    ax2 = axes[0, 1]
    ax2.hist(transformed_data[humans_transformed].dropna(), bins=40, alpha=0.5, label="human", density=True, color='blue')
    ax2.hist(transformed_data[bots_transformed].dropna(), bins=40, alpha=0.5, label="bot", density=True, color='red')
    ax2.set_title(f"{column} (Transformed)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    
    # Raw data box plot
    ax3 = axes[1, 0]
    data_to_plot = [raw_data[humans_raw].dropna(), raw_data[bots_raw].dropna()]
    bp = ax3.boxplot(data_to_plot, labels=['human', 'bot'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.5)
    ax3.set_title(f"{column} (Raw) - Box Plot", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Value")
    
    # Transformed data box plot
    ax4 = axes[1, 1]
    data_to_plot = [transformed_data[humans_transformed].dropna(), transformed_data[bots_transformed].dropna()]
    bp = ax4.boxplot(data_to_plot, labels=['human', 'bot'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.5)
    ax4.set_title(f"{column} (Transformed) - Box Plot", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Value")
    
    plt.tight_layout()
    
    # Save the plot
    safe_column_name = column.replace('/', '_').replace('\\', '_')
    plot_path = os.path.join(output_dir, f"distribution_{safe_column_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {plot_path}")
    
    return plot_path


def save_statistics_summary(all_stats, output_dir):
    """Save all statistics to a text file."""
    summary_path = os.path.join(output_dir, "statistics_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("STATISTICS SUMMARY FOR ENTROPY AND RATIO COLUMNS\n")
        f.write("="*60 + "\n\n")
        
        for col_name, stats in all_stats.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Column: {col_name}\n")
            f.write(f"{'='*60}\n")
            
            f.write(f"{'Metric':<15} {'Raw':<15} {'Transformed':<15}\n")
            f.write("-" * 45 + "\n")
            
            raw_stats = stats['raw']
            trans_stats = stats['transformed']
            
            for metric in ['count', 'non_null', 'null_count', 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75']:
                raw_val = raw_stats[metric]
                trans_val = trans_stats[metric]
                
                if metric in ['count', 'non_null', 'null_count']:
                    raw_str = f"{raw_val:.0f}"
                    trans_str = f"{trans_val:.0f}"
                else:
                    raw_str = f"{raw_val:.4f}" if pd.notna(raw_val) else "NaN"
                    trans_str = f"{trans_val:.4f}" if pd.notna(trans_val) else "NaN"
                
                f.write(f"{metric:<15} {raw_str:<15} {trans_str:<15}\n")
            
            f.write("\n")
    
    print(f"\nSaved statistics summary to: {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze entropy and ratio columns from raw and transformed datasets"
    )
    parser.add_argument("--raw-dataset", required=True, help="Path to raw train dataset parquet file")
    parser.add_argument("--transformed-dataset", required=True, help="Path to transformed train dataset parquet file")
    parser.add_argument("--output-dir", default="./entropy_ratio_analysis", help="Output directory for plots and summaries")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    df_raw, df_transformed = load_data(args.raw_dataset, args.transformed_dataset)
    
    # Identify target columns
    target_columns = identify_target_columns(df_raw)
    
    if not target_columns:
        print("No columns with 'entropy' or 'ratio' found. Exiting.")
        return
    
    # Check if transformed dataset has the same columns
    missing_cols = set(target_columns) - set(df_transformed.columns)
    if missing_cols:
        print(f"Warning: The following columns are missing from transformed dataset: {missing_cols}")
        target_columns = [col for col in target_columns if col in df_transformed.columns]
    
    print(f"\nAnalyzing {len(target_columns)} columns...")
    
    # Analyze each column
    all_stats = {}
    
    for col in target_columns:
        print(f"\n{'='*60}")
        print(f"Analyzing: {col}")
        print(f"{'='*60}")
        
        # Calculate statistics
        raw_stats = calculate_statistics(df_raw[col], col)
        transformed_stats = calculate_statistics(df_transformed[col], col)
        
        # Print statistics comparison
        print_statistics_comparison(raw_stats, transformed_stats)
        
        # Store statistics
        all_stats[col] = {
            'raw': raw_stats,
            'transformed': transformed_stats
        }
        
        # Create distribution plots
        try:
            plot_distribution_comparison(df_raw, df_transformed, col, args.output_dir)
        except Exception as e:
            print(f"  Error creating plot for {col}: {e}")
    
    # Save statistics summary
    save_statistics_summary(all_stats, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
