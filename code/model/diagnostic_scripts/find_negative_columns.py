import pandas as pd
import sys

def find_negative_columns(parquet_path):
    """
    Read a parquet file and return a list of columns that contain negative values.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        List of column names that have negative values
    """
    # Read the parquet file
    print(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    # Find columns with negative values
    negative_columns = []
    
    for col in df.columns:
        # Only check numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if there are any negative values
            if (df[col] < 0).any():
                negative_count = (df[col] < 0).sum()
                negative_columns.append(col)
                print(f"  - {col}: {negative_count} negative values")
    
    return negative_columns

if __name__ == "__main__":
    # Default path to the parquet file
    default_path = r"C:\Users\Arohi Shiraskar\Documents\uncc\Projects\Scam detection\code\model\output\2025\combine-cleaned-data\combined_dataset_1to10.parquet"
    
    # Use command line argument if provided, otherwise use default
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    try:
        negative_cols = find_negative_columns(parquet_path)
        
        print("\n" + "="*50)
        print(f"Columns with negative values: {len(negative_cols)}")
        print("="*50)
        
        if negative_cols:
            print("\nColumn names:")
            for col in negative_cols:
                print(f"  - {col}")
        else:
            print("\nNo columns with negative values found.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
