## Usage: python strip_parquet_column.py path/to/input.parquet path/to/output_folder text (where text is the name of the column to keep)

import os
import sys
import pandas as pd

def strip_parquet_column(input_path, output_folder, keep_column):
    # Load only the desired column
    df = pd.read_parquet(input_path, columns=[keep_column])
    # Prepare output path
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(input_path))
    # Save to new parquet file
    df.to_parquet(output_path, index=False)
    print(f"Saved stripped parquet to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python strip_parquet_column.py <input_parquet> <output_folder> <column_to_keep>")
        sys.exit(1)
    input_parquet = sys.argv[1]
    output_folder = sys.argv[2]
    column_to_keep = sys.argv[3]
    strip_parquet_column(input_parquet, output_folder, column_to_keep)