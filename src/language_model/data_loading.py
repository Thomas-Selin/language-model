"""Data loading utilities for parquet files.

This module handles loading text data from parquet files, including
file monitoring, stability checks, and batch processing.
"""

import os
import time
import logging
import pandas as pd
from typing import List, Optional
from language_model.constants import STABLE_CHECK_INTERVAL, STABLE_CHECK_COUNT


def is_file_fully_uploaded(
    file_path: str,
    check_interval: int = STABLE_CHECK_INTERVAL,
    checks: int = STABLE_CHECK_COUNT
) -> bool:
    """Check if a file has finished uploading by verifying size stability.
    
    Monitors file size over multiple checks to ensure the file is not
    still being written to. Useful for detecting when uploads are complete.
    
    Args:
        file_path: Path to the file to check
        check_interval: Seconds between size checks (default from constants)
        checks: Number of consecutive stable checks required (default from constants)
        
    Returns:
        True if file size is stable across all checks, False otherwise
    """
    prev_size = -1
    for _ in range(checks):
        size = os.path.getsize(file_path)
        if size == prev_size:
            return True
        prev_size = size
        time.sleep(check_interval)
    return False


def poll_for_new_parquet_file(
    parquet_dir: str,
    poll_interval: int = 5
) -> Optional[str]:
    """Poll directory for new, fully uploaded parquet files.
    
    Args:
        parquet_dir: Directory to monitor
        poll_interval: Seconds between polling attempts
        
    Returns:
        Filename of new parquet file, or None
    """
    seen_files = set()
    
    while True:
        current_files = set(
            f for f in os.listdir(parquet_dir) if f.endswith('.parquet')
        )
        new_files = current_files - seen_files
        
        for file in new_files:
            file_path = os.path.join(parquet_dir, file)
            if is_file_fully_uploaded(file_path):
                logging.info(f"Detected new, fully uploaded file: {file}")
                seen_files.add(file)
                return file
            else:
                logging.debug(f"File {file} is still being uploaded, waiting...")
        
        time.sleep(poll_interval)


def load_text_from_parquet(
    parquet_file: str,
    text_column: str = 'text'
) -> str:
    """Load text data from a parquet file.
    
    Args:
        parquet_file: Path to the parquet file
        text_column: Name of the column containing text data
        
    Returns:
        Concatenated text from all rows
        
    Raises:
        InvalidDataFormatError: If text column is not found
    """
    from language_model.exceptions import InvalidDataFormatError
    
    logging.info(f"Loading parquet dataset from {parquet_file}...")
    
    try:
        df = pd.read_parquet(parquet_file)
        
        # Check if the text column exists
        if text_column not in df.columns:
            available_columns = ', '.join(df.columns)
            raise InvalidDataFormatError(
                f"Column '{text_column}' not found in parquet file. "
                f"Available columns: {available_columns}"
            )
        
        # Extract text from the specified column
        text_data = ' '.join(df[text_column].fillna('').astype(str).tolist())
        logging.info(f"Parquet dataset loaded successfully. {len(df)} rows processed.")
        return text_data
        
    except Exception as e:
        if isinstance(e, InvalidDataFormatError):
            raise
        logging.error(f"Error loading parquet file: {e}")
        return ""


def get_parquet_files(directory: str) -> List[str]:
    """Get all parquet files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of full paths to parquet files
    """
    if not os.path.isdir(directory):
        return []
    
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.parquet')
    ]
    
    return sorted(files, key=os.path.getctime)


def validate_parquet_file(
    file_path: str,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> bool:
    """Validate that a parquet file meets requirements.
    
    Args:
        file_path: Path to parquet file
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_parquet(file_path)
        
        # Check minimum rows
        if len(df) < min_rows:
            logging.warning(
                f"File {file_path} has only {len(df)} rows, "
                f"minimum {min_rows} required"
            )
            return False
        
        # Check required columns
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                logging.warning(
                    f"File {file_path} is missing required columns: "
                    f"{missing_columns}"
                )
                return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating parquet file {file_path}: {e}")
        return False
