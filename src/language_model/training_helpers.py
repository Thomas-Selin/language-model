"""
Training helper functions for file monitoring, preloading, and training utilities.
"""

import os
import time
import logging
import glob
import threading
from typing import Tuple, Optional, Set
from data_handler import load_and_process_data
import config


def is_file_ready_for_training(file_path: str, min_size_bytes: int, stable_count_threshold: int,
                              size_state: dict) -> bool:
    """
    Check if a file is ready for training (stable size and minimum size).
    
    Args:
        file_path: Path to the file to check
        min_size_bytes: Minimum file size in bytes
        stable_count_threshold: Number of consecutive stable checks required
        size_state: Dictionary tracking file sizes and stability
        
    Returns:
        True if file is ready for training
    """
    if not os.path.exists(file_path):
        return False
        
    file = os.path.basename(file_path)
    curr_size = os.path.getsize(file_path)
    prev_size, stable_count = size_state.get(file, (None, 0))
    
    logging.debug(f"Checking file '{file}': size={curr_size/1024/1024:.2f} MB, "
                 f"previous size={prev_size}, stable_count={stable_count}")

    if curr_size >= min_size_bytes and prev_size is not None and curr_size == prev_size:
        stable_count += 1
        logging.debug(f"File '{file}' size is stable and >={min_size_bytes/1024/1024:.1f} MB. "
                     f"Stable count: {stable_count}/{stable_count_threshold}")
    else:
        if curr_size < min_size_bytes:
            logging.debug(f"File '{file}' is too small ({curr_size/1024/1024:.2f} MB). "
                         "Waiting for upload to finish.")
        elif prev_size is not None and curr_size != prev_size:
            logging.debug(f"File '{file}' size changed from {prev_size} to {curr_size}. "
                         "Waiting for upload to finish.")
        else:
            logging.debug(f"First observation for file '{file}'. Waiting for size to stabilize.")
        stable_count = 0

    size_state[file] = (curr_size, stable_count)
    return stable_count >= stable_count_threshold


def wait_for_new_files_or_stop(parquet_dir_path: str, trained_files: Set[str], 
                              stop_file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Wait for new parquet files or stop signal.
    
    Args:
        parquet_dir_path: Directory to monitor for parquet files
        trained_files: Set of files already used for training
        stop_file_path: Path to stop signal file
        
    Returns:
        Tuple of (user_interrupted, ready_file_name)
    """
    # Remove any existing stop file at the start
    if os.path.exists(stop_file_path):
        os.remove(stop_file_path)
    
    logging.info("\nAll files in folder processed. Waiting for new files...\n")
    logging.info("=" * 60)
    logging.info("WAITING FOR NEW FILES OR USER INPUT")
    logging.info("=" * 60)
    logging.info("Options:")
    logging.info("1. Add new .parquet files to continue training")
    logging.info("2. Create a file named 'STOP_TRAINING' in the data/output/ folder to stop")
    logging.info("   Command: touch data/output/STOP_TRAINING")
    logging.info("=" * 60)
    
    size_state = {}
    check_counter = 0
    
    while True:
        # Check for stop file first
        if os.path.exists(stop_file_path):
            logging.info(f"\n🛑 Stop file detected: {stop_file_path}")
            logging.info("User requested to stop base training.")
            logging.info("Removing stop file and proceeding to save model...")
            os.remove(stop_file_path)
            return True, None
        
        # Check for new files
        current_files = set(f for f in os.listdir(parquet_dir_path) if f.endswith('.parquet'))
        new_files = current_files - trained_files
        
        if new_files:
            for file in new_files:
                file_path = os.path.join(parquet_dir_path, file)
                if is_file_ready_for_training(file_path, config.MIN_FILE_SIZE_BYTES, 
                                            config.STABLE_COUNT_THRESHOLD, size_state):
                    logging.info(f"New file detected and size stabilized (>={config.MIN_FILE_SIZE_BYTES/1024/1024:.1f} MB): "
                               f"{file} ({os.path.getsize(file_path)/1024/1024:.1f} MB)")
                    logging.info(f"File '{file}' appears to have finished uploading. "
                               "Resuming training with new file...")
                    return False, file
        
        # Show periodic status
        if check_counter % 300 == 0:
            logging.debug(f"Current .parquet files detected: {current_files}")
            logging.debug(f"New files since last training: {new_files}")
            logging.debug("Still waiting... (Create 'data/output/STOP_TRAINING' file to stop)")
        
        check_counter += 1
        time.sleep(10)


def preload_parquet_data(parquet_file: str, vocab_size: int, parquet_dir_path: str, 
                        text_column: str, vocab_path: str, batch_size: int) -> Tuple[threading.Thread, dict]:
    """
    Start preloading parquet data in a background thread.
    
    Args:
        parquet_file: Name of the parquet file to preload
        vocab_size: Vocabulary size
        parquet_dir_path: Directory containing parquet files
        text_column: Name of text column
        vocab_path: Path to vocabulary file
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (thread, result_dict)
    """
    result = {}
    
    def loader():
        try:
            result['data'] = load_and_process_data(
                vocab_size=vocab_size,
                parquet_dir_path=parquet_dir_path,
                text_column=text_column,
                vocab_path=vocab_path,
                batch_size=batch_size,
                single_file=parquet_file
            )
        except Exception as e:
            logging.error(f"Error preloading {parquet_file}: {e}")
            result['error'] = str(e)
    
    thread = threading.Thread(target=loader)
    thread.start()
    return thread, result


def cleanup_processed_file(file_path: str) -> None:
    """
    Safely delete a processed file.
    
    Args:
        file_path: Path to the file to delete
    """
    try:
        os.remove(file_path)
        logging.info(f"Deleted file after training: {file_path}")
    except FileNotFoundError:
        logging.info(f"File not found when deleting {file_path}")
    except PermissionError as e:
        logging.info(f"Permission error when deleting {file_path}: {e}")
    except Exception as e:
        logging.info(f"Error deleting file {file_path}: {e}")


def get_parquet_files(parquet_dir_path: str) -> list:
    """
    Get sorted list of parquet files in directory.
    
    Args:
        parquet_dir_path: Directory to search
        
    Returns:
        Sorted list of parquet file paths
    """
    return sorted(
        glob.glob(os.path.join(parquet_dir_path, '*.parquet')),
        key=lambda x: os.path.basename(x)
    )


def setup_output_directory(output_dir: str) -> str:
    """
    Create output directory and return path.
    
    Args:
        output_dir: Desired output directory path
        
    Returns:
        Created output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
