"""Data processing utilities for model training.

This module contains core data processing functions for loading, tokenizing,
and batching data for training. Functions for file loading and QA processing
have been moved to data_loading.py and qa_processing.py respectively.
"""

import threading
import pandas as pd
import torch
import os
import datetime
import gc
import time
import logging
from language_model.config import LOG_LEVEL, BLOCK_SIZE
from language_model.helpers import print_memory_usage, configure_colored_logging
from language_model.subword_tokenizer import SubwordTokenizer

# Configure logging
configure_colored_logging(LOG_LEVEL)


def load_and_process_data(vocab_size, parquet_dir_path, text_column='text', vocab_path='data/output/vocab_subword.json', batch_size=10, single_file=None):
    """Load and process text data from multiple parquet files in batches for training, or a single file if specified."""
    thread_label = threading.current_thread().name
    tokenizer_path = vocab_path
    if single_file is not None:
        # Only process the specified file
        all_parquet_files = [single_file]
        logging.info(f"[{thread_label}] Processing single parquet file: {single_file}")
    else:
        # List all parquet files in the directory
        all_parquet_files = [f for f in os.listdir(parquet_dir_path) if f.endswith('.parquet')]
        logging.info(f"[{thread_label}] Found {len(all_parquet_files)} parquet files in {parquet_dir_path}")
        
        # Sort the files to process them in a consistent order
        all_parquet_files.sort()
    
    # Calculate total batches
    total_batches = len(all_parquet_files) // batch_size
    if len(all_parquet_files) % batch_size > 0:
        total_batches += 1
    
    logging.info(f"[{thread_label}] Will process files in {total_batches} batches of up to {batch_size} files each")
    
    # Only build vocab if it doesn't exist
    first_batch_idx = 0
    tokenizer_built = os.path.exists(tokenizer_path)
    while first_batch_idx < total_batches:
        first_batch = all_parquet_files[first_batch_idx*batch_size:(first_batch_idx+1)*batch_size]
        logging.info(f"Processing batch {first_batch_idx+1} of {total_batches}, {len(first_batch)} files...")
        skipped_files = []
        # Build vocab if needed
        if not tokenizer_built:
            logging.info("Need to build vocabulary with subword tokenizer...")
            import random
            sample_texts = []

            PER_FILE_CHAR_CAP = 1_000_000          # ~1M chars max from any single file
            TOTAL_VOCAB_SAMPLE_CHARS = 20_000_000  # target total chars across files (10–20M is plenty)

            files_for_vocab = list(all_parquet_files)  # sample from the entire corpus, not just the first batch
            random.shuffle(files_for_vocab)

            total_chars = 0
            for file in files_for_vocab:
                file_path = os.path.join(parquet_dir_path, file)
                try:
                    df = pd.read_parquet(file_path, columns=[text_column])
                    if text_column not in df.columns:
                        logging.info(f"Warning: Column '{text_column}' not found in {file}")
                        skipped_files.append(file)
                        continue
                    # small, cheap per-file row sample; increase if rows are very short
                    df_sample = df.sample(min(100, len(df))) if len(df) > 100 else df
                    texts = df_sample[text_column].fillna('').astype(str).tolist()
                    text_sample = ' '.join(texts)
                    if len(text_sample) > PER_FILE_CHAR_CAP:
                        text_sample = text_sample[:PER_FILE_CHAR_CAP]
                    sample_texts.append(text_sample)
                    total_chars += len(text_sample)
                    logging.debug(f"Added {len(text_sample)} chars from {file} (total {total_chars})")
                    if total_chars >= TOTAL_VOCAB_SAMPLE_CHARS:
                        break
                except Exception as e:
                    logging.warning(f"Error sampling file {file} for vocab: {e}")
                    skipped_files.append(file)

            if not sample_texts:
                logging.info(f"No data available to build vocabulary. Skipped files: {skipped_files}")
                first_batch_idx += 1
                continue

            combined_samples = ' '.join(sample_texts)
            logging.info(f"Building vocabulary from {len(combined_samples)} characters of sample text...")
            tokenizer_obj = SubwordTokenizer.build_vocab(combined_samples, vocab_size=vocab_size)
            logging.info(f"Vocabulary built successfully. Size: {tokenizer_obj.get_vocab_size()}")
            SubwordTokenizer.save_vocab(tokenizer_obj, path=tokenizer_path)
            del sample_texts, combined_samples
            tokenizer_built = True
        else:
            logging.info("Using existing subword vocabulary...")
        tokenizer = SubwordTokenizer(vocab_file=tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        logging.info(f"Vocabulary loaded. Size: {vocab_size}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}")
        logging.debug("Processing batch files one by one...")
        print_memory_usage()
        chunk_tensors = []
        skipped_files = []
        # Keep track of successfully processed files
        successfully_processed_files = []
        for file_idx, file in enumerate(first_batch):
            file_path = os.path.join(parquet_dir_path, file)
            logging.info(f"Processing file {file_idx+1}/{len(first_batch)}: {file}")
            logging.debug(f"Chunk size: 100 rows")
            max_retries = 1000
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    df = pd.read_parquet(file_path, columns=[text_column])
                    if text_column not in df.columns:
                        logging.info(f"Warning: Column '{text_column}' not found in {file}, skipping")
                        skipped_files.append(file)
                        break
                    chunk_size_rows = 100
                    for i in range(0, len(df), chunk_size_rows):
                        end_idx = min(i + chunk_size_rows, len(df))
                        chunk_df = df.iloc[i:end_idx]
                        chunk_text = ' '.join(chunk_df[text_column].fillna('').astype(str).tolist())
                        if chunk_text:
                            chunk_tokens = tokenizer.encode(chunk_text)
                            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device='cpu')
                            chunk_tensors.append(chunk_tensor)
                        del chunk_text
                        del chunk_df
                        # Log progress every 100 chunks, with thread name
                        if (i // chunk_size_rows) % 100 == 0:
                            percent_done = end_idx / len(df) * 100
                            thread_label = threading.current_thread().name
                            logging.debug(f"[{thread_label}] Processing {file}: {percent_done:.1f}% done")
                            if (i // chunk_size_rows) % 50 == 0:
                                gc.collect()
                    del df
                    gc.collect()
                    if (file_idx + 1) % 3 == 0:
                        print_memory_usage()
                    # After successfully processing the file (outside the retry loop):
                    if file not in skipped_files:
                        successfully_processed_files.append(file)
                    break
                except Exception as e:
                    retry_count += 1
                    logging.error(f"Error processing file {file} at {datetime.datetime.now().strftime('%H:%M:%S')}: {e}")
                    if retry_count > max_retries:
                        logging.warning(f"Max retries reached for file {file}, skipping.")
                        skipped_files.append(file)
                        break
                    logging.warning(f"Retrying file {file} in 1 minute (attempt {retry_count}/{max_retries}) at {datetime.datetime.now().strftime('%H:%M')}")
                    time.sleep(60)
            logging.info("")
        if not chunk_tensors:
            logging.info(f"No tokens could be extracted from batch {first_batch_idx+1}. Skipped files: {skipped_files}")
            first_batch_idx += 1
            continue
        if skipped_files:
            logging.info(f"Skipped files in batch {first_batch_idx+1}: {skipped_files}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_usage()
        logging.info("Converting tokens to tensor...")
        data = torch.cat(chunk_tensors)
        del chunk_tensors
        print_memory_usage()
        logging.info(f"Batch {first_batch_idx+1} encoded. Length: {len(data)}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # Validate that we have enough data (block_size from config)
        MIN_DATA_LENGTH = BLOCK_SIZE  # Must match block_size from config
        if len(data) <= MIN_DATA_LENGTH:
            error_msg = (
                f"Insufficient data for training. Encoded data length ({len(data)} tokens) must be greater than "
                f"{MIN_DATA_LENGTH}. The input file may be too small or contain insufficient text content. "
                f"Please provide a larger dataset or combine multiple files."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        n = int(0.9*len(data))
        train_data = data[:n]
        val_data = data[n:]
        del data
        print_memory_usage()
        if single_file is not None:
            if single_file in successfully_processed_files:
                remaining_batches = []
                logging.info(f"File {single_file} has been successfully processed. No remaining batches.")
            else:
                error_msg = f"File {single_file} was found but could not be properly processed."
                logging.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            remaining_batches = [all_parquet_files[i:i+batch_size] for i in range((first_batch_idx+1)*batch_size, len(all_parquet_files), batch_size)]
        return train_data, val_data, tokenizer, vocab_size, remaining_batches
    raise ValueError("No tokens could be extracted from any batch of files.")

def load_next_batch(batch_files, parquet_dir_path, text_column, tokenizer, train_data, val_data):
    """Load and process the next batch of parquet files"""
    logging.info(f"Loading next batch of {len(batch_files)} files...")
    print_memory_usage()  # Print initial memory usage
    chunk_tensors = []
    for file_idx, file in enumerate(batch_files):
        file_path = os.path.join(parquet_dir_path, file)
        logging.info(f"Processing file {file_idx+1}/{len(batch_files)}: {file}")
        logging.debug(f"Chunk size: 100 rows")
        try:
            df = pd.read_parquet(file_path, columns=[text_column])
            if text_column not in df.columns:
                logging.info(f"Warning: Column '{text_column}' not found in {file}, skipping")
                continue
            chunk_size_rows = 100  # Process 100 rows at a time
            for i in range(0, len(df), chunk_size_rows):
                end_idx = min(i + chunk_size_rows, len(df))
                chunk_df = df.iloc[i:end_idx]
                chunk_text = ' '.join(chunk_df[text_column].fillna('').astype(str).tolist())
                if chunk_text:
                    chunk_tokens = tokenizer.encode(chunk_text)
                    chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device='cpu')
                    chunk_tensors.append(chunk_tensor)
                del chunk_text
                del chunk_df
                # Log progress every 100 chunks
                if (i // chunk_size_rows) % 100 == 0:
                    percent_done = end_idx / len(df) * 100
                    logging.debug(f"Processing {file}: {percent_done:.1f}% done")
                    
                    # Add explicit garbage collection periodically
                    if (i // chunk_size_rows) % 50 == 0:
                        gc.collect()
            logging.info("")
            del df
            gc.collect()
            if (file_idx + 1) % 3 == 0:
                print_memory_usage()
        except Exception as e:
            logging.error(f"Error processing file {file} at {datetime.datetime.now().strftime('%H:%M:%S')}: {e}")
    if not chunk_tensors:
        logging.info("Warning: No tokens could be extracted from this batch of files.")
        return train_data, val_data
    logging.info("Converting tokens to tensor...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_usage()  # Print memory usage before tensor conversion
    data = torch.cat(chunk_tensors)
    del chunk_tensors  # Free memory
    print_memory_usage()  # Check memory after tensor conversion
    logging.info(f"Batch encoded. Length: {len(data)}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    # Validate that we have enough data (block_size from config)
    MIN_DATA_LENGTH = BLOCK_SIZE  # Must match block_size from config
    if len(data) <= MIN_DATA_LENGTH:
        error_msg = (
            f"Insufficient data in batch. Encoded data length ({len(data)} tokens) must be greater than "
            f"{MIN_DATA_LENGTH}. The batch files may be too small or contain insufficient text content. "
            f"Skipping this batch and returning existing data."
        )
        logging.warning(error_msg)
        del data
        return train_data, val_data
    
    # Split into train and validation sets
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]  # Replace existing data instead of concatenating
    val_data = data[n:]
    
    # Clean up to free memory
    del data
    
    logging.info(f"New batch data size: train={len(train_data)}, val={len(val_data)}")
    print_memory_usage()  # Final memory check
    
    return train_data, val_data


def get_batch(block_size, batch_size, data_split, train_data, val_data, device):
    """Generate a random batch of data.
    
    For evaluation, we randomly sample batches. For training, use get_sequential_batches
    to ensure the model sees all data.
    """
    data = train_data if data_split == 'train' else val_data
    
    # Check if we have enough data for the requested block size
    if len(data) <= block_size:
        raise ValueError(
            f"Insufficient data for training. Data length ({len(data)}) must be greater than "
            f"block_size ({block_size}). Please ensure your input file contains enough text data."
        )
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def get_sequential_batches(block_size, batch_size, data, device, shuffle=True):
    """Generate sequential batches that cover all data systematically.
    
    This ensures the model sees all training data in each epoch.
    
    Args:
        block_size: Context window size
        batch_size: Number of sequences per batch
        data: The dataset tensor
        device: Device to place tensors on
        shuffle: Whether to shuffle the data at the start of each epoch
        
    Yields:
        Tuples of (input_batch, target_batch)
    """
    if len(data) <= block_size:
        raise ValueError(
            f"Insufficient data. Data length ({len(data)}) must be greater than "
            f"block_size ({block_size})."
        )
    
    # Calculate number of possible sequences
    max_start_idx = len(data) - block_size - 1
    
    # Create all possible starting indices
    indices = torch.arange(0, max_start_idx)
    
    if shuffle:
        # Shuffle indices for better training
        perm = torch.randperm(len(indices))
        indices = indices[perm]
    
    # Yield batches
    num_batches = (len(indices) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]
        
        # Create batch
        x = torch.stack([data[i:i+block_size] for i in batch_indices])
        y = torch.stack([data[i+1:i+block_size+1] for i in batch_indices])
        x, y = x.to(device), y.to(device)
        
        yield x, y
