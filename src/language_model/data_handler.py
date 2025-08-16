from helpers import print_memory_usage, get_device, configure_colored_logging
from subword_tokenizer import SubwordTokenizer
import pandas as pd
import torch
import os
import datetime
import gc
import time
import logging
from config import LOG_LEVEL

# Configure logging
configure_colored_logging(LOG_LEVEL)

device = get_device()

def is_file_fully_uploaded(file_path, check_interval=2, checks=3):
    """Returns True if file size is stable for a few checks (not being uploaded)."""
    prev_size = -1
    for _ in range(checks):
        size = os.path.getsize(file_path)
        if size == prev_size:
            return True
        prev_size = size
        time.sleep(check_interval)
    return False

def poll_for_new_parquet_file(parquet_dir, poll_interval=5):
    """Polls for a new, fully uploaded parquet file in the directory."""
    seen_files = set()  # Start with empty set to process all files present at startup
    while True:
        current_files = set(f for f in os.listdir(parquet_dir) if f.endswith('.parquet'))
        new_files = current_files - seen_files
        for file in new_files:
            file_path = os.path.join(parquet_dir, file)
            if is_file_fully_uploaded(file_path):
                logging.info(f"Detected new, fully uploaded file: {file}")
                seen_files.add(file)  # Mark as seen
                return file
            else:
                logging.debug(f"File {file} is still being uploaded, waiting...")
        time.sleep(poll_interval)

def load_text_from_parquet(parquet_file, text_column='text'):
    """Load text data from a parquet file"""
    logging.info(f"Loading parquet dataset from {parquet_file}...")
    try:
        # Read the parquet file
        df = pd.read_parquet(parquet_file)
        
        # Check if the text column exists
        if text_column not in df.columns:
            available_columns = ', '.join(df.columns)
            raise ValueError(f"Column '{text_column}' not found in parquet file. Available columns: {available_columns}")
        
        # Extract text from the specified column
        text_data = ' '.join(df[text_column].fillna('').astype(str).tolist())
        logging.info(f"Parquet dataset loaded successfully. {len(df)} rows processed.")
        return text_data
    except Exception as e:
        logging.info(f"Error loading parquet file: {e}")
        return ""

def load_and_process_data(vocab_size, parquet_dir_path, text_column='text', vocab_path='data/output/vocab_subword.json', batch_size=10, single_file=None):
    """Load and process text data from multiple parquet files in batches for training, or a single file if specified."""
    tokenizer_path = vocab_path
    if single_file is not None:
        # Only process the specified file
        all_parquet_files = [single_file]
        logging.info(f"Processing single parquet file: {single_file}")
    else:
        # List all parquet files in the directory
        all_parquet_files = [f for f in os.listdir(parquet_dir_path) if f.endswith('.parquet')]
        logging.info(f"Found {len(all_parquet_files)} parquet files in {parquet_dir_path}")
        
        # Sort the files to process them in a consistent order
        all_parquet_files.sort()
    
    # Calculate total batches
    total_batches = len(all_parquet_files) // batch_size
    if len(all_parquet_files) % batch_size > 0:
        total_batches += 1
    
    logging.info(f"Will process files in {total_batches} batches of up to {batch_size} files each")
    
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
                    logging.info(f"Error sampling file {file} for vocab: {e}")
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
        for file_idx, file in enumerate(first_batch):
            file_path = os.path.join(parquet_dir_path, file)
            logging.info(f"Processing file {file_idx+1}/{len(first_batch)}: {file}")
            max_retries = 65
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    df = pd.read_parquet(file_path, columns=[text_column])
                    if text_column not in df.columns:
                        logging.info(f"Warning: Column '{text_column}' not found in {file}, skipping")
                        skipped_files.append(file)
                        break
                    chunk_size_rows = 100
                    logging.debug(f"Chunk size: {chunk_size_rows} rows")
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
                        if (i // chunk_size_rows) % 100 == 0:
                            percent_done = end_idx / len(df) * 100
                            bar_length = 20
                            filled_length = int(bar_length * end_idx // len(df))
                            bar = '█' * filled_length + '░' * (bar_length - filled_length)
                            print(f"\r  Processing {file}: [{bar}] {percent_done:.1f}%", end='', flush=True)
                            if (i // chunk_size_rows) % 50 == 0:
                                gc.collect()
                    print("\n")
                    del df
                    gc.collect()
                    if (file_idx + 1) % 3 == 0:
                        print_memory_usage()
                    break
                except Exception as e:
                    retry_count += 1
                    logging.info(f"Error processing file {file} at {datetime.datetime.now().strftime('%H:%M:%S')}: {e}")
                    if retry_count > max_retries:
                        logging.info(f"Max retries reached for file {file}, skipping.")
                        skipped_files.append(file)
                        break
                    logging.info(f"Retrying file {file} in 10 minutes (attempt {retry_count}/{max_retries}) at {datetime.datetime.now().strftime('%H:%M')}...")
                    time.sleep(600)
            logging.info("")
        if not chunk_tensors:
            logging.info(f"No tokens could be extracted from batch {first_batch_idx+1}. Skipped files: {skipped_files}")
            first_batch_idx += 1
            continue
        if skipped_files:
            logging.info(f"Skipped files in batch {first_batch_idx+1}: {skipped_files}")
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_usage()
        logging.info("Converting tokens to tensor...")
        data = torch.cat(chunk_tensors)
        del chunk_tensors
        print_memory_usage()
        logging.info(f"Batch {first_batch_idx+1} encoded. Length: {len(data)}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}")
        n = int(0.9*len(data))
        train_data = data[:n]
        val_data = data[n:]
        del data
        print_memory_usage()
        if single_file is not None:
            remaining_batches = []
            logging.debug(f"Remaining batches for single file {single_file}: {remaining_batches}")
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
                # Update progress bar every 100 chunks
                if (i // chunk_size_rows) % 100 == 0:
                    percent_done = end_idx / len(df) * 100
                    bar_length = 20
                    filled_length = int(bar_length * end_idx // len(df))
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"\r  Processing {file}: [{bar}] {percent_done:.1f}%", end='', flush=True)
                    
                    # Add explicit garbage collection periodically
                    if (i // chunk_size_rows) % 50 == 0:
                        gc.collect()
            logging.info("")
            del df
            gc.collect()
            if (file_idx + 1) % 3 == 0:
                print_memory_usage()
        except Exception as e:
            logging.info(f"Error processing file {file} at {datetime.datetime.now().strftime('%H:%M:%S')}: {e}")
    if not chunk_tensors:
        logging.info("Warning: No tokens could be extracted from this batch of files.")
        return train_data, val_data
    logging.info("Converting tokens to tensor...")
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_usage()  # Print memory usage before tensor conversion
    data = torch.cat(chunk_tensors)
    del chunk_tensors  # Free memory
    print_memory_usage()  # Check memory after tensor conversion
    logging.info(f"Batch encoded. Length: {len(data)}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    # Split into train and validation sets
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]  # Replace existing data instead of concatenating
    val_data = data[n:]
    
    # Clean up to free memory
    del data
    
    logging.info(f"New batch data size: train={len(train_data)}, val={len(val_data)}")
    print_memory_usage()  # Final memory check
    
    return train_data, val_data


def get_batch(block_size, batch_size, data_split, train_data, val_data):
    # generate a small batch of data of inputs x and targets y
    data = train_data if data_split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def process_qa_pairs_dataset(qa_parquet_path, tokenizer, max_length):
    """Process a single-turn question-answer pair dataset for chat alignment from a parquet file with 'answers' column."""
    import pandas as pd
    import ast
    qa_df = pd.read_parquet(qa_parquet_path)
    if not {'question', 'answers'}.issubset(qa_df.columns):
        raise ValueError("Parquet must contain 'question' and 'answers' columns.")
    pairs = []
    for _, row in qa_df.iterrows():
        question = str(row['question']) if 'question' in row else ''
        answers_field = row['answers']
        # Parse answers field (can be dict or stringified dict)
        if isinstance(answers_field, str):
            answers_dict = ast.literal_eval(answers_field)
        else:
            answers_dict = answers_field
        answer_text = ''
        if isinstance(answers_dict, dict) and 'text' in answers_dict and len(answers_dict['text']) > 0:
            answer_text = str(answers_dict['text'][0])
        pairs.append((question, answer_text))
    examples = [f"Question: {q} Answer: {a}" for q, a in pairs]
    tokenized = [tokenizer.encode(ex[:max_length]) for ex in examples]
    tensor = torch.tensor([t + [0]*(max_length-len(t)) if len(t)<max_length else t[:max_length] for t in tokenized], dtype=torch.long)
    del tokenized, examples, pairs
    gc.collect()
    return tensor

def prepare_context_data_for_training(qa_parquet_path, output_parquet_path, text_column='context'):
    """
    Extract context data from a QA dataset and prepare it for base training.
    Creates a new parquet file with the context data that can be used in the regular training pipeline.
    
    Args:
        qa_parquet_path (str): Path to the QA dataset parquet file
        output_parquet_path (str): Path to save the context data parquet file
        text_column (str): Name of the column to use in the output parquet file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import pandas as pd
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        
        # Load QA dataset
        qa_df = pd.read_parquet(qa_parquet_path)
        
        if 'context' not in qa_df.columns:
            logging.info(f"Error: 'context' column not found in {qa_parquet_path}")
            return False
            
        # Extract context data, filter out empty contexts, and drop duplicates
        contexts = qa_df['context'].dropna().astype(str)
        contexts = contexts[contexts.str.strip().str.len() > 0]
        contexts = contexts.drop_duplicates()
        
        if len(contexts) == 0:
            logging.info("No valid context data found in the dataset")
            return False
            
        # Create a new dataframe with the context data
        context_df = pd.DataFrame({text_column: contexts})
        
        # Save to parquet
        context_df.to_parquet(output_parquet_path, index=False)
        logging.info(f"Successfully saved {len(context_df)} qa context to base training data. Path: {output_parquet_path}")
        return True
        
    except Exception as e:
        logging.info(f"Error preparing context data: {e}")
        return False
