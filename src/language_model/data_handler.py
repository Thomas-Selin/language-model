from math import e
from helpers import print_memory_usage, get_device
from subword_tokenizer import SubwordTokenizer
import pandas as pd
import torch
import os
import datetime
import gc

device = get_device()

def load_text_from_parquet(parquet_file, text_column='text'):
    """Load text data from a parquet file"""
    print(f"Loading parquet dataset from {parquet_file}...")
    try:
        # Read the parquet file
        df = pd.read_parquet(parquet_file)
        
        # Check if the text column exists
        if text_column not in df.columns:
            available_columns = ', '.join(df.columns)
            raise ValueError(f"Column '{text_column}' not found in parquet file. Available columns: {available_columns}")
        
        # Extract text from the specified column
        text_data = ' '.join(df[text_column].fillna('').astype(str).tolist())
        print(f"✅ Parquet dataset loaded successfully. {len(df)} rows processed.")
        return text_data
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return ""

def load_and_process_data(vocab_size, parquet_dir_path, text_column='text', vocab_path='data/output/vocab_subword.json', batch_size=10, single_file=None):
    """Load and process text data from multiple parquet files in batches for training, or a single file if specified."""
    tokenizer_path = vocab_path
    if single_file is not None:
        # Only process the specified file
        all_parquet_files = [single_file]
        print(f"Processing single parquet file: {single_file}")
    else:
        # List all parquet files in the directory
        all_parquet_files = [f for f in os.listdir(parquet_dir_path) if f.endswith('.parquet')]
        print(f"Found {len(all_parquet_files)} parquet files in {parquet_dir_path}")
        
        # Sort the files to process them in a consistent order
        all_parquet_files.sort()
    
    # Calculate total batches
    total_batches = len(all_parquet_files) // batch_size
    if len(all_parquet_files) % batch_size > 0:
        total_batches += 1
    
    print(f"Will process files in {total_batches} batches of up to {batch_size} files each")
    
    # Process first batch to build vocabulary if needed
    first_batch = all_parquet_files[:batch_size]
    print(f"Processing first batch of {len(first_batch)} files...")
    
    # Only build vocab if it doesn't exist
    if not os.path.exists(tokenizer_path):
        print("Need to build vocabulary with subword tokenizer...")
        # Collect text samples for vocab building (use less memory)
        # We don't need the full text for building vocabulary, just representative samples
        sample_texts = []
        sample_size = 1000000  # Limit sample size per file to save memory
        total_samples = 0
        
        for file in first_batch:
            file_path = os.path.join(parquet_dir_path, file)
            try:
                # Read the parquet file but only load what we need for the sample
                df = pd.read_parquet(file_path)
                if text_column not in df.columns:
                    print(f"Warning: Column '{text_column}' not found in {file}")
                    continue
                
                # Take a sample of rows instead of all rows
                if len(df) > 100:
                    df_sample = df.sample(min(100, len(df)))
                else:
                    df_sample = df
                
                texts = df_sample[text_column].fillna('').astype(str).tolist()
                text_sample = ' '.join(texts)
                
                # Limit sample size
                if len(text_sample) > sample_size:
                    text_sample = text_sample[:sample_size]
                
                sample_texts.append(text_sample)
                total_samples += len(text_sample)
                print(f"Added {len(text_sample)} chars from {file} for vocabulary building")
                
                # If we have enough samples, stop collecting
                if total_samples >= 5000000:  # 5MB of text should be enough for vocab
                    break
                    
            except Exception as e:
                print(f"Error sampling file {file} for vocab: {e}")
        
        if not sample_texts:
            raise ValueError("No data could be loaded from first batch of files for vocabulary building.")
        
        # Combine samples
        combined_samples = ' '.join(sample_texts)
        print(f"Building vocabulary from {len(combined_samples)} characters of sample text...")
        
        # Build vocab
        tokenizer_obj = SubwordTokenizer.build_vocab(combined_samples, vocab_size=vocab_size)
        print(f"✅ Vocabulary built successfully. Size: {tokenizer_obj.get_vocab_size()}")
        SubwordTokenizer.save_vocab(tokenizer_obj, path=tokenizer_path)
        
        # Free memory
        del sample_texts
        del combined_samples
    else:
        print("Using existing subword vocabulary...")

    # Load tokenizer from saved file
    tokenizer = SubwordTokenizer(vocab_file=tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"✅ Vocabulary loaded. Size: {vocab_size}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}.")
    
    # Process files one by one to save memory
    print("Processing first batch files one by one...")
    print_memory_usage()  # Print memory usage before processing
    
    chunk_tensors = []
    for file_idx, file in enumerate(first_batch):
        file_path = os.path.join(parquet_dir_path, file)
        print(f"Processing file {file_idx+1}/{len(first_batch)}: {file}")
        
        try:
            df = pd.read_parquet(file_path)
            if text_column not in df.columns:
                print(f"Warning: Column '{text_column}' not found in {file}, skipping")
                continue
            chunk_size_rows = 1  # Process 1 row at a time
            print(f"Chunk size: {chunk_size_rows} rows")
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
                    
            del df
            # Perform garbage collection after each file
            gc.collect()
            if (file_idx + 1) % 3 == 0:
                print_memory_usage()
        except Exception as e:
            print(f"Error processing file {file}: {e}")
        print()
    if not chunk_tensors:
        raise ValueError("No tokens could be extracted from the first batch of files.")
    print("Converting tokens to tensor...")
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_usage()  # Print memory usage before tensor conversion
    data = torch.cat(chunk_tensors)
    del chunk_tensors  # Free memory
    print_memory_usage()  # Print memory usage after tensor conversion
    
    print(f"✅ First batch encoded. Length: {len(data)}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}.")
    
    # Split into train and validation sets
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    # Free memory
    del data
    print_memory_usage()  # Final memory check
    
    # If single_file, no remaining batches
    if single_file is not None:
        remaining_batches = []
        print(f"Remaining batches for single file {single_file}: {remaining_batches}")
    else:
        remaining_batches = [all_parquet_files[i:i+batch_size] for i in range(batch_size, len(all_parquet_files), batch_size)]
    
    # Return the first batch data, the remaining file batches, and tokenizer info
    return train_data, val_data, tokenizer, vocab_size, remaining_batches

def load_next_batch(batch_files, parquet_dir_path, text_column, tokenizer, train_data, val_data):
    """Load and process the next batch of parquet files"""
    print(f"Loading next batch of {len(batch_files)} files...")
    print_memory_usage()  # Print initial memory usage
    
    chunk_tensors = []
    for file_idx, file in enumerate(batch_files):
        file_path = os.path.join(parquet_dir_path, file)
        print(f"Processing file {file_idx+1}/{len(batch_files)}: {file}")
        try:
            df = pd.read_parquet(file_path)
            if text_column not in df.columns:
                print(f"Warning: Column '{text_column}' not found in {file}, skipping")
                continue
            chunk_size_rows = 200  # Process 200 rows at a time
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
            
            print()  # New line after progress bar
            del df
            # Perform garbage collection after each file
            gc.collect()
            if (file_idx + 1) % 3 == 0:
                print_memory_usage()
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    if not chunk_tensors:
        print("Warning: No tokens could be extracted from this batch of files.")
        return train_data, val_data
    print("Converting tokens to tensor...")
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_usage()  # Print memory usage before tensor conversion
    data = torch.cat(chunk_tensors)
    del chunk_tensors  # Free memory
    print_memory_usage()  # Check memory after tensor conversion
    
    print(f"✅ Batch encoded. Length: {len(data)}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}.")
    
    # Split into train and validation sets
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]  # Replace existing data instead of concatenating
    val_data = data[n:]
    
    # Clean up to free memory
    del data
    
    print(f"New batch data size: train={len(train_data)}, val={len(val_data)}")
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
            print(f"Error: 'context' column not found in {qa_parquet_path}")
            return False
            
        # Extract context data, filter out empty contexts, and drop duplicates
        contexts = qa_df['context'].dropna().astype(str)
        contexts = contexts[contexts.str.strip().str.len() > 0]
        contexts = contexts.drop_duplicates()
        
        if len(contexts) == 0:
            print("No valid context data found in the dataset")
            return False
            
        # Create a new dataframe with the context data
        context_df = pd.DataFrame({text_column: contexts})
        
        # Save to parquet
        context_df.to_parquet(output_parquet_path, index=False)
        print(f"✅ Successfully saved {len(context_df)} qa context to base training data. Path: {output_parquet_path}")
        return True
        
    except Exception as e:
        print(f"Error preparing context data: {e}")
        return False
