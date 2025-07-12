import torch
import torch.nn as nn
from torch.nn import functional as F
from subword_tokenizer import SubwordTokenizer
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import pyarrow.parquet as pq
import math

torch.manual_seed(1337)

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_epochs = 2 # how many epochs to train for as max, early stopping will stop training if no improvement is seen
eval_interval = 5 # how many steps between evaluations?
learning_rate = 3e-4
eval_iters = 100
n_embd = 768
n_head = 8
n_layer = 8
dropout = 0.15
early_stopping_patience = 25  # Number of epochs to wait for improvement
max_vocab_size = 3000 # Maximum vocabulary size
warmup_steps = 1000  # Adjust based on dataset size
lr_decay = "linear"  # Options: "linear", "cosine", "constant"

print("\033[94mAll hyperparameters set:\033[0m")
print(f"Batch size: {batch_size}")
print(f"Block size: {block_size}")
print(f"Max epochs: {max_epochs}")
print(f"Eval interval: {eval_interval}")
print(f"Learning rate: {learning_rate}")
print(f"Eval iters: {eval_iters}")
print(f"Embedding dimension: {n_embd}")
print(f"Number of heads: {n_head}")
print(f"Number of layers: {n_layer}")
print(f"Dropout: {dropout}")
print(f"Early stopping patience: {early_stopping_patience}")
print(f"Max vocab size: {max_vocab_size}")
print("\033[94m____________________________\033[0m\n")

# Device selection - prioritize CUDA, then Metal, fall back to CPU
if torch.cuda.is_available():
    print("CUDA GPU will be used.")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Apple Metal GPU will be used.")
    device = torch.device("mps")
else:
    print("No GPU available, CPU will be used.")
    device = torch.device("cpu")

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        attn = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(attn)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        if return_attention:
            return out, attn
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            outs, attns = zip(*(h(x, return_attention=True) for h in self.heads))
            out = torch.cat(outs, dim=-1)
            return self.dropout(self.proj(out)), attns
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, return_attention=False):
        if return_attention:
            sa_out, attns = self.sa(self.ln1(x), return_attention=True)
            x = x + sa_out
            x = x + self.ffwd(self.ln2(x))
            return x, attns
        else:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Tie weights between token embedding and output projection
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attention=False):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        attentions = []
        if return_attention:
            for block in self.blocks:
                x, attn = block(x, return_attention=True)
                attentions.append(attn)
            x = self.ln_f(x) # (B,T,C)
            logits = self.lm_head(x) # (B,T,vocab_size)
        else:
            x = self.blocks(x) # (B,T,C)
            x = self.ln_f(x) # (B,T,C)
            logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        if return_attention:
            return logits, loss, attentions
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, return_attention=False, eos_token_id=None):
        # idx is (B, T) array of indices in the current context
        all_attentions = []
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            if return_attention:
                logits, _, attentions = self(idx_cond, return_attention=True)
                # Check logits shape before indexing
                if len(logits.shape) != 3:
                    raise ValueError(f"Expected logits to have shape (B, T, C) but got {logits.shape}")
            else:
                logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply temperature
            logits = logits / temperature
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Check if the generated token is the EOS token
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                # Stop generation for sequences that produced EOS
                for b in range(idx_next.size(0)):
                    if idx_next[b, 0] == eos_token_id:
                        # We still append this final token before stopping
                        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
                        if return_attention and attentions:
                            all_attentions.append([[a.detach().cpu() for a in layer] for layer in attentions])
                        if return_attention:
                            return idx, all_attentions
                        return idx
        
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if return_attention:
                # Keep attention for all tokens, not just the last one
                all_attentions.append([[a.detach().cpu() for a in layer] for layer in attentions])
            
        if return_attention:
            return idx, all_attentions
        return idx

def print_gpu_memory_summary():
    if torch.cuda.is_available():
        # Get memory usage in GB (1GB = 1024MB = 1024*1024*1024 bytes)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        free = total - reserved
        print(f"GPU memory summary: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free, {total:.2f}GB total, {max_allocated:.2f}GB max allocated, {max_reserved:.2f}GB max reserved")
    else:
        print("No CUDA GPU available.")

def get_batch(data_split, train_data, val_data):
    # generate a small batch of data of inputs x and targets y
    data = train_data if data_split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # Use mixed precision for evaluation too
        with autocast(device_type=device.type):
            for k in range(eval_iters):
                X, Y = get_batch(split, train_data, val_data)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Define scheduler
def get_lr_scheduler(optimizer, warmup_steps, lr_decay, total_steps):
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Decay phase
        if lr_decay == "linear":
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        elif lr_decay == "cosine":
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:  # constant
            return 1.0
            
    return LambdaLR(optimizer, lr_lambda)

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

def load_and_process_data(parquet_dir_path, text_column='text', vocab_path='data/output/vocab.json', batch_size=10):
    """Load and process text data from multiple parquet files in batches for training"""
    # Change the file extension for subword tokenizer
    tokenizer_path = vocab_path.replace('.json', '_subword.json')
    
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
        tokenizer_obj = SubwordTokenizer.build_vocab(combined_samples, vocab_size=max_vocab_size)
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
                if (i // chunk_size_rows) % 10 == 0:
                    print(f"  Processed {end_idx}/{len(df)} rows from {file}")
            del df
            if (file_idx + 1) % 3 == 0:
                print_memory_usage()
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    if not chunk_tensors:
        raise ValueError("No tokens could be extracted from the first batch of files.")
    print("Converting tokens to tensor...")
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
    
    # Return the first batch data, the remaining file batches, and tokenizer info
    remaining_batches = [all_parquet_files[i:i+batch_size] for i in range(batch_size, len(all_parquet_files), batch_size)]
    
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
                if (i // chunk_size_rows) % 10 == 0:
                    print(f"  Processed {end_idx}/{len(df)} rows from {file}")
            del df
            if (file_idx + 1) % 3 == 0:
                print_memory_usage()
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    if not chunk_tensors:
        print("Warning: No tokens could be extracted from this batch of files.")
        return train_data, val_data
    print("Converting tokens to tensor...")
    data = torch.cat(chunk_tensors)
    del chunk_tensors  # Free memory
    print_memory_usage()  # Check memory after tensor conversion
    
    print(f"✅ Batch encoded. Length: {len(data)}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}.")
    
    # Split into train and validation sets
    n = int(0.9*len(data))  # first 90% will be train, rest val
    new_train_data = data[:n]
    new_val_data = data[n:]
    
    print("Concatenating with existing data...")
    # Combine with existing data
    train_data = torch.cat([train_data, new_train_data])
    val_data = torch.cat([val_data, new_val_data])
    
    # Clean up to free memory
    del data
    del new_train_data
    del new_val_data
    
    print(f"Combined train data size: {len(train_data)}, val data size: {len(val_data)}")
    print_memory_usage()  # Final memory check
    
    return train_data, val_data

def wait_for_keypress():
    """Wait for user to press Enter before continuing"""
    print("\n\033[93m=========================================\033[0m")
    print("\033[93mBatch processing complete. Time to upload the next batch of files.\033[0m")
    print("\033[93mPress Enter when you've uploaded the next batch and are ready to continue...\033[0m")
    print("\033[93m=========================================\033[0m\n")
    input()  # Wait for Enter key
    print("Continuing with next batch...")

def train_model_on_batched_data(parquet_dir_path, text_column='text', vocab_path='data/output/vocab.json', batch_size_files=10):
    """Main function to train model on batched parquet files"""
    # Create directory if it doesn't exist
    os.makedirs(parquet_dir_path, exist_ok=True)
    
    # Check if there are any parquet files in the directory
    parquet_files = [f for f in os.listdir(parquet_dir_path) if f.endswith('.parquet')]
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir_path}. Please upload some parquet files and run again.")
        return
        
    print(f"Found {len(parquet_files)} parquet files in {parquet_dir_path}")
    
    # Load and process data for training (first batch)
    train_data, val_data, tokenizer, vocab_size, remaining_batches = load_and_process_data(
        parquet_dir_path=parquet_dir_path,
        text_column=text_column,
        vocab_path=vocab_path,
        batch_size=batch_size_files
    )
    
    # Create TensorBoard writer
    log_dir = os.path.join('data', 'output', 'tensorboard_logs', 
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    writer.add_text('Session Information', f"""
    ## Hyperparameters
    - Batch size: {batch_size}
    - Block size: {block_size}
    - Max epochs: {max_epochs}
    - Eval interval: {eval_interval}
    - Learning rate: {learning_rate}
    - Eval iters: {eval_iters}
    - Embedding dimension: {n_embd}
    - Number of heads: {n_head}
    - Number of layers: {n_layer}
    - Dropout: {dropout}
    - Early stopping patience: {early_stopping_patience}
    - Max vocab size: {max_vocab_size}
    
    ## Environment
    - Device: {device}
    
    ## Data
    - Tokenizer vocab size: {vocab_size}
    - Tokenizer vocab path: {vocab_path}
    - Train data size: {len(train_data)} tokens
    - Validation data size: {len(val_data)} tokens
    
    ## Timing
    - Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)

    # Initialize model with vocabulary size
    model = GPTLanguageModel(vocab_size).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Calculate total steps (adjust if using early stopping)
    total_steps = max_epochs * (len(train_data) // batch_size)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, lr_decay, total_steps)
    
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Create a wrapper model for visualization that only returns logits
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # Only return the first part (logits) from the model output
            logits, _ = self.model(x)
            return logits
    
    # Add model graph to TensorBoard using the wrapper
    sample_input = torch.zeros((1, block_size), dtype=torch.long).to(device)
    vis_model = ModelWrapper(model).to(device)
    writer.add_graph(vis_model, sample_input)
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    global_iter = 0
    batch_counter = 0
    
    try:
        # Main training loop with support for batch processing
        for batch_counter, batch_files in enumerate([None] + remaining_batches):
            # For each batch (first one is already loaded)
            if batch_counter > 0:
                # Load next batch of files
                wait_for_keypress()  # Wait for user to upload new files
                train_data, val_data = load_next_batch(
                    batch_files=batch_files,
                    parquet_dir_path=parquet_dir_path,
                    text_column=text_column,
                    tokenizer=tokenizer,
                    train_data=train_data,
                    val_data=val_data
                )
                print(f"Batch {batch_counter} loaded. Training will continue.")
            
            # Train on the current batch
            for iter in range(max_epochs):
                epoch_start_time = time.time()
    
                # Evaluation logic
                if iter % eval_interval == 0 or iter == max_epochs - 1:
                    losses = estimate_loss(model, train_data, val_data)
                    print("\033[94m____________________________\033[0m\n")
                    print(f"Batch {batch_counter}, Step {iter} of max {max_epochs}: train loss {losses['train']:.4f}, 📏 val loss \033[94m{losses['val']:.4f}\033[0m")
                    print_gpu_memory_summary()
    
                    # Early stopping logic
                    if losses['val'] < best_val_loss:
                        best_val_loss = losses['val']
                        epochs_without_improvement = 0
                        # Optionally save the best model so far
                        torch.save(model.state_dict(), "data/output/best_model.pt")
                    else:
                        epochs_without_improvement += 1
                        print(f"No improvement for {epochs_without_improvement} epoch(s).")
                        if epochs_without_improvement >= early_stopping_patience:
                            print(f"Early stopping triggered at epoch {iter}. Best val loss: {best_val_loss:.4f}")
                            break
                
                    # Log losses to TensorBoard
                    writer.add_scalar('Loss/train', losses['train'], global_iter)
                    writer.add_scalar('Loss/val', losses['val'], global_iter)
                    
                    # Log memory to TensorBoard
                    if torch.cuda.is_available():
                        writer.add_scalar('Memory/allocated_GB', torch.cuda.memory_allocated() / 1024**3, global_iter)
                        writer.add_scalar('Memory/reserved_GB', torch.cuda.memory_reserved() / 1024**3, global_iter)
                        writer.add_scalar('Memory/free_GB', (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3, global_iter)
                    
                    # Log histograms of model parameters
                    # Only log histograms occasionally to reduce overhead
                    if global_iter % (eval_interval * 10) == 0:  # Log much less frequently
                        for name, param in model.named_parameters():
                            writer.add_histogram(f'Parameters/{name}', param, global_iter)
                            if param.grad is not None:
                                writer.add_histogram(f'Gradients/{name}', param.grad, global_iter)
                            
                    # Optional: Generate and log a sample text every few iterations
                    if global_iter % (eval_interval * 5) == 0 or (iter == max_epochs - 1 and batch_counter == len(remaining_batches)):
                        context = torch.zeros((1, 1), dtype=torch.long, device=device)
                        sample_text = tokenizer.decode(model.generate(context, max_new_tokens=100, temperature=1.0)[0].tolist())
                        writer.add_text('Generated Text 1.0 temp', sample_text, global_iter)
                        sample_text = tokenizer.decode(model.generate(context, max_new_tokens=100, temperature=0.5)[0].tolist())
                        writer.add_text('Generated Text 0.5 temp', sample_text, global_iter)
                        input_ids = torch.tensor([tokenizer.encode("What color is the ball?")], dtype=torch.long, device=device)
                        sample_text = tokenizer.decode(model.generate(input_ids, max_new_tokens=100, temperature=0.5)[0].tolist())
                        writer.add_text('Generated Text: "What color is the ball?" 0.5 temp', sample_text, global_iter)
                        input_ids = torch.tensor([tokenizer.encode("What color is the ball?")], dtype=torch.long, device=device)
                        sample_text = tokenizer.decode(model.generate(input_ids, max_new_tokens=100, temperature=1.0)[0].tolist())
                        writer.add_text('Generated Text: "What color is the ball?" 1.0 temp', sample_text, global_iter)
                        input_ids = torch.tensor([tokenizer.encode("There was a")], dtype=torch.long, device=device)
                        sample_text = tokenizer.decode(model.generate(input_ids, max_new_tokens=100, temperature=0.5)[0].tolist())
                        writer.add_text('Generated Text: "There was a" 0.5 temp', sample_text, global_iter)
                
                # sample a batch of data
                data_time = time.time()
                xb, yb = get_batch('train', train_data, val_data)
                data_time = time.time() - data_time
    
                # Measure forward/backward time
                train_time = time.time()
                with autocast(device_type=device.type):  # Uses float16 where appropriate
                    logits, loss = model(xb, yb)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                train_time = time.time() - train_time
    
                # Step the scheduler
                scheduler.step()
    
                epoch_duration = time.time() - epoch_start_time  # End timing the epoch
    
                # Print GPU memory usage for each epoch (not just evaluation intervals)
                print(f"Batch {batch_counter}, Epoch {iter}: Data time: {data_time:.4f}s, Train time: {train_time:.4f}s, Total epoch time: {epoch_duration:.4f}s")
                writer.add_scalar('Total/EpochTime', epoch_duration, global_iter)
                # Log the learning rate
                writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_iter)
                if iter % eval_interval == 0:
                    print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
                
                global_iter += 1
                
                # Check if we've reached early stopping criteria
                if epochs_without_improvement >= early_stopping_patience:
                    print("Early stopping triggered. Moving to next batch if available.")
                    break
            
            # After processing a batch, reset early stopping counter but keep best_val_loss
            epochs_without_improvement = 0
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model checkpoint...")
        torch.save(model.state_dict(), "data/output/model_interrupted.pt")
        print("Model saved to data/output/model_interrupted.pt")
    else:
        torch.save(model.state_dict(), "data/output/model_checkpoint.pt")
        print("Model saved to data/output/model_checkpoint.pt")
    
    # Close the TensorBoard writer
    writer.close()
    print(f"TensorBoard logging complete. View logs with: tensorboard --logdir={log_dir}")

def print_memory_usage():
    """Print current memory usage of the Python process as a percentage of system memory"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        total_mem = psutil.virtual_memory().total
        rss_percent = memory_info.rss / total_mem * 100
        vms_percent = memory_info.vms / total_mem * 100
        print(f"Memory usage: RSS = {memory_info.rss / (1024 * 1024):.2f} MB ({rss_percent:.2f}%), VMS = {memory_info.vms / (1024 * 1024):.2f} MB ({vms_percent:.2f}%) of system memory")
    except ImportError:
        print("Could not import psutil. Install with 'pip install psutil' to monitor memory usage.")
    except Exception as e:
        print(f"Error checking memory usage: {e}")

if __name__ == "__main__":
    # Data paths
    parquet_dir_path = 'data/input/parquet_files_test'  # Directory containing parquet files
    text_column = 'text'  # Column in the parquet file that contains the text
    vocab_path = os.path.join('data', 'output', 'vocab.json')
    batch_size_files = 10  # Number of parquet files to process in each batch
    
    train_model_on_batched_data(parquet_dir_path, text_column, vocab_path, batch_size_files)
