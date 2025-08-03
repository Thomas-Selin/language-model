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
import math
import gc
import torch.utils.checkpoint

from helpers import print_gpu_memory_summary, wait_for_keypress, get_device, count_parameters
from data_handler import load_and_process_data, get_batch, prepare_context_data_for_training, process_qa_pairs_dataset

torch.manual_seed(1337)

# hyperparameters
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
base_training_max_epochs = 32 # how many epochs to train for as max, early stopping will stop training if no improvement is seen
finetuning_max_epochs = 10  # Number of epochs for chat alignment fine-tuning
eval_interval = 5 # how many steps between evaluations?
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 4
dropout = 0.15
early_stopping_patience = 25  # Number of epochs to wait for improvement
max_vocab_size = 3000 # Maximum vocabulary size
warmup_steps = 1000  # Adjust based on dataset size
lr_decay = "linear"  # Options: "linear", "cosine", "constant"

print("\033[94mAll hyperparameters set:\033[0m")
print(f"Batch size: {batch_size}")
print(f"Block size: {block_size}")
print(f"Max epochs: {base_training_max_epochs}")
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

# Device selection - prioritize CUDA, then Apple Metal, fall back to CPU
device = get_device()

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
    def __init__(self, vocab_size, use_checkpoint=True):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Tie weights between token embedding and output projection
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)
        self.use_checkpoint = use_checkpoint  # <-- Add this line

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, return_attention=False):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        attentions = [] if return_attention else None
        
        for block in self.blocks:
            if self.use_checkpoint:
                if return_attention:
                    x, attn = torch.utils.checkpoint.checkpoint(block, x, return_attention, use_reentrant=False)
                    attentions.append(attn)
                else:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                if return_attention:
                    x, attn = block(x, return_attention)
                    attentions.append(attn)
                else:
                    x = block(x)
    
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
        all_attentions = [] if return_attention else None
        
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
                
            # focus only on the last time step and apply temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
            if return_attention:
                all_attentions.append([[a.detach().cpu() for a in layer] for layer in attentions])
            
            # Check if the generated token is the EOS token
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                break
        
        return (idx, all_attentions) if return_attention else idx


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # Use mixed precision for evaluation too
        with autocast(device_type=device.type):
            for k in range(eval_iters):
                X, Y = get_batch(block_size, batch_size, split, train_data, val_data)
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
        
        # Decay phase - progress calculation
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        
        # Different decay strategies
        if lr_decay == "linear":
            return max(0.0, 1.0 - progress)
        elif lr_decay == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:  # constant
            return 1.0
            
    return LambdaLR(optimizer, lr_lambda)

def base_train_model(parquet_dir_path, text_column='text', vocab_path='data/output/vocab.json', batch_size_files=1, training_start_time=None):
    """Main function to train model on parquet files one at a time, waiting for keypress after all files in folder are processed."""
    import glob
    # Get all parquet files in the directory
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir_path}")
        return

    file_name_single = os.path.basename(parquet_files[0])

    # Load tokenizer and vocab size
    train_data, val_data, tokenizer, vocab_size, _ = load_and_process_data(
        vocab_size=max_vocab_size,
        parquet_dir_path=parquet_dir_path,
        text_column=text_column,
        vocab_path=vocab_path,
        batch_size=batch_size,
        single_file=file_name_single # Load first file to get tokenizer and vocab size
    )

    # Create TensorBoard writer
    log_dir = os.path.join('data', 'output', 'tensorboard_logs', training_start_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    writer.add_text('Session Information', f"""
    ## Hyperparameters
    - Batch size: {batch_size}
    - Block size: {block_size}
    - Max epochs: {base_training_max_epochs}
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
    print(f"{count_parameters(model)/1e6:.2f} M parameters")
    print(f"Model is on device: {next(model.parameters()).device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = base_training_max_epochs * (len(train_data) // batch_size)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, lr_decay, total_steps)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            # Temporarily disable checkpointing for graph tracing
            orig = self.model.use_checkpoint
            self.model.use_checkpoint = False
            logits, _ = self.model(x)
            self.model.use_checkpoint = orig
            return logits
    sample_input = torch.zeros((1, block_size), dtype=torch.long).to(device)
    vis_model = ModelWrapper(model).to(device)
    writer.add_graph(vis_model, sample_input)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    global_iter = 0
    batch_counter = 0
    total_epochs_run = 0
    global_max_epochs = base_training_max_epochs

    try:
        while True:
            parquet_files = sorted(glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
            if not parquet_files:
                print(f"No parquet files found in {parquet_dir_path}. Waiting for new files...")
                wait_for_keypress()
                # Check again after keypress
                parquet_files = sorted(glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
                if not parquet_files:
                    print("No files present after keypress. Base training finished.")
                    break
                continue
            for file_idx, parquet_file in enumerate(parquet_files):
                # --- MEMORY CLEANUP BEFORE PROCESSING NEW FILE ---
                print("Performing memory cleanup before loading new file...")
                # Delete any large variables from previous file
                try:
                    del train_data
                except Exception:
                    pass
                try:
                    del val_data
                except Exception:
                    pass
                gc.collect()
                gc.collect()  # Call twice for extra effect
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print_gpu_memory_summary()
                # --- END MEMORY CLEANUP ---

                print(f"Processing file {file_idx+1}/{len(parquet_files)}: {parquet_file}. Setting best_val_loss to infinity.")
                best_val_loss = float('inf')
                # Pass directory and file name separately
                file_dir = os.path.dirname(parquet_file)
                file_name = os.path.basename(parquet_file)
                train_data, val_data, _, _, _ = load_and_process_data(
                    vocab_size=max_vocab_size,
                    parquet_dir_path=file_dir,
                    text_column=text_column,
                    vocab_path=vocab_path,
                    batch_size=batch_size,
                    single_file=file_name  # new argument to specify single file
                )
                for iter in range(base_training_max_epochs):
                    if total_epochs_run >= global_max_epochs:
                        print(f"Reached global epoch limit of {global_max_epochs}. Stopping training.")
                        break
                    if iter % eval_interval == 0 or iter == base_training_max_epochs - 1:
                        losses = estimate_loss(model, train_data, val_data)
                        print("\033[94m____________________________\033[0m\n")
                        print(f"File {file_idx+1}, Step {iter} of max {base_training_max_epochs}: train loss {losses['train']:.4f}, \U0001F4CF val loss \033[94m{losses['val']:.4f}\033[0m")
                        print_gpu_memory_summary()
                        if losses['val'] < best_val_loss:
                            best_val_loss = losses['val']
                            epochs_without_improvement = 0
                            torch.save(model.state_dict(), "data/output/best_model.pt")
                        else:
                            epochs_without_improvement += 1 * eval_interval
                            print(f"No improvement for {epochs_without_improvement} epoch(s).")
                            if epochs_without_improvement >= early_stopping_patience:
                                print(f"Early stopping triggered at epoch {iter}. Best val loss: {best_val_loss:.4f}")
                                # Restore best model weights
                                print("Restoring best model weights...")
                                model.load_state_dict(torch.load("data/output/best_model.pt"))
                                break
                        writer.add_scalar('Loss/train', losses['train'], global_iter)
                        writer.add_scalar('Loss/val', losses['val'], global_iter)
                        if torch.cuda.is_available():
                            writer.add_scalar('Memory/allocated_GB', torch.cuda.memory_allocated() / 1024**3, global_iter)
                            writer.add_scalar('Memory/reserved_GB', torch.cuda.memory_reserved() / 1024**3, global_iter)
                            writer.add_scalar('Memory/free_GB', (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3, global_iter)
                        if global_iter % (eval_interval * 10) == 0:
                            for name, param in model.named_parameters():
                                writer.add_histogram(f'Parameters/{name}', param, global_iter)
                                if param.grad is not None:
                                    writer.add_histogram(f'Gradients/{name}', param.grad, global_iter)
                        if global_iter % (eval_interval * 5) == 0 or (iter == base_training_max_epochs - 1 and file_idx == len(parquet_files)-1):
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
                    epoch_start_time = time.time()
                    data_time = time.time()
                    xb, yb = get_batch(block_size, batch_size, 'train', train_data, val_data)
                    data_time = time.time() - data_time
                    train_time = time.time()
                    with autocast(device_type=device.type):
                        logits, loss = model(xb, yb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    # Add explicit cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if global_iter % 3 == 0:
                        gc.collect()
                    train_time = time.time() - train_time
                    scheduler.step()
                    epoch_duration = time.time() - epoch_start_time
                    print(f"File {file_idx+1}, Epoch {iter}: Data time: {data_time:.4f}s, Train time: {train_time:.4f}s, Total epoch time: {epoch_duration:.4f}s")
                    writer.add_scalar('Total/EpochTime', epoch_duration, global_iter)
                    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_iter)
                    if iter % eval_interval == 0:
                        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
                    global_iter += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        print("Early stopping triggered. Moving to next file if available.")
                        break
                epochs_without_improvement = 0
            # After all files in folder processed, wait for keypress
            print("\n\033[94mAll files in folder processed. Please delete old files and upload new ones, then press Enter to continue, or type 'q' and Enter to finish base training.\033[0m\n")
            user_input = input()
            if user_input.strip().lower() == 'q':
                print("Base training finished by user request.")
                break
            # Check again after keypress
            parquet_files = sorted(glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
            if not parquet_files:
                print("No files present after keypress. Base training finished.")
                break
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model checkpoint...")
        torch.save(model.state_dict(), "data/output/model_interrupted.pt")
        print("Model saved to data/output/model_interrupted.pt")
    else:
        torch.save(model.state_dict(), "data/output/model_checkpoint.pt")
        print("Model saved to data/output/model_checkpoint.pt")
    writer.close()
    print(f"\033[92mTensorBoard logging complete. View logs with: tensorboard --logdir={log_dir}\033[0m")

def train_chat_alignment(model, qa_tensor, epochs=1, lr=1e-4, batch_size=batch_size, val_split=0.1, tensorboard_logdir=None):
    """Fine-tune the pre-trained model on QA chat alignment data with train/val split and verbose printing."""
    model.train()
    device = next(model.parameters()).device
    
    # 1. Split data first
    num_samples = qa_tensor.size(0)
    split_idx = int((1.0 - val_split) * num_samples)
    train_tensor = qa_tensor[:split_idx]
    val_tensor = qa_tensor[split_idx:]
    
    # 2. Calculate total steps
    total_steps = epochs * ((train_tensor.size(0) + batch_size - 1) // batch_size)
    
    # 3. Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=100, lr_decay="cosine", total_steps=total_steps)
    
    scaler = GradScaler(enabled=torch.cuda.is_available())
    global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    print(f"Total samples: {num_samples}, Train: {train_tensor.size(0)}, Val: {val_tensor.size(0)}")
    # TensorBoard writer
    writer = None
    if tensorboard_logdir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tensorboard_logdir)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} START")
        # Training
        model.train()
        train_loss = 0.0
        num_train_batches = (train_tensor.size(0) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, train_tensor.size(0), batch_size)):
            batch = train_tensor[i:i+batch_size]
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            with autocast(device_type=device.type):
                logits, loss = model(inputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * inputs.size(0)
            if writer:
                writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
            if batch_idx % 1000 == 0:
                print(f"  Train Batch {batch_idx+1}/{num_train_batches} - Batch Loss: {loss.item():.4f}")
        avg_train_loss = train_loss / train_tensor.size(0)
        print(f"Epoch {epoch+1} TRAINING DONE. Avg Train Loss: {avg_train_loss:.4f}")
        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = (val_tensor.size(0) + batch_size - 1) // batch_size
        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, val_tensor.size(0), batch_size)):
                batch = val_tensor[i:i+batch_size]
                inputs = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)
                with autocast(device_type=device.type):
                    logits, loss = model(inputs, targets)
                val_loss += loss.item() * inputs.size(0)
                if writer:
                    writer.add_scalar('Loss/val', loss.item(), global_step)
                global_step += 1
                if batch_idx % 250 == 0:
                    print(f"  Val Batch {batch_idx+1}/{num_val_batches} - Batch Loss: {loss.item():.4f}")
        avg_val_loss = val_loss / val_tensor.size(0)
        print(f"Epoch {epoch+1} VALIDATION DONE. Avg Val Loss: {avg_val_loss:.4f}")
        # Save best model and implement early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "data/output/chat_aligned_best_model.pt")
            print("Best model saved to data/output/chat_aligned_best_model.pt")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                # Restore best model weights
                print("Restoring best model weights...")
                model.load_state_dict(torch.load("data/output/chat_aligned_best_model.pt"))
                break
    # Save final model
    torch.save(model.state_dict(), "data/output/chat_aligned_model.pt")
    print("Final chat-aligned model saved to data/output/chat_aligned_model.pt")
    if writer:
        writer.close()
        print(f"\033[92mTensorBoard logging complete. View logs with: tensorboard --logdir={tensorboard_logdir}\033[0m")


if __name__ == "__main__":
    # Data paths
    parquet_dir_path = 'data/input/parquet_files'  # Directory containing parquet files
    text_column = 'text'  # Column in the parquet file that contains the text
    vocab_path = os.path.join('data', 'output', 'vocab_subword.json')
    batch_size_files = 1  # Number of parquet files to process in each batch
    
    # Get QA dataset path
    qa_parquet_path = 'data/input/chat-align/train-00000-of-00001.parquet'
    
    # Extract context data for base training
    print("\n=== Extracting context data from QA dataset for base training ===")
    # context_parquet_path = os.path.join(parquet_dir_path, 'context_data.parquet')
    context_parquet_path = os.path.join('data/input/', 'context_data.parquet')
    prepare_context_data_for_training(qa_parquet_path, context_parquet_path, text_column=text_column)
    print(f"Context data extracted to {context_parquet_path}")
    print("This file should be included in base training at the end by moving it to the parquet_files directory.")

    # # Get current time for logging
    training_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Base training - will now include the context data file
    base_train_model(parquet_dir_path, text_column, vocab_path, batch_size_files, training_start_time)

    # Now process QA dataset for fine-tuning
    tokenizer = SubwordTokenizer(vocab_file=vocab_path)
    print("\n=== Creating QA dataset for fine-tuning ===")
    qa_tensor = process_qa_pairs_dataset(
        qa_parquet_path, 
        tokenizer,
        max_length=block_size
    )
    print(f'QA tensor shape: {qa_tensor.shape}')
    
    # Load pre-trained model
    vocab_size = tokenizer.get_vocab_size()
    model = GPTLanguageModel(vocab_size).to(device)
    model.load_state_dict(torch.load('data/output/model_checkpoint.pt', map_location=device))
    print('Pre-trained model loaded.')
    
    # Fine-tune on QA pairs
    print("\n=== Starting fine-tuning on QA pairs ===")
    qa_logdir = f'data/output/tensorboard_logs/{training_start_time}'
    train_chat_alignment(
        model, 
        qa_tensor, 
        epochs=finetuning_max_epochs, 
        lr=1e-4, 
        batch_size=4, 
        val_split=0.1, 
        tensorboard_logdir=qa_logdir
    )
    print("Fine-tuning complete. Model ready for use.")
