import torch
import torch.nn as nn
from torch.nn import functional as F
from subword_tokenizer import SubwordTokenizer
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from torch.amp import GradScaler, autocast

torch.manual_seed(1337)

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_epochs = 3000 # how many epochs to train for as max, early stoxxpping will stop training if no improvement is seen
eval_interval = 5 # how many steps between evaluations?
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
early_stopping_patience = 20  # Number of epochs to wait for improvement
min_word_frequency = 8 # Minimum frequency for words to be included in the vocabulary
max_vocab_size = 5000 # Maximum vocabulary size

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
print(f"Min word frequency: {min_word_frequency}")
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

    def generate(self, idx, max_new_tokens, temperature=1.0, return_attention=False):
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

def load_and_process_data(file_path, vocab_path):
    """Load and process text data for training"""
    print("Loading dataset...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        print("✅ Dataset loaded successfully.")

    # Change the file extension for subword tokenizer
    tokenizer_path = vocab_path.replace('.json', '_subword.json')
    
    # Only build vocab if it doesn't exist
    if not os.path.exists(tokenizer_path):
        print("Building vocabulary with subword tokenizer...")
        # Subword tokenization:
        tokenizer_obj = SubwordTokenizer.build_vocab(text, vocab_size=max_vocab_size)
        print(f"✅ Vocabulary built successfully. Size: {tokenizer_obj.get_vocab_size()}")
        SubwordTokenizer.save_vocab(tokenizer_obj, path=tokenizer_path)
    else:
        print("Using existing subword vocabulary...")

    # Load tokenizer from saved file
    tokenizer = SubwordTokenizer(vocab_file=tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"✅ Vocabulary loaded. Size: {vocab_size}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}.")
    
    # Process dataset in chunks
    print("Encoding dataset in chunks...")
    chunk_size = 2500000  # Adjust based on your available memory
    tokens = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        tokens.extend(tokenizer.encode(chunk))
        print(f"Processed chunk {i // chunk_size + 1}")
    data = torch.tensor(tokens, dtype=torch.long, device='cpu')
    del tokens  # Free memory
    print(f"✅ Dataset encoded. Length: {len(data)}. Current time is {datetime.datetime.now().strftime('%H:%M:%S')}.")
    
    # Split into train and validation sets
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, tokenizer, vocab_size

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

if __name__ == "__main__":
    # Data paths
    input_file = 'data/input/half_of_TinyStories.txt'
    print(f"Input file: {input_file}. Nr of lines in input file: {sum(1 for _ in open(input_file, 'r', encoding='utf-8'))}. Size of input file in MB: {os.path.getsize(input_file) / (1024 * 1024):.2f} MB")
    vocab_path = os.path.join('data', 'output', 'vocab.json')
    
    # Load and process data for training
    train_data, val_data, tokenizer, vocab_size = load_and_process_data(input_file, vocab_path)
    
    # Create TensorBoard writer
    log_dir = os.path.join('data', 'output', 'tensorboard_logs', 
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    
    # Initialize model with vocabulary size
    model = GPTLanguageModel(vocab_size).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.07)
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
    
    try:
        for iter in range(max_epochs):
            epoch_start_time = time.time()

            # Evaluation logic
            if iter % eval_interval == 0 or iter == max_epochs - 1:
                losses = estimate_loss(model, train_data, val_data)
                print("\033[94m____________________________\033[0m\n")
                print(f"Step {iter} of max {max_epochs}: train loss {losses['train']:.4f}, 📏 val loss \033[94m{losses['val']:.4f}\033[0m")
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
                writer.add_scalar('Loss/train', losses['train'], iter)
                writer.add_scalar('Loss/val', losses['val'], iter)
                
                # Log memory to TensorBoard
                if torch.cuda.is_available():
                    writer.add_scalar('Memory/allocated_GB', torch.cuda.memory_allocated() / 1024**3, iter)
                    writer.add_scalar('Memory/reserved_GB', torch.cuda.memory_reserved() / 1024**3, iter)
                    writer.add_scalar('Memory/free_GB', (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3, iter)
                
                # Log histograms of model parameters
                # Only log histograms occasionally to reduce overhead
                if iter % (eval_interval * 10) == 0:  # Log much less frequently
                    for name, param in model.named_parameters():
                        writer.add_histogram(f'Parameters/{name}', param, iter)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, iter)
                        
                # Optional: Generate and log a sample text every few iterations
                if iter % (eval_interval * 5) == 0 or iter == max_epochs - 1:
                    context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    sample_text = tokenizer.decode(model.generate(context, max_new_tokens=100, temperature=1.0)[0].tolist())
                    writer.add_text('Generated Text 1.0 temp', sample_text, iter)
                    sample_text = tokenizer.decode(model.generate(context, max_new_tokens=100, temperature=0.5)[0].tolist())
                    writer.add_text('Generated Text 0.5 temp', sample_text, iter)
            
            # sample a batch of data
            data_time = time.time()
            xb, yb = get_batch('train', train_data, val_data)
            data_time = time.time() - data_time

            # Measure forward/backward time
            train_time = time.time()
            with autocast(device_type=device.type):  # Uses float16 where appropriate
                logits, loss = model(xb, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_time = time.time() - train_time

            epoch_duration = time.time() - epoch_start_time  # End timing the epoch

            # Print GPU memory usage for each epoch (not just evaluation intervals)
            print(f"Epoch {iter}: Data time: {data_time:.4f}s, Train time: {train_time:.4f}s, Total epoch time: {epoch_duration:.4f}s")
            writer.add_scalar('Total/EpochTime', epoch_duration, iter)
            
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
