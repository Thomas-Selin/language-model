import torch
import torch.nn as nn
from torch.nn import functional as F
from char_tokenizer import CharTokenizer
from word_tokenizer import WordTokenizer
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/input/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Save charset before instantiating tokenizer
chars = sorted(list(set(text)))
chars_json_path = os.path.join('data', 'output', 'chars.json')
CharTokenizer.save_charset(chars, path=chars_json_path)

# instantiate tokenizer
tokenizer = CharTokenizer(chars_file=chars_json_path)
vocab_size = len(tokenizer.character_set)

# Alternatively, for word-level tokenization:
vocab = WordTokenizer.build_vocab(text, vocab_size=10000)
WordTokenizer.save_vocab(vocab, path=os.path.join('data', 'output', 'vocab.json'))
tokenizer = WordTokenizer(vocab_file=os.path.join('data', 'output', 'vocab.json'))
vocab_size = len(vocab)


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 400
eval_interval = 10
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

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

# Create TensorBoard writer
log_dir = os.path.join('data', 'output', 'tensorboard_logs', 
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs will be saved to {log_dir}")

torch.manual_seed(1337)

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
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
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
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

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
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
    
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log losses to TensorBoard
            writer.add_scalar('Loss/train', losses['train'], iter)
            writer.add_scalar('Loss/val', losses['val'], iter)
            
            # Log histograms of model parameters
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, iter)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, iter)
                    
            # Optional: Generate and log a sample text every few iterations
            if iter % (eval_interval * 5) == 0 or iter == max_iters - 1:
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                sample_text = tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist())
                writer.add_text('Generated Text', sample_text, iter)
        
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "data/output/model_checkpoint.pt")
    print("Model saved to data/output/model_checkpoint.pt")
    
    # Close the TensorBoard writer
    writer.close()
    print(f"TensorBoard logging complete. View logs with: tensorboard --logdir={log_dir}")
