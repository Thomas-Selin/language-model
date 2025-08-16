from config import N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint

n_embd = N_EMBD
n_head = N_HEAD
n_layer = N_LAYER
block_size = BLOCK_SIZE
dropout = DROPOUT

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
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(wei, dim=-1)
        wei = self.dropout(attn)
        v = self.value(x)
        out = wei @ v
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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)
        self.use_checkpoint = use_checkpoint

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, return_attention=False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
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
        x = self.ln_f(x)
        logits = self.lm_head(x)
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
    
    def generate(self, idx, temperature=0.8, top_p=0.9, return_attention=False):
        """
        Generate tokens from the model.
        Args:
            idx: input tensor of token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold (float)
            return_attention: whether to return attention weights
        """
        all_attentions = [] if return_attention else None
        for _ in range(1024):
            idx_cond = idx[:, -block_size:]
            if return_attention:
                logits, _, attentions = self(idx_cond, return_attention=True)
                if len(logits.shape) != 3:
                    raise ValueError(f"Expected logits to have shape (B, T, C) but got {logits.shape}")
            else:
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # Nucleus (top-p) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Create mask for tokens to keep
            sorted_mask = cumulative_probs <= top_p
            # Always keep at least one token
            sorted_mask[..., 0] = True
            # Set probabilities of tokens outside top_p to zero
            masked_probs = torch.zeros_like(probs)
            for b in range(probs.size(0)):
                masked_probs[b, sorted_indices[b][sorted_mask[b]]] = probs[b, sorted_indices[b][sorted_mask[b]]]
            # Renormalize
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(masked_probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if return_attention:
                all_attentions.append([[a.detach().cpu() for a in layer] for layer in attentions])
            if (idx_next == 2).any(): # Check if end of sequence token (id 2) is generated
                break
        return (idx, all_attentions) if return_attention else idx
