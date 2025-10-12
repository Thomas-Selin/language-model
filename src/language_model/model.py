from language_model.config import N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint
from typing import Tuple, Optional, List

n_embd = N_EMBD
n_head = N_HEAD
n_layer = N_LAYER
block_size = BLOCK_SIZE
dropout = DROPOUT

class Head(nn.Module):
    """Single head of self-attention mechanism.
    
    Implements scaled dot-product attention with causal masking for
    autoregressive language modeling.
    
    Args:
        head_size: Dimension of the attention head.
    """
    
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of attention head.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            If return_attention is False:
                Output tensor of shape (batch_size, sequence_length, head_size)
            If return_attention is True:
                Tuple of (output tensor, attention weights)
        """
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Compute attention scores with scaled dot-product
        attention_weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)
        
        # Apply causal mask (prevent attending to future tokens)
        attention_weights = attention_weights.masked_fill(
            self.tril[:T, :T] == 0, float('-inf')
        )
        
        # Apply softmax and dropout
        attn = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attn)
        
        # Apply attention to values
        v = self.value(x)  # (B, T, head_size)
        out = attention_weights @ v  # (B, T, head_size)
        
        if return_attention:
            return out, attn
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention computed in parallel.
    
    Implements multi-head attention mechanism where multiple attention heads
    attend to different representation subspaces.
    
    Args:
        num_heads: Number of attention heads to use
        head_size: Dimension of each attention head
    """
    
    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            return_attention: Whether to return attention weights from all heads
            
        Returns:
            If return_attention is False:
                Output tensor of shape (batch_size, sequence_length, embedding_dim)
            If return_attention is True:
                Tuple of (output tensor, list of attention weights per head)
        """
        if return_attention:
            outs, attns = zip(*(h(x, return_attention=True) for h in self.heads))
            out = torch.cat(outs, dim=-1)
            return self.dropout(self.proj(out)), attns
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    """Position-wise feed-forward network.
    
    Two-layer MLP with ReLU activation and dropout, applied independently
    to each position in the sequence. Expands dimension by factor of 4
    in the hidden layer following the Transformer architecture.
    
    Args:
        n_embd: Embedding dimension
    """
    
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        from language_model.constants import FEEDFORWARD_EXPANSION_FACTOR
        
        hidden_dim = FEEDFORWARD_EXPANSION_FACTOR * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            
        Returns:
            Output tensor of same shape as input
        """
        return self.net(x)

class Block(nn.Module):
    """Transformer decoder block.
    
    Implements a single transformer block with pre-normalization:
    1. Multi-head self-attention with residual connection
    2. Feed-forward network with residual connection
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
    """
    
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            If return_attention is False:
                Output tensor of same shape as input
            If return_attention is True:
                Tuple of (output tensor, attention weights)
        """
        if return_attention:
            sa_out, attns = self.self_attention(self.ln1(x), return_attention=True)
            x = x + sa_out
            x = x + self.feed_forward(self.ln2(x))
            return x, attns
        else:
            x = x + self.self_attention(self.ln1(x))
            x = x + self.feed_forward(self.ln2(x))
            return x

class GPTLanguageModel(nn.Module):
    """GPT-style decoder-only transformer language model.
    
    Implements a generative pre-trained transformer with:
    - Token and position embeddings
    - Stack of transformer decoder blocks
    - Language modeling head tied to token embeddings
    
    Args:
        vocab_size: Size of the vocabulary
        use_checkpoint: Whether to use gradient checkpointing to save memory
    """
    
    def __init__(self, vocab_size: int, use_checkpoint: bool = True) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Tie weights between token embeddings and output layer
        self.lm_head.weight = self.token_embedding_table.weight
        
        self.apply(self._init_weights)
        self.use_checkpoint = use_checkpoint

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with normal distribution.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List]]:
        """Forward pass of the language model.
        
        Args:
            idx: Input token indices of shape (batch_size, sequence_length)
            targets: Target token indices for computing loss, same shape as idx
            return_attention: Whether to return attention weights from all layers
            
        Returns:
            If return_attention is False:
                Tuple of (logits, loss)
            If return_attention is True:
                Tuple of (logits, loss, attention_weights)
            
            Where:
                logits: Token predictions of shape (batch_size, sequence_length, vocab_size)
                loss: Cross-entropy loss (None if targets not provided)
                attention_weights: List of attention weights per layer (only if return_attention=True)
        """
        B, T = idx.shape
        
        # Generate embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Pass through transformer blocks
        attentions = [] if return_attention else None
        for block in self.blocks:
            if self.use_checkpoint:
                if return_attention:
                    x, attn = torch.utils.checkpoint.checkpoint(
                        block, x, return_attention, use_reentrant=False
                    )
                    attentions.append(attn)
                else:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                if return_attention:
                    x, attn = block(x, return_attention)
                    attentions.append(attn)
                else:
                    x = block(x)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        if return_attention:
            return logits, loss, attentions
        return logits, loss
    
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int = 1024, 
        temperature: float = 0.8, 
        top_p: float = 0.9, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """Generate new tokens autoregressively from input sequence.
        
        Uses nucleus (top-p) sampling for generation with temperature scaling.
        Stops early if end-of-sequence token (id=2) is generated.
        
        Args:
            idx: Input token indices of shape (batch_size, sequence_length)
            max_new_tokens: Maximum number of tokens to generate (default: 1024)
            temperature: Sampling temperature, higher = more random (default: 0.8)
            top_p: Nucleus sampling threshold, only sample from top p cumulative
                   probability mass (default: 0.9)
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            If return_attention is False:
                Generated token indices of shape (batch_size, sequence_length + num_generated)
            If return_attention is True:
                Tuple of (generated indices, list of attention weights per step)
        """
        # Ensure input tokens are on the same device as model parameters to avoid device mismatch
        model_device = next(self.parameters()).device
        idx = idx.to(model_device)

        # Disable checkpointing for generation to avoid unnecessary recomputation
        use_ckpt_original = self.use_checkpoint
        self.use_checkpoint = False

        all_attentions = [] if return_attention else None

        for _ in range(max_new_tokens):
            # Crop to last block_size tokens for efficiency
            idx_conditioned = idx[:, -block_size:]
            
            # Forward pass
            if return_attention:
                logits, _, attentions = self(idx_conditioned, return_attention=True)
                if len(logits.shape) != 3:
                    raise ValueError(
                        f"Expected logits to have shape (B, T, C) but got {logits.shape}"
                    )
            else:
                logits, _ = self(idx_conditioned)
            
            # Focus on last position and apply temperature
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Nucleus (top-p) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Create mask for tokens within top-p cumulative probability
            sorted_mask = cumulative_probs <= top_p
            sorted_mask[..., 0] = True  # Always keep at least one token
            
            # Apply mask and renormalize
            masked_probs = torch.zeros_like(probs)
            for b in range(probs.size(0)):
                keep_indices = sorted_indices[b][sorted_mask[b]]
                masked_probs[b, keep_indices] = probs[b, keep_indices]
            
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            
            # Sample next token
            next_token = torch.multinomial(masked_probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            
            if return_attention:
                all_attentions.append(
                    [[a.detach().cpu() for a in layer] for layer in attentions]
                )
            
            # Check for end-of-sequence token (id 2)
            if (next_token == 2).any():
                break

        # Restore checkpointing flag
        self.use_checkpoint = use_ckpt_original
        return (idx, all_attentions) if return_attention else idx
