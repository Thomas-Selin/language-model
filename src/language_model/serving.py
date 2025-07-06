from safetensors import safe_open
from gpt import GPTLanguageModel
import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from word_tokenizer import WordTokenizer

# Find the latest model file based on timestamp in the filename
def find_latest_model():
    model_files = glob.glob(os.path.join('data', 'output', 'hf_model_*_*_*_*'))
    
    if not model_files:
        raise FileNotFoundError("No model files found with timestamp pattern in data/output/")
    
    # Sort files by timestamp
    latest_file = max(model_files, key=lambda x: 
        [int(n) for n in os.path.basename(x).replace('hf_model_', '').split('_')])
    
    print(f"Loading latest model: {latest_file}")
    return latest_file

# Device selection - prioritize CUDA, then Metal, fall back to CPU
if torch.cuda.is_available():
    print("CUDA GPU is available.")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Apple Metal GPU is available.")
    device = torch.device("mps")
else:
    print("No GPU available, using CPU.")
    device = torch.device("cpu")

def load_tokenizer(tokenizer_type, model_dir):
    if tokenizer_type == 'word':
        vocab_path = os.path.join('data/output/', 'vocab.json')
        return WordTokenizer(vocab_file=vocab_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

def visualize_attention(generated_text, all_attentions, tokenizer, step_idx=-1, layer_idx=0, head_idx=0):
    """
    Visualize attention patterns using matplotlib.
    
    Args:
        generated_text: The full generated text
        all_attentions: Attention values from model.generate
        tokenizer: Tokenizer object for encoding/decoding
        step_idx: Which generation step to visualize (-1 for last step)
        layer_idx: Which transformer layer to visualize
        head_idx: Which attention head to visualize
    """
    # Get tokens for labeling
    tokens = [tokenizer.decode([t]) for t in tokenizer.encode(generated_text)]
    
    # Select attention from specified step, layer, and head
    step_attention = all_attentions[step_idx]
    layer_attention = step_attention[layer_idx]
    
    # Extract the 2D attention matrix, handling different possible shapes
    attention_tensor = layer_attention[head_idx]
    if len(attention_tensor.shape) == 3 and attention_tensor.shape[0] == 1:
        # Handle case where tensor has shape (1, seq_len, seq_len)
        head_attention = attention_tensor.squeeze(0).numpy()
    else:
        # Normal case with shape (seq_len, seq_len)
        head_attention = attention_tensor.numpy()
    
    # Ensure we only use as many tokens as we have in the attention matrix
    attention_size = head_attention.shape[0]
    tokens = tokens[:attention_size]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(head_attention, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention weight", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title and labels
    ax.set_title(f"Attention - Layer {layer_idx}, Head {head_idx}")
    ax.set_xlabel("Attended to")
    ax.set_ylabel("From token")
    
    # Loop over data dimensions and create text annotations for important weights
    for i in range(head_attention.shape[0]):
        for j in range(head_attention.shape[1]):
            if head_attention[i, j] > 0.1:  # Only annotate significant attention weights
                text = ax.text(j, i, f"{head_attention[i, j]:.2f}",
                              ha="center", va="center", color="white" if head_attention[i, j] > 0.5 else "black")
    
    fig.tight_layout()
    return fig

def visualize_combined_attention(generated_text, all_attentions, tokenizer, step_idx=-1, aggregation='mean'):
    """
    Visualize combined attention patterns across all layers and heads.
    
    Args:
        generated_text: The full generated text
        all_attentions: Attention values from model.generate
        tokenizer: Tokenizer object for encoding/decoding
        step_idx: Which generation step to visualize (-1 for last step)
        aggregation: How to combine attentions ('mean', 'max', or 'sum')
    """
    # Get tokens for labeling
    tokens = [tokenizer.decode([t]) for t in tokenizer.encode(generated_text)]
    
    # Select attention from specified step
    step_attention = all_attentions[step_idx]
    
    # Collect all attention matrices
    all_matrices = []
    for layer_idx in range(len(step_attention)):
        for head_idx in range(len(step_attention[layer_idx])):
            attention_tensor = step_attention[layer_idx][head_idx]
            if len(attention_tensor.shape) == 3 and attention_tensor.shape[0] == 1:
                attention_matrix = attention_tensor.squeeze(0).numpy()
            else:
                attention_matrix = attention_tensor.numpy()
            all_matrices.append(attention_matrix)
    
    # Combine attention matrices
    if aggregation == 'mean':
        combined_attention = np.mean(all_matrices, axis=0)
    elif aggregation == 'max':
        combined_attention = np.max(all_matrices, axis=0)
    elif aggregation == 'sum':
        combined_attention = np.sum(all_matrices, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Ensure we only use as many tokens as we have in the attention matrix
    attention_size = combined_attention.shape[0]
    tokens = tokens[:attention_size]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(combined_attention, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f"{aggregation.capitalize()} attention weight", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title and labels
    ax.set_title(f"Combined Attention ({aggregation})")
    ax.set_xlabel("Attended to")
    ax.set_ylabel("From token")
    
    # Highlight important connections
    threshold = 0.1 if aggregation != 'sum' else 0.3
    for i in range(combined_attention.shape[0]):
        for j in range(combined_attention.shape[1]):
            if combined_attention[i, j] > threshold:
                text = ax.text(j, i, f"{combined_attention[i, j]:.2f}",
                              ha="center", va="center", 
                              color="white" if combined_attention[i, j] > threshold*2 else "black")
    
    fig.tight_layout()
    return fig

def generate_text(prompt, max_new_tokens=200, temperature=1.0, tokenizer_type='word'):
    # Find latest model dir
    latest_model = find_latest_model()
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_type, latest_model)
    # Move model to device and eval mode
    model = GPTLanguageModel(vocab_size=tokenizer.get_vocab_size())
    with safe_open(f'{latest_model}/model.safetensors', framework='pt') as f:
        for k in f.keys():
            model.state_dict()[k].copy_(f.get_tensor(k))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output, all_attentions = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, return_attention=True)
        generated = tokenizer.decode(output[0].tolist())
        # Create visualization directory if it doesn't exist
        os.makedirs('data/output/attention_vis', exist_ok=True)
        # Create combined attention plots
        mean_fig = visualize_combined_attention(generated, all_attentions, tokenizer, aggregation='mean')
        max_fig = visualize_combined_attention(generated, all_attentions, tokenizer, aggregation='max')
        # Create per-layer visualizations
        layer_figs = []
        for layer_idx in range(2):  # Assuming 2 layers
            fig = visualize_attention(generated, all_attentions, tokenizer, layer_idx=layer_idx)
            layer_figs.append(fig)
        return generated, mean_fig, max_fig, layer_figs
