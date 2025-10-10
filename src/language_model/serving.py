from gpt import GPTLanguageModel
import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from subword_tokenizer import SubwordTokenizer
from helpers import get_device

# Find the latest chat aligned model (.pt file)
def find_latest_model(model_type="chat"):
    model_files = []
    # Get the project root directory (two levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dirs = glob.glob(os.path.join(project_root, 'data', 'output', '*'))
    
    for output_dir in output_dirs:
        if os.path.isdir(output_dir):
            if model_type == "chat":
                model_file = os.path.join(output_dir, 'chat_aligned_model.pt')
            else:  # pre-trained
                model_file = os.path.join(output_dir, 'best_model_resized_vocab_12856.pt')
            
            if os.path.exists(model_file):
                model_files.append(model_file)
    
    if not model_files:
        model_name = "chat_aligned_model.pt" if model_type == "chat" else "model.pt"
        raise FileNotFoundError(f"No {model_name} files found in data/output/")
    
    # Sort files by modification time (most recent first)
    latest_file = max(model_files, key=os.path.getmtime)
    
    model_description = "chat aligned" if model_type == "chat" else "pre-trained"
    print(f"Loading latest {model_description} model: {latest_file}")
    return latest_file

# Device selection - prioritize CUDA, then Metal, fall back to CPU
device = get_device()

def load_tokenizer(tokenizer_type, model_dir):
    if tokenizer_type == 'subword':
        # Get the project root directory (two levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        vocab_path = os.path.join(project_root, 'data', 'output', 'vocab_subword.json')
        return SubwordTokenizer(vocab_file=vocab_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

# Add model quantization helper
def quantize_model(model, device):
    """Apply quantization for better energy efficiency"""
    if device.type == 'cuda':
        # Use half precision on GPU
        return model.half()
    elif device.type == 'cpu':
        # Use dynamic quantization on CPU
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    else:  # MPS
        # MPS supports float16
        return model.half()

def aggregate_tokens_to_words(tokens, attention_matrix):
    """
    Aggregate subword tokens into words and combine their attention weights.
    Returns word-level tokens and aggregated attention matrix.
    """
    import re
    
    # Group tokens into words (tokens starting with special chars like Ġ or ## are continuations)
    words = []
    word_indices = []  # Maps word index to list of token indices
    current_word = ""
    current_indices = []
    
    for i, token in enumerate(tokens):
        # Clean token for display (remove BPE markers)
        clean_token = token.replace('Ġ', ' ').replace('##', '').strip()
        
        # If token starts with space or is the first token, start new word
        if token.startswith('Ġ') or token.startswith(' ') or i == 0:
            if current_word:  # Save previous word
                words.append(current_word.strip())
                word_indices.append(current_indices)
            current_word = clean_token
            current_indices = [i]
        else:
            # Continue current word
            current_word += clean_token
            current_indices.append(i)
    
    # Don't forget the last word
    if current_word:
        words.append(current_word.strip())
        word_indices.append(current_indices)
    
    # Aggregate attention weights by summing over subword tokens within each word
    word_attention = np.zeros((len(words), len(words)))
    
    for i, from_indices in enumerate(word_indices):
        for j, to_indices in enumerate(word_indices):
            # Sum attention from all tokens in word i to all tokens in word j
            attention_sum = 0
            for from_idx in from_indices:
                for to_idx in to_indices:
                    if from_idx < attention_matrix.shape[0] and to_idx < attention_matrix.shape[1]:
                        attention_sum += attention_matrix[from_idx, to_idx]
            # Average by number of token pairs to normalize
            word_attention[i, j] = attention_sum / (len(from_indices) * len(to_indices))
    
    return words, word_attention

def visualize_input_attention(prompt, generated_text, all_attentions, tokenizer, step_idx=-1, layer_idx=0, head_idx=0):
    """
    Visualize attention only for the input prompt, aggregated by words.
    Shows what parts of the input the model focused on during generation.
    """
    # Get input tokens
    input_tokens = tokenizer.encode(prompt)
    input_length = len(input_tokens)
    
    if input_length == 0:
        return None
    
    # Get attention for the last generation step
    step_attention = all_attentions[step_idx]
    layer_attention = step_attention[layer_idx]
    attention_tensor = layer_attention[head_idx]
    
    if len(attention_tensor.shape) == 3 and attention_tensor.shape[0] == 1:
        head_attention = attention_tensor.squeeze(0).cpu().numpy()
    else:
        head_attention = attention_tensor.cpu().numpy()
    
    # Only look at attention TO the input tokens (columns)
    # and FROM the last generated token (last row)
    if head_attention.shape[0] > input_length:
        input_attention = head_attention[-1, :input_length]  # Last token's attention to input
    else:
        input_attention = head_attention[:input_length, :input_length]
        input_attention = np.mean(input_attention, axis=0)  # Average attention to each input token
    
    # Convert tokens to words and aggregate attention
    input_token_strings = [tokenizer.decode([t]) for t in input_tokens]
    
    # Simple word aggregation for input attention (1D)
    words = []
    word_attentions = []
    current_word = ""
    current_attention = 0
    token_count = 0
    
    for i, token in enumerate(input_token_strings):
        clean_token = token.replace('Ġ', ' ').replace('##', '').strip()
        
        if token.startswith('Ġ') or token.startswith(' ') or i == 0:
            if current_word and token_count > 0:  # Save previous word
                words.append(current_word.strip())
                word_attentions.append(current_attention / token_count)
            current_word = clean_token
            current_attention = input_attention[i] if i < len(input_attention) else 0
            token_count = 1
        else:
            current_word += clean_token
            current_attention += input_attention[i] if i < len(input_attention) else 0
            token_count += 1
    
    # Don't forget the last word
    if current_word and token_count > 0:
        words.append(current_word.strip())
        word_attentions.append(current_attention / token_count)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(words)), word_attentions, color='viridis')
    
    # Color bars by attention strength
    max_attention = max(word_attentions) if word_attentions else 1
    for i, (bar, attention) in enumerate(zip(bars, word_attentions)):
        bar.set_color(plt.cm.viridis(attention / max_attention))
    
    ax.set_xlabel('Input Words')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'Input Attention - Layer {layer_idx}, Head {head_idx}\n(What the model focused on in your prompt)')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (word, attention) in enumerate(zip(words, word_attentions)):
        ax.text(i, attention + max_attention * 0.01, f'{attention:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def visualize_combined_input_attention(prompt, generated_text, all_attentions, tokenizer, step_idx=-1, aggregation='mean'):
    """
    Visualize combined attention to input across all layers and heads, aggregated by words.
    """
    input_tokens = tokenizer.encode(prompt)
    input_length = len(input_tokens)
    
    if input_length == 0:
        return None
    
    step_attention = all_attentions[step_idx]
    all_input_attentions = []
    
    # Collect attention to input from all layers and heads
    for layer_idx in range(len(step_attention)):
        for head_idx in range(len(step_attention[layer_idx])):
            attention_tensor = step_attention[layer_idx][head_idx]
            if len(attention_tensor.shape) == 3 and attention_tensor.shape[0] == 1:
                attention_matrix = attention_tensor.squeeze(0).cpu().numpy()
            else:
                attention_matrix = attention_tensor.cpu().numpy()
            
            # Get attention to input tokens from the last generated token
            if attention_matrix.shape[0] > input_length:
                input_attention = attention_matrix[-1, :input_length]
            else:
                input_attention = attention_matrix[:input_length, :input_length]
                input_attention = np.mean(input_attention, axis=0)
            
            all_input_attentions.append(input_attention[:input_length])
    
    # Aggregate across all heads and layers
    if aggregation == 'mean':
        combined_input_attention = np.mean(all_input_attentions, axis=0)
    elif aggregation == 'max':
        combined_input_attention = np.max(all_input_attentions, axis=0)
    elif aggregation == 'sum':
        combined_input_attention = np.sum(all_input_attentions, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Convert to words and aggregate
    input_token_strings = [tokenizer.decode([t]) for t in input_tokens]
    words = []
    word_attentions = []
    current_word = ""
    current_attention = 0
    token_count = 0
    
    for i, token in enumerate(input_token_strings):
        clean_token = token.replace('Ġ', ' ').replace('##', '').strip()
        
        if token.startswith('Ġ') or token.startswith(' ') or i == 0:
            if current_word and token_count > 0:
                words.append(current_word.strip())
                word_attentions.append(current_attention / token_count)
            current_word = clean_token
            current_attention = combined_input_attention[i] if i < len(combined_input_attention) else 0
            token_count = 1
        else:
            current_word += clean_token
            current_attention += combined_input_attention[i] if i < len(combined_input_attention) else 0
            token_count += 1
    
    if current_word and token_count > 0:
        words.append(current_word.strip())
        word_attentions.append(current_attention / token_count)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(range(len(words)), word_attentions)
    
    # Color bars by attention strength
    max_attention = max(word_attentions) if word_attentions else 1
    for i, (bar, attention) in enumerate(zip(bars, word_attentions)):
        bar.set_color(plt.cm.plasma(attention / max_attention))
    
    ax.set_xlabel('Input Words')
    ax.set_ylabel(f'{aggregation.capitalize()} Attention Weight')
    ax.set_title(f'Combined Input Attention ({aggregation})\n(Overall focus on each word in your prompt)')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (word, attention) in enumerate(zip(words, word_attentions)):
        ax.text(i, attention + max_attention * 0.01, f'{attention:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def visualize_word_to_word_attention(generated_text, all_attentions, tokenizer, step_idx=-1, layer_idx=0, head_idx=0):
    """
    Visualize word-to-word attention to show relationships like 'it' -> 'bear'.
    Aggregates subword tokens into words and shows attention between words.
    """
    # Get all tokens for the generated text
    all_tokens = tokenizer.encode(generated_text)
    max_tokens = min(len(all_tokens), 32)  # Limit for readability
    tokens = [tokenizer.decode([t]) for t in all_tokens[:max_tokens]]
    
    # Get attention matrix
    step_attention = all_attentions[step_idx]
    layer_attention = step_attention[layer_idx]
    attention_tensor = layer_attention[head_idx]
    
    if len(attention_tensor.shape) == 3 and attention_tensor.shape[0] == 1:
        head_attention = attention_tensor.squeeze(0).cpu().numpy()
    else:
        head_attention = attention_tensor.cpu().numpy()
    
    attention_size = min(head_attention.shape[0], len(tokens))
    tokens = tokens[:attention_size]
    head_attention = head_attention[:attention_size, :attention_size]
    
    # Aggregate tokens into words
    words, word_attention = aggregate_tokens_to_words(tokens, head_attention)
    
    # Apply power transformation to increase contrast for small differences
    word_attention_enhanced = np.power(word_attention, 0.7)  # Enhance contrast
    
    # Apply row-wise normalization to make patterns clearer
    row_sums = word_attention_enhanced.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    word_attention_normalized = word_attention_enhanced / row_sums
    
    # Create heatmap with better contrast
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use a more contrasting colormap and add gridlines
    im = ax.imshow(word_attention_normalized, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Add grid for better separation
    ax.set_xticks(np.arange(len(words)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(words)) + 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1, alpha=0.7)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized attention weight", rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(len(words)))
    ax.set_yticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(words, fontsize=10)
    
    ax.set_title(f"Word-to-Word Attention - Layer {layer_idx}, Head {head_idx}\n(Normalized per row for better contrast)", fontsize=12)
    ax.set_xlabel("Attended to (what is being focused on)", fontsize=11)
    ax.set_ylabel("From word (what is doing the attending)", fontsize=11)
    
    # Add attention values as text with black color
    threshold = 0.1  # Show values above 10% of normalized range
    for i in range(word_attention_normalized.shape[0]):
        for j in range(word_attention_normalized.shape[1]):
            value = word_attention_normalized[i, j]
            original_value = word_attention[i, j]
            if value > threshold:
                # Use black text for all values
                ax.text(j, i, f"{original_value:.3f}",
                       ha="center", va="center", 
                       color="black",
                       fontsize=8, weight='bold')
    
    plt.tight_layout()
    return fig

def visualize_combined_word_attention(generated_text, all_attentions, tokenizer, step_idx=-1, aggregation='mean'):
    """
    Combined word-to-word attention across all layers and heads with enhanced contrast.
    """
    all_tokens = tokenizer.encode(generated_text)
    max_tokens = min(len(all_tokens), 32)
    tokens = [tokenizer.decode([t]) for t in all_tokens[:max_tokens]]
    
    step_attention = all_attentions[step_idx]
    all_matrices = []
    
    # Collect attention from all layers and heads
    for layer_idx in range(len(step_attention)):
        for head_idx in range(len(step_attention[layer_idx])):
            attention_tensor = step_attention[layer_idx][head_idx]
            if len(attention_tensor.shape) == 3 and attention_tensor.shape[0] == 1:
                attention_matrix = attention_tensor.squeeze(0).cpu().numpy()
            else:
                attention_matrix = attention_tensor.cpu().numpy()
            
            attention_size = min(attention_matrix.shape[0], len(tokens))
            tokens_subset = tokens[:attention_size]
            attention_matrix = attention_matrix[:attention_size, :attention_size]
            
            # Convert to word-level
            words, word_attention = aggregate_tokens_to_words(tokens_subset, attention_matrix)
            all_matrices.append(word_attention)
    
    # Aggregate across all heads and layers
    if not all_matrices:
        return None
        
    # Ensure all matrices have the same size
    sizes = [m.shape[0] for m in all_matrices]
    most_common_size = max(set(sizes), key=sizes.count)
    filtered_matrices = [m for m in all_matrices if m.shape[0] == most_common_size]
    
    if not filtered_matrices:
        return None
    
    if aggregation == 'mean':
        combined_attention = np.mean(filtered_matrices, axis=0)
    elif aggregation == 'max':
        combined_attention = np.max(filtered_matrices, axis=0)
    elif aggregation == 'sum':
        combined_attention = np.sum(filtered_matrices, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Get words for the display
    sample_matrix = filtered_matrices[0]
    tokens_for_size = tokens[:sample_matrix.shape[0]]
    words, _ = aggregate_tokens_to_words(tokens_for_size, sample_matrix)
    
    # Apply logarithmic scaling to enhance small differences
    combined_attention_log = np.log1p(combined_attention * 100)  # log(1 + 100*x) for better scaling
    
    # Row-wise normalization for better patterns
    row_sums = combined_attention_log.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    combined_attention_normalized = combined_attention_log / row_sums
    
    # Create enhanced heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use diverging colormap with better contrast
    im = ax.imshow(combined_attention_normalized, cmap='RdBu_r', aspect='auto')
    
    # Add grid lines
    ax.set_xticks(np.arange(len(words)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(words)) + 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5, alpha=0.8)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f"Log-normalized {aggregation} attention", rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(len(words)))
    ax.set_yticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(words, fontsize=10)
    
    ax.set_title(f"Combined Word Attention ({aggregation}) - Enhanced Contrast\n(Log-scaled and normalized for better visibility)", fontsize=12)
    ax.set_xlabel("Attended to (what is being focused on)", fontsize=11)
    ax.set_ylabel("From word (what is doing the attending)", fontsize=11)
    
    # Add values with black text
    threshold = np.percentile(combined_attention_normalized, 75)  # Top 25% of values
    for i in range(combined_attention_normalized.shape[0]):
        for j in range(combined_attention_normalized.shape[1]):
            norm_value = combined_attention_normalized[i, j]
            orig_value = combined_attention[i, j]
            if norm_value > threshold and orig_value > 0.01:  # Only show significant values
                # Use black text for all values
                ax.text(j, i, f"{orig_value:.3f}",
                       ha="center", va="center", 
                       color="black",
                       fontsize=7, weight='bold')
    
    plt.tight_layout()
    return fig

def visualize_attention(generated_text, all_attentions, tokenizer, step_idx=-1, layer_idx=0, head_idx=0):
    # Limit tokens for visualization
    max_tokens = 64
    tokens = [tokenizer.decode([t]) for t in tokenizer.encode(generated_text)[:max_tokens]]

    step_attention = all_attentions[step_idx]
    layer_attention = step_attention[layer_idx]
    attention_tensor = layer_attention[head_idx]
    if len(attention_tensor.shape) == 3 and attention_tensor.shape[0] == 1:
        head_attention = attention_tensor.squeeze(0).cpu().numpy()
    else:
        head_attention = attention_tensor.cpu().numpy()
    attention_size = min(head_attention.shape[0], len(tokens))
    tokens = tokens[:attention_size]
    head_attention = head_attention[:attention_size, :attention_size]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(head_attention, cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention weight", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f"Attention - Layer {layer_idx}, Head {head_idx}")
    ax.set_xlabel("Attended to")
    ax.set_ylabel("From token")
    for i in range(head_attention.shape[0]):
        for j in range(head_attention.shape[1]):
            if head_attention[i, j] > 0.1:
                text = ax.text(j, i, f"{head_attention[i, j]:.2f}",
                              ha="center", va="center", color="white" if head_attention[i, j] > 0.5 else "black")
    fig.tight_layout()
    return fig

def visualize_combined_attention(generated_text, all_attentions, tokenizer, step_idx=-1, aggregation='mean'):
    max_tokens = 64
    tokens = [tokenizer.decode([t]) for t in tokenizer.encode(generated_text)[:max_tokens]]
    step_attention = all_attentions[step_idx]
    all_matrices = []
    for layer_idx in range(len(step_attention)):
        for head_idx in range(len(step_attention[layer_idx])):
            attention_tensor = step_attention[layer_idx][head_idx]
            if len(attention_tensor.shape) == 3 and attention_tensor.shape[0] == 1:
                attention_matrix = attention_tensor.squeeze(0).cpu().numpy()
            else:
                attention_matrix = attention_tensor.cpu().numpy()
            attention_matrix = attention_matrix[:max_tokens, :max_tokens]
            all_matrices.append(attention_matrix)
    if aggregation == 'mean':
        combined_attention = np.mean(all_matrices, axis=0)
    elif aggregation == 'max':
        combined_attention = np.max(all_matrices, axis=0)
    elif aggregation == 'sum':
        combined_attention = np.sum(all_matrices, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    attention_size = combined_attention.shape[0]
    tokens = tokens[:attention_size]
    combined_attention = combined_attention[:attention_size, :attention_size]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(combined_attention, cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f"{aggregation.capitalize()} attention weight", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f"Combined Attention ({aggregation})")
    ax.set_xlabel("Attended to")
    ax.set_ylabel("From token")
    threshold = 0.1 if aggregation != 'sum' else 0.3
    for i in range(combined_attention.shape[0]):
        for j in range(combined_attention.shape[1]):
            if combined_attention[i, j] > threshold:
                text = ax.text(j, i, f"{combined_attention[i, j]:.2f}",
                              ha="center", va="center", 
                              color="white" if combined_attention[i, j] > threshold*2 else "black")
    fig.tight_layout()
    return fig

def generate_text(prompt, max_new_tokens=50, temperature=0.8, tokenizer_type='subword', 
                 model=None, tokenizer=None, device=None, enable_kv_cache=True):
    # Only load model if not provided (for backward compatibility)
    if model is None or tokenizer is None or device is None:
        print("⚠️  Loading model in generate_text - consider passing pre-loaded model for better performance")
        latest_model = find_latest_model()
        tokenizer = load_tokenizer(tokenizer_type, os.path.dirname(latest_model))
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        model = GPTLanguageModel(vocab_size=tokenizer.get_vocab_size())
        
        # Load the PyTorch model directly
        checkpoint = torch.load(latest_model, map_location=device)
        model.load_state_dict(checkpoint)
        
        model = model.to(device)
        # Apply quantization for energy efficiency
        model = quantize_model(model, device)
        model.eval()
        
        # Enable torch optimizations
        if hasattr(torch, 'compile') and device.type != 'mps':  # torch.compile not supported on MPS yet
            model = torch.compile(model, mode='reduce-overhead')
    else:
        print("✅ Using pre-loaded model for generation")
    
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    
    # Set inference mode and disable gradient computation
    with torch.inference_mode():
        # Limit max tokens to prevent excessive computation and memory usage
        max_new_tokens = min(max_new_tokens, 500)
        
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        # Use autocast for mixed precision on supported devices
        autocast_context = torch.autocast(device.type) if device.type in ['cuda', 'cpu'] else torch.no_grad()
        
        with autocast_context:
            try:
                output, all_attentions = model.generate(
                    input_ids,
                    # max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    return_attention=True,
                    # eos_token_id=eos_token_id,
                    # use_cache=enable_kv_cache  # Enable KV caching if supported
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    cleanup_memory()
                    print(f"❌ OOM Error during generation: {e}")
                    raise RuntimeError("Out of memory during text generation. Try reducing max_new_tokens or using a smaller prompt.")
                else:
                    raise e
        
        # Clean up intermediate tensors
        del input_ids
        cleanup_memory()
        
        generated = tokenizer.decode(output[0].tolist())
        return generated, all_attentions, tokenizer

# Add memory cleanup function
def cleanup_memory():
    """Clean up GPU memory after inference"""
    import gc
    
    # Force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()