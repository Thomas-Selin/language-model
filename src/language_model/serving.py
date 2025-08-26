from gpt import GPTLanguageModel
import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from subword_tokenizer import SubwordTokenizer

# Find the latest chat aligned model (.pt file)
def find_latest_model():
    # Look for chat_aligned_model.pt files in output directories
    model_files = []
    # Get the project root directory (two levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dirs = glob.glob(os.path.join(project_root, 'data', 'output', '*'))
    
    for output_dir in output_dirs:
        if os.path.isdir(output_dir):
            chat_model = os.path.join(output_dir, 'chat_aligned_model.pt')
            if os.path.exists(chat_model):
                model_files.append(chat_model)
    
    if not model_files:
        raise FileNotFoundError("No chat_aligned_model.pt files found in data/output/")
    
    # Sort files by modification time (most recent first)
    latest_file = max(model_files, key=os.path.getmtime)
    
    print(f"Loading latest chat aligned model: {latest_file}")
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

def generate_text(prompt, max_new_tokens=200, temperature=0.8, tokenizer_type='subword', 
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