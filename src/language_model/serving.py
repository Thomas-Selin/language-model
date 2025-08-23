from safetensors import safe_open
from gpt import GPTLanguageModel
import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from subword_tokenizer import SubwordTokenizer

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
    if tokenizer_type == 'subword':
        vocab_path = os.path.join('data/output/', 'vocab_subword.json')
        return SubwordTokenizer(vocab_file=vocab_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

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

def generate_text(prompt, max_new_tokens=200, temperature=0.8, tokenizer_type='subword', model=None, tokenizer=None, device=None):
    if model is None or tokenizer is None or device is None:
        latest_model = find_latest_model()
        tokenizer = load_tokenizer(tokenizer_type, latest_model)
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        model = GPTLanguageModel(vocab_size=tokenizer.get_vocab_size())
        with safe_open(f'{latest_model}/model.safetensors', framework='pt') as f:
            for k in f.keys():
                model.state_dict()[k].copy_(f.get_tensor(k))
        model = model.to(device)
        if device.type == 'cuda':
            model = model.half()  # Use half-precision on GPU
        model.eval()
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output, all_attentions = model.generate(
            input_ids,
            # max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_attention=True,
            # eos_token_id=eos_token_id
        )
        generated = tokenizer.decode(output[0].tolist())
        return generated, all_attentions, tokenizer