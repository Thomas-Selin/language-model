# Load the model manually
from transformers import AutoConfig
from safetensors import safe_open
from gpt import GPTLanguageModel  # Your original model implementation
from gpt import decode
import torch
import glob
import os

# Find the latest model file based on timestamp in the filename
def find_latest_model():
    # The pattern should be a single argument to glob.glob
    model_files = glob.glob(os.path.join('data', 'output', 'hf_model_*_*_*'))
    
    if not model_files:
        raise FileNotFoundError("No model files found with timestamp pattern in data/output/")
    
    # Sort files by timestamp components (day, hour, minute)
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

# First find the latest model
latest_model = find_latest_model()

# Load configuration from local path
config = AutoConfig.from_pretrained(latest_model, local_files_only=True)

# Initialize model
model = GPTLanguageModel()

# Load safetensors weights
with safe_open(f'{latest_model}/model.safetensors', framework='pt') as f:
    for k in f.keys():
        model.state_dict()[k].copy_(f.get_tensor(k))

# Move model to the correct device
model = model.to(device)

# Use model for inference
model.eval()

# Generate text from the model
print("Starting text generation...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=500)
print(decode(output[0].tolist()))
