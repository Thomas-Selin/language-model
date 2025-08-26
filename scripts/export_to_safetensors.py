import os
import json
import torch
import shutil
from safetensors.torch import save_model
from src.language_model.gpt import GPTLanguageModel
from datetime import datetime
from src.language_model.config import BLOCK_SIZE, TRAINING_START_TIME

CURRENT_MODEL_FOLDER = "20250810-095241"

def export_model_as_safetensors():
     # Create export directory structure with timestamp
     current_time = datetime.now()
     export_path = f"data/output/hf_model_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}"
     os.makedirs(export_path, exist_ok=True)
     
     # Load vocabulary size from the saved file
     with open(os.path.join('data', 'output', CURRENT_MODEL_FOLDER, 'vocab_subword.json'), 'r', encoding='utf-8') as f:
          vocab_data = json.load(f)
     vocab_size = len(vocab_data["model"]["vocab"])  # Access the nested vocabulary dictionary
     print(f"Vocabulary size: {vocab_size}")

     # Load the trained model
     model = GPTLanguageModel(vocab_size=vocab_size)
     if torch.cuda.is_available():
          device = torch.device('cuda')
     elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
          device = torch.device('mps')
     else:
          device = torch.device('cpu')
     model.load_state_dict(torch.load(f"data/output/{CURRENT_MODEL_FOLDER}/chat_aligned_model.pt", map_location=device))
     model.to(device)
     model.eval()
     
     # Extract model configuration from the model instance
     n_embd = model.token_embedding_table.weight.shape[1]
     n_layer = len(model.blocks)
     # Alternative ways to get n_head
     n_head = len(model.blocks[0].sa.heads)  # If sa.heads is a ModuleList
     
     # Convert to safetensors format
     save_model(model, os.path.join(export_path, "model.safetensors"))
     print(f"Model saved as safetensors to {os.path.join(export_path, 'model.safetensors')}")
     
     # Create a config.json file for HF compatibility with extracted values
     config = {
          "architectures": ["GPTLanguageModel"],
          "model_type": "gpt2",
          "vocab_size": vocab_size,
          "hidden_size": n_embd,
          "num_hidden_layers": n_layer,
          "num_attention_heads": n_head,
          "intermediate_size": 4 * n_embd,  # Standard size for feedforward layer
          "hidden_act": "relu",  # You might want to extract this too if variable
          "max_position_embeddings": BLOCK_SIZE,
          "initializer_range": 0.02,
          "layer_norm_epsilon": 1e-5,
          "use_cache": True,
          "pad_token_id": 0,
          "bos_token_id": 0,
          "eos_token_id": 0
     }
     
     with open(os.path.join(export_path, "config.json"), "w") as f:
          json.dump(config, f, indent=2)
     
     # Load the vocabulary from the saved file
     with open(os.path.join('data', 'output', CURRENT_MODEL_FOLDER, 'vocab_subword.json'), 'r', encoding='utf-8') as f:
          vocab_data = json.load(f)

     # Save vocabulary for tokenizer
     with open(os.path.join(export_path, "vocab_subword.json"), "w") as f:
          json.dump(vocab_data, f)
     
     # Create tokenizer files
     tokenizer_config = {
          "model_type": "gpt2",
          "tokenizer_class": "SubwordTokenizer",
          "vocab_file": "vocab_subword.json"
     }
     
     with open(os.path.join(export_path, "tokenizer_config.json"), "w") as f:
          json.dump(tokenizer_config, f, indent=2)
     
     # Save the tokenizer implementation
     shutil.copy("src/language_model/subword_tokenizer.py", os.path.join(export_path, "tokenizer.py"))

     # Create generation_config.json
     generation_config = {
          "max_length": 1000,
          "temperature": 0.7,
          "top_p": 0.9,
          "do_sample": True
     }
     
     with open(os.path.join(export_path, "generation_config.json"), "w") as f:
          json.dump(generation_config, f, indent=2)
     
     # Copy the model implementation for reference
     shutil.copy("src/language_model/gpt.py", os.path.join(export_path, "model.py"))
     print(f"Model exported to {export_path} directory with safetensors format")

if __name__ == "__main__":
       export_model_as_safetensors()
