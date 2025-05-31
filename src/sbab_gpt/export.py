import os
import json
import torch
import shutil
from safetensors.torch import save_file
from gpt import GPTLanguageModel, block_size, vocab_size, chars
from datetime import datetime

def export_model_as_safetensors():
     # Create export directory structure with timestamp
     current_time = datetime.now()
     export_path =  f"data/output/hf_model_{current_time.day}_{current_time.hour}_{current_time.minute}"
     os.makedirs(export_path, exist_ok=True)
     
     # Load the trained model
     model = GPTLanguageModel()
     model.load_state_dict(torch.load("model_checkpoint.pt"))
     model.eval()
     
     # Convert to safetensors format
     state_dict = model.state_dict()
     save_file(state_dict, os.path.join(export_path, "model.safetensors"))
     print(f"Model saved as safetensors to {os.path.join(export_path, 'model.safetensors')}")
     
     # Create a config.json file for HF compatibility
     config = {
          "architectures": ["GPTLanguageModel"],
          "model_type": "gpt2",
          "vocab_size": vocab_size,
          "hidden_size": 384,  # n_embd
          "num_hidden_layers": 6,  # n_layer
          "num_attention_heads": 6,  # n_head
          "intermediate_size": 1536,  # 4 * n_embd (feedforward size)
          "hidden_act": "relu",  # Your model uses ReLU
          "max_position_embeddings": block_size,
          "initializer_range": 0.02,
          "layer_norm_epsilon": 1e-5,
          "use_cache": True,
          "pad_token_id": 0,
          "bos_token_id": 0,
          "eos_token_id": 0
     }
     
     with open(os.path.join(export_path, "config.json"), "w") as f:
          json.dump(config, f, indent=2)
     
     # Save character set for tokenizer
     with open(os.path.join(export_path, "chars.json"), "w") as f:
          json.dump(chars, f)
     
     # Create tokenizer files
     tokenizer_config = {
          "model_type": "gpt2",
          "tokenizer_class": "CharTokenizer",
          "char_file": "chars.json"
     }
     
     with open(os.path.join(export_path, "tokenizer_config.json"), "w") as f:
          json.dump(tokenizer_config, f, indent=2)
     
     # Save the tokenizer implementation
     shutil.copy("src/sbab_gpt/char_tokenizer.py", os.path.join(export_path, "tokenizer.py"))
     
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
     shutil.copy("gpt.py", os.path.join(export_path, "model.py"))
     
     print(f"Model exported to {export_path} directory with safetensors format")
     print("To load this model:")
     print("  from transformers import AutoConfig")
     print("  from safetensors import safe_open")
     print("  from model import GPTLanguageModel")
     print(f"  config = AutoConfig.from_pretrained({export_path})")
     print("  model = GPTLanguageModel()")
     print("  # Load safetensors weights")
     print(f"  with safe_open({export_path}/model.safetensors', framework='pt') as f:")
     print("      for k in f.keys():")
     print("          model.state_dict()[k].copy_(f.get_tensor(k))")

if __name__ == "__main__":
     try:
          import safetensors
     except ImportError:
          print("safetensors package not found. Installing...")
          import subprocess
          subprocess.check_call(["pip", "install", "safetensors"])
          print("safetensors installed successfully.")


     export_model_as_safetensors()
