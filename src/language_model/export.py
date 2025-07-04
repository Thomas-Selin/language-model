import os
import json
import torch
import shutil
from safetensors.torch import save_file
from gpt import GPTLanguageModel, block_size, vocab_size
from datetime import datetime

def export_model_as_safetensors():
     # Create export directory structure with timestamp
     current_time = datetime.now()
     export_path = f"data/output/hf_model_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}"
     os.makedirs(export_path, exist_ok=True)
     
     # Load the trained model
     model = GPTLanguageModel()
     model.load_state_dict(torch.load("data/output/model_checkpoint.pt"))
     model.eval()
     
     # Extract model configuration from the model instance
     n_embd = model.token_embedding_table.weight.shape[1]
     n_layer = len(model.blocks)
     # Alternative ways to get n_head
     n_head = len(model.blocks[0].sa.heads)  # If sa.heads is a ModuleList
     
     # Convert to safetensors format
     state_dict = model.state_dict()
     save_file(state_dict, os.path.join(export_path, "model.safetensors"))
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
     
     # Save vocabulary for tokenizer
     with open(os.path.join(export_path, "vocab.json"), "w") as f:
          json.dump(vocab, f)
     
     # Create tokenizer files
     tokenizer_config = {
          "model_type": "gpt2",
          "tokenizer_class": "WordTokenizer",
          "vocab_file": "vocab.json"
     }
     
     with open(os.path.join(export_path, "tokenizer_config.json"), "w") as f:
          json.dump(tokenizer_config, f, indent=2)
     
     # Save the tokenizer implementation
     shutil.copy("src/language_model/word_tokenizer.py", os.path.join(export_path, "tokenizer.py"))

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
