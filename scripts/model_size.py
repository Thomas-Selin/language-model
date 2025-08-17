import torch
import os

def print_model_size(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    total_bytes = 0
    print(f"Model checkpoint: {model_path}")
    print("Tensor shapes:")
    for k, v in checkpoint.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
            total_bytes += v.element_size() * v.nelement()
    print(f"Total parameter size: {total_bytes / (1024 ** 2):.2f} MB ({total_bytes} bytes)")

if __name__ == "__main__":
    model_path = "data/output/20250810-095241/best_model.pt"
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
    else:
        print_model_size(model_path)