import torch

def get_device():
    if torch.cuda.is_available():
        print("CUDA GPU will be used.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Apple Metal GPU will be used.")
        return torch.device("mps")
    else:
        print("No GPU available, CPU will be used.")
        return torch.device("cpu")

def print_gpu_memory_summary():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        free = total - reserved
        print(f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free, {total:.2f}GB total")
    else:
        print("No CUDA GPU available.")
def print_memory_usage():
    """Print percentage of RAM and GPU memory used."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        total_mem = psutil.virtual_memory().total
        used_mem = psutil.virtual_memory().used
        ram_percent = used_mem / total_mem * 100
        print(f"RAM usage: {ram_percent:.2f}% of system memory")
    except ImportError:
        print("Could not import psutil. Install with 'pip install psutil' to monitor memory usage.")
        return
    except Exception as e:
        print(f"Error checking RAM usage: {e}")
        return

    try:
        import torch
        if torch.cuda.is_available():
            total_gpu = torch.cuda.get_device_properties(0).total_memory
            allocated_gpu = torch.cuda.memory_allocated()
            gpu_percent = allocated_gpu / total_gpu * 100
            print(f"GPU usage: {gpu_percent:.2f}% of GPU memory")
        else:
            print("No CUDA GPU available.")
    except ImportError:
        print("Could not import torch. Install with 'pip install torch' to monitor GPU usage.")
    except Exception as e:
        print(f"Error checking GPU usage: {e}")


def wait_for_keypress():
    """Wait for user to press Enter before continuing"""
    print("\n\033[93m=========================================\033[0m")
    print("\033[93mBatch processing complete. Time to upload the next batch of files.\033[0m")
    print("\033[93mPress Enter when you've uploaded the next batch and are ready to continue...\033[0m")
    print("\033[93m=========================================\033[0m\n")
    input()  # Wait for Enter key
    print("Continuing with next batch...")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())