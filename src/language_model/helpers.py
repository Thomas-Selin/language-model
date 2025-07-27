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
        # Get memory usage in GB (1GB = 1024MB = 1024*1024*1024 bytes)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        free = total - reserved
        print(f"GPU memory summary: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free, {total:.2f}GB total, {max_allocated:.2f}GB max allocated, {max_reserved:.2f}GB max reserved")
    else:
        print("No CUDA GPU available.")

def print_memory_usage():
    """Print current memory usage of the Python process as a percentage of system memory"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        total_mem = psutil.virtual_memory().total
        rss_percent = memory_info.rss / total_mem * 100
        vms_percent = memory_info.vms / total_mem * 100
        print(f"Memory usage: RSS = {memory_info.rss / (1024 * 1024):.2f} MB ({rss_percent:.2f}%), VMS = {memory_info.vms / (1024 * 1024):.2f} MB ({vms_percent:.2f}%) of system memory")
    except ImportError:
        print("Could not import psutil. Install with 'pip install psutil' to monitor memory usage.")
    except Exception as e:
        print(f"Error checking memory usage: {e}")


def wait_for_keypress():
    """Wait for user to press Enter before continuing"""
    print("\n\033[93m=========================================\033[0m")
    print("\033[93mBatch processing complete. Time to upload the next batch of files.\033[0m")
    print("\033[93mPress Enter when you've uploaded the next batch and are ready to continue...\033[0m")
    print("\033[93m=========================================\033[0m\n")
    input()  # Wait for Enter key
    print("Continuing with next batch...")
