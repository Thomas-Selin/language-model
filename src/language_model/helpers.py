import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_device():
    if torch.cuda.is_available():
        logging.info("CUDA GPU will be used.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Apple Metal GPU will be used.")
        return torch.device("mps")
    else:
        logging.info("No GPU available, CPU will be used.")
        return torch.device("cpu")

def print_gpu_memory_summary():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        free = total - reserved
        logging.info(f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free, {total:.2f}GB total")
    else:
        logging.info("No CUDA GPU available.")
def print_memory_usage():
    """Print percentage of RAM and GPU memory used."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        total_mem = psutil.virtual_memory().total
        used_mem = psutil.virtual_memory().used
        ram_percent = used_mem / total_mem * 100
        logging.info(f"RAM usage: {ram_percent:.2f}% of system memory")
    except ImportError:
        logging.info("Could not import psutil. Install with 'pip install psutil' to monitor memory usage.")
        return
    except Exception as e:
        logging.info(f"Error checking RAM usage: {e}")
        return

    try:
        import torch
        if torch.cuda.is_available():
            total_gpu = torch.cuda.get_device_properties(0).total_memory
            allocated_gpu = torch.cuda.memory_allocated()
            gpu_percent = allocated_gpu / total_gpu * 100
            logging.info(f"GPU usage: {gpu_percent:.2f}% of GPU memory")
        else:
            logging.info("No CUDA GPU available.")
    except ImportError:
        logging.info("Could not import torch. Install with 'pip install torch' to monitor GPU usage.")
    except Exception as e:
        logging.info(f"Error checking GPU usage: {e}")


def wait_for_keypress():
    """Wait for user to press Enter before continuing"""
    logging.info("\n\033[93m=========================================\033[0m")
    logging.info("\033[93mBatch processing complete. Time to upload the next batch of files.\033[0m")
    logging.info("\033[93mPress Enter when you've uploaded the next batch and are ready to continue...\033[0m")
    logging.info("\033[93m=========================================\033[0m\n")
    input()  # Wait for Enter key
    logging.info("Continuing with next batch...")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())