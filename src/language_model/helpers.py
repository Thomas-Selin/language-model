import torch
import logging
from config import LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='\033[95m[%(levelname)s]\033[0m %(message)s'
)

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
        
def print_memory_usage():
    """Print percentage of RAM, GPU memory used, and disk usage for current directory."""
    try:
        import psutil
        import os
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
            logging.info(f"GPU usage: \033[93m{gpu_percent:.2f}%\033[0m of GPU memory")
        else:
            logging.info("No CUDA GPU available.")
    except ImportError:
        logging.info("Could not import torch. Install with 'pip install torch' to monitor GPU usage.")
    except Exception as e:
        logging.info(f"Error checking GPU usage: {e}")

    # Disk usage for current directory
    try:
        current_path = os.getcwd()
        usage = psutil.disk_usage(current_path)
        logging.info(
            f"Disk usage for {current_path}: {usage.percent:.2f}% used "
            f"({usage.used / (1024 ** 3):.2f} GiB used / {usage.total / (1024 ** 3):.2f} GiB total, "
            f"{usage.free / (1024 ** 3):.2f} GiB free)"
        )
    except Exception as e:
        logging.info(f"Error checking disk usage: {e}")


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