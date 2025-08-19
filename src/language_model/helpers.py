import torch
import logging
import os
import json
from torch.optim.lr_scheduler import LambdaLR
import math

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',   # Blue
        'INFO': '\033[92m',    # Green
        'WARNING': '\033[91m', # Red
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[91m' # Red
    }
    RESET = '\033[0m'
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname_colored = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def configure_colored_logging(level):
    formatter = ColorFormatter('[%(levelname_colored)s] [%(threadName)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler])


def update_log_level(new_level: str):
    """
    Dynamically update the logging level during runtime.
    
    Args:
        new_level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    try:
        # Convert string to logging level
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        if new_level.upper() in level_map:
            numeric_level = level_map[new_level.upper()]
            
            # Update the root logger level
            logging.getLogger().setLevel(numeric_level)
            
            # Update all existing handlers
            for handler in logging.getLogger().handlers:
                handler.setLevel(numeric_level)
                
            logging.info(f"[RUNTIME] Log level changed to {new_level.upper()}")
        else:
            logging.warning(f"[RUNTIME] Invalid log level: {new_level}. Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    except Exception as e:
        logging.error(f"[RUNTIME] Failed to update log level: {e}")

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


def apply_runtime_overrides(optimizer, scheduler, params: dict, runtime_overrides_file: str) -> tuple[dict, dict]:
    """
    Reads runtime overrides JSON file and applies any overrides to optimizer/scheduler and params dict.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (unused but kept for compatibility)
        params: Dictionary of training parameters
        runtime_overrides_file: Path to the JSON file containing overrides
        
    Returns:
        Tuple of (updated params dict, extra counters dict)
    """
    extra = {}
    
    if os.path.exists(runtime_overrides_file):
        try:
            with open(runtime_overrides_file, "r") as f:
                overrides = json.load(f)
                
            # Iteration counters
            if "global_iter" in overrides:
                extra['global_iter'] = int(overrides["global_iter"])
                logging.info(f"[RUNTIME] Set global_iter to {extra['global_iter']}")
            if "total_epochs_run" in overrides:
                extra['total_epochs_run'] = int(overrides["total_epochs_run"])
                logging.info(f"[RUNTIME] Set total_epochs_run to {extra['total_epochs_run']}")
                
            # Optimizer parameters
            if "learning_rate" in overrides:
                lr = float(overrides["learning_rate"])
                for g in optimizer.param_groups:
                    g['lr'] = lr
                params['learning_rate'] = lr
                logging.info(f"[RUNTIME] Set learning rate to {lr}")
                
            if "weight_decay" in overrides:
                wd = float(overrides["weight_decay"])
                for g in optimizer.param_groups:
                    g['weight_decay'] = wd
                params['weight_decay'] = wd
                logging.info(f"[RUNTIME] Set weight decay to {wd}")
                
            # Training parameters
            for param_name in ["eval_interval", "batch_size", "block_size", "early_stopping_patience", "warmup_steps", "base_training_max_epochs", "finetuning_max_epochs"]:
                if param_name in overrides:
                    value = int(overrides[param_name])
                    if value > 0:
                        params[param_name] = value
                        logging.info(f"[RUNTIME] Set {param_name} to {value}")
                        
            # Float parameters
            for param_name in ["grad_clip_norm", "dropout"]:
                if param_name in overrides:
                    value = float(overrides[param_name])
                    params[param_name] = value
                    logging.info(f"[RUNTIME] Set {param_name} to {value}")
                    
            # String parameters
            if "lr_decay" in overrides:
                lrd = str(overrides["lr_decay"])
                params['lr_decay'] = lrd
                logging.info(f"[RUNTIME] Set lr_decay to {lrd}")
                
            # Log level parameter
            if "log_level" in overrides:
                new_log_level = str(overrides["log_level"]).upper()
                update_log_level(new_log_level)
                params['log_level'] = new_log_level
                
        except Exception as e:
            logging.warning(f"[RUNTIME] Failed to apply runtime overrides: {e}")
            
    return params, extra


def get_lr_scheduler(optimizer, warmup_steps: int, lr_decay: str, total_steps: int) -> LambdaLR:
    """
    Create a learning rate scheduler with warmup and decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        lr_decay: Type of decay ("linear", "cosine", or "none")
        total_steps: Total number of training steps
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if lr_decay == "linear":
            return max(0.0, 1.0 - progress)
        elif lr_decay == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            return 1.0
    return LambdaLR(optimizer, lr_lambda)