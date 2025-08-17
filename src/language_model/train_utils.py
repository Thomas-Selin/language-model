import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import math
import gc
from torch.utils.tensorboard import SummaryWriter
import os
import time
import datetime
from helpers import configure_colored_logging, print_memory_usage, wait_for_keypress, get_device, count_parameters
from data_handler import get_batch, load_and_process_data
from model import GPTLanguageModel
import config
import logging
import glob
import threading
from subword_tokenizer import SubwordTokenizer
import json

# Magic numbers as constants
MIN_FILE_SIZE_BYTES = 200 * 1024  # 200 KB
STABLE_COUNT_THRESHOLD = 30

# Configure logging
configure_colored_logging(config.LOG_LEVEL)

# Hyperparameters (should be imported or passed in real use)
batch_size = config.BATCH_SIZE
block_size = config.BLOCK_SIZE
base_training_max_epochs = config.BASE_TRAINING_MAX_EPOCHS
finetuning_max_epochs = config.FINETUNING_MAX_EPOCHS
eval_interval = config.EVAL_INTERVAL
learning_rate = config.LEARNING_RATE
eval_iters = config.EVAL_ITERS
n_embd = config.N_EMBD
n_head = config.N_HEAD
n_layer = config.N_LAYER
dropout = config.DROPOUT
early_stopping_patience = config.EARLY_STOPPING_PATIENCE
max_vocab_size = config.MAX_VOCAB_SIZE
warmup_steps = config.WARMUP_STEPS
lr_decay = config.LR_DECAY
device = get_device()

# Runtime override config file
RUNTIME_OVERRIDES_FILE = "data/output/RUNTIME_OVERRIDES.json"

def apply_runtime_overrides(optimizer: torch.optim.Optimizer, scheduler, params: dict) -> tuple[dict, dict]:
    # Optionally return global_iter and total_epochs_run if present
    extra = {}
    """
    Reads RUNTIME_OVERRIDES.json and applies any overrides to optimizer/scheduler and params dict.
    Supported keys: learning_rate, weight_decay, eval_interval, batch_size, block_size, grad_clip_norm,
    dropout, early_stopping_patience, warmup_steps, lr_decay
    """
    if os.path.exists(RUNTIME_OVERRIDES_FILE):
        try:
            with open(RUNTIME_OVERRIDES_FILE, "r") as f:
                overrides = json.load(f)
            # Iteration counters
            if "global_iter" in overrides:
                extra['global_iter'] = int(overrides["global_iter"])
                logging.info(f"[RUNTIME] Set global_iter to {extra['global_iter']}")
            if "total_epochs_run" in overrides:
                extra['total_epochs_run'] = int(overrides["total_epochs_run"])
                logging.info(f"[RUNTIME] Set total_epochs_run to {extra['total_epochs_run']}")
            # Learning rate
            if "learning_rate" in overrides:
                lr = float(overrides["learning_rate"])
                for g in optimizer.param_groups:
                    g['lr'] = lr
                params['learning_rate'] = lr
                logging.info(f"[RUNTIME] Set learning rate to {lr}")
            # Weight decay
            if "weight_decay" in overrides:
                wd = float(overrides["weight_decay"])
                for g in optimizer.param_groups:
                    g['weight_decay'] = wd
                params['weight_decay'] = wd
                logging.info(f"[RUNTIME] Set weight decay to {wd}")
            # Eval interval
            if "eval_interval" in overrides:
                new_eval = int(overrides["eval_interval"])
                if new_eval > 0:
                    params['eval_interval'] = new_eval
                    logging.info(f"[RUNTIME] Set eval_interval to {new_eval}")
            # Batch size
            if "batch_size" in overrides:
                bs = int(overrides["batch_size"])
                if bs > 0:
                    params['batch_size'] = bs
                    logging.info(f"[RUNTIME] Set batch_size to {bs}")
            # Block size
            if "block_size" in overrides:
                bls = int(overrides["block_size"])
                if bls > 0:
                    params['block_size'] = bls
                    logging.info(f"[RUNTIME] Set block_size to {bls}")
            # Gradient clipping
            if "grad_clip_norm" in overrides:
                gcn = float(overrides["grad_clip_norm"])
                params['grad_clip_norm'] = gcn
                logging.info(f"[RUNTIME] Set grad_clip_norm to {gcn}")
            # Dropout
            if "dropout" in overrides:
                dp = float(overrides["dropout"])
                params['dropout'] = dp
                logging.info(f"[RUNTIME] Set dropout to {dp}")
            # Early stopping patience
            if "early_stopping_patience" in overrides:
                esp = int(overrides["early_stopping_patience"])
                if esp > 0:
                    params['early_stopping_patience'] = esp
                    logging.info(f"[RUNTIME] Set early_stopping_patience to {esp}")
            # Warmup steps
            if "warmup_steps" in overrides:
                ws = int(overrides["warmup_steps"])
                if ws > 0:
                    params['warmup_steps'] = ws
                    logging.info(f"[RUNTIME] Set warmup_steps to {ws}")
            # LR decay
            if "lr_decay" in overrides:
                lrd = str(overrides["lr_decay"])
                params['lr_decay'] = lrd
                logging.info(f"[RUNTIME] Set lr_decay to {lrd}")
        except Exception as e:
            logging.warning(f"[RUNTIME] Failed to apply runtime overrides: {e}")
    return params, extra

@torch.no_grad()
def estimate_loss(model: nn.Module, train_data, val_data) -> dict:
    """
    Estimate average train and validation loss over eval_iters batches.
    Args:
        model: The language model.
        train_data: Training data.
        val_data: Validation data.
    Returns:
        Dictionary with mean losses for 'train' and 'val'.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        with autocast(device_type=device.type):
            for k in range(eval_iters):
                X, Y = get_batch(block_size, batch_size, split, train_data, val_data)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, lr_decay: str, total_steps: int) -> LambdaLR:
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

def preload_parquet(parquet_file: str, vocab_size: int, parquet_dir_path: str, text_column: str, vocab_path: str, batch_size: int):
    # This function will run in a background thread
    result = {}
    def loader():
        result['data'] = load_and_process_data(
            vocab_size=vocab_size,
            parquet_dir_path=parquet_dir_path,
            text_column=text_column,
            vocab_path=vocab_path,
            batch_size=batch_size,
            single_file=parquet_file
        )
    thread = threading.Thread(target=loader)
    thread.start()
    return thread, result

def base_train_model(
    parquet_dir_path: str,
    text_column: str = 'text',
    vocab_path: str = 'data/output/vocab_subword.json',
    training_start_time: str = None,
    output_dir: str = None,
    checkpoint_path: str = None
) -> None:
    """
    Main training loop for base model. Handles file watching, training, evaluation, early stopping, and TensorBoard logging.
    Args:
        parquet_dir_path: Directory containing .parquet files.
        text_column: Name of text column in parquet files.
        vocab_path: Path to vocabulary file.
        training_start_time: Timestamp string for output dir naming.
        output_dir: Output directory for logs and models.
        checkpoint_path: Optional path to resume model weights.
    """
    parquet_files = sorted(
        glob.glob(os.path.join(parquet_dir_path, '*.parquet')),
        key=lambda x: os.path.basename(x)
    )
    if not parquet_files:
        logging.info(f"No parquet files found in {parquet_dir_path}")
        return
    if output_dir is None:
        output_dir = os.path.join('data', 'output', training_start_time)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logging.info(f"\033[92mTensorBoard logging started. View logs with: tensorboard --logdir={log_dir}\033[0m")
    logging.info(f"........max vocab size: {max_vocab_size}") #TODO: Remove
    model = GPTLanguageModel(max_vocab_size).to(device) #TODO: should be vocab_size
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logging.info(f"\033[92mResumed model weights loaded from {checkpoint_path}\033[0m")
        except (OSError, RuntimeError) as e:
            logging.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    logging.info(f"{count_parameters(model)/1e6:.2f} M parameters")
    logging.info(f"Model is on device: {next(model.parameters()).device}")
    learning_rate = config.LEARNING_RATE
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    scaler = GradScaler(enabled=torch.cuda.is_available())
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            orig = self.model.use_checkpoint
            self.model.use_checkpoint = False
            logits, _ = self.model(x)
            self.model.use_checkpoint = orig
            return logits
    block_size = config.BLOCK_SIZE
    sample_input = torch.zeros((1, block_size), dtype=torch.long).to(device)
    vis_model = ModelWrapper(model).to(device)
    writer.add_graph(vis_model, sample_input)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    global_iter = None  # Will be set by config or runtime overrides
    total_epochs_run = 0
    global_max_epochs = base_training_max_epochs
    tensorboard_logged = False  # Initialize flag
    # Initialize runtime_params before the main training loop
    runtime_params = {
        'eval_interval': config.EVAL_INTERVAL,
        'batch_size': config.BATCH_SIZE,
        'block_size': config.BLOCK_SIZE,
        'grad_clip_norm': 1.0,
        'dropout': config.DROPOUT,
        'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        'warmup_steps': config.WARMUP_STEPS,
        'lr_decay': config.LR_DECAY,
        'learning_rate': config.LEARNING_RATE
    }
    try:
        while True:
            parquet_files = sorted(
                glob.glob(os.path.join(parquet_dir_path, '*.parquet')),
                key=lambda x: os.path.basename(x)
            )
            # Remove the first file if present (avoid double processing)
            # parquet_files = [f for f in parquet_files if os.path.basename(f) != file_name_single]
            if not parquet_files:
                logging.info(f"No parquet files found in {parquet_dir_path}. Waiting for new files...")
                wait_for_keypress()
                parquet_files = sorted(
                    glob.glob(os.path.join(parquet_dir_path, '*.parquet')),
                    key=lambda x: os.path.basename(x)
                )
                # parquet_files = [f for f in parquet_files if os.path.basename(f) != file_name_single]
                if not parquet_files:
                    logging.info("No files present after keypress. Base training finished.")
                    break
                continue
            for file_idx, parquet_file in enumerate(parquet_files):
                logging.debug("Performing memory cleanup before loading new file...")
                try:
                    del train_data
                except Exception:
                    pass
                try:
                    del val_data
                except Exception:
                    pass
                gc.collect()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                logging.debug("Memory usage after cleanup:")
                print_memory_usage()
                logging.info(f"Processing file {file_idx+1}/{len(parquet_files)}: {parquet_file}. Setting best_val_loss to infinity.")
                logging.debug(f"\nChunk size: 100 rows")
                best_val_loss = float('inf')
                file_dir = os.path.dirname(parquet_file)
                file_name = os.path.basename(parquet_file)

                # Start preloading the next file (if any)
                next_file = parquet_files[file_idx + 1] if file_idx + 1 < len(parquet_files) else None
                preload_thread, preload_result = None, None
                if next_file:
                    next_file_dir = os.path.dirname(next_file)
                    next_file_name = os.path.basename(next_file)
                    preload_thread, preload_result = preload_parquet(
                        next_file_name, max_vocab_size, next_file_dir, text_column, vocab_path, runtime_params['batch_size']
                    )

                # Load current file (blocking)
                train_data, val_data, tokenizer, vocab_size, _ = load_and_process_data(
                    vocab_size=max_vocab_size,
                    parquet_dir_path=file_dir,
                    text_column=text_column,
                    vocab_path=vocab_path,
                    batch_size=runtime_params['batch_size'],
                    single_file=file_name
                )

                # If preloading was started, wait for it to finish and use the result for the next iteration
                if preload_thread:
                    preload_thread.join()
                    train_data, val_data, tokenizer, vocab_size, _ = preload_result['data']

                # Apply runtime overrides ONCE per file
                runtime_params, extra_counters = apply_runtime_overrides(optimizer, None, runtime_params)
                # Use possibly updated params for this file
                eval_interval = runtime_params['eval_interval']
                batch_size = runtime_params['batch_size']
                block_size = runtime_params['block_size']
                grad_clip_norm = runtime_params['grad_clip_norm']
                dropout = runtime_params['dropout']
                early_stopping_patience = runtime_params['early_stopping_patience']
                warmup_steps = runtime_params['warmup_steps']
                lr_decay = runtime_params['lr_decay']
                learning_rate = runtime_params['learning_rate']
                # Log hyperparameters to TensorBoard after batch_size is set (once per session)
                if not tensorboard_logged:
                    writer.add_text('Session Information', f"""
                    ## Hyperparameters
                    - Batch size: {batch_size}
                    - Block size: {block_size}
                    - Max epochs: {base_training_max_epochs}
                    - Eval interval: {eval_interval}
                    - Learning rate: {learning_rate}
                    - Eval iters: {eval_iters}
                    - Embedding dimension: {n_embd}
                    - Number of heads: {n_head}
                    - Number of layers: {n_layer}
                    - Dropout: {dropout}
                    - Early stopping patience: {early_stopping_patience}
                    - Max vocab size: {max_vocab_size}
                    ## Environment
                    - Device: {device}
                    ## Data
                    - Tokenizer vocab path: {vocab_path}
                    ## Timing
                    - Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """)
                    tensorboard_logged = True
                # Optionally update global_iter/total_epochs_run between files
                if 'global_iter' in extra_counters:
                    global_iter = extra_counters['global_iter']
                elif global_iter is None and hasattr(config, 'GLOBAL_ITER'):
                    global_iter = getattr(config, 'GLOBAL_ITER')
                if 'total_epochs_run' in extra_counters:
                    total_epochs_run = extra_counters['total_epochs_run']
                # Set up LR scheduler for each file.
                # One optimizer step per loop iteration, so steps_per_file == base_training_max_epochs.
                steps_per_file = base_training_max_epochs
                warmup_steps_local = max(1, int(0.02 * steps_per_file))  # ~1–2% warmup
                scheduler = get_lr_scheduler(optimizer, warmup_steps_local, lr_decay, steps_per_file)
                for iter in range(base_training_max_epochs):
                    if total_epochs_run >= global_max_epochs:
                        logging.info(f"Reached global epoch limit of {global_max_epochs}. Stopping training.")
                        break
                    if iter % eval_interval == 0 or iter == base_training_max_epochs - 1:
                        losses = estimate_loss(model, train_data, val_data)
                        logging.info("____________________________")
                        logging.debug("Memory usage before evaluation:")
                        print_memory_usage()
                        logging.info(f"File {file_idx+1}, Step {iter} of max {base_training_max_epochs}: train loss {losses['train']:.4f}, 📏 val loss {losses['val']:.4f}")
                        # Save best model if val loss improves
                        if losses['val'] < best_val_loss:
                            logging.info(f"Validation loss improved ({losses['val']:.4f} < {best_val_loss:.4f}). Saving best model.")
                            best_val_loss = losses['val']
                            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1
                            logging.info(f"Epochs without improvement: {epochs_without_improvement}")
                        if epochs_without_improvement >= early_stopping_patience:
                            logging.info(f"Early stopping triggered at epoch {iter}. Best val loss: {best_val_loss:.4f}")
                            logging.info("Restoring best model weights...")
                            model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
                            break
                        # Only log to TensorBoard every eval_interval * 5 steps or at the last step
                        if global_iter is not None and (global_iter % (eval_interval * 5) == 0 or (iter == base_training_max_epochs - 1 and file_idx == len(parquet_files)-1)):
                            logging.debug(f"Writing metrics and generated text to TensorBoard at step {global_iter}")
                            writer.add_scalar('Loss/train', losses['train'], global_iter)
                            writer.add_scalar('Loss/val', losses['val'], global_iter)
                            writer.add_scalar('Perplexity/train', float(torch.exp(losses['train'])), global_iter)
                            writer.add_scalar('Perplexity/val', float(torch.exp(losses['val'])), global_iter)
                            if torch.cuda.is_available():
                                writer.add_scalar('Memory/allocated_GB', torch.cuda.memory_allocated() / 1024**3, global_iter)
                                writer.add_scalar('Memory/reserved_GB', torch.cuda.memory_reserved() / 1024**3, global_iter)
                                writer.add_scalar('Memory/free_GB', (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3, global_iter)
                            # Log parameter and gradient histograms less frequently
                            if global_iter % (eval_interval * 10) == 0:
                                for name, param in model.named_parameters():
                                    writer.add_histogram(f'Parameters/{name}', param, global_iter)
                                    if param.grad is not None:
                                        writer.add_histogram(f'Gradients/{name}', param.grad, global_iter)
                            # Log generated text samples
                            log_generated_samples(model, tokenizer, writer, global_iter, device)
                    epoch_start_time = time.time()
                    data_time = time.time()
                    xb, yb = get_batch(block_size, batch_size, 'train', train_data, val_data)
                    data_time = time.time() - data_time
                    train_time = time.time()
                    with autocast(device_type=device.type):
                        logits, loss = model(xb, yb)
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if global_iter % 3 == 0:
                        gc.collect()
                    train_time = time.time() - train_time
                    scheduler.step()
                    epoch_duration = time.time() - epoch_start_time
                    writer.add_scalar('Total/EpochTime', epoch_duration, global_iter)
                    # logging.debug(f"Writing learning rate to TensorBoard: {scheduler.get_last_lr()[0]} at step {global_iter}")
                    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_iter)
                    if iter % eval_interval == 0:
                        logging.info(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
                    if global_iter is not None:
                        global_iter += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        logging.info("Early stopping triggered. Moving to next file if available.")
                        break
                epochs_without_improvement = 0
                # Delete the file after training
                try:
                    os.remove(parquet_file)
                    logging.info(f"Deleted file after training: {parquet_file}")
                except FileNotFoundError as e:
                    logging.info(f"File not found when deleting {parquet_file}: {e}")
                except PermissionError as e:
                    logging.info(f"Permission error when deleting {parquet_file}: {e}")
                except Exception as e:
                    logging.info(f"Error deleting file {parquet_file}: {e}")
            logging.info("\nAll files in folder processed. Waiting for new files...\n")
            logging.info("=" * 60)
            logging.info("WAITING FOR NEW FILES OR USER INPUT")
            logging.info("=" * 60)
            logging.info("Options:")
            logging.info("1. Add new .parquet files to continue training")
            logging.info("2. Create a file named 'STOP_TRAINING' in the data/output/ folder to stop")
            logging.info("   Command: touch data/output/STOP_TRAINING")
            logging.info("=" * 60)
            
            stop_file_path = "data/output/STOP_TRAINING"
            # Remove any existing stop file at the start
            if os.path.exists(stop_file_path):
                os.remove(stop_file_path)
            
            stop_file_path = "data/output/STOP_TRAINING"
            # Remove any existing stop file at the start
            if os.path.exists(stop_file_path):
                os.remove(stop_file_path)
            
            user_interrupted = False
            check_counter = 0
            
            # Keep track of files we've already seen to detect new ones
            seen_files = set(os.path.basename(f) for f in glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
            # Require at least 200 Kb and stable size across consecutive checks
            size_state = {}  # file -> (last_size, stable_count)
            trained_files = set()  # Files that have been used for training

            while True:
                # Check for stop file FIRST and FREQUENTLY (every iteration)
                if os.path.exists(stop_file_path):
                    logging.info(f"\n🛑 Stop file detected: {stop_file_path}")
                    logging.info("User requested to stop base training.")
                    logging.info("Removing stop file and proceeding to save model...")
                    os.remove(stop_file_path)  # Clean up
                    user_interrupted = True
                    break
                
                # Check for new files manually (non-blocking)
                current_files = set(f for f in os.listdir(parquet_dir_path) if f.endswith('.parquet'))
                new_files = current_files - trained_files  # Only files not yet trained

                if new_files:
                    ready_file_found = False
                    for file in new_files:
                        file_path = os.path.join(parquet_dir_path, file)
                        if not os.path.exists(file_path):
                            logging.debug(f"File {file_path} does not exist (may have been deleted/moved). Skipping.")
                            continue
                        curr_size = os.path.getsize(file_path)
                        prev_size, stable_count = size_state.get(file, (None, 0))
                        logging.debug(f"Checking file '{file}': size={curr_size/1024/1024:.2f} MB, previous size={prev_size}, stable_count={stable_count}")

                        if curr_size >= MIN_FILE_SIZE_BYTES and prev_size is not None and curr_size == prev_size:
                            stable_count += 1  # one more stable check
                            logging.debug(f"File '{file}' size is stable and >=200 KB ({curr_size/1024/1024:.2f} MB). Stable count: {stable_count}/{STABLE_COUNT_THRESHOLD}")
                        else:
                            if curr_size < MIN_FILE_SIZE_BYTES:
                                logging.debug(f"File '{file}' is too small ({curr_size/1024/1024:.2f} MB). Waiting for upload to finish.")
                            elif prev_size is not None and curr_size != prev_size:
                                logging.debug(f"File '{file}' size changed from {prev_size} to {curr_size}. Waiting for upload to finish.")
                            else:
                                logging.debug(f"First observation for file '{file}'. Waiting for size to stabilize.")
                            stable_count = 0  # size changed/too small/first observation

                        size_state[file] = (curr_size, stable_count)

                        # Require N consecutive stable checks (e.g., 30 loops ≈ 30 seconds)
                        if stable_count >= STABLE_COUNT_THRESHOLD:
                            logging.info(f"New file detected and size stabilized (>=200 KB): {file} ({curr_size/1024/1024:.1f} MB)")
                            logging.info(f"File '{file}' appears to have finished uploading. Resuming training with new file...")
                            trained_files.add(file)  # Mark as trained AFTER processing
                            ready_file_found = True
                            break
                    
                    if ready_file_found:
                        break
                
                # Show periodic status (less frequent to reduce spam)
                if check_counter % 300 == 0:
                    logging.debug(f"Current .parquet files detected: {current_files}")
                    logging.debug(f"New files since last training: {new_files}")
                    logging.debug("Still waiting... (Create 'data/output/STOP_TRAINING' file to stop)")
                
                check_counter += 1
                time.sleep(10)  # Check stop file every ten seconds

            if user_interrupted:
                break
    except Exception as e:
        # Handle any other exceptions (but not KeyboardInterrupt)
        logging.error(f"An error occurred during training: {e}")
        torch.save(model.state_dict(), os.path.join(output_dir, "model_error.pt"))
        logging.info("Model saved to data/output/model_error.pt due to error")
    
    if writer:
        logging.debug("[DEBUG] Closing TensorBoard writer.")
        writer.close()

def log_generated_samples(
    model: nn.Module,
    tokenizer: SubwordTokenizer,
    writer: SummaryWriter,
    global_step: int,
    device: torch.device
) -> None:
    """
    Log unconditional and prompted generations to TensorBoard.
    Args:
        model: The language model.
        tokenizer: Tokenizer for decoding outputs.
        writer: TensorBoard SummaryWriter.
        global_step: Current global step for logging.
        device: Device to run generation on.
    """
    for temp in [0.5, 0.8, 1.0]:
        # Unconditional generation (no prompt)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        sample_text = tokenizer.decode(model.generate(context, temperature=temp)[0].tolist())
        writer.add_text(f'Generated Text (no prompt) {temp} temp', sample_text, global_step)
        # Prompted generation 1
        prompt1 = "Which Emmy award was American Idol nominated for nine times?"
        input_ids1 = torch.tensor([tokenizer.encode(prompt1)], dtype=torch.long, device=device)
        sample_text_prompt1 = tokenizer.decode(model.generate(input_ids1, temperature=temp)[0].tolist())
        writer.add_text(f'Generated Text: "{prompt1}" {temp} temp', sample_text_prompt1, global_step)
        # Prompted generation 2
        prompt2 = "Where is Paris?"
        input_ids2 = torch.tensor([tokenizer.encode(prompt2)], dtype=torch.long, device=device)
        sample_text_prompt2 = tokenizer.decode(model.generate(input_ids2, temperature=temp)[0].tolist())
        writer.add_text(f'Generated Text: "{prompt2}" {temp} temp', sample_text_prompt2, global_step)

def train_chat_alignment(
    model: nn.Module,
    qa_tensor: torch.Tensor,
    tensorboard_logdir: str,
    output_dir: str,
    lr: float = 1e-4,
    batch_size: int = batch_size,
    val_split: float = 0.1
) -> None:
    """
    Finetune model for chat alignment using QA tensor data. Logs metrics and samples to TensorBoard.
    Args:
        model: The language model.
        qa_tensor: Tensor of QA data.
        tensorboard_logdir: Directory for TensorBoard logs.
        output_dir: Directory to save model.
        lr: Learning rate.
        batch_size: Batch size.
        val_split: Fraction of data for validation.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = finetuning_max_epochs
    model.train()
    device = next(model.parameters()).device
    num_samples = qa_tensor.size(0)
    split_idx = int((1.0 - val_split) * num_samples)
    train_tensor = qa_tensor[:split_idx]
    val_tensor = qa_tensor[split_idx:]

    total_steps = epochs * ((train_tensor.size(0) + batch_size - 1) // batch_size)
    warmup_steps = max(10, min(200, int(0.02 * total_steps)))  # ~2% with floor=10, cap=200

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, lr_decay="cosine", total_steps=total_steps)
    patience = config.FINETUNE_EARLY_STOPPING_PATIENCE
    logging.info(f"Chat alignment: total_steps={total_steps}, warmup_steps={warmup_steps}, patience={patience}")
    scaler = GradScaler(enabled=torch.cuda.is_available())
    global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    logging.info(f"Total samples: {num_samples}, Train: {train_tensor.size(0)}, Val: {val_tensor.size(0)}")
    tokenizer = SubwordTokenizer(vocab_file=config.VOCAB_PATH)

    writer = SummaryWriter(tensorboard_logdir)
    logging.info(f"TensorBoard logging started. View logs with: tensorboard --logdir={tensorboard_logdir}")

    for epoch in range(epochs):
        logging.info(f"\nEpoch {epoch+1}/{epochs} START")
        model.train()
        train_loss = 0.0
        num_train_batches = (train_tensor.size(0) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, train_tensor.size(0), batch_size)):
            batch = train_tensor[i:i+batch_size]
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            with autocast(device_type=device.type):
                logits, loss = model(inputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * inputs.size(0)
            if global_step % 100 == 0 or batch_idx == num_train_batches - 1:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                log_generated_samples(model, tokenizer, writer, global_step, device)
            global_step += 1
            if batch_idx % 1000 == 0:
                logging.info(f"  Train Batch {batch_idx+1}/{num_train_batches} - Batch Loss: {loss.item():.4f}")
                logging.debug("Memory usage:")
                print_memory_usage()
        avg_train_loss = train_loss / train_tensor.size(0)
        logging.info(f"Epoch {epoch+1} TRAINING DONE. Avg Train Loss: {avg_train_loss:.4f}")
        model.eval()
        val_loss = 0.0
        num_val_batches = (val_tensor.size(0) + batch_size - 1) // batch_size
        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, val_tensor.size(0), batch_size)):
                batch = val_tensor[i:i+batch_size]
                inputs = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)
                with autocast(device_type=device.type):
                    logits, loss = model(inputs, targets)
                val_loss += loss.item() * inputs.size(0)
                if global_step % 100 == 0 or batch_idx == num_val_batches - 1:
                    writer.add_scalar('Loss/val', loss.item(), global_step)
                    log_generated_samples(model, tokenizer, writer, global_step, device)
                global_step += 1
                if batch_idx % 250 == 0:
                    logging.info(f"  Val Batch {batch_idx+1}/{num_val_batches} - Batch Loss: {loss.item():.4f}")
        avg_val_loss = val_loss / val_tensor.size(0)
        logging.info(f"Epoch {epoch+1} VALIDATION DONE. Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logging.info("Early stopping triggered for chat alignment.")
            break
    torch.save(model.state_dict(), os.path.join(output_dir, "chat_aligned_model.pt"))
    logging.info("Final chat-aligned model saved to data/output/chat_aligned_model.pt")