import torch
import torch.nn as nn
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast, GradScaler
import gc
import os
import time
import logging
import traceback
import threading
from typing import Optional
from language_model.helpers import configure_colored_logging, print_memory_usage, get_device, count_parameters
from language_model.helpers import apply_runtime_overrides, get_lr_scheduler
from language_model.data_handler import load_and_process_data, get_batch, get_sequential_batches
from language_model.evaluation import estimate_loss
from language_model.checkpointing import save_best_model
from language_model.model import GPTLanguageModel
from language_model.training_helpers import (
    wait_for_new_files_or_stop, preload_parquet_data, cleanup_processed_file,
    get_parquet_files, setup_output_directory
)
from language_model.tensorboard_utils import (
    setup_tensorboard_logging, log_hyperparameters, log_model_graph,
    log_training_metrics, log_generated_samples, log_model_parameters, log_epoch_time
)
import language_model.config as config
from language_model.subword_tokenizer import SubwordTokenizer
from language_model.constants import (
    GPU_MEMORY_FRACTION, 
    HIGH_MEMORY_THRESHOLD_GB, 
    RESERVED_MEMORY_THRESHOLD_GB,
)

# Configure logging
configure_colored_logging(config.LOG_LEVEL)

# Set device
device = get_device()

# Add a lock for thread safety
preload_lock = threading.Lock()

def optimize_memory_settings() -> None:
    """Configure optimal memory settings for training.
    
    Applies several optimizations:
    - Enables flash attention if available (CUDA)
    - Sets GPU memory fraction to prevent OOM
    - Enables cudNN benchmarking for speed
    """
    # Enable memory-efficient attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Set memory fraction to leave some headroom
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        logging.info(f"Set GPU memory fraction to {GPU_MEMORY_FRACTION:.1%}")
    
    # Enable cudNN benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    logging.info("Memory optimization settings applied")

def aggressive_memory_cleanup():
    """Aggressive memory cleanup between training steps"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations complete
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

def log_memory_usage(step_name: str = "", global_iter: Optional[int] = None) -> None:
    """Log current GPU memory usage for debugging.
    
    Args:
        step_name: Descriptive name for the current step (e.g., "before_eval")
        global_iter: Current training iteration number for context
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        step_info = f"[Step {global_iter}] " if global_iter else ""
        logging.debug(
            f"{step_info}GPU Memory {step_name}: "
            f"Alloc={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB"
        )
        
        # Warning if memory usage is high
        if allocated > HIGH_MEMORY_THRESHOLD_GB:
            logging.warning(f"High GPU memory usage detected: {allocated:.2f}GB allocated")
        if reserved > RESERVED_MEMORY_THRESHOLD_GB:
            logging.warning(f"High reserved GPU memory: {reserved:.2f}GB reserved")

def base_train_model(
    parquet_dir_path: str,
    text_column: str = 'text',
    vocab_path: str = 'data/output/vocab_subword.json',
    output_dir: str = None,
    checkpoint_path: str = None
) -> int:
    """
    Main training loop for base model. Handles file watching, training, evaluation, early stopping, and TensorBoard logging.
    Returns:
        int: The final global_iter value after training.
    """
    # Setup directories and logging
    if output_dir is None:
        output_dir = os.path.join('data', 'output')
    setup_output_directory(output_dir)
    
    # Initialize TensorBoard
    writer = setup_tensorboard_logging(output_dir)
    
    # Apply memory optimizations
    optimize_memory_settings()

    # Create / restore model using configured vocab size (do not mutate config at runtime)
    vocab_size = getattr(config, 'MAX_VOCAB_SIZE')
    model = GPTLanguageModel(vocab_size).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logging.info(f"\033[92mResumed model weights loaded from {checkpoint_path}\033[0m")
        except (OSError, RuntimeError) as e:
            logging.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    
    logging.info(f"{count_parameters(model)/1e6:.2f} M parameters")
    
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.DEFAULT_WEIGHT_DECAY)
    # Enable mixed precision only for CUDA (MPS has issues with GradScaler in some PyTorch versions)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Log model graph
    log_model_graph(writer, model, config.BLOCK_SIZE, device)
    
    # Training state
    global_iter = getattr(config, 'GLOBAL_ITER', 0)
    tensorboard_logged = False
    
    # Runtime parameters
    runtime_params = {
        'eval_step_interval': max(1, config.BATCH_SIZE),  # evaluate every roughly one batch worth of sequences (can be overridden)
        'batch_size': config.BATCH_SIZE,
        'block_size': config.BLOCK_SIZE,
        'dropout': config.DROPOUT,
        'early_stopping_patience_steps': config.EARLY_STOPPING_PATIENCE * 10,  # convert epoch patience heuristic to steps default
        'lr_decay': config.LR_DECAY,
        'learning_rate': config.LEARNING_RATE,
        'log_level': config.LOG_LEVEL
    }

    # Track best validation and early stopping in steps
    best_val_loss_global = float('inf')
    steps_without_improvement = 0
    
    stop_file_path = "data/output/STOP_TRAINING"
    
    try:
        preload_thread, preload_result, preload_file_name = None, None, None
        while True:
            parquet_files = get_parquet_files(parquet_dir_path)
            if not parquet_files:
                logging.info(f"No parquet files found in {parquet_dir_path}. Waiting for new files...")
                user_interrupted, _ = wait_for_new_files_or_stop(parquet_dir_path, stop_file_path)
                if user_interrupted:
                    break
                continue

            # Sort files by creation time (oldest first)
            parquet_files = sorted(parquet_files, key=lambda f: os.path.getctime(f))

            file_count = len(parquet_files)
            file_idx = 0
            while file_idx < file_count:
                parquet_file = parquet_files[file_idx]
                file_name = os.path.basename(parquet_file)
                _cleanup_memory()
                file_start_time = time.time()
                # ==================== TRAINING START DIVIDER ====================
                logging.info("\n" + "="*80)
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                logging.info(f"\033[96m🚀 STARTING TRAINING ON FILE {file_idx+1}/{file_count} | Time: {current_time}\033[0m")
                logging.info(f"\033[96m📁 FILE: {os.path.basename(parquet_file)}\033[0m")
                logging.info(f"\033[96m📂 PATH: {parquet_file}\033[0m")
                logging.info("="*80 + "\n")

                # Check if we have preloaded data for this file
                train_data, val_data, tokenizer, vocab_size = None, None, None, None
                if preload_thread is not None and preload_file_name == file_name:
                    logging.info(f"Waiting for preloaded data for file: {file_name}")
                    preload_thread.join()
                    with preload_lock:
                        if 'error' in preload_result:
                            logging.warning(f"Preload failed: {preload_result['error']}. Loading synchronously.")
                            train_data, val_data, tokenizer, vocab_size, _ = load_and_process_data(
                                vocab_size=config.MAX_VOCAB_SIZE,
                                parquet_dir_path=os.path.dirname(parquet_file),
                                text_column=text_column,
                                vocab_path=vocab_path,
                                batch_size=runtime_params['batch_size'],
                                single_file=file_name
                            )
                        elif 'data' in preload_result:
                            logging.info("Using preloaded data successfully")
                            train_data, val_data, tokenizer, vocab_size, _ = preload_result['data']
                        else:
                            logging.warning("Preload completed but no data found. Loading synchronously.")
                            train_data, val_data, tokenizer, vocab_size, _ = load_and_process_data(
                                vocab_size=config.MAX_VOCAB_SIZE,
                                parquet_dir_path=os.path.dirname(parquet_file),
                                text_column=text_column,
                                vocab_path=vocab_path,
                                batch_size=runtime_params['batch_size'],
                                single_file=file_name
                            )
                    # Reset preload state after using
                    preload_thread, preload_result, preload_file_name = None, None, None
                else:
                    # No preloaded data available, load synchronously
                    logging.info(f"Loading data synchronously for file: {file_name}")
                    train_data, val_data, tokenizer, vocab_size, _ = load_and_process_data(
                        vocab_size=config.MAX_VOCAB_SIZE,
                        parquet_dir_path=os.path.dirname(parquet_file),
                        text_column=text_column,
                        vocab_path=vocab_path,
                        batch_size=runtime_params['batch_size'],
                        single_file=file_name
                    )

                # Start preloading the next file immediately (true parallelism)
                next_file = parquet_files[file_idx + 1] if file_idx + 1 < file_count else None
                if next_file and preload_thread is None:
                    next_file_dir = os.path.dirname(next_file)
                    next_file_name = os.path.basename(next_file)
                    logging.info(f"Starting preload of next file: {next_file_name}")
                    preload_thread, preload_result = preload_parquet_data(
                        next_file_name, config.MAX_VOCAB_SIZE, next_file_dir,
                        text_column, vocab_path, runtime_params['batch_size']
                    )
                    preload_file_name = next_file_name

                # Apply runtime overrides
                runtime_params, extra_counters = apply_runtime_overrides(
                    optimizer, None, runtime_params, config.RUNTIME_OVERRIDES_FILE
                )

                if 'global_iter' in extra_counters:
                    global_iter = extra_counters['global_iter']

                if not tensorboard_logged:
                    config_dict = {
                        'batch_size': runtime_params['batch_size'],
                        'block_size': runtime_params['block_size'],
                        'max_epochs': runtime_params['base_training_max_epochs'],
                        'eval_interval': runtime_params['eval_interval'],
                        'learning_rate': runtime_params['learning_rate'],
                        'eval_iters': config.EVAL_ITERS,
                        'n_embd': config.N_EMBD,
                        'n_head': config.N_HEAD,
                        'n_layer': config.N_LAYER,
                        'dropout': runtime_params['dropout'],
                        'early_stopping_patience': runtime_params['early_stopping_patience'],
                        'max_vocab_size': config.MAX_VOCAB_SIZE,
                        'device': device,
                        'vocab_path': vocab_path
                    }
                    log_hyperparameters(writer, config_dict)
                    tensorboard_logged = True

                # Per-file training: treat each file as a restart for cosine if desired.
                # Cosine restart: reinstantiate scheduler per file if lr_decay is cosine.
                steps_in_file = max(1, (len(train_data) - runtime_params['block_size'] - 1) // runtime_params['batch_size'])
                warmup_steps_local = max(1, int(0.02 * steps_in_file))
                scheduler = get_lr_scheduler(optimizer, warmup_steps_local, runtime_params['lr_decay'], steps_in_file)

                file_best_val = float('inf')
                file_steps = 0
                model.train()
                for xb, yb in get_sequential_batches(runtime_params['block_size'], runtime_params['batch_size'], train_data, device, shuffle=True):
                    file_steps += 1
                    global_iter += 1
                    with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                        logits, loss = model(xb, yb)
                    if torch.isnan(loss):
                        raise RuntimeError("NaN loss detected")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # Step-based evaluation
                    if global_iter % runtime_params['eval_step_interval'] == 0:
                        losses = estimate_loss(model, train_data, val_data, config.EVAL_ITERS,
                                               runtime_params['block_size'], runtime_params['batch_size'], device)
                        current_val = losses['val']
                        current_train = losses['train']
                        log_training_metrics(writer, losses, global_iter, scheduler.get_last_lr()[0])
                        if current_val < file_best_val:
                            file_best_val = current_val
                        # Global improvement tracking
                        if current_val < best_val_loss_global:
                            best_val_loss_global = current_val
                            steps_without_improvement = 0
                            save_best_model(model, output_dir, "best_model.pt")
                        else:
                            steps_without_improvement += runtime_params['eval_step_interval']
                        # Early stopping by steps
                        if steps_without_improvement >= runtime_params['early_stopping_patience_steps']:
                            logging.info(f"Early stopping triggered at step {global_iter}. Best global val loss: {best_val_loss_global:.4f}")
                            raise StopIteration

                    if file_steps >= steps_in_file:
                        break

                # ==================== TRAINING COMPLETION DIVIDER ====================
                logging.info("\n" + "="*80)
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                logging.info(f"\033[92m✅ COMPLETED TRAINING ON FILE {file_idx+1}/{file_count} | Time: {current_time}\033[0m")
                logging.info(f"\033[92m📁 FILE: {os.path.basename(parquet_file)}\033[0m")
                logging.info(f"\033[92m🎯 BEST VAL (FILE): {file_best_val:.4f}\033[0m")
                logging.info(f"\033[92m🧮 GLOBAL STEP: {global_iter}\033[0m")
                file_end_time = time.time()
                file_duration = file_end_time - file_start_time
                logging.info(f"\033[92m⏰ TOTAL DURATION: {file_duration:.2f} seconds ({file_duration/60:.2f} min)\033[0m")
                logging.info("="*80 + "\n")

                if getattr(config, 'AUTO_DELETE_USED_FILES', False):
                    cleanup_processed_file(parquet_file)

                file_idx += 1

            user_interrupted, _ = wait_for_new_files_or_stop(parquet_dir_path, stop_file_path)
            if user_interrupted:
                break

    except StopIteration:
        logging.info("Training stopped early due to early stopping condition.")
    except Exception as e:
        logging.error("An error occurred during training:\n" + traceback.format_exc())
        try:
            save_best_model(model, output_dir, "model_error.pt")
            logging.info("Model saved to model_error.pt due to error")
        finally:
            raise
    finally:
        # Clean up any running preload thread
        if preload_thread is not None and preload_thread.is_alive():
            logging.info("Waiting for preload thread to finish before exit...")
            preload_thread.join()
        if writer:
            writer.close()
    return global_iter


def _cleanup_memory():
    """Clean up memory between file processing."""
    aggressive_memory_cleanup()


def _train_on_file(model, train_data, val_data, optimizer, scaler, writer, 
                  runtime_params, global_iter, output_dir):
    """Train the model on a single file's data using proper epoch-based training.
    
    Each epoch iterates through the entire training dataset systematically,
    ensuring the model sees all data.
    """
    from language_model.data_handler import get_sequential_batches
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Calculate total steps per epoch
    max_start_idx = len(train_data) - runtime_params['block_size'] - 1
    steps_per_epoch = (max_start_idx + runtime_params['batch_size'] - 1) // runtime_params['batch_size']
    total_steps = runtime_params['base_training_max_epochs'] * steps_per_epoch
    
    # Setup scheduler for this file - warmup is 2% of total steps (standard practice)
    warmup_steps_local = max(1, int(0.02 * total_steps))
    scheduler = get_lr_scheduler(optimizer, warmup_steps_local,
                               runtime_params['lr_decay'], total_steps)
    
    logging.info(f"Training setup: {runtime_params['base_training_max_epochs']} epochs, "
                f"{steps_per_epoch} steps/epoch, {total_steps} total steps")

    for epoch in range(runtime_params['base_training_max_epochs']):
        # Evaluation at start of epoch and last epoch
        if epoch % runtime_params['eval_interval'] == 0 or epoch == runtime_params['base_training_max_epochs'] - 1:
            log_memory_usage("before_eval", global_iter)
            
            losses = estimate_loss(model, train_data, val_data, config.EVAL_ITERS,
                                 runtime_params['block_size'], runtime_params['batch_size'], device)

            logging.info(f"Epoch {epoch}/{runtime_params['base_training_max_epochs']}: "
                        f"train loss {losses['train']:.4f}, val loss 📉 {losses['val']:.4f}")
            log_memory_usage("after_eval", global_iter)

            # Save best model
            if losses['val'] < best_val_loss:
                logging.info(f"🎉 Validation loss improved. Saving best model.")
                best_val_loss = losses['val']
                save_best_model(model, output_dir, "best_model.pt")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if epochs_without_improvement >= runtime_params['early_stopping_patience']:
                logging.info(f"Early stopping triggered after {epoch} epochs. Best val loss: {best_val_loss:.4f}")
                model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
                break

            # Log to TensorBoard
            if global_iter is not None:
                log_memory_usage("before_tensorboard", global_iter)
                log_training_metrics(writer, losses, global_iter, scheduler.get_last_lr()[0])
                log_memory_usage("after_tensorboard", global_iter)

                # Log samples and parameters periodically
                if epoch % (runtime_params['eval_interval'] * 5) == 0:
                    tokenizer = SubwordTokenizer(vocab_file=config.VOCAB_PATH)
                    log_generated_samples(model, tokenizer, writer, global_iter, device)
                    log_model_parameters(writer, model, global_iter)
                    
                # Aggressive cleanup after TensorBoard logging (where OOM occurred)
                aggressive_memory_cleanup()
                log_memory_usage("after_cleanup", global_iter)

        # Training epoch - iterate through entire dataset
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Get gradient accumulation steps from config
        gradient_accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        
        model.train()
        
        # Iterate through all data sequentially (with shuffling)
        for xb, yb in get_sequential_batches(runtime_params['block_size'], 
                                            runtime_params['batch_size'],
                                            train_data, device, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            
            # Note: We process one batch at a time here. If you want gradient accumulation
            # with sequential batches, you'd need to batch multiple sequential batches together
            with autocast(device_type=device.type):
                logits, loss = model(xb, yb)

            scaler.scale(loss).backward()
            total_loss = loss.item()
            
            # Clean up intermediate tensors
            del xb, yb, logits
            
            # Gradient clipping and optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=runtime_params['grad_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            epoch_loss += total_loss
            num_batches += 1
            
            if global_iter is not None:
                global_iter += 1
            
            # Periodic cleanup
            if num_batches % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Log epoch timing and average loss
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logging.info(f"Epoch {epoch} completed: avg loss {avg_epoch_loss:.4f}, "
                    f"{num_batches} batches processed")
        
        if global_iter is not None:
            epoch_duration = time.time() - epoch_start_time
            log_epoch_time(writer, epoch_duration, global_iter)

        # Aggressive memory cleanup between epochs
        aggressive_memory_cleanup()

    return best_val_loss, global_iter


def train_chat_alignment(
    model: nn.Module,
    qa_tensor: torch.Tensor,
    output_dir: str,
    lr: float = 1e-4,
    batch_size: int = None,
    val_split: float = 0.1,
    global_step: int = 0,
    runtime_params: dict = None
) -> int:
    """
    Finetune model for chat alignment using QA tensor data.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    # Use runtime parameters for finetuning max epochs if provided
    if runtime_params and 'finetuning_max_epochs' in runtime_params:
        epochs = runtime_params['finetuning_max_epochs']
        logging.info(f"Using runtime override for finetuning_max_epochs: {epochs}")
    else:
        epochs = config.FINETUNING_MAX_EPOCHS
        
    os.makedirs(output_dir, exist_ok=True)
    model.train()
    device = next(model.parameters()).device
    
    # Split data
    num_samples = qa_tensor.size(0)
    split_idx = int((1.0 - val_split) * num_samples)
    train_tensor = qa_tensor[:split_idx]
    val_tensor = qa_tensor[split_idx:]

    # Setup training
    total_steps = epochs * ((train_tensor.size(0) + batch_size - 1) // batch_size)
    warmup_steps = max(10, min(200, int(0.02 * total_steps)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, lr_decay="cosine", total_steps=total_steps)
    patience = config.FINETUNE_EARLY_STOPPING_PATIENCE
    
    logging.info(f"Chat alignment: total_steps={total_steps}, warmup_steps={warmup_steps}, patience={patience}")
    
    # Enable mixed precision only for CUDA (MPS has issues with GradScaler in some PyTorch versions)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    # global_step is now passed as an argument and persists across calls
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    current_val_loss = None  # Track current validation loss
    
    logging.info(f"Total samples: {num_samples}, Train: {train_tensor.size(0)}, Val: {val_tensor.size(0)}")
    
    # ==================== FINE-TUNING START DIVIDER ====================
    logging.info("\n" + "="*80)
    logging.info(f"\033[93m🎯 STARTING CHAT ALIGNMENT FINE-TUNING\033[0m")
    logging.info(f"\033[93m⏰ START TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
    logging.info(f"\033[93m📊 TOTAL SAMPLES: {num_samples:,}\033[0m")
    logging.info(f"\033[93m🏋️  TRAIN SAMPLES: {train_tensor.size(0):,}\033[0m")
    logging.info(f"\033[93m✅ VAL SAMPLES: {val_tensor.size(0):,}\033[0m")
    logging.info(f"\033[93m🔄 EPOCHS: {epochs}\033[0m")
    logging.info(f"\033[93m📈 TOTAL STEPS: {total_steps:,}\033[0m")
    logging.info("="*80 + "\n")
    
    tokenizer = SubwordTokenizer(vocab_file=config.VOCAB_PATH)
    writer = setup_tensorboard_logging(output_dir)

    for epoch in range(epochs):
        logging.info(f"\nEpoch {epoch+1}/{epochs} START")
        model.train()
        train_loss = 0.0
        num_train_batches = (train_tensor.size(0) + batch_size - 1) // batch_size
        
        # Training loop
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
                # Log training metrics - validation loss will only be logged if current_val_loss is not None/0
                losses_dict = {'train': loss.item()}
                if current_val_loss is not None:
                    losses_dict['val'] = current_val_loss
                log_training_metrics(writer, losses_dict, global_step, scheduler.get_last_lr()[0])
                log_generated_samples(model, tokenizer, writer, global_step, device)
            
            global_step += 1
            
            if batch_idx % 1000 == 0:
                logging.info(f"  Train Batch {batch_idx+1}/{num_train_batches} - Batch Loss: {loss.item():.4f}")
                print_memory_usage()
        
        avg_train_loss = train_loss / train_tensor.size(0)
        logging.info(f"Epoch {epoch+1} TRAINING DONE. Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation loop
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
                
                if batch_idx % 250 == 0:
                    logging.info(f"  Val Batch {batch_idx+1}/{num_val_batches} - Batch Loss: {loss.item():.4f}")
        
        avg_val_loss = val_loss / val_tensor.size(0)
        current_val_loss = avg_val_loss  # Update current validation loss
        logging.info(f"Epoch {epoch+1} VALIDATION DONE. Avg Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logging.info("Early stopping triggered for chat alignment.")
            break
    
    # ==================== FINE-TUNING COMPLETION DIVIDER ====================
    logging.info("\n" + "="*80)
    logging.info(f"\033[92m🎉 COMPLETED CHAT ALIGNMENT FINE-TUNING\033[0m")
    logging.info(f"\033[92m🏆 BEST VALIDATION LOSS: {best_val_loss:.4f}\033[0m")
    logging.info(f"\033[92m🔢 FINAL GLOBAL STEP: {global_step:,}\033[0m")
    logging.info(f"\033[92m📊 EPOCHS COMPLETED: {epoch+1}/{epochs}\033[0m")
    logging.info("="*80 + "\n")
    
    # Save final model
    save_best_model(model, output_dir, "chat_aligned_model.pt")
    logging.info("Final chat-aligned model saved to chat_aligned_model.pt")
    
    if writer:
        writer.close()
    return global_step