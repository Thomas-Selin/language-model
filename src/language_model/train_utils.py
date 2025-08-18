import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import gc
import os
import time
import datetime
from helpers import configure_colored_logging, print_memory_usage, get_device, count_parameters
from helpers import apply_runtime_overrides, get_lr_scheduler
from data_handler import estimate_loss, load_and_process_data, get_batch
from model import GPTLanguageModel
from training_helpers import (
    wait_for_new_files_or_stop, preload_parquet_data, cleanup_processed_file,
    get_parquet_files, setup_output_directory
)
from tensorboard_utils import (
    setup_tensorboard_logging, log_hyperparameters, log_model_graph,
    log_training_metrics, log_generated_samples, log_model_parameters, log_epoch_time
)
import config
import logging
from subword_tokenizer import SubwordTokenizer

# Configure logging
configure_colored_logging(config.LOG_LEVEL)

# Set device
device = get_device()


def base_train_model(
    parquet_dir_path: str,
    text_column: str = 'text',
    vocab_path: str = 'data/output/vocab_subword.json',
    training_start_time: str = None,
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
        output_dir = os.path.join('data', 'output', training_start_time)
    setup_output_directory(output_dir)
    
    # Initialize TensorBoard
    writer = setup_tensorboard_logging(output_dir)
    
    # Create model
    model = GPTLanguageModel(config.MAX_VOCAB_SIZE).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logging.info(f"\033[92mResumed model weights loaded from {checkpoint_path}\033[0m")
        except (OSError, RuntimeError) as e:
            logging.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    
    logging.info(f"{count_parameters(model)/1e6:.2f} M parameters")
    
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.DEFAULT_WEIGHT_DECAY)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Log model graph
    log_model_graph(writer, model, config.BLOCK_SIZE, device)
    
    # Training state
    global_iter = getattr(config, 'GLOBAL_ITER', 0)
    total_epochs_run = 0
    tensorboard_logged = False
    
    # Runtime parameters
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
    
    stop_file_path = "data/output/STOP_TRAINING"
    trained_files = set()
    
    try:

        preloaded_data = None  # Store preloaded data from previous iteration
        preload_thread, preload_result, next_file_name = None, None, None
        while True:
            parquet_files = get_parquet_files(parquet_dir_path)
            if not parquet_files:
                logging.info(f"No parquet files found in {parquet_dir_path}. Waiting for new files...")
                user_interrupted, ready_file = wait_for_new_files_or_stop(parquet_dir_path, trained_files, stop_file_path)
                if user_interrupted:
                    break
                if ready_file:
                    trained_files.add(ready_file)
                continue

            file_count = len(parquet_files)
            file_idx = 0
            while file_idx < file_count:
                parquet_file = parquet_files[file_idx]
                file_name = os.path.basename(parquet_file)
                if file_name in trained_files:
                    file_idx += 1
                    continue

                trained_files.add(file_name)
                _cleanup_memory()
                logging.info(f"Processing file {file_idx+1}/{file_count}: {parquet_file}")

                # Use preloaded data if available
                if preloaded_data and preloaded_data.get('file_name') == file_name:
                    logging.info(f"Using preloaded data for file: {file_name}")
                    train_data, val_data, tokenizer, vocab_size, _ = preloaded_data['data']
                    preloaded_data = None  # Clear preloaded data after use
                else:
                    train_data, val_data, tokenizer, vocab_size, _ = load_and_process_data(
                        vocab_size=config.MAX_VOCAB_SIZE,
                        parquet_dir_path=os.path.dirname(parquet_file),
                        text_column=text_column,
                        vocab_path=vocab_path,
                        batch_size=runtime_params['batch_size'],
                        single_file=file_name
                    )

                # Start preloading the next file as soon as training for current file begins
                next_file = parquet_files[file_idx + 1] if file_idx + 1 < file_count else None
                next_file_name = None
                if next_file and (preload_thread is None):
                    next_file_dir = os.path.dirname(next_file)
                    next_file_name = os.path.basename(next_file)
                    if next_file_name not in trained_files:
                        logging.info(f"Starting preload of next file: {next_file_name}")
                        preload_thread, preload_result = preload_parquet_data(
                            next_file_name, config.MAX_VOCAB_SIZE, next_file_dir,
                            text_column, vocab_path, runtime_params['batch_size']
                        )

                # Apply runtime overrides
                runtime_params, extra_counters = apply_runtime_overrides(
                    optimizer, None, runtime_params, config.RUNTIME_OVERRIDES_FILE
                )

                if 'global_iter' in extra_counters:
                    global_iter = extra_counters['global_iter']
                if 'total_epochs_run' in extra_counters:
                    total_epochs_run = extra_counters['total_epochs_run']

                if not tensorboard_logged:
                    config_dict = {
                        'batch_size': runtime_params['batch_size'],
                        'block_size': runtime_params['block_size'],
                        'max_epochs': config.BASE_TRAINING_MAX_EPOCHS,
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

                best_val_loss, global_iter = _train_on_file(
                    model, train_data, val_data, optimizer, scaler, writer,
                    runtime_params, global_iter, output_dir
                )

                # Wait for preloading to complete (if it was started)
                if preload_thread is not None and next_file_name is not None:
                    logging.info("Waiting for next file preload to complete...")
                    preload_thread.join()
                    if 'error' in preload_result:
                        logging.warning(f"Preload failed: {preload_result['error']}")
                        preloaded_data = None
                    elif 'data' in preload_result:
                        logging.info("Next file preload completed successfully")
                        preloaded_data = {
                            'file_name': next_file_name,
                            'data': preload_result['data']
                        }
                    else:
                        logging.warning("Preload completed but no data found")
                        preloaded_data = None
                    preload_thread, preload_result, next_file_name = None, None, None

                cleanup_processed_file(parquet_file)

                if total_epochs_run >= config.BASE_TRAINING_MAX_EPOCHS:
                    logging.info(f"Reached global epoch limit. Stopping training.")
                    break

                file_idx += 1

            user_interrupted, ready_file = wait_for_new_files_or_stop(parquet_dir_path, trained_files, stop_file_path)
            if user_interrupted:
                break
            if ready_file:
                trained_files.add(ready_file)

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        torch.save(model.state_dict(), os.path.join(output_dir, "model_error.pt"))
        logging.info("Model saved to model_error.pt due to error")
    finally:
        if writer:
            writer.close()
    return global_iter


def _cleanup_memory():
    """Clean up memory between file processing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _train_on_file(model, train_data, val_data, optimizer, scaler, writer, 
                  runtime_params, global_iter, output_dir):
    """Train the model on a single file's data."""
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Setup scheduler for this file
    steps_per_file = config.BASE_TRAINING_MAX_EPOCHS
    warmup_steps_local = max(1, int(0.02 * steps_per_file))
    scheduler = get_lr_scheduler(optimizer, warmup_steps_local,
                               runtime_params['lr_decay'], steps_per_file)

    for iter in range(config.BASE_TRAINING_MAX_EPOCHS):
        # Evaluation
        if iter % runtime_params['eval_interval'] == 0 or iter == config.BASE_TRAINING_MAX_EPOCHS - 1:
            losses = estimate_loss(model, train_data, val_data, config.EVAL_ITERS,
                                 runtime_params['block_size'], runtime_params['batch_size'])

            logging.info(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Save best model
            if losses['val'] < best_val_loss:
                logging.info(f"Validation loss improved. Saving best model.")
                best_val_loss = losses['val']
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if epochs_without_improvement >= runtime_params['early_stopping_patience']:
                logging.info(f"Early stopping triggered. Best val loss: {best_val_loss:.4f}")
                model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
                break

            # Log to TensorBoard
            if global_iter is not None:
                log_training_metrics(writer, losses, global_iter, scheduler.get_last_lr()[0])

                # Log samples and parameters periodically
                if global_iter % (runtime_params['eval_interval'] * 25) == 0:
                    tokenizer = SubwordTokenizer(vocab_file=config.VOCAB_PATH)
                    log_generated_samples(model, tokenizer, writer, global_iter, device)
                    log_model_parameters(writer, model, global_iter)

        # Training step
        epoch_start_time = time.time()
        xb, yb = get_batch(runtime_params['block_size'], runtime_params['batch_size'],
                          'train', train_data, val_data)

        with autocast(device_type=device.type):
            logits, loss = model(xb, yb)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=runtime_params['grad_clip_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # Log timing
        if global_iter is not None:
            epoch_duration = time.time() - epoch_start_time
            log_epoch_time(writer, epoch_duration, global_iter)
            global_iter += 1

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if global_iter is not None and global_iter % 5 == 0:
            gc.collect()

    return best_val_loss, global_iter


def train_chat_alignment(
    model: nn.Module,
    qa_tensor: torch.Tensor,
    output_dir: str,
    lr: float = 1e-4,
    batch_size: int = None,
    val_split: float = 0.1,
    global_step: int = 0
) -> int:
    """
    Finetune model for chat alignment using QA tensor data.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    os.makedirs(output_dir, exist_ok=True)
    epochs = config.FINETUNING_MAX_EPOCHS
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
    
    scaler = GradScaler(enabled=torch.cuda.is_available())
    # global_step is now passed as an argument and persists across calls
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    current_val_loss = None  # Track current validation loss
    
    logging.info(f"Total samples: {num_samples}, Train: {train_tensor.size(0)}, Val: {val_tensor.size(0)}")
    
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
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "chat_aligned_model.pt"))
    logging.info("Final chat-aligned model saved to chat_aligned_model.pt")
    
    if writer:
        writer.close()
    return global_step