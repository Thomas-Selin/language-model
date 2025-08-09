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
from helpers import print_memory_usage, wait_for_keypress, get_device, count_parameters
from data_handler import get_batch, load_and_process_data, poll_for_new_parquet_file
from model import GPTLanguageModel
from config import BATCH_SIZE, BLOCK_SIZE, BASE_TRAINING_MAX_EPOCHS, FINETUNING_MAX_EPOCHS, EVAL_INTERVAL, LEARNING_RATE, EVAL_ITERS, N_EMBD, N_HEAD, N_LAYER, DROPOUT, EARLY_STOPPING_PATIENCE, MAX_VOCAB_SIZE, WARMUP_STEPS, LR_DECAY, LOG_LEVEL
import logging

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='\033[95m[%(levelname)s]\033[0m %(message)s'
)

# Hyperparameters (should be imported or passed in real use)
batch_size = BATCH_SIZE
block_size = BLOCK_SIZE
base_training_max_epochs = BASE_TRAINING_MAX_EPOCHS
finetuning_max_epochs = FINETUNING_MAX_EPOCHS
eval_interval = EVAL_INTERVAL
learning_rate = LEARNING_RATE
eval_iters = EVAL_ITERS
n_embd = N_EMBD
n_head = N_HEAD
n_layer = N_LAYER
dropout = DROPOUT
early_stopping_patience = EARLY_STOPPING_PATIENCE
max_vocab_size = MAX_VOCAB_SIZE
warmup_steps = WARMUP_STEPS
lr_decay = LR_DECAY
device = get_device()

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
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

def get_lr_scheduler(optimizer, warmup_steps, lr_decay, total_steps):
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

def base_train_model(parquet_dir_path, text_column='text', vocab_path='data/output/vocab.json', batch_size_files=1, training_start_time=None, output_dir=None):
    import glob
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
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
    model = GPTLanguageModel(max_vocab_size).to(device)
    logging.info(f"{count_parameters(model)/1e6:.2f} M parameters")
    logging.info(f"Model is on device: {next(model.parameters()).device}")
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
    sample_input = torch.zeros((1, block_size), dtype=torch.long).to(device)
    vis_model = ModelWrapper(model).to(device)
    writer.add_graph(vis_model, sample_input)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    global_iter = 0
    batch_counter = 0
    total_epochs_run = 0
    global_max_epochs = base_training_max_epochs
    try:
        while True:
            parquet_files = sorted(glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
            # Remove the first file if present (avoid double processing)
            # parquet_files = [f for f in parquet_files if os.path.basename(f) != file_name_single]
            if not parquet_files:
                logging.info(f"No parquet files found in {parquet_dir_path}. Waiting for new files...")
                wait_for_keypress()
                parquet_files = sorted(glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
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
                best_val_loss = float('inf')
                file_dir = os.path.dirname(parquet_file)
                file_name = os.path.basename(parquet_file)
                train_data, val_data, tokenizer, vocab_size, _ = load_and_process_data(
                    vocab_size=max_vocab_size,
                    parquet_dir_path=file_dir,
                    text_column=text_column,
                    vocab_path=vocab_path,
                    batch_size=batch_size,
                    single_file=file_name
                )
                # Set up learning rate scheduler for each file
                total_steps = base_training_max_epochs * (len(train_data) // batch_size)
                scheduler = get_lr_scheduler(optimizer, warmup_steps, lr_decay, total_steps)
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
                        if global_iter % (eval_interval * 5) == 0 or (iter == base_training_max_epochs - 1 and file_idx == len(parquet_files)-1):
                            logging.debug(f"Writing metrics and generated text to TensorBoard at step {global_iter}")
                            writer.add_scalar('Loss/train', losses['train'], global_iter)
                            writer.add_scalar('Loss/val', losses['val'], global_iter)
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
                            context = torch.zeros((1, 1), dtype=torch.long, device=device)
                            sample_text_1 = tokenizer.decode(model.generate(context, max_new_tokens=100, temperature=1.0)[0].tolist())
                            sample_text_05 = tokenizer.decode(model.generate(context, max_new_tokens=100, temperature=0.5)[0].tolist())
                            writer.add_text('Generated Text 1.0 temp', sample_text_1, global_iter)
                            writer.add_text('Generated Text 0.5 temp', sample_text_05, global_iter)
                            for prompt, temp in [("What color is the ball?", 0.5), ("What color is the ball?", 1.0), ("There was a", 0.5)]:
                                input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
                                sample_text = tokenizer.decode(model.generate(input_ids, max_new_tokens=100, temperature=temp)[0].tolist())
                                writer.add_text(f'Generated Text: "{prompt}" {temp} temp', sample_text, global_iter)
                    epoch_start_time = time.time()
                    data_time = time.time()
                    xb, yb = get_batch(block_size, batch_size, 'train', train_data, val_data)
                    data_time = time.time() - data_time
                    train_time = time.time()
                    with autocast(device_type=device.type):
                        logits, loss = model(xb, yb)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    logging.debug(f"Writing epoch duration to TensorBoard: {epoch_duration} at step {global_iter}")
                    writer.add_scalar('Total/EpochTime', epoch_duration, global_iter)
                    logging.debug(f"Writing learning rate to TensorBoard: {scheduler.get_last_lr()[0]} at step {global_iter}")
                    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_iter)
                    if iter % eval_interval == 0:
                        logging.info(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
                    global_iter += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        logging.info("Early stopping triggered. Moving to next file if available.")
                        break
                epochs_without_improvement = 0
                # Delete the file after training
                try:
                    os.remove(parquet_file)
                    logging.info(f"Deleted file after training: {parquet_file}")
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
            import glob
            seen_files = set(os.path.basename(f) for f in glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
            # Require at least 8 MB and stable size across consecutive checks
            MIN_FILE_SIZE_BYTES = 8 * 1024 * 1024  # 8 MB
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
                logging.debug(f"Current .parquet files detected: {current_files}")
                new_files = current_files - trained_files  # Only files not yet trained
                logging.debug(f"New files since last training: {new_files}")

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
                            logging.debug(f"File '{file}' size is stable and >=8 MB ({curr_size/1024/1024:.2f} MB). Stable count: {stable_count}/30")
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
                        if stable_count >= 30:
                            logging.info(f"New file detected and size stabilized (>=8 MB): {file} ({curr_size/1024/1024:.1f} MB)")
                            logging.info(f"File '{file}' appears to have finished uploading. Resuming training with new file...")
                            trained_files.add(file)  # Mark as trained AFTER processing
                            ready_file_found = True
                            break
                    
                    if ready_file_found:
                        break
                
                # Show periodic status (less frequent to reduce spam)
                if check_counter % 5 == 0:
                    logging.debug("Still waiting... (Create 'data/output/STOP_TRAINING' file to stop)")
                
                check_counter += 1
                time.sleep(1)  # Short sleep, check stop file every second
            
            if user_interrupted:
                break
    except Exception as e:
        # Handle any other exceptions (but not KeyboardInterrupt)
        logging.error(f"An error occurred during training: {e}")
        torch.save(model.state_dict(), os.path.join(output_dir, "model_error.pt"))
        logging.info("Model saved to data/output/model_error.pt due to error")
    else:
        # Normal completion
        torch.save(model.state_dict(), os.path.join(output_dir, "model_checkpoint.pt"))
        logging.info("Model saved to data/output/model_checkpoint.pt")
    
    # Always ensure we have a checkpoint for the next phase
    if not os.path.exists(os.path.join(output_dir, "model_checkpoint.pt")):
        if os.path.exists(os.path.join(output_dir, "best_model.pt")):
            # Copy best model as checkpoint for fine-tuning
            import shutil
            shutil.copy(os.path.join(output_dir, "best_model.pt"), os.path.join(output_dir, "model_checkpoint.pt"))
            logging.info("Copied best model as checkpoint for fine-tuning.")
        elif os.path.exists(os.path.join(output_dir, "model_error.pt")):
            # Copy error model as last resort
            import shutil
            shutil.copy(os.path.join(output_dir, "model_error.pt"), os.path.join(output_dir, "model_checkpoint.pt"))
            logging.info("Copied error model as checkpoint for fine-tuning.")
        else:
            # Save current state
            torch.save(model.state_dict(), os.path.join(output_dir, "model_checkpoint.pt"))
            logging.info("Saved current model state as checkpoint for fine-tuning.")
    
    if writer:
        logging.debug("[DEBUG] Closing TensorBoard writer.")
        writer.close()

def train_chat_alignment(model, qa_tensor, lr=1e-4, batch_size=batch_size, val_split=0.1, tensorboard_logdir=None, output_dir=None):
    if output_dir is None:
        output_dir = 'data/output'
    os.makedirs(output_dir, exist_ok=True)
    epochs = finetuning_max_epochs
    model.train()
    device = next(model.parameters()).device
    num_samples = qa_tensor.size(0)
    split_idx = int((1.0 - val_split) * num_samples)
    train_tensor = qa_tensor[:split_idx]
    val_tensor = qa_tensor[split_idx:]
    total_steps = epochs * ((train_tensor.size(0) + batch_size - 1) // batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=100, lr_decay="cosine", total_steps=total_steps)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    logging.info(f"Total samples: {num_samples}, Train: {train_tensor.size(0)}, Val: {val_tensor.size(0)}")
    writer = None
    if tensorboard_logdir is not None:
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
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * inputs.size(0)
            # Only log train loss every 100 steps and at the last batch
            if writer and (global_step % 100 == 0 or batch_idx == num_train_batches - 1):
                writer.add_scalar('Loss/train', loss.item(), global_step)
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
                # Only log val loss every 100 steps and at the last batch
                if writer and (global_step % 100 == 0 or batch_idx == num_val_batches - 1):
                    writer.add_scalar('Loss/val', loss.item(), global_step)
                global_step += 1
                if batch_idx % 250 == 0:
                    logging.info(f"  Val Batch {batch_idx+1}/{num_val_batches} - Batch Loss: {loss.item():.4f}")
        avg_val_loss = val_loss / val_tensor.size(0)
        logging.info(f"Epoch {epoch+1} VALIDATION DONE. Avg Val Loss: {avg_val_loss:.4f}")

        # Save best model if val loss improves
        if avg_val_loss < best_val_loss:
            logging.info(f"Validation loss improved ({avg_val_loss:.4f} < {best_val_loss:.4f}). Saving best model.")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "chat_aligned_best_model.pt"))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logging.info(f"Epochs without improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= early_stopping_patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
            logging.info("Restoring best model weights...")
            model.load_state_dict(torch.load(os.path.join(output_dir, "chat_aligned_best_model.pt")))
            break
    torch.save(model.state_dict(), os.path.join(output_dir, "chat_aligned_model.pt"))
    logging.info("Final chat-aligned model saved to data/output/chat_aligned_model.pt")
