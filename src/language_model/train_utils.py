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
from config import BATCH_SIZE, BLOCK_SIZE, BASE_TRAINING_MAX_EPOCHS, FINETUNING_MAX_EPOCHS, EVAL_INTERVAL, LEARNING_RATE, EVAL_ITERS, N_EMBD, N_HEAD, N_LAYER, DROPOUT, EARLY_STOPPING_PATIENCE, MAX_VOCAB_SIZE, WARMUP_STEPS, LR_DECAY

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

def base_train_model(parquet_dir_path, text_column='text', vocab_path='data/output/vocab.json', batch_size_files=1, training_start_time=None):
    import glob
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir_path}")
        return
    file_name_single = os.path.basename(parquet_files[0])
    train_data, val_data, tokenizer, vocab_size, _ = load_and_process_data(
        vocab_size=max_vocab_size,
        parquet_dir_path=parquet_dir_path,
        text_column=text_column,
        vocab_path=vocab_path,
        batch_size=batch_size,
        single_file=file_name_single
    )
    # Remove the first file from the list to avoid double processing
    parquet_files = [f for f in parquet_files if os.path.basename(f) != file_name_single]
    log_dir = os.path.join('data', 'output', 'tensorboard_logs', training_start_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"\033[92mTensorBoard logging started View logs with: tensorboard --logdir={log_dir}\033[0m")
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
    - Tokenizer vocab size: {vocab_size}
    - Tokenizer vocab path: {vocab_path}
    - Train data size: {len(train_data)} tokens
    - Validation data size: {len(val_data)} tokens
    ## Timing
    - Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    model = GPTLanguageModel(vocab_size).to(device)
    print(f"{count_parameters(model)/1e6:.2f} M parameters")
    print(f"Model is on device: {next(model.parameters()).device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = base_training_max_epochs * (len(train_data) // batch_size)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, lr_decay, total_steps)
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
                print(f"No parquet files found in {parquet_dir_path}. Waiting for new files...")
                wait_for_keypress()
                parquet_files = sorted(glob.glob(os.path.join(parquet_dir_path, '*.parquet')))
                parquet_files = [f for f in parquet_files if os.path.basename(f) != file_name_single]
                if not parquet_files:
                    print("No files present after keypress. Base training finished.")
                    break
                continue
            for file_idx, parquet_file in enumerate(parquet_files):
                print("Performing memory cleanup before loading new file...")
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
                print_memory_usage()
                print(f"Processing file {file_idx+1}/{len(parquet_files)}: {parquet_file}. Setting best_val_loss to infinity.")
                best_val_loss = float('inf')
                file_dir = os.path.dirname(parquet_file)
                file_name = os.path.basename(parquet_file)
                train_data, val_data, _, _, _ = load_and_process_data(
                    vocab_size=max_vocab_size,
                    parquet_dir_path=file_dir,
                    text_column=text_column,
                    vocab_path=vocab_path,
                    batch_size=batch_size,
                    single_file=file_name
                )
                for iter in range(base_training_max_epochs):
                    if total_epochs_run >= global_max_epochs:
                        print(f"Reached global epoch limit of {global_max_epochs}. Stopping training.")
                        break
                    if iter % eval_interval == 0 or iter == base_training_max_epochs - 1:
                        losses = estimate_loss(model, train_data, val_data)
                        print("\033[94m____________________________\033[0m\n")
                        print_memory_usage()
                        print(f"File {file_idx+1}, Step {iter} of max {base_training_max_epochs}: train loss {losses['train']:.4f}, \U0001F4CF val loss \033[94m{losses['val']:.4f}\033[0m")
                        
                        # Save best model if val loss improves
                        if losses['val'] < best_val_loss:
                            print(f"Validation loss improved ({losses['val']:.4f} < {best_val_loss:.4f}). Saving best model.")
                            best_val_loss = losses['val']
                            torch.save(model.state_dict(), "data/output/best_model.pt")
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1
                            print(f"Epochs without improvement: {epochs_without_improvement}")
                        
                        if epochs_without_improvement >= early_stopping_patience:
                            print(f"Early stopping triggered at epoch {iter}. Best val loss: {best_val_loss:.4f}")
                            print("Restoring best model weights...")
                            model.load_state_dict(torch.load("data/output/best_model.pt"))
                            break
                        writer.add_scalar('Loss/train', losses['train'], global_iter)
                        writer.add_scalar('Loss/val', losses['val'], global_iter)
                        if torch.cuda.is_available():
                            writer.add_scalar('Memory/allocated_GB', torch.cuda.memory_allocated() / 1024**3, global_iter)
                            writer.add_scalar('Memory/reserved_GB', torch.cuda.memory_reserved() / 1024**3, global_iter)
                            writer.add_scalar('Memory/free_GB', (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3, global_iter)
                        if global_iter % (eval_interval * 10) == 0:
                            for name, param in model.named_parameters():
                                writer.add_histogram(f'Parameters/{name}', param, global_iter)
                                if param.grad is not None:
                                    writer.add_histogram(f'Gradients/{name}', param.grad, global_iter)
                        if global_iter % (eval_interval * 5) == 0 or (iter == base_training_max_epochs - 1 and file_idx == len(parquet_files)-1):
                            context = torch.zeros((1, 1), dtype=torch.long, device=device)
                            sample_text = tokenizer.decode(model.generate(context, max_new_tokens=100, temperature=1.0)[0].tolist())
                            writer.add_text('Generated Text 1.0 temp', sample_text, global_iter)
                            sample_text = tokenizer.decode(model.generate(context, max_new_tokens=100, temperature=0.5)[0].tolist())
                            writer.add_text('Generated Text 0.5 temp', sample_text, global_iter)
                            input_ids = torch.tensor([tokenizer.encode("What color is the ball?")], dtype=torch.long, device=device)
                            sample_text = tokenizer.decode(model.generate(input_ids, max_new_tokens=100, temperature=0.5)[0].tolist())
                            writer.add_text('Generated Text: "What color is the ball?" 0.5 temp', sample_text, global_iter)
                            input_ids = torch.tensor([tokenizer.encode("What color is the ball?")], dtype=torch.long, device=device)
                            sample_text = tokenizer.decode(model.generate(input_ids, max_new_tokens=100, temperature=1.0)[0].tolist())
                            writer.add_text('Generated Text: "What color is the ball?" 1.0 temp', sample_text, global_iter)
                            input_ids = torch.tensor([tokenizer.encode("There was a")], dtype=torch.long, device=device)
                            sample_text = tokenizer.decode(model.generate(input_ids, max_new_tokens=100, temperature=0.5)[0].tolist())
                            writer.add_text('Generated Text: "There was a" 0.5 temp', sample_text, global_iter)
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
                    # print(f"File {file_idx+1}, Epoch {iter}: Data time: {data_time:.4f}s, Train time: {train_time:.4f}s, Total epoch time: {epoch_duration:.4f}s")
                    writer.add_scalar('Total/EpochTime', epoch_duration, global_iter)
                    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_iter)
                    if iter % eval_interval == 0:
                        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
                    global_iter += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        print("Early stopping triggered. Moving to next file if available.")
                        break
                epochs_without_improvement = 0
                # Delete the file after training
                try:
                    os.remove(parquet_file)
                    print(f"Deleted file after training: {parquet_file}")
                except Exception as e:
                    print(f"Error deleting file {parquet_file}: {e}")
            print("\n\033[94mAll files in folder processed. Waiting for new files... (Press 'q' then Enter to finish base training)\033[0m\n")
            # Poll for new files, but allow user to quit by pressing 'q' and Enter
            import sys
            import select
            print("Polling for new files. Press 'q' then Enter at any time to finish base training.")
            user_input = None  # Fix: ensure user_input is always defined
            while True:
                # Check for user input (non-blocking)
                print("Waiting for new files...", end='\r', flush=True)
                if sys.stdin in select.select([sys.stdin], [], [], 5)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input == 'q':
                        print("Base training finished by user request.")
                        break
                # Use poll_for_new_parquet_file to wait for a fully uploaded file
                new_file = poll_for_new_parquet_file(parquet_dir_path, poll_interval=2)
                if new_file:
                    break
            if user_input == 'q':
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model checkpoint...")
        torch.save(model.state_dict(), "data/output/model_interrupted.pt")
        print("Model saved to data/output/model_interrupted.pt")
    else:
        torch.save(model.state_dict(), "data/output/model_checkpoint.pt")
        print("Model saved to data/output/model_checkpoint.pt")
    writer.close()

def train_chat_alignment(model, qa_tensor, epochs=1, lr=1e-4, batch_size=batch_size, val_split=0.1, tensorboard_logdir=None):
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
    print(f"Total samples: {num_samples}, Train: {train_tensor.size(0)}, Val: {val_tensor.size(0)}")
    writer = None
    if tensorboard_logdir is not None:
        writer = SummaryWriter(tensorboard_logdir)
        print(f"\033[92mTensorBoard logging started. View logs with: tensorboard --logdir={tensorboard_logdir}\033[0m")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} START")
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
            if writer:
                writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
            if batch_idx % 1000 == 0:
                print(f"  Train Batch {batch_idx+1}/{num_train_batches} - Batch Loss: {loss.item():.4f}")
                print_memory_usage()
        avg_train_loss = train_loss / train_tensor.size(0)
        print(f"Epoch {epoch+1} TRAINING DONE. Avg Train Loss: {avg_train_loss:.4f}")
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
                if writer:
                    writer.add_scalar('Loss/val', loss.item(), global_step)
                global_step += 1
                if batch_idx % 250 == 0:
                    print(f"  Val Batch {batch_idx+1}/{num_val_batches} - Batch Loss: {loss.item():.4f}")
        avg_val_loss = val_loss / val_tensor.size(0)
        print(f"Epoch {epoch+1} VALIDATION DONE. Avg Val Loss: {avg_val_loss:.4f}")

        # Save best model if val loss improves
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved ({avg_val_loss:.4f} < {best_val_loss:.4f}). Saving best model.")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "data/output/chat_aligned_best_model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Epochs without improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
            print("Restoring best model weights...")
            model.load_state_dict(torch.load("data/output/chat_aligned_best_model.pt"))
            break
    torch.save(model.state_dict(), "data/output/chat_aligned_model.pt")
    print("Final chat-aligned model saved to data/output/chat_aligned_model.pt")


def train_and_poll(parquet_dir_path, text_column='text', vocab_path='data/output/vocab_subword.json', batch_size_files=1, training_start_time=None):
    """
    Polls for new parquet files, trains on each, deletes after training, then exits when no new files remain.
    Allows user to quit by pressing 'q' and Enter while waiting for new files.
    """
    import sys
    import select
    import time
    while True:
        # Poll for a new, fully uploaded parquet file
        print("Polling for new files. Press 'q' then Enter at any time to finish base training.")
        while True:
            # Non-blocking check for user input
            print("Waiting for new files...", end='\r', flush=True)
            if sys.stdin in select.select([sys.stdin], [], [], 5)[0]:
                user_input = sys.stdin.readline().strip().lower()
                if user_input == 'q':
                    print("Base training finished by user request.")
                    return
            new_file = poll_for_new_parquet_file(parquet_dir_path, poll_interval=2)
            if new_file:
                break
        # Train on the single file
        print(f"Training on file: {new_file}")
        base_train_model(parquet_dir_path, text_column, vocab_path, batch_size_files, training_start_time)
        print(f"Finished training on {new_file}")
        # Delete the file after training
        new_file_path = os.path.join(parquet_dir_path, new_file)
        try:
            os.remove(new_file_path)
            print(f"Deleted file after training: {new_file_path}")
        except Exception as e:
            print(f"Error deleting file {new_file_path}: {e}")
        # Wait a short time before polling again
        time.sleep(2)
    print("All files processed. Proceeding to fine-tuning.")