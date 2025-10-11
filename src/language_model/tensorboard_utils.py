"""
TensorBoard utilities for logging metrics, generated samples, and model information.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime
import logging
from language_model.subword_tokenizer import SubwordTokenizer


def setup_tensorboard_logging(output_dir: str) -> SummaryWriter:
    """
    Set up TensorBoard logging and return the writer.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        TensorBoard SummaryWriter instance
    """
    import os
    log_dir = os.path.join(output_dir, 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logging.info(f"\033[92mTensorBoard logging started. View logs with: tensorboard --logdir={log_dir}\033[0m")
    return writer


def log_hyperparameters(writer: SummaryWriter, config_dict: dict) -> None:
    """
    Log hyperparameters and session information to TensorBoard.
    
    Args:
        writer: TensorBoard writer
        config_dict: Dictionary containing hyperparameters and settings
    """
    hyperparams_text = f"""
    ## Hyperparameters
    - Batch size: {config_dict.get('batch_size', 'N/A')}
    - Block size: {config_dict.get('block_size', 'N/A')}
    - Max epochs: {config_dict.get('max_epochs', 'N/A')}
    - Eval interval: {config_dict.get('eval_interval', 'N/A')}
    - Learning rate: {config_dict.get('learning_rate', 'N/A')}
    - Eval iters: {config_dict.get('eval_iters', 'N/A')}
    - Embedding dimension: {config_dict.get('n_embd', 'N/A')}
    - Number of heads: {config_dict.get('n_head', 'N/A')}
    - Number of layers: {config_dict.get('n_layer', 'N/A')}
    - Dropout: {config_dict.get('dropout', 'N/A')}
    - Early stopping patience: {config_dict.get('early_stopping_patience', 'N/A')}
    - Max vocab size: {config_dict.get('max_vocab_size', 'N/A')}
    ## Environment
    - Device: {config_dict.get('device', 'N/A')}
    ## Data
    - Tokenizer vocab path: {config_dict.get('vocab_path', 'N/A')}
    ## Timing
    - Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    writer.add_text('Session Information', hyperparams_text)


def log_model_graph(writer: SummaryWriter, model: nn.Module, block_size: int, device: torch.device) -> None:
    """
    Log the model computational graph to TensorBoard.
    
    Args:
        writer: TensorBoard writer
        model: The language model
        block_size: Input sequence length
        device: Device to create sample input on
    """
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            orig = getattr(self.model, 'use_checkpoint', False)
            if hasattr(self.model, 'use_checkpoint'):
                self.model.use_checkpoint = False
            logits, _ = self.model(x)
            if hasattr(self.model, 'use_checkpoint'):
                self.model.use_checkpoint = orig
            return logits
    
    sample_input = torch.zeros((1, block_size), dtype=torch.long).to(device)
    vis_model = ModelWrapper(model).to(device)
    writer.add_graph(vis_model, sample_input)


def log_training_metrics(writer: SummaryWriter, losses: dict, global_iter: int, lr: float = None) -> None:
    """
    Log training metrics to TensorBoard.
    
    Args:
        writer: TensorBoard writer
        losses: Dictionary with 'train' and optionally 'val' losses
        global_iter: Current global iteration
        lr: Current learning rate (optional)
    """
    writer.add_scalar('Loss/train', losses['train'], global_iter)
    writer.add_scalar('Perplexity/train', float(torch.exp(losses['train'].detach().clone() if torch.is_tensor(losses['train']) else torch.tensor(losses['train']))), global_iter)
    
    # Only log validation metrics if validation loss is provided and not 0
    if 'val' in losses and losses['val'] != 0:
        writer.add_scalar('Loss/val', losses['val'], global_iter)
        writer.add_scalar('Perplexity/val', float(torch.exp(losses['val'].detach().clone() if torch.is_tensor(losses['val']) else torch.tensor(losses['val']))), global_iter)
    
    if lr is not None:
        writer.add_scalar('Learning Rate', lr, global_iter)
    
    # Log GPU memory usage if available
    if torch.cuda.is_available():
        writer.add_scalar('Memory/allocated_GB', torch.cuda.memory_allocated() / 1024**3, global_iter)
        writer.add_scalar('Memory/reserved_GB', torch.cuda.memory_reserved() / 1024**3, global_iter)
        writer.add_scalar('Memory/free_GB', 
                         (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3, 
                         global_iter)
        
    logging.debug(f"Logged metrics at global step {global_iter}: {losses}")


def log_generated_samples(model: nn.Module, tokenizer: SubwordTokenizer, writer: SummaryWriter, 
                         global_step: int, device: torch.device) -> None:
    """
    Log unconditional and prompted generations to TensorBoard.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for decoding outputs
        writer: TensorBoard SummaryWriter
        global_step: Current global step for logging
        device: Device to run generation on
    """
    model.eval()
    
    prompts = [
        ("Which Emmy award was American Idol nominated for nine times?", "Emmy Question"),
        ("Where is Paris?", "Geography Question")
    ]
    
    for temp in [0.5, 0.8, 1.0]:
        # Unconditional generation (no prompt)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        try:
            sample_text = tokenizer.decode(model.generate(context, temperature=temp)[0].tolist())
            writer.add_text(f'Generated Text (no prompt) {temp} temp', sample_text, global_step)
        except Exception as e:
            logging.debug(f"Error generating unconditional sample: {e}")
        
        # Prompted generation
        for prompt, prompt_name in prompts:
            try:
                input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
                sample_text = tokenizer.decode(model.generate(input_ids, temperature=temp)[0].tolist())
                writer.add_text(f'Generated Text: "{prompt}" {temp} temp', sample_text, global_step)
            except Exception as e:
                logging.debug(f"Error generating sample for prompt '{prompt}': {e}")
    
    model.train()


def log_model_parameters(writer: SummaryWriter, model: nn.Module, global_iter: int) -> None:
    """
    Log model parameter and gradient histograms to TensorBoard.
    
    Args:
        writer: TensorBoard writer
        model: The language model
        global_iter: Current global iteration
    """
    for name, param in model.named_parameters():
        writer.add_histogram(f'Parameters/{name}', param, global_iter)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, global_iter)


def log_epoch_time(writer: SummaryWriter, epoch_duration: float, global_iter: int) -> None:
    """
    Log epoch duration to TensorBoard.
    
    Args:
        writer: TensorBoard writer
        epoch_duration: Duration of the epoch in seconds
        global_iter: Current global iteration
    """
    writer.add_scalar('Total/EpochTime', epoch_duration, global_iter)
