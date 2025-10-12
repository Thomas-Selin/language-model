"""Model evaluation utilities for the language model.

This module contains functions for evaluating model performance during training,
including loss estimation and validation metrics.
"""

import torch
from torch.amp import autocast
from typing import Dict
import logging


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_iters: int,
    block_size: int,
    batch_size: int,
    device: torch.device
) -> Dict[str, float]:
    """Estimate average train and validation loss over eval_iters batches.
    
    Args:
        model: The language model to evaluate
        train_data: Training data tensor
        val_data: Validation data tensor
        eval_iters: Number of evaluation iterations
        block_size: Block size for sequences
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary with mean losses for 'train' and 'val' splits
    """
    from language_model.data_handler import get_batch
    
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        with autocast(device_type=device.type):
            for k in range(eval_iters):
                X, Y = get_batch(block_size, batch_size, split, train_data, val_data, device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()
    return out


def check_early_stopping(
    current_loss: float,
    best_loss: float,
    patience: int,
    epochs_without_improvement: int
) -> tuple[bool, int]:
    """Check if early stopping should be triggered.
    
    Args:
        current_loss: Current validation loss
        best_loss: Best validation loss seen so far
        patience: Number of epochs to wait before stopping
        epochs_without_improvement: Current count of epochs without improvement
        
    Returns:
        Tuple of (should_stop, new_epochs_without_improvement)
    """
    if current_loss < best_loss:
        # Improvement found
        return False, 0
    else:
        # No improvement
        epochs_without_improvement += 1
        should_stop = epochs_without_improvement >= patience
        return should_stop, epochs_without_improvement


def log_evaluation_metrics(
    step: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float
) -> None:
    """Log evaluation metrics in a formatted way.
    
    Args:
        step: Current training step
        train_loss: Training loss
        val_loss: Validation loss
        learning_rate: Current learning rate
    """
    logging.info(
        f"Step {step}: "
        f"train loss {train_loss:.4f}, "
        f"val loss 📉 {val_loss:.4f}, "
        f"lr {learning_rate:.6f}"
    )
