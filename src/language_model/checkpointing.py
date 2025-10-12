"""Checkpoint management for model training.

This module handles saving and loading model checkpoints, including
model state, optimizer state, and training metadata.
"""

import torch
import os
import logging
from typing import Dict, Optional, Any
from datetime import datetime


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    output_dir: str,
    filename: str = "checkpoint.pt",
    include_optimizer: bool = True
) -> str:
    """Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch number
        global_step: Current global step
        best_val_loss: Best validation loss achieved
        output_dir: Directory to save checkpoint
        filename: Name of checkpoint file
        include_optimizer: Whether to include optimizer and scheduler state
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    if include_optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    logging.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with checkpoint metadata (epoch, global_step, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model state loaded from {checkpoint_path}")
    
    # Load optimizer state if available and requested
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("Optimizer state loaded")
    
    # Load scheduler state if available and requested
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info("Scheduler state loaded")
    
    # Return metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
    }
    
    return metadata


def save_best_model(
    model: torch.nn.Module,
    output_dir: str,
    filename: str = "best_model.pt"
) -> str:
    """Save the best model (state dict only, no optimizer).
    
    Args:
        model: Model to save
        output_dir: Directory to save model
        filename: Name of model file
        
    Returns:
        Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), model_path)
    logging.info(f"Best model saved to {model_path}")
    return model_path


def cleanup_old_checkpoints(
    output_dir: str,
    keep_best_n: int = 3,
    pattern: str = "checkpoint_*.pt"
) -> None:
    """Remove old checkpoints, keeping only the N most recent.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_best_n: Number of checkpoints to keep
        pattern: File pattern to match checkpoints
    """
    import glob
    
    checkpoint_files = glob.glob(os.path.join(output_dir, pattern))
    
    if len(checkpoint_files) <= keep_best_n:
        return
    
    # Sort by modification time (oldest first)
    checkpoint_files.sort(key=os.path.getmtime)
    
    # Remove oldest checkpoints
    files_to_remove = checkpoint_files[:-keep_best_n]
    for filepath in files_to_remove:
        try:
            os.remove(filepath)
            logging.debug(f"Removed old checkpoint: {filepath}")
        except OSError as e:
            logging.warning(f"Failed to remove checkpoint {filepath}: {e}")
    
    if files_to_remove:
        logging.info(f"Cleaned up {len(files_to_remove)} old checkpoints")
