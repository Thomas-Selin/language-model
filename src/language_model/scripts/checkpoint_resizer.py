import torch
import logging
from language_model.subword_tokenizer import SubwordTokenizer
from language_model.model import GPTLanguageModel
import os


def resize_model_checkpoint(
    checkpoint_path: str,
    old_vocab_size: int,
    new_vocab_size: int,
    output_path: str = None,
    vocab_file: str = None,
    remove_lm_head_bias: bool = True
) -> str:
    """
    Resize a model checkpoint to use a different vocabulary size.
    
    Args:
        checkpoint_path: Path to the existing checkpoint
        old_vocab_size: Original vocabulary size the model was trained with
        new_vocab_size: New vocabulary size to resize to
        output_path: Where to save the resized checkpoint (optional)
        vocab_file: Path to vocabulary file for token mapping (optional)
        remove_lm_head_bias: Whether to remove lm_head bias (for weight tying compatibility)
    
    Returns:
        Path to the resized checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if output_path is None:
        base_name = os.path.splitext(checkpoint_path)[0]
        output_path = f"{base_name}.pt"
    
    logging.info(f"Resizing checkpoint from vocab_size {old_vocab_size} to {new_vocab_size}")
    logging.info(f"Input: {checkpoint_path}")
    logging.info(f"Output: {output_path}")
    
    # Load the original checkpoint
    device = torch.device('cpu')  # Load on CPU to avoid memory issues
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract embedding and language model head weights and bias
    token_embedding_weight = checkpoint['token_embedding_table.weight']  # Shape: [old_vocab_size, n_embd]
    lm_head_weight = checkpoint['lm_head.weight']  # Shape: [old_vocab_size, n_embd] (same as embedding due to weight tying)
    
    logging.info(f"Original embedding shape: {token_embedding_weight.shape}")
    logging.info(f"Original lm_head shape: {lm_head_weight.shape}")
    
    # Check if lm_head has bias
    has_lm_head_bias = 'lm_head.bias' in checkpoint
    if has_lm_head_bias:
        lm_head_bias = checkpoint['lm_head.bias']  # Shape: [old_vocab_size]
        logging.info(f"Original lm_head bias shape: {lm_head_bias.shape}")
    
    # Resize the embeddings
    if new_vocab_size <= old_vocab_size:
        # Truncate - keep the first new_vocab_size tokens
        logging.info(f"Truncating vocabulary from {old_vocab_size} to {new_vocab_size}")
        new_token_embedding = token_embedding_weight[:new_vocab_size, :]
        new_lm_head = lm_head_weight[:new_vocab_size, :]
        if has_lm_head_bias:
            new_lm_head_bias = lm_head_bias[:new_vocab_size]
    else:
        # Expand - need to add new token embeddings
        logging.info(f"Expanding vocabulary from {old_vocab_size} to {new_vocab_size}")
        n_embd = token_embedding_weight.shape[1]
        
        # Keep existing embeddings
        new_token_embedding = torch.zeros(new_vocab_size, n_embd, dtype=token_embedding_weight.dtype)
        new_lm_head = torch.zeros(new_vocab_size, n_embd, dtype=lm_head_weight.dtype)
        
        # Copy existing embeddings
        new_token_embedding[:old_vocab_size, :] = token_embedding_weight
        new_lm_head[:old_vocab_size, :] = lm_head_weight
        
        # Initialize new embeddings with small random values (same as original initialization)
        std = 0.02  # Same as in model._init_weights
        new_token_embedding[old_vocab_size:, :] = torch.normal(0.0, std, (new_vocab_size - old_vocab_size, n_embd))
        new_lm_head[old_vocab_size:, :] = torch.normal(0.0, std, (new_vocab_size - old_vocab_size, n_embd))
        
        # Handle bias if it exists
        if has_lm_head_bias:
            new_lm_head_bias = torch.zeros(new_vocab_size, dtype=lm_head_bias.dtype)
            new_lm_head_bias[:old_vocab_size] = lm_head_bias
            # New bias entries remain zero (standard initialization)
    
    # Update the checkpoint
    checkpoint['token_embedding_table.weight'] = new_token_embedding
    checkpoint['lm_head.weight'] = new_lm_head
    if has_lm_head_bias and not remove_lm_head_bias:
        checkpoint['lm_head.bias'] = new_lm_head_bias
    elif has_lm_head_bias and remove_lm_head_bias:
        # Remove the bias for weight tying compatibility
        logging.info("Removing lm_head.bias for weight tying compatibility")
        del checkpoint['lm_head.bias']
    
    logging.info(f"New embedding shape: {new_token_embedding.shape}")
    logging.info(f"New lm_head shape: {new_lm_head.shape}")
    if has_lm_head_bias:
        if remove_lm_head_bias:
            logging.info("Removed lm_head bias for weight tying compatibility")
        else:
            logging.info(f"New lm_head bias shape: {new_lm_head_bias.shape}")
    else:
        logging.info("No lm_head bias found in checkpoint")
    
    # Save the resized checkpoint
    torch.save(checkpoint, output_path)
    logging.info(f"Resized checkpoint saved to: {output_path}")
    
    return output_path


def verify_resized_checkpoint(
    resized_checkpoint_path: str,
    expected_vocab_size: int,
    original_checkpoint_path: str = None
) -> bool:
    """
    Verify that the resized checkpoint is valid and can be loaded.
    
    Args:
        resized_checkpoint_path: Path to the resized checkpoint
        expected_vocab_size: Expected vocabulary size
        original_checkpoint_path: Original checkpoint for comparison (optional)
    
    Returns:
        True if verification passes
    """
    try:
        logging.info(f"Verifying resized checkpoint: {resized_checkpoint_path}")
        
        # Try to create a model with the new vocab size and load the checkpoint
        model = GPTLanguageModel(expected_vocab_size)
        checkpoint = torch.load(resized_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        
        # Check shapes
        embedding_shape = model.token_embedding_table.weight.shape
        lm_head_shape = model.lm_head.weight.shape
        
        logging.info(f"Verified embedding shape: {embedding_shape}")
        logging.info(f"Verified lm_head shape: {lm_head_shape}")
        
        if embedding_shape[0] != expected_vocab_size:
            logging.error(f"Embedding vocab size mismatch: expected {expected_vocab_size}, got {embedding_shape[0]}")
            return False
        
        if lm_head_shape[0] != expected_vocab_size:
            logging.error(f"LM head vocab size mismatch: expected {expected_vocab_size}, got {lm_head_shape[0]}")
            return False
        
        # Test a forward pass
        test_input = torch.randint(0, expected_vocab_size, (1, 10))
        with torch.no_grad():
            logits, _ = model(test_input)
        
        if logits.shape[-1] != expected_vocab_size:
            logging.error(f"Output vocab size mismatch: expected {expected_vocab_size}, got {logits.shape[-1]}")
            return False
        
        logging.info("✅ Checkpoint verification passed!")
        return True
        
    except Exception as e:
        logging.error(f"Checkpoint verification failed: {e}")
        return False


def resize_checkpoint_for_actual_vocab(
    checkpoint_path: str,
    vocab_file: str,
    output_path: str = None
) -> str:
    """
    Convenience function to resize a checkpoint to match the actual vocabulary size.
    
    Args:
        checkpoint_path: Path to the existing checkpoint
        vocab_file: Path to the vocabulary file
        output_path: Where to save the resized checkpoint (optional)
    
    Returns:
        Path to the resized checkpoint
    """
    # Load tokenizer to get actual vocab size
    tokenizer = SubwordTokenizer(vocab_file=vocab_file)
    actual_vocab_size = tokenizer.get_vocab_size()
    
    logging.info(f"Detected actual vocabulary size: {actual_vocab_size}")
    
    # Determine the original vocab size from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    original_vocab_size = checkpoint['token_embedding_table.weight'].shape[0]
    
    logging.info(f"Original checkpoint vocabulary size: {original_vocab_size}")
    
    if original_vocab_size == actual_vocab_size:
        logging.info("Vocabulary sizes already match - no resizing needed")
        return checkpoint_path
    
    # Resize the checkpoint
    resized_path = resize_model_checkpoint(
        checkpoint_path=checkpoint_path,
        old_vocab_size=original_vocab_size,
        new_vocab_size=actual_vocab_size,
        output_path=output_path,
        vocab_file=vocab_file
    )
    
    # Verify the resized checkpoint
    if verify_resized_checkpoint(resized_path, actual_vocab_size, checkpoint_path):
        logging.info("✅ Successfully resized and verified checkpoint")
    else:
        logging.error("❌ Checkpoint verification failed")
        raise RuntimeError("Resized checkpoint verification failed")
    
    return resized_path


if __name__ == "__main__":
    # Example usage
    import sys
    from language_model.config import VOCAB_PATH
    
    if len(sys.argv) < 2:
        print("Usage: python checkpoint_resizer.py <checkpoint_path> [output_path]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        resized_path = resize_checkpoint_for_actual_vocab(
            checkpoint_path=checkpoint_path,
            vocab_file=VOCAB_PATH,
            output_path=output_path
        )
        print(f"✅ Successfully resized checkpoint: {resized_path}")
    except Exception as e:
        print(f"❌ Error resizing checkpoint: {e}")
        sys.exit(1)
