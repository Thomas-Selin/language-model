#!/usr/bin/env python3
"""
Test script to verify memory optimizations are working
"""
import os

import torch
from language_model.train_utils import optimize_memory_settings, aggressive_memory_cleanup
from language_model.model import GPTLanguageModel
from language_model.config import MAX_VOCAB_SIZE, BATCH_SIZE, BLOCK_SIZE, N_EMBD
from language_model.helpers import get_device

def test_memory_setup():
    """Test memory configuration and allocation"""
    print("=== MEMORY OPTIMIZATION TEST ===")
    
    # Apply memory optimizations
    optimize_memory_settings()
    
    device = get_device()
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        # Print current memory settings
        print(f"GPU Memory before model creation:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
    
    # Create model (same as training)
    model = GPTLanguageModel(12856).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    if torch.cuda.is_available():
        print(f"GPU Memory after model creation:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # Test batch allocation
    try:
        batch_size = BATCH_SIZE
        sequence_length = BLOCK_SIZE
        
        print(f"\nTesting batch allocation: batch_size={batch_size}, seq_len={sequence_length}")
        
        # Create a test batch
        x = torch.randint(0, MAX_VOCAB_SIZE, (batch_size, sequence_length)).to(device)
        y = torch.randint(0, MAX_VOCAB_SIZE, (batch_size, sequence_length)).to(device)
        
        print("✅ Batch allocation successful")
        
        if torch.cuda.is_available():
            print(f"GPU Memory after batch allocation:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        
        # Test forward pass
        with torch.amp.autocast(device_type=device.type):
            logits, loss = model(x, y)
        
        print("✅ Forward pass successful")
        
        if torch.cuda.is_available():
            print(f"GPU Memory after forward pass:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        
        # Clean up
        del x, y, logits, loss
        aggressive_memory_cleanup()
        
        if torch.cuda.is_available():
            print(f"GPU Memory after cleanup:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        
        print("\n✅ Memory test completed successfully!")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ Memory test failed: {e}")
            print("\nSuggestions:")
            print("1. Reduce BATCH_SIZE in config.py")
            print("2. Enable gradient accumulation")
            print("3. Use gradient checkpointing")
            assert False, f"Out of memory error: {e}"
        else:
            raise e

if __name__ == "__main__":
    # Set the CUDA allocator config programmatically as backup
    cuda_config = "expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:8"
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', cuda_config)
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
    
    success = test_memory_setup()
    if success:
        print("\n🚀 Ready for training!")
    else:
        print("\n⚠️  Consider reducing batch size or enabling gradient accumulation")
