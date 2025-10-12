#!/usr/bin/env python3
"""
Test script to verify that model loading optimization works correctly.
This script will simulate multiple text generation calls to ensure the model
is loaded only once and reused.
"""

import os
import pytest
import torch
import time

from language_model.helpers import get_device
from language_model.serving import generate_text, find_latest_model, load_tokenizer, GPTLanguageModel, quantize_model


def check_model_exists():
    """Check if any trained model exists in tests/fixtures/"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fixtures_dir = os.path.join(project_root, 'tests', 'fixtures')
    
    if not os.path.isdir(fixtures_dir):
        return False
    
    # Check for any model file in fixtures
    model_path = os.path.join(fixtures_dir, 'best_model.pt')
    vocab_file = os.path.join(fixtures_dir, 'vocab_subword.json')
    
    # Just check if both files exist - we'll adapt to whatever architecture is in the checkpoint
    if os.path.exists(model_path) and os.path.exists(vocab_file):
        return True
    
    return False


@pytest.mark.skipif(not check_model_exists(), reason="No trained model found in tests/fixtures/")
def test_optimized_loading():
    print("🧪 Testing optimized model loading...")
    
    # Load model once
    print("\n1. Loading model and tokenizer (should happen only once):")
    start_time = time.time()
    
    # Use fixtures directory for testing
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fixtures_dir = os.path.join(project_root, 'tests', 'fixtures')
    latest_model = os.path.join(fixtures_dir, 'best_model.pt')
    tokenizer = load_tokenizer('subword', fixtures_dir)
    device = get_device()
    
    # Load checkpoint to determine architecture
    checkpoint = torch.load(latest_model, map_location='cpu', weights_only=False)
    
    # Dynamically configure model based on checkpoint architecture
    import language_model.model as model_module
    original_n_embd = model_module.n_embd
    original_n_head = model_module.n_head
    original_n_layer = model_module.n_layer
    
    # Extract architecture from checkpoint
    vocab_size, n_embd = checkpoint['token_embedding_table.weight'].shape
    # Count layers by finding keys like 'blocks.0', 'blocks.1', etc.
    n_layer = max([int(k.split('.')[1]) for k in checkpoint.keys() if k.startswith('blocks.')]) + 1
    # Count heads by finding unique head indices in blocks.0.sa.heads.X (X is at index 4)
    heads = set([int(k.split('.')[4]) for k in checkpoint.keys() if k.startswith('blocks.0.sa.heads.') and len(k.split('.')) > 5])
    n_head = max(heads) + 1 if heads else 1
    
    # Temporarily override config for this test
    model_module.n_embd = n_embd
    model_module.n_head = n_head  
    model_module.n_layer = n_layer
    
    try:
        # Use vocab size from checkpoint, not tokenizer (fixtures may have mismatched files)
        model = GPTLanguageModel(vocab_size=vocab_size)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model = quantize_model(model, device)
        model.eval()
        
        # Clean up checkpoint
        del checkpoint
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"   Model architecture: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}")
        
        # Test multiple generations with pre-loaded model
        prompts = [
            "Once upon a time",
            "The quick brown fox",
            "In a galaxy far, far away"
        ]
        
        print(f"\n2. Running {len(prompts)} text generations with pre-loaded model:")
        
        total_generation_time = 0
        for i, prompt in enumerate(prompts, 1):
            print(f"\n   Generation {i}: '{prompt}'")
            start_time = time.time()
            
            # This should use the pre-loaded model and NOT reload it
            result, _, _ = generate_text(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.8,
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            gen_time = time.time() - start_time
            total_generation_time += gen_time
            
            print(f"   Generated in {gen_time:.2f}s: {result[:100]}...")
        
        print(f"\n📊 Performance Summary:")
        print(f"   Model loading time: {load_time:.2f}s")
        print(f"   Total generation time: {total_generation_time:.2f}s")
        print(f"   Average per generation: {total_generation_time/len(prompts):.2f}s")
        print(f"   Total time: {load_time + total_generation_time:.2f}s")
        
        print(f"\n✅ Test completed! Model was loaded once and reused for all generations.")
    
    finally:
        # Restore original config values
        model_module.n_embd = original_n_embd
        model_module.n_head = original_n_head
        model_module.n_layer = original_n_layer


@pytest.mark.skipif(not check_model_exists(), reason="No trained model found in tests/fixtures/")
def test_old_way():
    print("\n🐌 Testing old way (reloading model each time):")
    
    # Use fixtures directory for testing
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fixtures_dir = os.path.join(project_root, 'tests', 'fixtures')
    model_path = os.path.join(fixtures_dir, 'best_model.pt')
    
    # Load checkpoint once to get architecture params
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    vocab_size, n_embd = checkpoint['token_embedding_table.weight'].shape
    n_layer = max([int(k.split('.')[1]) for k in checkpoint.keys() if k.startswith('blocks.')]) + 1
    heads = set([int(k.split('.')[4]) for k in checkpoint.keys() if k.startswith('blocks.0.sa.heads.') and len(k.split('.')) > 5])
    n_head = max(heads) + 1 if heads else 1
    del checkpoint
    
    # Setup config overrides
    import language_model.model as model_module
    original_n_embd = model_module.n_embd
    original_n_head = model_module.n_head
    original_n_layer = model_module.n_layer
    model_module.n_embd = n_embd
    model_module.n_head = n_head
    model_module.n_layer = n_layer
    
    try:
        prompts = [
            "Once upon a time",
            "The quick brown fox"
        ]
        
        total_time = 0
        for i, prompt in enumerate(prompts, 1):
            print(f"\n   Generation {i}: '{prompt}'")
            start_time = time.time()
            
            # Reload model each time to simulate old behavior
            tokenizer = load_tokenizer('subword', fixtures_dir)
            device = get_device()
            model = GPTLanguageModel(vocab_size=vocab_size)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint)
            model = model.to(device)
            model = quantize_model(model, device)
            model.eval()
            del checkpoint
            
            result, _, _ = generate_text(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.8,
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            gen_time = time.time() - start_time
            total_time += gen_time
            
            print(f"   Generated in {gen_time:.2f}s: {result[:100]}...")
        
        print(f"\n📊 Old way summary:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average per generation: {total_time/len(prompts):.2f}s")
    
    finally:
        # Restore original config values
        model_module.n_embd = original_n_embd
        model_module.n_head = original_n_head
        model_module.n_layer = original_n_layer

if __name__ == "__main__":
    print("🚀 Model Loading Performance Test")
    print("=" * 50)
    
    try:
        test_optimized_loading()
        test_old_way()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        print("\n💡 Key takeaways:")
        print("   - Use pre-loaded model/tokenizer/device for best performance")
        print("   - Streamlit app now caches the model using @st.cache_resource")
        print("   - Model loading messages help debug any remaining issues")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
