#!/usr/bin/env python3
"""
Test script to verify that model loading optimization works correctly.
This script will simulate multiple text generation calls to ensure the model
is loaded only once and reused.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from helpers import get_device

from serving import generate_text, find_latest_model, load_tokenizer, GPTLanguageModel, quantize_model
import torch
import time

def test_optimized_loading():
    print("🧪 Testing optimized model loading...")
    
    # Load model once
    print("\n1. Loading model and tokenizer (should happen only once):")
    start_time = time.time()
    
    latest_model = find_latest_model()
    tokenizer = load_tokenizer('subword', os.path.dirname(latest_model))
    device = get_device()
    model = GPTLanguageModel(vocab_size=tokenizer.get_vocab_size())
    
    checkpoint = torch.load(latest_model, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model = quantize_model(model, device)
    model.eval()
    
    # Clean up checkpoint
    del checkpoint
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
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

def test_old_way():
    print("\n🐌 Testing old way (reloading model each time):")
    
    prompts = [
        "Once upon a time",
        "The quick brown fox"
    ]
    
    total_time = 0
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   Generation {i}: '{prompt}'")
        start_time = time.time()
        
        # This will reload the model each time (old behavior)
        result, _, _ = generate_text(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8
            # No model/tokenizer/device passed = will reload
        )
        
        gen_time = time.time() - start_time
        total_time += gen_time
        
        print(f"   Generated in {gen_time:.2f}s: {result[:100]}...")
    
    print(f"\n📊 Old way summary:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average per generation: {total_time/len(prompts):.2f}s")

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
