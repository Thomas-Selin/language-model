#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'language_model'))

from serving import generate_text, find_latest_model

def test_serving():
    print("Testing chat_aligned_model.pt serving...")
    
    # Test model finding
    try:
        model_path = find_latest_model()
        print(f"✅ Found model: {model_path}")
    except Exception as e:
        print(f"❌ Error finding model: {e}")
        return
    
    # Test text generation
    try:
        result, attentions, tokenizer = generate_text(
            prompt="Hello, how are you?",
            max_new_tokens=50,
            temperature=0.8
        )
        print(f"✅ Generated text: {result[:100]}...")
        print(f"✅ Attention data available: {len(attentions) if attentions else 0} steps")
    except Exception as e:
        print(f"❌ Error generating text: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_serving()
