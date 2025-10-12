"""Integration tests for the language model training and inference pipeline.

These tests verify that different components work together correctly in
realistic end-to-end scenarios.
"""

import pytest
import torch
import tempfile
import os
import shutil
import pandas as pd
from unittest.mock import patch
from language_model.subword_tokenizer import SubwordTokenizer
from language_model.model import GPTLanguageModel
from language_model.data_handler import load_text_from_parquet, get_batch
from language_model.helpers import get_device


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for the complete training pipeline."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.vocab_path = os.path.join(self.test_dir, 'vocab_test.json')
        self.model_path = os.path.join(self.test_dir, 'model_test.pt')
        
    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_end_to_end_training_and_generation(self):
        """Test complete pipeline: data loading, training, saving, loading, and generation.
        
        This test verifies:
        1. Tokenizer can be built from text
        2. Model can be created with correct architecture
        3. Training data can be processed
        4. Model can perform forward pass
        5. Model can be saved and loaded
        6. Model can generate text
        """
        # 1. Create sample training data
        sample_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a sample text for training. "
            "Machine learning is fascinating. "
            "Natural language processing enables computers to understand text. "
            "Deep learning models can learn complex patterns. "
        ) * 20  # Repeat to get enough data
        
        # 2. Build tokenizer
        tokenizer = SubwordTokenizer.build_vocab(
            sample_text, 
            vocab_size=500,  # Small vocab for testing
            min_frequency=1
        )
        SubwordTokenizer.save_vocab(tokenizer, path=self.vocab_path)
        
        # Reload tokenizer
        tokenizer = SubwordTokenizer(vocab_file=self.vocab_path)
        vocab_size = tokenizer.get_vocab_size()
        
        assert vocab_size > 0, "Vocabulary should not be empty"
        assert vocab_size <= 500, "Vocabulary should not exceed max size"
        
        # 3. Create model with small architecture for testing
        import language_model.model as model_module
        original_n_embd = model_module.n_embd
        original_n_head = model_module.n_head
        original_n_layer = model_module.n_layer
        original_block_size = model_module.block_size
        
        try:
            # Use tiny model for fast testing
            model_module.n_embd = 32
            model_module.n_head = 2
            model_module.n_layer = 2
            model_module.block_size = 16
            
            device = torch.device('cpu')  # Use CPU for tests
            model = GPTLanguageModel(vocab_size=vocab_size, use_checkpoint=False)
            model = model.to(device)
            model.train()
            
            # 4. Prepare training data
            encoded_text = tokenizer.encode(sample_text)
            data = torch.tensor(encoded_text, dtype=torch.long)
            
            # Ensure we have enough data
            assert len(data) > model_module.block_size, "Need more data than block size"
            
            # Split into train/val
            split_idx = int(0.9 * len(data))
            train_data = data[:split_idx]
            val_data = data[split_idx:]
            
            # 5. Run a few training steps
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            initial_loss = None
            for step in range(5):
                # Get batch
                batch_size = 2
                block_size = model_module.block_size
                x, y = get_batch(block_size, batch_size, 'train', train_data, val_data, device)
                
                # Forward pass
                logits, loss = model(x, y)
                
                if initial_loss is None:
                    initial_loss = loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Basic assertions
                assert logits.shape == (batch_size, block_size, vocab_size)
                assert loss.item() > 0, "Loss should be positive"
            
            final_loss = loss.item()
            print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
            
            # 6. Save model
            torch.save(model.state_dict(), self.model_path)
            assert os.path.exists(self.model_path), "Model file should be saved"
            
            # 7. Load model
            model_loaded = GPTLanguageModel(vocab_size=vocab_size, use_checkpoint=False)
            model_loaded.load_state_dict(torch.load(self.model_path, map_location=device))
            model_loaded = model_loaded.to(device)
            model_loaded.eval()
            
            # 8. Generate text
            prompt = "The quick brown"
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                generated_ids = model_loaded.generate(
                    input_tensor,
                    max_new_tokens=10,
                    temperature=0.8,
                    top_p=0.9
                )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            
            assert len(generated_text) > len(prompt), "Should generate additional tokens"
            assert generated_text.startswith(prompt), "Should start with prompt"
            
            print(f"Prompt: '{prompt}'")
            print(f"Generated: '{generated_text}'")
            
        finally:
            # Restore original config
            model_module.n_embd = original_n_embd
            model_module.n_head = original_n_head
            model_module.n_layer = original_n_layer
            model_module.block_size = original_block_size
    
    def test_checkpoint_save_and_resume(self):
        """Test that training can be saved and resumed from checkpoint.
        
        This test verifies:
        1. Model state can be saved
        2. Optimizer state can be saved
        3. Training can resume from saved state
        4. Resumed training continues correctly
        """
        # Create small dataset
        sample_text = "Hello world. " * 100
        
        # Build tokenizer
        tokenizer = SubwordTokenizer.build_vocab(sample_text, vocab_size=200)
        SubwordTokenizer.save_vocab(tokenizer, path=self.vocab_path)
        tokenizer = SubwordTokenizer(vocab_file=self.vocab_path)
        vocab_size = tokenizer.get_vocab_size()
        
        # Create model
        import language_model.model as model_module
        original_n_embd = model_module.n_embd
        original_n_head = model_module.n_head
        original_n_layer = model_module.n_layer
        original_block_size = model_module.block_size
        
        try:
            model_module.n_embd = 16
            model_module.n_head = 2
            model_module.n_layer = 1
            model_module.block_size = 8
            
            device = torch.device('cpu')
            model = GPTLanguageModel(vocab_size=vocab_size, use_checkpoint=False)
            model = model.to(device)
            
            # Prepare data
            encoded = tokenizer.encode(sample_text)
            data = torch.tensor(encoded, dtype=torch.long)
            split_idx = int(0.9 * len(data))
            train_data, val_data = data[:split_idx], data[split_idx:]
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            # Train for a few steps
            for _ in range(3):
                x, y = get_batch(model_module.block_size, 2, 'train', train_data, val_data, device)
                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            loss_before_save = loss.item()
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': 3,
            }
            checkpoint_path = os.path.join(self.test_dir, 'checkpoint.pt')
            torch.save(checkpoint, checkpoint_path)
            
            # Create new model and optimizer
            model_resumed = GPTLanguageModel(vocab_size=vocab_size, use_checkpoint=False)
            model_resumed = model_resumed.to(device)
            optimizer_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3)
            
            # Load checkpoint
            checkpoint_loaded = torch.load(checkpoint_path, map_location=device)
            model_resumed.load_state_dict(checkpoint_loaded['model_state_dict'])
            optimizer_resumed.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
            
            # Continue training
            x, y = get_batch(model_module.block_size, 2, 'train', train_data, val_data, device)
            logits, loss_resumed = model_resumed(x, y)
            
            # Loss should be similar (same model state)
            assert abs(loss_before_save - loss_resumed.item()) < 5.0, \
                "Loss after resume should be in similar range"
            
            print(f"Loss before save: {loss_before_save:.4f}")
            print(f"Loss after resume: {loss_resumed.item():.4f}")
            
        finally:
            model_module.n_embd = original_n_embd
            model_module.n_head = original_n_head
            model_module.n_layer = original_n_layer
            model_module.block_size = original_block_size
    
    def test_parquet_data_loading(self):
        """Test loading and processing data from parquet files.
        
        This test verifies:
        1. Parquet files can be created and read
        2. Text data can be extracted from parquet
        3. Data can be tokenized and prepared for training
        """
        # Create test parquet file
        df = pd.DataFrame({
            'text': [
                'This is the first sentence.',
                'Here is another sentence.',
                'And a third one for good measure.',
                'More text to ensure we have enough data.',
                'Even more text here.'
            ] * 10  # Repeat to get enough data
        })
        
        parquet_path = os.path.join(self.test_dir, 'test_data.parquet')
        df.to_parquet(parquet_path)
        
        # Load text from parquet
        text = load_text_from_parquet(parquet_path, text_column='text')
        
        assert len(text) > 0, "Should extract text from parquet"
        assert 'first sentence' in text, "Should contain expected text"
        
        # Build tokenizer and encode
        tokenizer = SubwordTokenizer.build_vocab(text, vocab_size=300)
        encoded = tokenizer.encode(text)
        
        assert len(encoded) > 0, "Should encode text to tokens"
        
        # Verify round-trip
        decoded = tokenizer.decode(encoded[:100])
        assert len(decoded) > 0, "Should decode tokens back to text"
        
        print(f"Loaded {len(text)} characters from parquet")
        print(f"Encoded to {len(encoded)} tokens")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])
