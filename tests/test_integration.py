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
from language_model.subword_tokenizer import SubwordTokenizer
from language_model.model import GPTLanguageModel
from language_model.data_handler import get_batch
from language_model.data_loading import load_text_from_parquet
from language_model.qa_processing import process_qa_pairs_dataset


class TestTrainingPipeline:
    """Integration tests for the complete training pipeline."""
    
    @classmethod
    def setup_class(cls):
        """Set up fixtures once for all tests in the class."""
        # Load sample text from parquet file once for all tests
        sample_text_path = os.path.join('tests', 'fixtures', 'test_text_data', 'base_training_data.parquet')
        cls.sample_text = load_text_from_parquet(sample_text_path, text_column='text')
    
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
        """Test complete pipeline: data loading, training, QA finetuning, saving, loading, and generation.
        
        This test verifies:
        1. Tokenizer can be built from text
        2. Model can be created with correct architecture
        3. Base training data can be processed
        4. Model can perform forward pass on base data
        5. Q&A data can be processed for finetuning
        6. Model can be finetuned on Q&A data
        7. Model can be saved and loaded
        8. Model can generate text after finetuning
        """
        # Use the shared sample text
        sample_text = self.sample_text
        
        # Step 1: Build tokenizer
        tokenizer = SubwordTokenizer.build_vocab(
            sample_text, 
            vocab_size=1000,  # Small vocab for testing
            min_frequency=1
        )
        SubwordTokenizer.save_vocab(tokenizer, path=self.vocab_path)
        tokenizer = SubwordTokenizer(vocab_file=self.vocab_path)
        vocab_size = tokenizer.get_vocab_size()
        
        # Step 2: Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = GPTLanguageModel(vocab_size=vocab_size, use_checkpoint=False)
        model = model.to(device)
        
        # Step 3: Prepare base training data
        encoded = tokenizer.encode(sample_text)
        train_data = torch.tensor(encoded, dtype=torch.long)
        
        # Step 4: Train model on base data (just a few steps for testing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        
        # Perform a few training steps on base data
        batch_size = 4
        block_size = 32
        for step in range(5):
            xb, yb = get_batch(block_size, batch_size, 'train', train_data, train_data, device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        base_loss = loss.item()
        print(f"Base training loss after 5 steps: {base_loss:.4f}")
        
        # Step 5: Process Q&A data for finetuning
        qa_path = os.path.join('tests', 'fixtures', 'test_text_data', 'question_answer_dataset.parquet')
        qa_data = process_qa_pairs_dataset(
            qa_parquet_path=qa_path,
            tokenizer=tokenizer,
            max_length=block_size
        )
        qa_data = qa_data.to(device)
        
        assert qa_data.shape[0] > 0, "Should have processed Q&A pairs"
        assert qa_data.shape[1] == block_size, "Q&A data should match block size"
        
        # Step 6: Finetune model on Q&A data
        model.train()
        qa_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        
        # Perform Q&A finetuning steps
        num_qa_batches = min(10, len(qa_data) // batch_size)
        for step in range(num_qa_batches):
            # Get batch from Q&A data
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(qa_data))
            
            if end_idx - start_idx < batch_size:
                break
                
            batch = qa_data[start_idx:end_idx, :-1].contiguous()  # Input: all but last token
            targets = qa_data[start_idx:end_idx, 1:].contiguous()  # Target: all but first token
            
            logits, loss = model(batch, targets)
            qa_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            qa_optimizer.step()
        
        qa_loss = loss.item()
        print(f"Q&A finetuning loss after {num_qa_batches} steps: {qa_loss:.4f}")
        
        # Step 7: Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'vocab_size': vocab_size,
            'optimizer_state_dict': qa_optimizer.state_dict()
        }
        torch.save(checkpoint, self.model_path)
        assert os.path.exists(self.model_path), "Model checkpoint should be saved"
        
        # Step 8: Load model
        loaded_checkpoint = torch.load(self.model_path, map_location=device)
        loaded_model = GPTLanguageModel(vocab_size=vocab_size, use_checkpoint=False)
        loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        loaded_model = loaded_model.to(device)
        loaded_model.eval()
        
        # Step 9: Generate text
        prompt = "Question: What is"
        encoded_prompt = tokenizer.encode(prompt)
        prompt_tokens = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated = loaded_model.generate(
                prompt_tokens,
                max_new_tokens=20,
                temperature=0.8,
                top_p=0.9
            )
        
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        
        assert len(generated_text) > len(prompt), "Should generate new text"
        print(f"Generated text: {generated_text}")
        
        # Verify model can still perform inference on both types of data
        # Test on base data
        with torch.no_grad():
            base_xb, base_yb = get_batch(block_size, 2, 'train', train_data, train_data, device)
            base_logits, base_test_loss = loaded_model(base_xb, base_yb)
            assert base_test_loss is not None, "Should compute loss on base data"
            
        # Test on Q&A data
        with torch.no_grad():
            qa_batch = qa_data[:2, :-1].contiguous()
            qa_targets = qa_data[:2, 1:].contiguous()
            qa_logits, qa_test_loss = loaded_model(qa_batch, qa_targets)
            assert qa_test_loss is not None, "Should compute loss on Q&A data"
        
        print(f"Test losses - Base: {base_test_loss:.4f}, Q&A: {qa_test_loss:.4f}")
    
    
    def test_parquet_data_loading(self):
        """Test loading and processing data from parquet files.
        
        This test verifies:
        1. Parquet files can be created and read
        2. Text data can be extracted from parquet
        3. Data can be tokenized and prepared for training
        """
        # Use the shared sample text for this test as well
        text = self.sample_text
        
        assert len(text) > 0, "Should have loaded sample text"
        
        # Build tokenizer and encode
        tokenizer = SubwordTokenizer.build_vocab(text, vocab_size=300)
        encoded = tokenizer.encode(text)
        
        # Get the actual token IDs from the Encoding object
        token_ids = encoded.ids
        
        assert len(token_ids) > 0, "Should encode text to tokens"
        
        # Verify round-trip
        decoded = tokenizer.decode(token_ids[:100])
        assert len(decoded) > 0, "Should decode tokens back to text"
        
        print(f"Loaded {len(text)} characters from parquet")
        print(f"Encoded to {len(token_ids)} tokens")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])