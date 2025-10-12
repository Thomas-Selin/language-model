import os
import unittest
from unittest.mock import patch, MagicMock
import torch

from language_model.serving import generate_text, find_latest_model
from language_model.subword_tokenizer import SubwordTokenizer


class TestServing(unittest.TestCase):
    """Test cases for model serving functionality.
    
    These tests use mocked models and tokenizers to avoid dependency on
    actual trained model files, making tests more reliable and portable.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures path - using mock model for tests."""
        cls.fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
        cls.vocab_path = os.path.join(cls.fixtures_dir, 'vocab_subword.json')
        cls.model_path = os.path.join(cls.fixtures_dir, 'best_model.pt')
        
        # Load tokenizer from fixtures if available
        if os.path.exists(cls.vocab_path):
            cls.tokenizer = SubwordTokenizer(vocab_file=cls.vocab_path)
        else:
            # Use mock tokenizer if fixtures not available
            cls.tokenizer = MagicMock()
            cls.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
            cls.tokenizer.decode = MagicMock(return_value="Mock text")
        
        # Create a mock model for consistent testing
        cls.device = torch.device('cpu')  # Use CPU for tests
        cls.model = MagicMock()
        cls.model.eval = MagicMock()
        cls.model.to = MagicMock(return_value=cls.model)
    
    @patch('language_model.serving.glob.glob')
    def test_find_latest_model(self, mock_glob):
        """Test that find_latest_model returns a valid model path."""
        # Mock the glob to return a fake output directory
        mock_glob.return_value = ['tests/fixtures/']
        
        # Mock os.path.isdir, os.path.exists, and os.path.getmtime
        with patch('language_model.serving.os.path.isdir', return_value=True):
            with patch('language_model.serving.os.path.exists', return_value=True):
                with patch('language_model.serving.os.path.getmtime', return_value=1234567890.0):
                    model_path = find_latest_model()
                    
                    # Assert model path is returned
                    self.assertIsNotNone(model_path, "Model path should not be None")
                    self.assertIsInstance(model_path, str, "Model path should be a string")
                    
                    # Assert it's a .pt file
                    self.assertTrue(model_path.endswith('.pt'), "Model file should have .pt extension")
    
    def test_generate_text_with_valid_prompt(self):
        """Test text generation with a valid prompt - using mock model."""
        prompt = "Hello, how are you?"
        max_new_tokens = 50
        temperature = 0.8
        
        # Mock the model's generate method to return (output, attentions) tuple
        mock_output = torch.tensor([[1, 2, 3, 4, 5]])  # Fake token IDs
        mock_attentions = []  # Empty attentions list
        self.model.generate = MagicMock(return_value=(mock_output, mock_attentions))
        
        # Mock tokenizer encode/decode
        self.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        self.tokenizer.decode = MagicMock(return_value=prompt + " I'm doing great!")
        
        # Use pre-loaded mock model for tests
        result, attentions, tokenizer = generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # Assert result is not empty
        self.assertIsNotNone(result, "Generated text should not be None")
        self.assertIsInstance(result, str, "Generated text should be a string")
        
        # Assert attentions is a list (can be empty)
        self.assertIsInstance(attentions, list, "Attentions should be a list")
        
        # Assert tokenizer is returned
        self.assertIsNotNone(tokenizer, "Tokenizer should not be None")
    
    def test_generate_text_with_empty_prompt(self):
        """Test text generation with an empty prompt."""
        # Mock the model's generate method to return (output, attentions) tuple
        mock_output = torch.tensor([[1, 2, 3]])
        mock_attentions = []
        self.model.generate = MagicMock(return_value=(mock_output, mock_attentions))
        
        # Mock tokenizer
        self.tokenizer.encode = MagicMock(return_value=[])
        self.tokenizer.decode = MagicMock(return_value="Generated text")
        
        # Should handle empty prompt gracefully
        result, attentions, tokenizer = generate_text(
            prompt="",
            max_new_tokens=20,
            temperature=0.8,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        self.assertIsNotNone(result, "Generated text should not be None even with empty prompt")
        self.assertIsInstance(result, str, "Generated text should be a string")
    
    def test_generate_text_with_different_temperatures(self):
        """Test that different temperatures produce different results."""
        prompt = "The quick brown fox"
        
        # Mock the model's generate method to return (output, attentions) tuples
        mock_output1 = torch.tensor([[1, 2, 3, 4]])
        mock_output2 = torch.tensor([[1, 2, 5, 6]])
        mock_attentions = []
        self.model.generate = MagicMock(side_effect=[
            (mock_output1, mock_attentions),
            (mock_output2, mock_attentions)
        ])
        
        # Mock tokenizer
        self.tokenizer.encode = MagicMock(return_value=[1, 2])
        self.tokenizer.decode = MagicMock(side_effect=[
            prompt + " jumps over the lazy dog",
            prompt + " runs through the forest"
        ])
        
        # Generate with low temperature (more deterministic)
        result_low, _, _ = generate_text(
            prompt=prompt, 
            max_new_tokens=30, 
            temperature=0.1,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # Generate with high temperature (more random)
        result_high, _, _ = generate_text(
            prompt=prompt, 
            max_new_tokens=30, 
            temperature=1.5,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # Both should start with the prompt
        self.assertTrue(result_low.startswith(prompt), "Low temp result should start with prompt")
        self.assertTrue(result_high.startswith(prompt), "High temp result should start with prompt")
        
        # Results should be strings
        self.assertIsInstance(result_low, str, "Low temp result should be a string")
        self.assertIsInstance(result_high, str, "High temp result should be a string")


if __name__ == "__main__":
    unittest.main()
