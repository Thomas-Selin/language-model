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
    
if __name__ == "__main__":
    unittest.main()
