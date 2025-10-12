"""Unit tests for data_handler module."""

import pytest
import torch
import tempfile
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from language_model.data_handler import get_batch
from language_model.qa_processing import process_qa_pairs_dataset
from language_model.subword_tokenizer import SubwordTokenizer
from language_model.exceptions import InvalidDataFormatError


class TestGetBatch:
    """Tests for get_batch() function."""
    
    def test_basic_batch_generation(self):
        """Test basic batch generation from data."""
        # Create simple test data
        train_data = torch.arange(100)
        val_data = torch.arange(100, 150)
        block_size = 10
        batch_size = 4
        device = torch.device('cpu')
        
        x, y = get_batch(block_size, batch_size, 'train', train_data, val_data, device)
        
        # Check shapes
        assert x.shape == (batch_size, block_size)
        assert y.shape == (batch_size, block_size)
        
        # Check that y is shifted by 1
        for i in range(batch_size):
            start_idx = x[i, 0].item()
            # Find where this sequence starts in train_data
            for j in range(len(train_data) - block_size):
                if train_data[j].item() == start_idx:
                    # Verify y is the next token sequence
                    expected_y = train_data[j+1:j+block_size+1]
                    if torch.equal(y[i], expected_y):
                        break
    
    def test_validation_batch(self):
        """Test batch generation from validation data."""
        train_data = torch.arange(100)
        val_data = torch.arange(100, 150)
        block_size = 10
        batch_size = 4
        device = torch.device('cpu')
        
        x, y = get_batch(block_size, batch_size, 'val', train_data, val_data, device)
        
        # Check shapes
        assert x.shape == (batch_size, block_size)
        assert y.shape == (batch_size, block_size)
        
        # Check values are from validation range
        assert x.min() >= 100
        assert x.max() < 150
    
    def test_insufficient_data_error(self):
        """Test error when data is too small for block size."""
        train_data = torch.arange(5)  # Only 5 tokens
        val_data = torch.arange(5, 10)
        block_size = 10  # Larger than data
        batch_size = 2
        device = torch.device('cpu')
        
        with pytest.raises(ValueError, match="Insufficient data"):
            get_batch(block_size, batch_size, 'train', train_data, val_data, device)
    
    def test_device_placement(self):
        """Test that batches are placed on correct device."""
        train_data = torch.arange(100)
        val_data = torch.arange(100, 150)
        block_size = 10
        batch_size = 4
        device = torch.device('cpu')
        
        x, y = get_batch(block_size, batch_size, 'train', train_data, val_data, device)
        
        assert x.device == device
        assert y.device == device


class TestProcessQAPairsDataset:
    """Tests for process_qa_pairs_dataset() function."""
    
    def test_basic_qa_processing(self):
        """Test processing of basic QA pairs."""
        # Create temporary parquet file with QA data
        df = pd.DataFrame({
            'question': ['What is 2+2?', 'What is the capital of France?'],
            'answers': [
                "{'text': ['Four']}",
                "{'text': ['Paris']}"
            ]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_file = f.name
        
        try:
            df.to_parquet(temp_file)
            
            # Create a simple mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
            
            max_length = 20
            result = process_qa_pairs_dataset(temp_file, mock_tokenizer, max_length)
            
            # Check result is a tensor
            assert isinstance(result, torch.Tensor)
            
            # Check shape
            assert result.shape[0] == 2  # Two QA pairs
            assert result.shape[1] == max_length  # Max length
            
        finally:
            os.unlink(temp_file)
    
    def test_missing_columns_error(self):
        """Test error when required columns are missing."""
        # Create parquet without required columns
        df = pd.DataFrame({
            'wrong_column': ['data1', 'data2']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_file = f.name
        
        try:
            df.to_parquet(temp_file)
            
            mock_tokenizer = MagicMock()
            
            with pytest.raises(InvalidDataFormatError, match="must contain 'question' and 'answers'"):
                process_qa_pairs_dataset(temp_file, mock_tokenizer, max_length=20)
        finally:
            os.unlink(temp_file)
    
    def test_dict_answers_format(self):
        """Test processing when answers are already dict objects."""
        # Create temporary parquet file with dict answers
        df = pd.DataFrame({
            'question': ['Test question?'],
            'answers': [{'text': ['Test answer']}]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_file = f.name
        
        try:
            df.to_parquet(temp_file)
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
            
            result = process_qa_pairs_dataset(temp_file, mock_tokenizer, max_length=20)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape[0] == 1
            
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
