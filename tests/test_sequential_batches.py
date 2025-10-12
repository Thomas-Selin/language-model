"""Unit tests for sequential batch generation."""

import pytest
import torch
from language_model.data_handler import get_sequential_batches


class TestSequentialBatches:
    """Tests for get_sequential_batches() function."""
    
    def test_full_data_coverage(self):
        """Test that sequential batches cover all data exactly once."""
        data = torch.arange(1000)
        block_size = 10
        batch_size = 16
        device = torch.device('cpu')
        
        # Track all starting positions seen
        seen_positions = set()
        
        for x, y in get_sequential_batches(block_size, batch_size, data, device, shuffle=False):
            for i in range(x.shape[0]):
                start_pos = x[i, 0].item()
                seen_positions.add(start_pos)
                
                # Verify sequence continuity
                expected_x = data[start_pos:start_pos + block_size]
                expected_y = data[start_pos + 1:start_pos + block_size + 1]
                assert torch.equal(x[i], expected_x)
                assert torch.equal(y[i], expected_y)
        
        # Verify complete coverage
        max_positions = len(data) - block_size - 1
        assert len(seen_positions) == max_positions
        assert seen_positions == set(range(max_positions))
    
    def test_batch_shapes(self):
        """Test that batches have correct shapes."""
        data = torch.arange(500)
        block_size = 20
        batch_size = 8
        device = torch.device('cpu')
        
        for x, y in get_sequential_batches(block_size, batch_size, data, device):
            # Last batch might be smaller
            assert x.shape[0] <= batch_size
            assert y.shape[0] <= batch_size
            assert x.shape[1] == block_size
            assert y.shape[1] == block_size
            assert x.shape == y.shape
    
    def test_with_shuffle(self):
        """Test that shuffling still covers all data."""
        data = torch.arange(500)
        block_size = 10
        batch_size = 16
        device = torch.device('cpu')
        
        seen_positions = set()
        
        for x, y in get_sequential_batches(block_size, batch_size, data, device, shuffle=True):
            for i in range(x.shape[0]):
                start_pos = x[i, 0].item()
                seen_positions.add(start_pos)
        
        # Even with shuffling, all positions should be seen
        max_positions = len(data) - block_size - 1
        assert len(seen_positions) == max_positions
    
    def test_insufficient_data_error(self):
        """Test error when data is too small."""
        data = torch.arange(5)
        block_size = 10  # Larger than data
        batch_size = 4
        device = torch.device('cpu')
        
        with pytest.raises(ValueError, match="Insufficient data"):
            # Consume the generator to trigger the error
            list(get_sequential_batches(block_size, batch_size, data, device))
    
    def test_device_placement(self):
        """Test that batches are placed on correct device."""
        data = torch.arange(100)
        block_size = 10
        batch_size = 4
        device = torch.device('cpu')
        
        for x, y in get_sequential_batches(block_size, batch_size, data, device):
            assert x.device == device
            assert y.device == device
            break  # Just check first batch
    
    def test_target_is_shifted(self):
        """Test that y is always x shifted by 1."""
        data = torch.arange(200)
        block_size = 15
        batch_size = 8
        device = torch.device('cpu')
        
        for x, y in get_sequential_batches(block_size, batch_size, data, device, shuffle=False):
            for i in range(x.shape[0]):
                # For each position in the sequence, y should be x shifted by 1
                start_pos = x[i, 0].item()
                for j in range(block_size):
                    assert x[i, j].item() == data[start_pos + j].item()
                    assert y[i, j].item() == data[start_pos + j + 1].item()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
