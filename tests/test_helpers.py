"""Unit tests for helper functions in language_model.helpers module."""

import pytest
import torch
import json
import tempfile
import os
import warnings
from unittest.mock import patch
from language_model.helpers import (
    get_device,
    count_parameters,
    apply_runtime_overrides,
    get_lr_scheduler
)


class TestGetDevice:
    """Tests for get_device() function."""
    
    def test_cuda_available(self):
        """Test device selection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device()
            assert device.type == 'cuda'
    
    def test_mps_available_no_cuda(self):
        """Test device selection when MPS is available but CUDA is not."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = get_device()
                assert device.type == 'mps'
    
    def test_cpu_fallback(self):
        """Test device fallback to CPU when no GPU available."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = get_device()
                assert device.type == 'cpu'


class TestCountParameters:
    """Tests for count_parameters() function."""
    
    def test_simple_model(self):
        """Test counting parameters in a simple model."""
        model = torch.nn.Linear(10, 5)
        # 10 * 5 weights + 5 biases = 55 parameters
        assert count_parameters(model) == 55
    
    def test_sequential_model(self):
        """Test counting parameters in a sequential model."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),  # 55 parameters
            torch.nn.Linear(5, 2)    # 12 parameters
        )
        assert count_parameters(model) == 67
    
    def test_model_without_parameters(self):
        """Test counting parameters in a model with no parameters."""
        model = torch.nn.Identity()
        assert count_parameters(model) == 0


class TestApplyRuntimeOverrides:
    """Tests for apply_runtime_overrides() function."""
    
    def test_no_overrides_file(self):
        """Test behavior when no overrides file exists."""
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(5))])
        params = {'learning_rate': 0.001, 'batch_size': 32}
        
        result_params, extra = apply_runtime_overrides(
            optimizer, None, params, 'nonexistent_file.json'
        )
        
        assert result_params == params
        assert extra == {}
    
    def test_learning_rate_override(self):
        """Test overriding learning rate."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'learning_rate': 0.0005}, f)
            temp_file = f.name
        
        try:
            optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(5))])
            params = {'learning_rate': 0.001}
            
            result_params, extra = apply_runtime_overrides(
                optimizer, None, params, temp_file
            )
            
            assert result_params['learning_rate'] == 0.0005
            assert optimizer.param_groups[0]['lr'] == 0.0005
        finally:
            os.unlink(temp_file)
    
    def test_multiple_overrides(self):
        """Test overriding multiple parameters."""
        overrides = {
            'learning_rate': 0.0005,
            'batch_size': 64,
            'eval_interval': 100,
            'dropout': 0.2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(overrides, f)
            temp_file = f.name
        
        try:
            optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(5))])
            params = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'eval_interval': 50,
                'dropout': 0.1
            }
            
            result_params, extra = apply_runtime_overrides(
                optimizer, None, params, temp_file
            )
            
            assert result_params['learning_rate'] == 0.0005
            assert result_params['batch_size'] == 64
            assert result_params['eval_interval'] == 100
            assert result_params['dropout'] == 0.2
        finally:
            os.unlink(temp_file)
    
    def test_extra_counters(self):
        """Test that extra counters like global_iter are returned separately."""
        overrides = {
            'global_iter': 1000,
            'total_epochs_run': 50
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(overrides, f)
            temp_file = f.name
        
        try:
            optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(5))])
            params = {'learning_rate': 0.001}
            
            result_params, extra = apply_runtime_overrides(
                optimizer, None, params, temp_file
            )
            
            assert extra['global_iter'] == 1000
            assert extra['total_epochs_run'] == 50
        finally:
            os.unlink(temp_file)


class TestGetLRScheduler:
    """Tests for get_lr_scheduler() function."""
    
    def test_linear_decay(self):
        """Test linear learning rate decay."""
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(5))], lr=1.0)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=10, lr_decay='linear', total_steps=100)
        
        # Test warmup phase
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
            for _ in range(10):
                scheduler.step()
        lr_after_warmup = optimizer.param_groups[0]['lr']
        assert lr_after_warmup == pytest.approx(1.0, rel=0.01)
        
        # Test decay phase
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
            for _ in range(45):  # Go to step 55 (middle of decay)
                scheduler.step()
        lr_mid_decay = optimizer.param_groups[0]['lr']
        assert 0.4 < lr_mid_decay < 0.6  # Should be around 0.5
    
    def test_cosine_decay(self):
        """Test cosine learning rate decay."""
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(5))], lr=1.0)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=10, lr_decay='cosine', total_steps=100)
        
        # Test warmup phase
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
            for _ in range(10):
                scheduler.step()
        lr_after_warmup = optimizer.param_groups[0]['lr']
        assert lr_after_warmup == pytest.approx(1.0, rel=0.01)
        
        # Test decay phase
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
            for _ in range(90):  # Complete the training
                scheduler.step()
        lr_end = optimizer.param_groups[0]['lr']
        assert lr_end < 0.1  # Should decay to near 0
    
    def test_no_decay(self):
        """Test constant learning rate (no decay)."""
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(5))], lr=1.0)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=10, lr_decay='none', total_steps=100)
        
        # Test warmup phase
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
            for _ in range(10):
                scheduler.step()
        lr_after_warmup = optimizer.param_groups[0]['lr']
        
        # Test that LR stays constant after warmup
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
            for _ in range(50):
                scheduler.step()
        lr_later = optimizer.param_groups[0]['lr']
        
        assert lr_after_warmup == pytest.approx(lr_later, rel=0.01)
    
    def test_warmup_from_zero(self):
        """Test that warmup starts from near-zero learning rate."""
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(5))], lr=1.0)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=10, lr_decay='linear', total_steps=100)
        
        # LR should be very small at step 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
            scheduler.step()
        lr_start = optimizer.param_groups[0]['lr']
        assert lr_start < 0.2  # Should be close to 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
