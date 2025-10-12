"""Shared test configuration and fixtures for pytest.

This file provides common test fixtures, mock objects, and utility functions
used across multiple test modules.
"""

import os
import pytest
import torch
from unittest.mock import MagicMock

try:
    from language_model.subword_tokenizer import SubwordTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def vocab_path(fixtures_dir):
    """Return path to test vocabulary file."""
    return os.path.join(fixtures_dir, 'vocab_subword.json')


@pytest.fixture
def model_path(fixtures_dir):
    """Return path to test model file."""
    return os.path.join(fixtures_dir, 'best_model.pt')


@pytest.fixture
def device():
    """Return CPU device for testing (avoids GPU requirements)."""
    return torch.device('cpu')


@pytest.fixture
def tokenizer(vocab_path):
    """Create a SubwordTokenizer instance for testing."""
    if TOKENIZER_AVAILABLE and os.path.exists(vocab_path):
        return SubwordTokenizer(vocab_file=vocab_path)
    return None


@pytest.fixture
def mock_model():
    """Create a mock GPT model for testing without loading real weights."""
    model = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    
    # Mock generate method
    def mock_generate(*args, **kwargs):
        # Return fake token IDs
        output = torch.tensor([[1, 2, 3, 4, 5]])
        attentions = [] if kwargs.get('return_attention', False) else None
        return (output, attentions) if attentions is not None else output
    
    model.generate = mock_generate
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing without real vocab."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    tokenizer.decode = MagicMock(return_value="Mock decoded text")
    tokenizer.get_vocab_size = MagicMock(return_value=1000)
    return tokenizer


def skip_if_no_fixtures():
    """Decorator to skip tests if fixture files don't exist."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    model_path = os.path.join(fixtures_dir, 'best_model.pt')
    vocab_path = os.path.join(fixtures_dir, 'vocab_subword.json')
    
    has_fixtures = os.path.exists(model_path) and os.path.exists(vocab_path)
    return pytest.mark.skipif(
        not has_fixtures,
        reason="Test fixtures not found. Run training first to generate model files."
    )
