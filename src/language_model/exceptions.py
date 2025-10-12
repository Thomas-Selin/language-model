"""Custom exceptions for the language model project.

This module defines domain-specific exceptions to provide better error messages
and make error handling more explicit throughout the codebase.
"""


class LanguageModelError(Exception):
    """Base exception for all language model errors."""
    pass


class DataError(LanguageModelError):
    """Base exception for data-related errors."""
    pass


class InsufficientDataError(DataError):
    """Raised when there is not enough data for training or processing.
    
    This can occur when:
    - Input file is too small
    - Data length is less than block_size
    - Not enough examples for train/val split
    """
    pass


class InvalidDataFormatError(DataError):
    """Raised when data format is invalid or unexpected.
    
    This can occur when:
    - Required columns are missing from parquet files
    - Data types are incorrect
    - File format is corrupted
    """
    pass


class ModelError(LanguageModelError):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails.
    
    This can occur when:
    - Checkpoint file is corrupted
    - Vocabulary size mismatch
    - Architecture incompatibility
    """
    pass


class CheckpointError(ModelError):
    """Raised when checkpoint operations fail.
    
    This can occur when:
    - Checkpoint file doesn't exist
    - Checkpoint is corrupted
    - Cannot save checkpoint (disk full, permissions)
    """
    pass


class ConfigurationError(LanguageModelError):
    """Raised when configuration is invalid.
    
    This can occur when:
    - Hyperparameters are out of valid range
    - Required config values are missing
    - Config types are incorrect
    """
    pass


class TokenizationError(LanguageModelError):
    """Raised when tokenization fails.
    
    This can occur when:
    - Vocabulary file is missing or corrupted
    - Tokenizer round-trip check fails
    - Invalid tokens in input
    """
    pass


class TrainingError(LanguageModelError):
    """Raised when training encounters an error.
    
    This can occur when:
    - Out of memory
    - NaN loss detected
    - Gradient explosion
    - Invalid training configuration
    """
    pass


class GenerationError(LanguageModelError):
    """Raised when text generation fails.
    
    This can occur when:
    - Invalid generation parameters
    - Model in wrong state
    - Out of memory during generation
    """
    pass
