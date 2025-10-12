"""Constants used throughout the language model project.

This module contains all magic numbers and constants that are used
in multiple places throughout the codebase.
"""

# Feed forward network expansion factor
FEEDFORWARD_EXPANSION_FACTOR = 4

# Data splitting ratios
TRAIN_VAL_SPLIT_RATIO = 0.9  # 90% train, 10% validation

# Data processing constants
CHUNK_SIZE_ROWS = 100  # Number of rows to process at once from parquet files
PER_FILE_CHAR_CAP = 1_000_000  # Max characters to use from a single file for vocab
TOTAL_VOCAB_SAMPLE_CHARS = 20_000_000  # Total characters to use for vocab building

# Training constants
GRADIENT_ACCUMULATION_DEFAULT = 1  # Default number of gradient accumulation steps

# Memory management constants
GPU_MEMORY_FRACTION = 0.90  # Fraction of GPU memory to use
HIGH_MEMORY_THRESHOLD_GB = 18.0  # Threshold for warning about high GPU usage
RESERVED_MEMORY_THRESHOLD_GB = 20.0  # Threshold for warning about high reserved memory

# Retry and monitoring constants
MAX_FILE_RETRIES = 1000
RETRY_DELAY_SECONDS = 60
STABLE_CHECK_INTERVAL = 2  # Seconds between file size stability checks
STABLE_CHECK_COUNT = 3  # Number of checks for file stability
