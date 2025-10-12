# Configuration file for all hyperparameters and paths

# General settings
LOG_LEVEL = "DEBUG"  # Set to "INFO" or "DEBUG" as needed

# Data paths
PARQUET_DIR_PATH = 'data/input/parquet_files'
TEXT_COLUMN = 'text'
QA_PARQUET_PATH = 'data/input/chat-align/question_answer_dataset.parquet'
CONTEXT_PARQUET_PATH = 'data/input/context_data.parquet'
VOCAB_PATH = "data/output/vocab_subword.json"

# Model and training hyperparameters
GLOBAL_ITER = 0
BATCH_SIZE = 80  # Reduced from 128 if OOM persists
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 80*8 = 640
BLOCK_SIZE = 512
FINETUNING_MAX_EPOCHS = 8  # Number of complete passes through the QA dataset
FINETUNE_EARLY_STOPPING_PATIENCE = 2
EVAL_INTERVAL = 5  # Evaluate every N epochs
LEARNING_RATE = 3e-4
EVAL_ITERS = 100
N_EMBD = 36
N_HEAD = 4
N_LAYER = 3
DROPOUT = 0.10
EARLY_STOPPING_PATIENCE = 6
MAX_VOCAB_SIZE = 32000
LR_DECAY = "cosine"

# Runtime configuration
RUNTIME_OVERRIDES_FILE = "data/output/RUNTIME_OVERRIDES.json"

# File monitoring constants
MIN_FILE_SIZE_BYTES = 200 * 1024  # 200 KB
STABLE_COUNT_THRESHOLD = 15

# Whether to automatically delete parquet files after they've been used for base training.
AUTO_DELETE_USED_FILES = False

# Default weight decay
DEFAULT_WEIGHT_DECAY = 0.01
