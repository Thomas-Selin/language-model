# Configuration file for all hyperparameters and paths

# General settings
LOG_LEVEL = "DEBUG"  # Set to "INFO" or "DEBUG" as needed

# Data paths
PARQUET_DIR_PATH = 'data/input/parquet_files'
TEXT_COLUMN = 'text'
QA_PARQUET_PATH = 'data/input/chat-align/train-00000-of-00001.parquet'
CONTEXT_PARQUET_PATH = 'data/input/context_data.parquet'

# Model and training hyperparameters
GLOBAL_ITER = 0
BATCH_SIZE = 80  # Reduced from 128 if OOM persists
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 80*8 = 640
BLOCK_SIZE = 512
BASE_TRAINING_MAX_EPOCHS = 100
FINETUNING_MAX_EPOCHS = 4
FINETUNE_EARLY_STOPPING_PATIENCE = 2
EVAL_INTERVAL = 5
LEARNING_RATE = 3e-4
EVAL_ITERS = 100
N_EMBD = 768
N_HEAD = 12
N_LAYER = 8
DROPOUT = 0.10
EARLY_STOPPING_PATIENCE = 15
MAX_VOCAB_SIZE = 32000
WARMUP_STEPS = 2000
LR_DECAY = "cosine"
TRAINING_START_TIME = "20250810-095241"  # Set to None to use current time, or specify a string like "20251001-120000"

# Runtime configuration
RUNTIME_OVERRIDES_FILE = "data/output/RUNTIME_OVERRIDES.json"

# File monitoring constants
MIN_FILE_SIZE_BYTES = 200 * 1024  # 200 KB
STABLE_COUNT_THRESHOLD = 15

# Default weight decay
DEFAULT_WEIGHT_DECAY = 0.01

import datetime
def get_vocab_path():
	folder = TRAINING_START_TIME if TRAINING_START_TIME else datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
	return f"data/output/{folder}/vocab_subword.json"
VOCAB_PATH = get_vocab_path()