# Configuration file for all hyperparameters and paths

# General settings
LOG_LEVEL = "DEBUG"  # Set to "INFO" or "DEBUG" as needed

# Data paths
PARQUET_DIR_PATH = 'data/input/parquet_files'
TEXT_COLUMN = 'text'
VOCAB_PATH = 'data/output/vocab_subword.json'
QA_PARQUET_PATH = 'data/input/chat-align/train-00000-of-00001.parquet'
CONTEXT_PARQUET_PATH = 'data/input/context_data.parquet'

# Model and training hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 512
BASE_TRAINING_MAX_EPOCHS = 100
FINETUNING_MAX_EPOCHS = 3
FINETUNE_EARLY_STOPPING_PATIENCE = 2
EVAL_INTERVAL = 5
LEARNING_RATE = 3e-4
EVAL_ITERS = 100
N_EMBD = 768
N_HEAD = 12
N_LAYER = 8
DROPOUT = 0.10
EARLY_STOPPING_PATIENCE = 40
MAX_VOCAB_SIZE = 32000
WARMUP_STEPS = 2000
LR_DECAY = "cosine"
