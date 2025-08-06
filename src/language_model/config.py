# Configuration file for all hyperparameters and paths

# Data paths
PARQUET_DIR_PATH = 'data/input/parquet_files'
TEXT_COLUMN = 'text'
VOCAB_PATH = 'data/output/vocab_subword.json'
QA_PARQUET_PATH = 'data/input/chat-align/train-00000-of-00001.parquet'
CONTEXT_PARQUET_PATH = 'data/input/context_data.parquet'

# Model and training hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 64
BASE_TRAINING_MAX_EPOCHS = 3
FINETUNING_MAX_EPOCHS = 2
EVAL_INTERVAL = 5
LEARNING_RATE = 3e-4
EVAL_ITERS = 100
N_EMBD = 384
N_HEAD = 6
N_LAYER = 4
DROPOUT = 0.15
EARLY_STOPPING_PATIENCE = 10
MAX_VOCAB_SIZE = 3000
WARMUP_STEPS = 1000
LR_DECAY = "linear"
