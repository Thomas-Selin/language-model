import torch
import datetime
import os
from subword_tokenizer import SubwordTokenizer
from helpers import get_device
from data_handler import prepare_context_data_for_training, process_qa_pairs_dataset
from model import GPTLanguageModel
from train_utils import base_train_model, train_chat_alignment
from config import PARQUET_DIR_PATH, TEXT_COLUMN, VOCAB_PATH, QA_PARQUET_PATH, CONTEXT_PARQUET_PATH, LOG_LEVEL, TRAINING_START_TIME, MAX_VOCAB_SIZE
import logging
from helpers import configure_colored_logging

# Configure logging
configure_colored_logging(LOG_LEVEL)

if __name__ == "__main__":
    training_start_time = TRAINING_START_TIME or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('data', 'output', training_start_time)
    # Data paths
    parquet_dir_path = PARQUET_DIR_PATH
    text_column = TEXT_COLUMN
    vocab_path = VOCAB_PATH
    batch_size_files = 1  # Number of parquet files to process in each batch
    qa_parquet_path = QA_PARQUET_PATH
    
    # Extract context data for base training
    logging.info("\n=== Extracting context data from QA dataset for base training ===")
    context_parquet_path = CONTEXT_PARQUET_PATH
    prepare_context_data_for_training(qa_parquet_path, context_parquet_path, text_column=text_column)
    logging.info(f"Context data extracted to {context_parquet_path}")
    logging.info("This file should be included in base training at the end by moving it to the parquet_files directory.")

    # Base training - now uses train_and_poll for polling and deletion
    logging.info("\n=== Starting base training with polling and auto-deletion ===")
    # Check for existing checkpoint
    checkpoint_path = os.path.join(output_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        logging.info(f"Resuming base training from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None
    # Pass checkpoint_path to base_train_model (update base_train_model to accept this argument)
    global_iter = base_train_model(parquet_dir_path, text_column, vocab_path, training_start_time, output_dir=output_dir, checkpoint_path=checkpoint_path)

    # Now process QA dataset for fine-tuning
    tokenizer = SubwordTokenizer(vocab_file=vocab_path)
    vocab_size = tokenizer.get_vocab_size()

    logging.info("\n=== Creating QA dataset for fine-tuning ===")
    from config import BLOCK_SIZE
    block_size = BLOCK_SIZE  # Should match model config
    qa_tensor = process_qa_pairs_dataset(
        qa_parquet_path, 
        tokenizer,
        max_length=block_size
    )
    logging.debug(f'QA tensor shape: {qa_tensor.shape}')
    
    # Load pre-trained model
    device = get_device()
    model = GPTLanguageModel(MAX_VOCAB_SIZE).to(device) #TODO: should be vocab_size
    
    # Try to load the best available model for fine-tuning
    model_loaded = False
    model_paths_to_try = [
        os.path.join(output_dir, 'best_model.pt'),
        os.path.join(output_dir, 'model_error.pt')
    ]
    
    for model_path in model_paths_to_try:
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                logging.info(f'\033[92mPre-trained model loaded from {model_path}\033[0m')
                model_loaded = True
                break
            except Exception as e:
                logging.warning(f'Failed to load model from {model_path}: {e}')
                continue
    
    if not model_loaded:
        logging.warning('No pre-trained model found. Starting fine-tuning with randomly initialized model.')
    
    logging.info('Model ready for fine-tuning.')
    
    # Fine-tune on QA pairs
    logging.info('\n=== Starting fine-tuning on QA pairs ===')
    global_step = train_chat_alignment(
        model, 
        qa_tensor,
        output_dir=output_dir,
        lr=1e-4, 
        batch_size=4, 
        val_split=0.1,
        global_step=global_iter
    )
    logging.info('Fine-tuning complete. Model ready for use.')
