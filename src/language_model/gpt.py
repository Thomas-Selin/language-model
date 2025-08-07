import torch
import os
import datetime
from subword_tokenizer import SubwordTokenizer
from helpers import get_device
from data_handler import prepare_context_data_for_training, process_qa_pairs_dataset
from model import GPTLanguageModel
from train_utils import base_train_model, train_chat_alignment
from config import PARQUET_DIR_PATH, TEXT_COLUMN, VOCAB_PATH, QA_PARQUET_PATH, CONTEXT_PARQUET_PATH
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
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

    # Get current time for logging
    training_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Base training - now uses train_and_poll for polling and deletion
    logging.info("\n=== Starting base training with polling and auto-deletion ===")
    base_train_model(parquet_dir_path, text_column, vocab_path, batch_size_files, training_start_time)

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
    model = GPTLanguageModel(vocab_size).to(device)
    model.load_state_dict(torch.load('data/output/model_checkpoint.pt', map_location=device))
    logging.info('Pre-trained model loaded.')
    
    # Fine-tune on QA pairs
    logging.info("\n=== Starting fine-tuning on QA pairs ===")
    qa_logdir = f'data/output/tensorboard_logs/{training_start_time}'
    train_chat_alignment(
        model, 
        qa_tensor, 
        epochs=10, 
        lr=1e-4, 
        batch_size=4, 
        val_split=0.1, 
        tensorboard_logdir=qa_logdir
    )
    logging.info("Fine-tuning complete. Model ready for use.")
