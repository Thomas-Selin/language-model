import torch
import time
import os
import logging

from torch.nn.parallel.distributed import logger
from language_model.subword_tokenizer import SubwordTokenizer
from language_model.helpers import get_device, apply_runtime_overrides
from language_model.model import GPTLanguageModel
from language_model.train_utils import base_train_model, train_chat_alignment
from language_model.config import PARQUET_DIR_PATH, TEXT_COLUMN, VOCAB_PATH, QA_PARQUET_PATH, CONTEXT_PARQUET_PATH, LOG_LEVEL, FINETUNING_MAX_EPOCHS, BLOCK_SIZE, RUNTIME_OVERRIDES_FILE
from language_model.qa_processing import prepare_context_data_for_training, process_qa_pairs_dataset
from language_model.helpers import configure_colored_logging

# Configure logging
configure_colored_logging(LOG_LEVEL)

if __name__ == "__main__":
    output_dir = os.path.join('data', 'output')
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
    logging.info("This file can be included in base training by moving it to the parquet_files directory. That would ensure that the model has seen the context text for the QA pairs during base training.")

    # Base training - now uses train_and_poll for polling and deletion
    # ==================== BASE TRAINING START DIVIDER ====================
    logging.info("\n" + "="*80)
    logging.info(f"\033[94m🚀 STARTING BASE MODEL TRAINING\033[0m")
    logging.info(f"\033[94m🕒 START TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
    logging.info(f"\033[94m📂 PARQUET DIR: {parquet_dir_path}\033[0m")
    logging.info(f"\033[94m📝 VOCAB PATH: {vocab_path}\033[0m")
    logging.info(f"\033[94m🎯 OUTPUT DIR: {output_dir}\033[0m")
    logging.info("="*80 + "\n")
    
    # Check for existing checkpoint
    checkpoint_path = os.path.join(output_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        logging.info(f"Resuming base training from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None
    # Pass checkpoint_path to base_train_model (update base_train_model to accept this argument)
    global_iter = base_train_model(parquet_dir_path, text_column, vocab_path, output_dir=output_dir, checkpoint_path=checkpoint_path)

    # ==================== BASE TRAINING COMPLETION DIVIDER ====================
    logging.info("\n" + "="*80)
    logging.info(f"\033[92m✅ COMPLETED BASE MODEL TRAINING\033[0m")
    logging.info(f"\033[92m🔢 FINAL GLOBAL ITERATION: {global_iter:,}\033[0m")
    logging.info(f"\033[92m📁 MODEL SAVED TO: {output_dir}\033[0m")
    logging.info("="*80 + "\n")

    # Now process QA dataset for fine-tuning
    logger.info(f"Looking for vocabulary in path: {vocab_path}")
    tokenizer = SubwordTokenizer(vocab_file=vocab_path)
    vocab_size = tokenizer.get_vocab_size()

    logging.info("\n" + "="*60)
    logging.info(f"\033[95m📋 CREATING QA DATASET FOR FINE-TUNING\033[0m")
    logging.info("="*60)
    block_size = BLOCK_SIZE  # Should match model config
    qa_tensor = process_qa_pairs_dataset(
        qa_parquet_path, 
        tokenizer,
        max_length=block_size
    )
    logging.debug(f'QA tensor shape: {qa_tensor.shape}')
    
    # Load pre-trained model with correct vocab size
    device = get_device()
    model = GPTLanguageModel(vocab_size).to(device)
    
    # Try to load the best available model for fine-tuning
    model_loaded = False
    model_paths_to_try = [
        os.path.join(output_dir, 'best_model.pt'),
        os.path.join(output_dir, 'model_error.pt')
    ]
    
    for model_path in model_paths_to_try:
        if os.path.exists(model_path):
            try:
                # Check if vocab sizes match
                checkpoint = torch.load(model_path, map_location='cpu')
                checkpoint_vocab_size = checkpoint['token_embedding_table.weight'].shape[0]
                
                if checkpoint_vocab_size != vocab_size:
                    logging.info(f'Vocab size mismatch: checkpoint has {checkpoint_vocab_size}, need {vocab_size}')
                    logging.info('Resizing checkpoint to match actual vocabulary size...')
                    
                    from language_model.scripts.checkpoint_resizer import resize_checkpoint_for_actual_vocab
                    resized_path = resize_checkpoint_for_actual_vocab(
                        checkpoint_path=model_path,
                        vocab_file=vocab_path,
                        output_path=model_path
                    )
                    model.load_state_dict(torch.load(resized_path, map_location=device))
                    logging.info(f'\033[92mPre-trained model loaded from resized checkpoint: {resized_path}\033[0m')
                else:
                    model.load_state_dict(checkpoint)
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
    
    # Create a dummy optimizer to satisfy the apply_runtime_overrides function signature
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Runtime parameters for fine-tuning
    finetuning_runtime_params = {
        'finetuning_max_epochs': FINETUNING_MAX_EPOCHS,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'log_level': LOG_LEVEL
    }
    
    # Apply runtime overrides
    finetuning_runtime_params, _ = apply_runtime_overrides(
        dummy_optimizer, None, finetuning_runtime_params, RUNTIME_OVERRIDES_FILE
    )
    
    global_step = train_chat_alignment(
        model, 
        qa_tensor,
        output_dir=output_dir,
        lr=finetuning_runtime_params['learning_rate'], 
        batch_size=finetuning_runtime_params.get('batch_size', 4), 
        val_split=0.1,
        global_step=global_iter,
        runtime_params=finetuning_runtime_params
    )
    
    # ==================== TRAINING PIPELINE COMPLETION DIVIDER ====================
    logging.info("\n" + "="*80)
    logging.info(f"\033[93m🎉 COMPLETE TRAINING PIPELINE FINISHED SUCCESSFULLY! 🎉\033[0m")
    logging.info(f"\033[93m🏆 FINAL GLOBAL STEP: {global_step:,}\033[0m")
    logging.info(f"\033[93m📁 OUTPUT DIRECTORY: {output_dir}\033[0m")
    logging.info(f"\033[93m🚀 MODEL READY FOR INFERENCE AND DEPLOYMENT! 🚀\033[0m")
    logging.info("="*80 + "\n")
