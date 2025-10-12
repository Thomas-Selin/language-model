"""Question-Answer dataset processing utilities.

This module handles processing of QA pairs for chat alignment and fine-tuning,
including context extraction and data formatting.
"""

import torch
import pandas as pd
import ast
import gc
import logging
import os
from typing import Optional
from language_model.subword_tokenizer import SubwordTokenizer
from language_model.exceptions import InvalidDataFormatError


def process_qa_pairs_dataset(
    qa_parquet_path: str,
    tokenizer: SubwordTokenizer,
    max_length: int
) -> torch.Tensor:
    """Process a single-turn question-answer pair dataset for chat alignment.
    
    Reads a parquet file with 'question' and 'answers' columns and formats
    them as "Question: ... Answer: ..." pairs for training.
    
    Args:
        qa_parquet_path: Path to parquet file containing QA pairs
        tokenizer: Tokenizer to use for encoding text
        max_length: Maximum sequence length (will pad/truncate)
        
    Returns:
        Tensor of tokenized QA pairs, shape (num_examples, max_length)
        
    Raises:
        InvalidDataFormatError: If required columns are missing
    """
    qa_df = pd.read_parquet(qa_parquet_path)
    
    if not {'question', 'answers'}.issubset(qa_df.columns):
        raise InvalidDataFormatError(
            "Parquet file must contain 'question' and 'answers' columns."
        )
    
    pairs = []
    for _, row in qa_df.iterrows():
        question = str(row['question']) if 'question' in row else ''
        answers_field = row['answers']
        
        # Parse answers field (can be dict or stringified dict)
        if isinstance(answers_field, str):
            answers_dict = ast.literal_eval(answers_field)
        else:
            answers_dict = answers_field
        
        answer_text = ''
        if isinstance(answers_dict, dict) and 'text' in answers_dict:
            if len(answers_dict['text']) > 0:
                answer_text = str(answers_dict['text'][0])
        
        pairs.append((question, answer_text))
    
    # Format as training examples
    examples = [f"Question: {q} Answer: {a}" for q, a in pairs]
    
    # Tokenize
    tokenized = [tokenizer.encode(ex[:max_length]) for ex in examples]
    
    # Pad/truncate to max_length
    tensor = torch.tensor(
        [
            t + [0] * (max_length - len(t)) if len(t) < max_length else t[:max_length]
            for t in tokenized
        ],
        dtype=torch.long
    )
    
    # Clean up
    del tokenized, examples, pairs
    gc.collect()
    
    logging.info(f"Processed {len(tensor)} QA pairs")
    return tensor


def prepare_context_data_for_training(
    qa_parquet_path: str,
    output_parquet_path: str,
    text_column: str = 'context'
) -> bool:
    """Extract context data from a QA dataset for base training.
    
    Creates a new parquet file with unique context data that can be used
    in the regular training pipeline.
    
    Args:
        qa_parquet_path: Path to the QA dataset parquet file
        output_parquet_path: Path to save the context data parquet file
        text_column: Name of the column to use in the output parquet file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        
        # Load QA dataset
        qa_df = pd.read_parquet(qa_parquet_path)
        
        if 'context' not in qa_df.columns:
            logging.error(f"Error: 'context' column not found in {qa_parquet_path}")
            return False
        
        # Extract context data, filter out empty contexts, and drop duplicates
        contexts = qa_df['context'].dropna().astype(str)
        contexts = contexts[contexts.str.strip().str.len() > 0]
        contexts = contexts.drop_duplicates()
        
        if contexts.empty:
            logging.info("No valid context data found in the dataset")
            return False
        
        # Create a new dataframe with the context data
        context_df = pd.DataFrame({text_column: contexts})
        
        # Save to parquet
        context_df.to_parquet(output_parquet_path, index=False)
        
        logging.info(
            f"Successfully saved {len(context_df)} unique contexts to {output_parquet_path}"
        )
        return True
        
    except Exception as e:
        logging.error(f"Error preparing context data: {e}")
        return False


def format_qa_for_chat(
    question: str,
    answer: str,
    system_prompt: Optional[str] = None
) -> str:
    """Format a QA pair for chat-style training.
    
    Args:
        question: The question text
        answer: The answer text
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Formatted chat string
    """
    if system_prompt:
        return f"System: {system_prompt}\nQuestion: {question}\nAnswer: {answer}"
    else:
        return f"Question: {question}\nAnswer: {answer}"


def validate_qa_dataset(
    qa_parquet_path: str,
    min_pairs: int = 1
) -> bool:
    """Validate that a QA dataset meets requirements.
    
    Args:
        qa_parquet_path: Path to QA parquet file
        min_pairs: Minimum number of QA pairs required
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_parquet(qa_parquet_path)
        
        # Check required columns
        if not {'question', 'answers'}.issubset(df.columns):
            logging.error("QA dataset missing required columns")
            return False
        
        # Check minimum rows
        if len(df) < min_pairs:
            logging.error(
                f"QA dataset has only {len(df)} pairs, "
                f"minimum {min_pairs} required"
            )
            return False
        
        # Check for empty questions/answers
        empty_questions = df['question'].isna().sum()
        if empty_questions > len(df) * 0.5:  # More than 50% empty
            logging.warning(
                f"QA dataset has {empty_questions} empty questions "
                f"out of {len(df)} total"
            )
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating QA dataset: {e}")
        return False
