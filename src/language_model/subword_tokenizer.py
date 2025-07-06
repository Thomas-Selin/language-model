from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os
import json
from typing import List

def create_bpe_tokenizer(text_files, vocab_size=3000):
    """Create a BPE tokenizer trained on the given text files"""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<UNK>", "<|endoftext|>"]
    )
    
    tokenizer.train(text_files, trainer)
    return tokenizer

class SubwordTokenizer:
    """
    Subword-level tokenizer for mapping between text and integer tokens.
    """
    UNK_TOKEN = "<UNK>"
    EOS_TOKEN = "<|endoftext|>"
    
    def __init__(self, vocab_file: str = "vocab.json"):
        """Initialize from a saved tokenizer file"""
        self.tokenizer = Tokenizer.from_file(vocab_file)
        self._vocab_size = self.tokenizer.get_vocab_size()
        
        # Get the vocabulary to find the EOS token ID
        self.vocab = self.tokenizer.get_vocab()
        # If EOS token exists in vocab, store its ID, otherwise use None
        self.eos_token_id = self.vocab.get(self.EOS_TOKEN)
    
    @staticmethod
    def build_vocab(text: str, vocab_size: int = 3000, min_frequency: int = 1) -> Tokenizer:
        """Build a BPE vocabulary from text"""
        # Save text to temporary file
        temp_file = "temp_training_file.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Create and train the tokenizer with the EOS token
        tokenizer = create_bpe_tokenizer([temp_file], vocab_size=vocab_size)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return tokenizer
    
    @staticmethod
    def save_vocab(tokenizer, path="data/output/tokenizer.json"):
        """Save the tokenizer to a file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tokenizer.save(path)
        print(f"Tokenizer saved to {path}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        # The encode method returns a dictionary with IDs
        encoded = self.tokenizer.encode(text)
        return encoded.ids
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(tokens)
    
    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary"""
        return self._vocab_size
