import json
import re
from typing import List

class WordTokenizer:
    """
    Word-level tokenizer for mapping between words and integer tokens.
    """
    # Special token for unknown words
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, vocab_file: str = "vocab.json"):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
            
        # Use only one dictionary and compute indices on-the-fly
        self.word_to_token_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.unk_idx = self.word_to_token_index.get(self.UNK_TOKEN, 0)
        
        # Don't store token_index_to_word in memory all the time
        # Instead, only generate it when decoding
    
    @staticmethod
    def build_vocab(text: str, vocab_size: int = 10000, min_frequency: int = 1) -> List[str]:
        # Simple whitespace and punctuation splitting
        words = re.findall(r"\b\w+\b", text.lower())
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        # Sort by frequency and take the most common
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        # Filter by min_frequency and limit to vocab_size-1 (to leave room for UNK)
        vocab = [w for w, count in sorted_words[:vocab_size-1] if count >= min_frequency]
        # Add UNK token at the beginning
        return [WordTokenizer.UNK_TOKEN] + vocab

    @staticmethod
    def save_vocab(vocab, path="data/output/vocab.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(vocab, f)
        print(f"Vocabulary saved to {path}")

    def encode(self, text: str) -> List[int]:
        words = re.findall(r"\b\w+\b", text.lower())
        # Use the UNK token index for words not in vocabulary
        return [self.word_to_token_index.get(w, self.unk_idx) for w in words]

    def decode(self, tokens: List[int]) -> str:
        # Create mapping on-demand instead of storing permanently
        return ' '.join([self.vocab[i] if 0 <= i < len(self.vocab) else self.UNK_TOKEN for i in tokens])
        
    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.
        
        Returns:
            int: The number of tokens in the vocabulary.
        """
        return len(self.vocab)
        