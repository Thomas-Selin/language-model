import json
import re
from typing import List

class WordTokenizer:
    """
    Word-level tokenizer for mapping between words and integer tokens.
    """
    def __init__(self, vocab_file: str = "vocab.json"):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.word_to_token_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.token_index_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    @staticmethod
    def build_vocab(text: str, vocab_size: int = 10000, min_frequency: int = 1) -> List[str]:
        # Simple whitespace and punctuation splitting
        words = re.findall(r"\b\w+\b", text.lower())
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        # Sort by frequency and take the most common
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        vocab = [w for w, count in sorted_words[:vocab_size] if count >= min_frequency]
        return vocab

    @staticmethod
    def save_vocab(vocab, path="data/output/vocab.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(vocab, f)
        print(f"Vocabulary saved to {path}")

    def encode(self, text: str) -> List[int]:
        words = re.findall(r"\b\w+\b", text.lower())
        return [self.word_to_token_index[w] for w in words if w in self.word_to_token_index]

    def decode(self, tokens: List[int]) -> str:
        return ' '.join([self.token_index_to_word[i] for i in tokens if i in self.token_index_to_word])
        
    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.
        
        Returns:
            int: The number of tokens in the vocabulary.
        """
        return len(self.vocab)
