import json
from typing import List

class CharTokenizer:
    """
    Character-level tokenizer for mapping between characters and integer tokens.
    """
    def __init__(self, chars_file: str = "chars.json"):
        with open(chars_file, 'r') as f:
            self.chars = json.load(f)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of integer tokens."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        """Decode a list of integer tokens back into a string."""
        return ''.join([self.itos[i] for i in tokens if i in self.itos])

    @staticmethod
    def save_charset(chars, path="data/output/chars.json"):
        """Save a list of characters to a JSON file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(chars, f)
        print(f"Character set saved to {path}")
