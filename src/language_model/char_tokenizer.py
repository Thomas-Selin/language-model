import json
from typing import List

class CharTokenizer:
    """
    Character-level tokenizer for mapping between characters and integer tokens.
    """
    def __init__(self, chars_file: str = "chars.json"):
        with open(chars_file, 'r') as f:
            self.character_set = json.load(f)
        self.char_to_token_index = {char: idx for idx, char in enumerate(self.character_set)}
        self.token_index_to_char = {idx: char for idx, char in enumerate(self.character_set)}

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of integer tokens."""
        return [self.char_to_token_index[c] for c in text if c in self.char_to_token_index]

    def decode(self, tokens: List[int]) -> str:
        """Decode a list of integer tokens back into a string."""
        return ''.join([self.token_index_to_char[i] for i in tokens if i in self.token_index_to_char])

    @staticmethod
    def save_charset(chars, path="data/output/chars.json"):
        """Save a list of characters to a JSON file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(chars, f)
        print(f"Character set saved to {path}")
