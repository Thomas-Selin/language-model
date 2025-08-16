import logging
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
import os
from typing import List
from config import LOG_LEVEL, MAX_VOCAB_SIZE
from src.language_model.helpers import configure_colored_logging

# Configure logging
configure_colored_logging(LOG_LEVEL)

def create_bpe_tokenizer(text_files, vocab_size=MAX_VOCAB_SIZE):
    """Create a ByteLevel BPE tokenizer trained on the given text files"""
    tokenizer = Tokenizer(models.BPE(unk_token=SubwordTokenizer.UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            SubwordTokenizer.UNK_TOKEN,
            SubwordTokenizer.BOS_TOKEN,
            SubwordTokenizer.EOS_TOKEN,
            SubwordTokenizer.PAD_TOKEN,
        ],
    )
    tokenizer.train(text_files, trainer)
    return tokenizer

class SubwordTokenizer:
    """Subword-level tokenizer for mapping between text and integer tokens."""
    UNK_TOKEN = "<|unk|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|endoftext|>"
    PAD_TOKEN = "<|pad|>"

    def __init__(self, vocab_file: str = "vocab.json"):
        self.tokenizer = Tokenizer.from_file(vocab_file)
        self._vocab_size = self.tokenizer.get_vocab_size()
        self.vocab = self.tokenizer.get_vocab()
        self.unk_token_id = self.vocab.get(self.UNK_TOKEN)
        self.bos_token_id = self.vocab.get(self.BOS_TOKEN)
        self.eos_token_id = self.vocab.get(self.EOS_TOKEN)
        self.pad_token_id = self.vocab.get(self.PAD_TOKEN)

    @staticmethod
    def build_vocab(text: str, vocab_size: int = MAX_VOCAB_SIZE, min_frequency: int = 2) -> Tokenizer:
        """Build a ByteLevel BPE vocabulary from text"""
        temp_file = "temp_training_file.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(text)
        tokenizer = create_bpe_tokenizer([temp_file], vocab_size=vocab_size)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return tokenizer

    @staticmethod
    def save_vocab(tokenizer, path="data/output/tokenizer.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tokenizer.save(path)
        logging.info(f"Tokenizer saved to {path}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens and self.bos_token_id is not None and self.eos_token_id is not None:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def get_vocab_size(self) -> int:
        return self._vocab_size