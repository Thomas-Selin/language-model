import sentencepiece as spm
import os
from typing import List

class SubwordTokenizer:
    """
    Subword-level tokenizer using SentencePiece (BPE or Unigram).
    """
    def __init__(self, model_file: str = "spm.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)

    @staticmethod
    def train(text_file: str, model_prefix: str = "spm", vocab_size: int = 8000, model_type: str = "bpe"):
        """Train a SentencePiece model and save it as model_prefix.model and model_prefix.vocab"""
        spm.SentencePieceTrainer.Train(
            input=text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            bos_id=0,
            eos_id=1,
            pad_id=2
        )
        print(f"SentencePiece model saved as {model_prefix}.model and {model_prefix}.vocab")

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: List[int]) -> str:
        return self.sp.decode(tokens)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
