"""
Vocabulary management for tokens.

Handles bidirectional mapping between tokens (characters/phonemes) and indices.
"""

from typing import Dict, List

from .constants import (
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    UNK_TOKEN,
    PAD_IDX,
    START_IDX,
    END_IDX,
    UNK_IDX,
    SPECIAL_TOKENS_COUNT,
)


class Vocabulary:
    """
    Manages bidirectional mapping between tokens and integer indices.

    Reserves the first 4 indices for special tokens: PAD, START, END, UNK.
    """

    def __init__(self) -> None:
        """Initialize vocabulary with special tokens."""
        self.token_to_idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            START_TOKEN: START_IDX,
            END_TOKEN: END_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        self.idx_to_token: Dict[int, str] = {v: k for k, v in self.token_to_idx.items()}
        self.vocab_size: int = SPECIAL_TOKENS_COUNT

    def add_token(self, token: str) -> None:
        """
        Add a single token to vocabulary if it doesn't exist.
        """
        if token not in self.token_to_idx:
            idx = self.vocab_size
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            self.vocab_size += 1

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add multiple tokens to vocabulary.
        """
        for token in tokens:
            self.add_token(token)

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert list of tokens to list of indices.
        Unknown tokens are mapped to UNK_IDX.
        """
        return [self.token_to_idx.get(token, UNK_IDX) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Convert list of indices back to tokens.
        """
        return [self.idx_to_token.get(idx, UNK_TOKEN) for idx in indices]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token_to_idx
