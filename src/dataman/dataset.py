"""
PyTorch Dataset implementation for Grapheme-to-Phoneme pairs.

Converts token sequences to tensor pairs with special tokens.
"""

from typing import List, Tuple

import torch
from typing import Any
from torch.utils.data import Dataset

from .vocabulary import Vocabulary
from .constants import START_IDX, END_IDX


class G2PDataset(Dataset[Any]):
    """
    PyTorch Dataset for Grapheme-to-Phoneme pairs.

    Converts token sequences to tensor pairs, automatically adding
    START and END tokens to each sequence.
    """

    def __init__(
        self,
        data_pairs: List[Tuple[List[str], List[str]]],
        grapheme_vocab: Vocabulary,
        phoneme_vocab: Vocabulary,
    ) -> None:
        self.data_pairs = data_pairs
        self.grapheme_vocab = grapheme_vocab
        self.phoneme_vocab = phoneme_vocab

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        """
        grapheme_list, phoneme_list = self.data_pairs[idx]

        grapheme_indices = self.grapheme_vocab.encode(grapheme_list)
        phoneme_indices = self.phoneme_vocab.encode(phoneme_list)

        grapheme_with_tokens = [START_IDX] + grapheme_indices + [END_IDX]
        phoneme_with_tokens = [START_IDX] + phoneme_indices + [END_IDX]

        grapheme_tensor = torch.tensor(grapheme_with_tokens, dtype=torch.long)
        phoneme_tensor = torch.tensor(phoneme_with_tokens, dtype=torch.long)

        return grapheme_tensor, phoneme_tensor
