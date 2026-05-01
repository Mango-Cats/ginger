"""
Collate function for batching sequences with padding.

Handles variable-length sequences by padding them to the same length
within each batch.
"""

from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from .constants import PAD_IDX


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads sequences to the same length within each batch using PAD_IDX.
    Returns in batch_first=False format: (seq_len, batch_size)
    """
    grapheme_tensors: List[torch.Tensor] = []
    phoneme_tensors: List[torch.Tensor] = []

    for grapheme_tensor, phoneme_tensor in batch:
        grapheme_tensors.append(grapheme_tensor)
        phoneme_tensors.append(phoneme_tensor)

    grapheme_padded = pad_sequence(
        grapheme_tensors,
        batch_first=False,
        padding_value=PAD_IDX,
    )
    phoneme_padded = pad_sequence(
        phoneme_tensors,
        batch_first=False,
        padding_value=PAD_IDX,
    )

    return grapheme_padded, phoneme_padded
