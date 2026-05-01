"""
G2P Data Management Package

Provides utilities for loading, cleaning, and preparing Grapheme-to-Phoneme data.
"""

from pathlib import Path
from typing import Tuple, Any

from torch.utils.data import DataLoader

from .vocabulary import Vocabulary
from .cleaning import clean_and_build_vocabularies
from .loading import load_tsv_file
from .dataset import G2PDataset
from .constants import (
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    UNK_TOKEN,
    PAD_IDX,
    START_IDX,
    END_IDX,
    UNK_IDX,
)


__all__ = [
    "Vocabulary",
    "create_dataloader",
    "load_tsv_file",
    "clean_and_build_vocabularies",
    "G2PDataset",
    "PAD_TOKEN",
    "START_TOKEN",
    "END_TOKEN",
    "UNK_TOKEN",
    "PAD_IDX",
    "START_IDX",
    "END_IDX",
    "UNK_IDX",
]


def create_dataloader(
    file_path: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader[Any], Vocabulary, Vocabulary]:
    """
    Complete pipeline: loads, cleans, builds vocabularies, creates dataset, and 
    created a DataLoader.
    """
    pairs = load_tsv_file(file_path)

    cleaned_pairs, grapheme_vocab, phoneme_vocab = clean_and_build_vocabularies(pairs)

    if not cleaned_pairs:
        raise ValueError("No valid pairs found after cleaning")

    dataset = G2PDataset(cleaned_pairs, grapheme_vocab, phoneme_vocab)

    from .collate import collate_batch
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )

    return dataloader, grapheme_vocab, phoneme_vocab
