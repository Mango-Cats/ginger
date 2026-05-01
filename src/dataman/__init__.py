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
    "create_dataloaders",
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


import random


def create_dataloaders(
    file_path: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], Vocabulary, Vocabulary]:
    """
    Complete pipeline: loads, cleans, builds vocabularies, creates dataset, and
    created a DataLoader.
    """
    pairs = load_tsv_file(file_path)

    cleaned_pairs, grapheme_vocab, phoneme_vocab = clean_and_build_vocabularies(pairs)

    if not cleaned_pairs:
        raise ValueError("No valid pairs found after cleaning")

    # Create splits
    random.seed(42)
    shuffled_pairs = cleaned_pairs.copy()
    random.shuffle(shuffled_pairs)

    n = len(shuffled_pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_pairs = shuffled_pairs[:train_end]
    val_pairs = shuffled_pairs[train_end:val_end]
    test_pairs = shuffled_pairs[val_end:]

    train_dataset = G2PDataset(
        data_pairs=train_pairs,
        grapheme_vocab=grapheme_vocab,
        phoneme_vocab=phoneme_vocab,
    )
    val_dataset = G2PDataset(
        data_pairs=val_pairs, grapheme_vocab=grapheme_vocab, phoneme_vocab=phoneme_vocab
    )
    test_dataset = G2PDataset(
        data_pairs=test_pairs,
        grapheme_vocab=grapheme_vocab,
        phoneme_vocab=phoneme_vocab,
    )

    from .collate import collate_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, grapheme_vocab, phoneme_vocab
