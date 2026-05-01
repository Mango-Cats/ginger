"""
Data loading utilities for TSV files.

Handles reading word-IPA transcription pairs from TSV format.
"""

import csv
from pathlib import Path
from typing import List, Tuple


def load_tsv_file(file_path: Path) -> List[Tuple[str, str]]:
    """
    Load TSV file with word and IPA transcription pairs.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    pairs: List[Tuple[str, str]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for _, row in enumerate(reader):
            if not row or not any(row):
                continue

            if len(row) < 2:
                continue

            word = row[0].strip()
            ipa = row[1].strip()

            if word and ipa:
                pairs.append((word, ipa))

    if not pairs:
        raise ValueError(f"No valid pairs found in {file_path}")

    return pairs
