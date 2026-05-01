from typing import Literal

from .levenshtein import levenshtein_distance


def compute_metrics(
    predictions, references
) -> tuple[float | Literal[0], float | Literal[0]]:
    """
    Compute PER (Phoneme Error Rate) and WER (Word Error Rate).
    """
    total_phoneme_errors = 0
    total_phonemes = 0
    word_errors = 0

    for pred, ref in zip(predictions, references):
        errors = levenshtein_distance(pred, ref)
        total_phoneme_errors += errors
        total_phonemes += len(ref)

        if errors > 0:
            word_errors += 1

    per: float | Literal[0] = (
        100.0 * total_phoneme_errors / total_phonemes if total_phonemes > 0 else 0
    )
    wer: float | Literal[0] = (
        100.0 * word_errors / len(references) if len(references) > 0 else 0
    )

    return per, wer
