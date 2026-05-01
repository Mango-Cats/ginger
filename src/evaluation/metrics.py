from typing import Tuple
from .levenshtein import levenshtein_distance


def compute_metrics(predictions, references) -> Tuple[float, float]:

    total_phoneme_errors = 0
    total_phonemes = 0
    word_errors = 0

    assert len(predictions) == len(references), "Mismatch in predictions/references"

    for pred, ref in zip(predictions, references):

        errors = levenshtein_distance(pred, ref)

        total_phoneme_errors += errors
        total_phonemes += len(ref)

        if errors > 0:
            word_errors += 1

    per = 100.0 * total_phoneme_errors / max(total_phonemes, 1)
    wer = 100.0 * word_errors / max(len(references), 1)

    return per, wer