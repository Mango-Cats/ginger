"""
Data cleaning and vocabulary building utilities.

Validates and normalizes word-IPA pairs, then builds vocabularies.
"""

from typing import List, Tuple

from .vocabulary import Vocabulary


def validate_pair(word: str, ipa: str) -> bool:
    """
    Validate a word-IPA pair is well-formed.
    """
    if not word or not ipa:
        return False
    if not word.isalpha():
        return False
    return True


def clean_word(word: str) -> str:
    """
    Clean and normalize a word.
    """
    return word.strip().lower()


def clean_ipa(ipa: str) -> List[str]:
    """
    Clean and normalize IPA transcription.
    Assumes phonemes are space-separated in the input.
    """
    phonemes = ipa.strip().split()
    return phonemes


def clean_and_build_vocabularies(
    pairs: List[Tuple[str, str]],
) -> Tuple[List[Tuple[List[str], List[str]]], Vocabulary, Vocabulary]:
    """
    Clean data and build grapheme and phoneme vocabularies.
    Validates each pair, normalizes words and IPA, and builds separate
    vocabularies for graphemes (characters) and phonemes.
    """
    grapheme_vocab = Vocabulary()
    phoneme_vocab = Vocabulary()
    cleaned_pairs: List[Tuple[List[str], List[str]]] = []

    skipped_count = 0

    for word, ipa in pairs:
        if not validate_pair(word, ipa):
            skipped_count += 1
            continue

        clean_w = clean_word(word)
        clean_p = clean_ipa(ipa)

        grapheme_list: List[str] = list(clean_w)

        cleaned_pairs.append((grapheme_list, clean_p))

        grapheme_vocab.add_tokens(grapheme_list)
        phoneme_vocab.add_tokens(clean_p)

    return cleaned_pairs, grapheme_vocab, phoneme_vocab
