import torch
import random

from src.architecture.inference import greedy_decode
from src.evaluation.metrics import compute_metrics


def evaluate_model(
    model,
    test_loader,
    grapheme_vocab,
    phoneme_vocab,
    device,
    model_path="best_g2p_model.pt",
    sample_size=5,
):
    """
    Full evaluation pipeline:
    - loads best checkpoint
    - runs inference on test set
    - computes PER / WER
    - prints samples
    """

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_pairs = test_loader.dataset.data_pairs

    predictions = []
    references = []

    for graphemes, true_phonemes in test_pairs:
        word = "".join(graphemes)

        pred, _ = greedy_decode(
            model=model,
            src_tokens=list(word),
            grapheme_vocab=grapheme_vocab,
            phoneme_vocab=phoneme_vocab,
            device=device,
        )

        predictions.append(pred)
        references.append(true_phonemes)

    per, wer = compute_metrics(predictions, references)

    print("\nMetrics:")
    print(f"Total Words Tested: {len(test_pairs)}")
    print(f"Phoneme Error Rate (PER): {per:.2f}%")
    print(f"Word Error Rate (WER):    {wer:.2f}%")

    print("\nSample Predictions:")
    sample_indices = random.sample(
        range(len(test_pairs)), k=min(sample_size, len(test_pairs))
    )

    for idx in sample_indices:
        graphemes = "".join(test_pairs[idx][0])

        print(f"\nWord:     {graphemes.upper()}")
        print(f"Expected: {' '.join(references[idx])}")
        print(f"Output:   {' '.join(predictions[idx])}")

    return {
        "per": per,
        "wer": wer,
        "predictions": predictions,
        "references": references,
    }
