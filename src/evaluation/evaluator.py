import torch
import random

from src.architecture.inference import batched_greedy_decode
from src.evaluation.metrics import compute_metrics




@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    grapheme_vocab,
    phoneme_vocab,
    device,
    model_path="best_g2p_model.pt",
    sample_size=5,
):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    model = torch.compile(model)

    predictions = []
    references = []
    sample_pairs = []  # Store some raw strings for the sample output

    for src_tensor, tgt_tensor in test_loader:
        src_tensor = src_tensor.to(device)

        batch_predictions = batched_greedy_decode(
            model=model, src_tensor=src_tensor, device=device, max_len=50
        )

        for i in range(len(src_tensor)):
            pred_tokens = phoneme_vocab.decode(batch_predictions[i])
            predictions.append(pred_tokens)

            true_tokens = [
                phoneme_vocab.idx_to_token[idx.item()]
                for idx in tgt_tensor[i]
                if idx.item() not in [0, 1, 2]
            ]
            references.append(true_tokens)

            src_word_tokens = [
                grapheme_vocab.idx_to_token[idx.item()]
                for idx in src_tensor[i]
                if idx.item() not in [0, 1, 2]
            ]
            sample_pairs.append("".join(src_word_tokens))

    per, wer = compute_metrics(predictions, references)

    print("\nMetrics:")
    print(f"Total Words Tested: {len(predictions)}")
    print(f"Phoneme Error Rate (PER): {per:.2f}%")
    print(f"Word Error Rate (WER):    {wer:.2f}%")

    print("\nSample Predictions:")
    sample_indices = random.sample(
        range(len(predictions)), k=min(sample_size, len(predictions))
    )

    for idx in sample_indices:
        print(f"\nWord:     {sample_pairs[idx].upper()}")
        print(f"Expected: {' '.join(references[idx])}")
        print(f"Output:   {' '.join(predictions[idx])}")

    return {
        "per": per,
        "wer": wer,
        "predictions": predictions,
        "references": references,
    }
