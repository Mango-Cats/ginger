from typing import List

import torch

from src.architecture.transformer import create_causal_mask
from src.dataman import Vocabulary
from src.dataman.constants import END_IDX, START_IDX


@torch.no_grad()
def greedy_decode(
    model,
    src_tokens: List[str],
    grapheme_vocab: Vocabulary,
    phoneme_vocab: Vocabulary,
    device: torch.device,
    max_length: int = 50,
):
    """
    Generate phoneme sequence greedily (pick highest probability at each step).
    """
    model.eval()

    src_indices = grapheme_vocab.encode(src_tokens)
    src_indices = [START_IDX] + src_indices + [END_IDX]
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(device)

    tgt_indices = [START_IDX]

    for _ in range(max_length):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(1).to(device)

        tgt_mask = create_causal_mask(tgt_tensor.size(0), device)

        output = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)

        next_token_logits = output[-1, 0, :]
        next_token_idx = next_token_logits.argmax(dim=-1).item()

        tgt_indices.append(next_token_idx)

        if next_token_idx == END_IDX:
            break

    tgt_indices = [idx for idx in tgt_indices if idx not in [START_IDX, END_IDX]]
    predicted_phonemes = phoneme_vocab.decode(tgt_indices)

    return predicted_phonemes, tgt_indices
