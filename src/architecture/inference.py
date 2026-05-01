from typing import List, Tuple
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
) -> Tuple[str, List[int]]:

    model.eval()

    src_indices = grapheme_vocab.encode(src_tokens)
    src_indices = [START_IDX] + src_indices + [END_IDX]

    src_tensor = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)

    tgt_indices = [START_IDX]

    for _ in range(max_length):
        tgt_tensor = torch.tensor(
            tgt_indices, dtype=torch.long, device=device
        ).unsqueeze(0)

        seq_len = tgt_tensor.size(1)
        tgt_mask = create_causal_mask(seq_len, device)

        output = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)

        next_token_logits = output[:, -1, :]  # (1, vocab)

        next_token_idx = torch.argmax(next_token_logits, dim=-1).item()

        tgt_indices.append(next_token_idx)

        if next_token_idx == END_IDX:
            break

    filtered = [i for i in tgt_indices if i not in (START_IDX, END_IDX)]

    return phoneme_vocab.decode(filtered), filtered
