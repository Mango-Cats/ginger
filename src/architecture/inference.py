import torch
from src.architecture.transformer import create_causal_mask
from src.dataman.constants import END_IDX, START_IDX


@torch.no_grad()
def batched_greedy_decode(
    model,
    src_tensor: torch.Tensor,
    device: torch.device,
    max_len: int = 50,
):
    model.eval()

    B = src_tensor.size(0)

    src_tensor = src_tensor.to(device)

    tgt_tokens = torch.full((B, 1), START_IDX, dtype=torch.long, device=device)

    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        seq_len = tgt_tokens.size(1)
        tgt_mask = create_causal_mask(seq_len, device)

        output = model(src_tensor, tgt_tokens, tgt_mask=tgt_mask)
        next_logits = output[:, -1, :]  # (B, V)

        next_tokens = torch.argmax(next_logits, dim=-1)  # (B,)

        tgt_tokens = torch.cat([tgt_tokens, next_tokens.unsqueeze(1)], dim=1)

        finished |= next_tokens == END_IDX

        if finished.all():
            break

    results = []
    for i in range(B):
        seq = tgt_tokens[i].tolist()
        filtered = [t for t in seq if t not in (START_IDX, END_IDX)]
        results.append(filtered)

    return results
