import torch

from src.architecture.transformer import create_causal_mask


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        tgt_expected = tgt[1:, :]

        tgt_mask = create_causal_mask(tgt_input.size(0), device)
        output = model(src, tgt_input, tgt_mask=tgt_mask)

        loss = criterion(output.view(-1, output.size(-1)), tgt_expected.view(-1))
        total_loss += loss.item()

    return total_loss / len(dataloader)
