import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from typing import Union

from src.architecture.transformer import create_causal_mask
from src.dataman.constants import PAD_IDX

DeviceType = Union[torch.device, str]

# PRE-COMPUTE MASK
MAX_SEQ_LEN: int = 500
GLOBAL_CAUSAL_MASK: torch.Tensor = create_causal_mask(MAX_SEQ_LEN, device="cpu")


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
):
    model.train()
    total_loss = 0.0

    global_mask = GLOBAL_CAUSAL_MASK.bool()

    for _, (src, tgt) in enumerate(dataloader):

        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # ✅ batch-first fix
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        src_padding_mask = (src == PAD_IDX).bool()
        tgt_padding_mask = (tgt_input == PAD_IDX).bool()

        seq_len = tgt_input.size(1)
        tgt_mask = global_mask[:seq_len, :seq_len].to(device)

        with torch.autocast("cuda"):
            output = model(
                src,
                tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_expected.reshape(-1),
            )

        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    global_mask = GLOBAL_CAUSAL_MASK.bool()

    for _, (src, tgt) in enumerate(dataloader):

        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        seq_len = tgt_input.size(1)
        tgt_mask = global_mask[:seq_len, :seq_len].to(device)

        with torch.autocast("cuda"):
            output = model(src, tgt_input, tgt_mask=tgt_mask)

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_expected.reshape(-1),
            )

        total_loss += loss.item()

    return total_loss / len(dataloader)