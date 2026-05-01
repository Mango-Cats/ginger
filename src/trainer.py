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

# pre-computation
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

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    num_epochs,
    patience,
    best_model_path="best_g2p_model.pt",
    log_every=5,
):
    no_improve = 0
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    window_best_val = float("inf")
    window_best_epoch = -1

    print(f"\nStarting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # GLOBAL BEST
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            status = "(Best)"
        else:
            no_improve += 1
            status = f"(No improvement for {no_improve} epochs)"

        if val_loss < window_best_val:
            window_best_val = val_loss
            window_best_epoch = epoch + 1

        if (epoch + 1) % log_every == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} {status} | "
                f"Best (last {log_every}): {window_best_val:.4f} @ epoch {window_best_epoch}"
            )

            window_best_val = float("inf")
            window_best_epoch = -1

        if no_improve >= patience:
            print(f"\nStopping early (patience={patience})")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print("Restoring best model...")

            model.load_state_dict(
                torch.load(best_model_path, map_location=device)
            )
            break

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }