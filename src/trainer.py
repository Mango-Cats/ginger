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
GLOBAL_CAUSAL_MASK: torch.Tensor = create_causal_mask(MAX_SEQ_LEN, device='cpu')

def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, # type: ignore
    criterion: nn.Module, 
    optimizer: Optimizer, 
    scheduler: LRScheduler, 
    device: DeviceType, 
    scaler: GradScaler
) -> float:
    model.train()
    total_loss: float = 0.0
    
    global_mask: torch.Tensor = GLOBAL_CAUSAL_MASK.to(device).bool()
    
    src: torch.Tensor
    tgt: torch.Tensor
    for _, (src, tgt) in enumerate(dataloader): # type: ignore
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        
        tgt_input: torch.Tensor = tgt[:-1, :]
        tgt_expected: torch.Tensor = tgt[1:, :]
        
        src_padding_mask: torch.Tensor = (src == PAD_IDX).transpose(0, 1).bool()
        tgt_padding_mask: torch.Tensor = (tgt_input == PAD_IDX).transpose(0, 1).bool()
        
        seq_len: int = tgt_input.size(0)
        tgt_mask: torch.Tensor = global_mask[:seq_len, :seq_len]
        
        with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
            output: torch.Tensor = model(
                src, 
                tgt_input, 
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )
            
            loss: torch.Tensor = criterion(
                output.view(-1, output.size(-1)),
                tgt_expected.view(-1)
            )
        
        optimizer.zero_grad(set_to_none=True)
        
        scaler.scale(loss).backward() #type: ignore
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader) #type: ignore


@torch.no_grad()
def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, # type: ignore
    criterion: nn.Module, 
    device: DeviceType
) -> float:
    model.eval()
    total_loss: float = 0.0
    
    global_mask: torch.Tensor = GLOBAL_CAUSAL_MASK.to(device).bool()

    src: torch.Tensor
    tgt: torch.Tensor
    for _, (src, tgt) in enumerate(dataloader): #type: ignore
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_input: torch.Tensor = tgt[:-1, :]
        tgt_expected: torch.Tensor = tgt[1:, :]

        seq_len: int = tgt_input.size(0)
        tgt_mask: torch.Tensor = global_mask[:seq_len, :seq_len]
        
        with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
            output: torch.Tensor = model(src, tgt_input, tgt_mask=tgt_mask)
            loss: torch.Tensor = criterion(output.view(-1, output.size(-1)), tgt_expected.view(-1))
            
        total_loss += loss.item()

    return total_loss / len(dataloader) #type: ignore