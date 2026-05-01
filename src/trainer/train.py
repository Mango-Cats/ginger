import torch
from src.architecture.transformer import create_causal_mask
from src.dataman.constants import PAD_IDX

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:-1, :]
        tgt_expected = tgt[1:, :]
        
        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt_input == PAD_IDX).transpose(0, 1)
        
        tgt_mask = create_causal_mask(tgt_input.size(0), device)
        
        output = model(
            src, 
            tgt_input, 
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask
        )
        
        loss = criterion(
            output.view(-1, output.size(-1)),
            tgt_expected.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)