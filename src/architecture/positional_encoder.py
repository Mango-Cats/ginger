import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds positional information to embeddings using sine and cosine waves.

    This allows the Transformer to know WHERE each token is in the sequence,
    since Transformers process all tokens in parallel (unlike RNNs).
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout_rate: float = 0.1,
        max_sequence_length: int = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Create the positional encoding matrix
        pe: torch.Tensor = torch.zeros(max_sequence_length, embedding_dim)
        position: torch.Tensor = torch.arange(
            0, max_sequence_length, dtype=torch.float
        ).unsqueeze(1)

        div_term: torch.Tensor = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to embeddings.
        """
        seq_len = x.size(1)

        pe_sliced = self.pe[:, :seq_len, :].to(x.device)

        x = x + pe_sliced
        return self.dropout(x)
