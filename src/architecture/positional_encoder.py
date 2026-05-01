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

        # position = [0, 1, 2, 3, ..., max_sequence_length-1]
        position: torch.Tensor = torch.arange(
            0, max_sequence_length, dtype=torch.float
        ).unsqueeze(1)

        # Compute the scaling factors for different dimensions
        # Lower dimensions get lower frequencies (vary slowly)
        # Higher dimensions get higher frequencies (vary quickly)
        div_term: torch.Tensor = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape from (max_len, d_model) to (max_len, 1, d_model)
        # This is needed because batches are: (seq_len, batch_size, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer (saved with model, not trained)
        self.register_buffer("pe", pe)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to embeddings.
        """
        seq_len: int = embeddings.size(0)

        # Add positional encoding to embeddings
        embeddings = embeddings + self.pe[:seq_len, :]

        # Apply dropout for regularization
        return self.dropout(embeddings)
