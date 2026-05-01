"""
Grapheme-to-Phoneme Transformer

A PyTorch implementation of a seq2seq Transformer for converting written text
(graphemes) to phonetic pronunciation (phonemes).

Example:
    "tough" → "tuff"
    "read" → "red" (present tense) or "red" (past tense)
"""

import math
from .positional_encoder import PositionalEncoding
import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Grapheme-to-Phoneme Transformer using encoder-decoder architecture.

    The encoder reads the input text (graphemes/letters) and creates a
    representation. The decoder then generates the phonetic output (phonemes)
    one token at a time, referencing the encoder's representation.
    """

    def __init__(
        self,
        num_graphemes: int,
        num_phonemes: int,
        embedding_dim: int = 128,
        num_attention_heads: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        feedforward_dim: int = 512,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.grapheme_embedding = nn.Embedding(num_graphemes, embedding_dim)
        self.phoneme_embedding = nn.Embedding(num_phonemes, embedding_dim)

        self.positional_encoder = PositionalEncoding(
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.output_projection = nn.Linear(embedding_dim, num_phonemes)

    def forward(
        self, src, tgt, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None
    ):
        src = self.grapheme_embedding(src) * math.sqrt(self.embedding_dim)
        src = self.positional_encoder(src)

        tgt = self.phoneme_embedding(tgt) * math.sqrt(self.embedding_dim)
        tgt = self.positional_encoder(tgt)

        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        return self.output_projection(out)


def create_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal mask for the decoder (prevents attending to future tokens).

    This ensures the model can only look at previous tokens when generating
    the current token during training.
    """
    return torch.triu(
        torch.full((length, length), float("-inf"), device=device),
        diagonal=1,
    )
