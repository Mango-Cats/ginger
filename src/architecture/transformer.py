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
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.grapheme_embedding = nn.Embedding(
            num_embeddings=num_graphemes,
            embedding_dim=embedding_dim,
        )
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=num_phonemes,
            embedding_dim=embedding_dim,
        )

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
            batch_first=False,
        )

        self.output_projection = nn.Linear(embedding_dim, num_phonemes)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the G2P Transformer.
        """

        grapheme_embeddings: torch.Tensor = self.grapheme_embedding(src)

        grapheme_embeddings = grapheme_embeddings * math.sqrt(self.embedding_dim)

        encoded_src: torch.Tensor = self.positional_encoder(grapheme_embeddings)

        phoneme_embeddings: torch.Tensor = self.phoneme_embedding(tgt)
        phoneme_embeddings = phoneme_embeddings * math.sqrt(self.embedding_dim)
        encoded_tgt: torch.Tensor = self.positional_encoder(phoneme_embeddings)

        decoder_output: torch.Tensor = self.transformer(
            src=encoded_src,
            tgt=encoded_tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        output_logits: torch.Tensor = self.output_projection(decoder_output)

        return output_logits


def create_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal mask for the decoder (prevents attending to future tokens).

    This ensures the model can only look at previous tokens when generating
    the current token during training.
    """
    mask: torch.Tensor = torch.triu(
        torch.ones(length, length, device=device) * float("-inf"),
        diagonal=1,
    )
    return mask
