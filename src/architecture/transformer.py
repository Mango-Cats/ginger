"""
Grapheme-to-Phoneme Transformer

A PyTorch implementation of a seq2seq Transformer for converting written text
(graphemes) to phonetic pronunciation (phonemes).

Example:
    "tough" → "tuff"
    "read" → "red" (present tense) or "red" (past tense)
"""

import math
from positional_encoder import PositionalEncoding
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

        # ============ COMPONENT 1: EMBEDDINGS ============
        # Convert token IDs to dense vectors
        # grapheme_emb(5) → [d1, d2, ..., d128] (learned during training)
        self.grapheme_embedding = nn.Embedding(
            num_embeddings=num_graphemes,
            embedding_dim=embedding_dim,
        )
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=num_phonemes,
            embedding_dim=embedding_dim,
        )

        # ============ COMPONENT 2: POSITIONAL ENCODING ============
        # Tells the model WHERE each token is in the sequence
        self.positional_encoder = PositionalEncoding(
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )

        # ============ COMPONENT 3: THE TRANSFORMER ============
        # Contains both the encoder and decoder
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout_rate,
            batch_first=False,  
        )

        # ============ COMPONENT 4: OUTPUT PROJECTION ============
        # Project from embedding space to phoneme probabilities
        # Takes (seq_len, batch, 128) → (seq_len, batch, num_phonemes)
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

        # Step 1: Embed graphemes and add positional encoding
        # ────────────────────────────────────────────────
        grapheme_embeddings: torch.Tensor = self.grapheme_embedding(src)

        # Scale embeddings by sqrt(d_model) before adding positional encodings
        # This is a training stability trick from the original paper
        grapheme_embeddings = grapheme_embeddings * math.sqrt(self.embedding_dim)

        # Add positional information
        encoded_src: torch.Tensor = self.positional_encoder(grapheme_embeddings)

        # Step 2: Embed phonemes and add positional encoding
        # ──────────────────────────────────────────────────
        phoneme_embeddings: torch.Tensor = self.phoneme_embedding(tgt)
        phoneme_embeddings = phoneme_embeddings * math.sqrt(self.embedding_dim)
        encoded_tgt: torch.Tensor = self.positional_encoder(phoneme_embeddings)

        # Step 3: Pass through Transformer
        # ────────────────────────────────
        # ENCODER: Reads source (graphemes), creates contextualized representation
        # DECODER: Reads target (phonemes) and encoder output, generates predictions
        decoder_output: torch.Tensor = self.transformer(
            src=encoded_src,
            tgt=encoded_tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        # Step 4: Project to phoneme logits
        # ─────────────────────────────────
        # Convert (tgt_len, batch, 128) → (tgt_len, batch, num_phonemes)
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


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("G2P Transformer Example")
    print("=" * 60)

    # Hyperparameters
    num_graphemes: int = 30  # Number of unique input characters
    num_phonemes: int = 50  # Number of unique output phonemes
    batch_size: int = 4
    src_length: int = 10  # Example: "elephant" = 8 chars
    tgt_length: int = 12  # Example: phonemes might be longer

    # Create model
    model = Transformer(
        num_graphemes=num_graphemes,
        num_phonemes=num_phonemes,
        embedding_dim=128,
        num_attention_heads=4,
        num_encoder_layers=2,  # Using 2 instead of 4 for faster demo
        num_decoder_layers=2,
        feedforward_dim=512,
        dropout_rate=0.1,
    )

    # Create dummy data
    src: torch.Tensor = torch.randint(0, num_graphemes, (src_length, batch_size))
    tgt: torch.Tensor = torch.randint(0, num_phonemes, (tgt_length, batch_size))

    # Create causal mask for decoder
    tgt_mask: torch.Tensor = create_causal_mask(tgt_length, device=src.device)

    # Forward pass
    output: torch.Tensor = model(src, tgt, tgt_mask=tgt_mask)

    print(f"\nInput shape (src):        {tuple(src.shape)}")
    print(f"Target shape (tgt):      {tuple(tgt.shape)}")
    print(f"Output shape:            {tuple(output.shape)}")
    print(f"\nExpected output shape:   ({tgt_length}, {batch_size}, {num_phonemes})")
    print(f"Match: {output.shape == (tgt_length, batch_size, num_phonemes)}")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
