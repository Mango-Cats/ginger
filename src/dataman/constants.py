"""
Special tokens and their indices for the G2P model.

These are defined in a separate module to avoid circular imports.
"""

# Special token strings
PAD_TOKEN: str = "<PAD>"
START_TOKEN: str = "<START>"
END_TOKEN: str = "<END>"
UNK_TOKEN: str = "<UNK>"

# Special token indices (must match vocabulary initialization)
PAD_IDX: int = 0
START_IDX: int = 1
END_IDX: int = 2
UNK_IDX: int = 3
SPECIAL_TOKENS_COUNT: int = 4
