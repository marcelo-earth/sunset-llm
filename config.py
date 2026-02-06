from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    vocab_size: int = 5000
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    context_len: int = 256
    dropout: float = 0.1
    bias: bool = False
