import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class Embeddings(nn.Module):
    """Token and positional embeddings."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(pos)
        return self.dropout(tok_emb + pos_emb)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model

        # Q, K, V projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_len, config.context_len))
            .view(1, 1, config.context_len, config.context_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        # Apply causal mask
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = 4 * config.d_model
        self.fc1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.fc2 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block: attention + ffn with residual connections."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT language model for generating Spanish poems."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(config)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.embeddings.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass.

        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T) for computing loss

        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        x = self.embeddings(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            idx: Starting tokens (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)
            eos_token_id: Stop generation at this token

        Returns:
            Generated token sequence (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx[:, -self.config.context_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return idx
