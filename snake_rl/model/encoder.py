"""Observation encoders for the actor: grid, scalars and delivered messages.

Shapes follow the spec: the conv stack ends at 256, the scalar MLP at 64 and the
message encoder at 64, and the three are fused back down to 256 before the GRU.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpatialEncoder(nn.Module):
    """Conv(C,32) -> Conv(32,64,s2) -> Conv(64,64,s2) -> Linear(.,256), SiLU throughout."""

    def __init__(self, in_channels: int, grid: int = 7, out_dim: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
        )
        with torch.no_grad():
            flat = self.body(torch.zeros(1, in_channels, grid, grid)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, out_dim), nn.SiLU())
        self.out_dim = out_dim

    def forward(self, grid: Tensor) -> Tensor:
        return self.head(self.body(grid))


class ScalarEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(),
                                 nn.Linear(out_dim, out_dim), nn.SiLU())
        self.out_dim = out_dim

    def forward(self, scalars: Tensor) -> Tensor:
        return self.net(scalars)


def bytes_to_symbols(messages: Tensor, vocab: int) -> Tensor:
    """Message bytes -> symbol ids in [0, vocab).

    The sonar payload is one uint32, carried as four bytes. With a vocabulary of
    16 only the low nibble of each byte is used, which is what the byte heads
    emit; a vocabulary of 256 uses the whole byte.
    """
    symbols = messages.long()
    if vocab < 256:
        symbols = symbols % vocab
    return symbols


class MessageEncoder(nn.Module):
    """Embeds delivered messages and pools them, with an explicit empty embedding.

    Messages arrive as a padded `[B, K, S]` block of symbols plus a `[B, K]`
    mask. Pooling is a masked mean, so the order the engine delivered them in
    does not matter; a dragon that heard nothing gets a learned vector of its
    own rather than a zero that looks like a real message.
    """

    def __init__(self, vocab: int = 16, n_symbols: int = 4, embed_dim: int = 16,
                 out_dim: int = 64):
        super().__init__()
        self.vocab = vocab
        self.n_symbols = n_symbols
        self.symbol = nn.Embedding(vocab, embed_dim)
        self.slot = nn.Parameter(torch.zeros(n_symbols, embed_dim))
        self.proj = nn.Sequential(nn.Linear(n_symbols * embed_dim, out_dim), nn.SiLU())
        self.empty = nn.Parameter(torch.zeros(out_dim))
        self.out_dim = out_dim
        nn.init.normal_(self.slot, std=0.02)
        nn.init.normal_(self.empty, std=0.02)

    def forward(self, symbols: Tensor, mask: Tensor) -> Tensor:
        # symbols [B, K, S] long, mask [B, K] bool
        embedded = self.symbol(symbols) + self.slot            # [B, K, S, E]
        per_message = self.proj(embedded.flatten(-2))          # [B, K, out]
        weight = mask.to(per_message.dtype).unsqueeze(-1)      # [B, K, 1]
        total = (per_message * weight).sum(dim=1)
        count = weight.sum(dim=1).clamp(min=1.0)
        pooled = total / count
        heard = mask.any(dim=1, keepdim=True)
        return torch.where(heard, pooled, self.empty.expand_as(pooled))
