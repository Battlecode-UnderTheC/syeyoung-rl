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


# ===========================================================================
# dragons-spatial-v1 encoders (spec 4.5, 5.1, 5.3, 6.5)
# ===========================================================================

from ..env.schema import (N_ENTITY_FEATURES, N_FACT_CHANNELS, N_HISTORY_FEATURES,
                          N_STATUS, HISTORY_TURNS, VISION)


class LocalEncoder(nn.Module):
    """Two 3x3 convolutions over the north-up 7x7 factual crop, 32 channels,
    keeping the [49, 32] per-cell features for the sparse writer."""

    def __init__(self, in_channels: int = N_FACT_CHANNELS, channels: int = 32,
                 out_dim: int = 128):
        super().__init__()
        # zero padding: the crop is already wrapped on the torus by construction
        # and its border is where the window ends, not where the board does
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj = nn.Sequential(nn.Linear(channels * VISION * VISION, out_dim), nn.SiLU())
        self.channels = channels
        self.out_dim = out_dim

    def forward(self, local: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (cells [B, 49, 32] in window row-major order, vector [B, 128])."""
        x = torch.nn.functional.silu(self.conv1(local))
        x = torch.nn.functional.silu(self.conv2(x))
        cells = x.flatten(2).transpose(1, 2)
        return cells, self.proj(x.flatten(1))


class StatusEncoder(nn.Module):
    def __init__(self, in_dim: int = N_STATUS, out_dim: int = 64, small: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(),
                                 nn.Linear(out_dim, out_dim), nn.SiLU())
        self.small = nn.Sequential(nn.Linear(in_dim, small), nn.SiLU())
        self.out_dim = out_dim

    def forward(self, status: Tensor) -> tuple[Tensor, Tensor]:
        return self.net(status), self.small(status)


class HistoryEncoder(nn.Module):
    """Masked temporal convolution over the last 32 completed commands."""

    def __init__(self, in_dim: int = N_HISTORY_FEATURES, turns: int = HISTORY_TURNS,
                 channels: int = 32, out_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.proj = nn.Sequential(nn.Linear(channels * turns, out_dim), nn.SiLU())
        self.empty = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.empty, std=0.02)
        self.out_dim = out_dim

    def forward(self, history: Tensor, mask: Tensor) -> Tensor:
        m = mask.to(history.dtype)[:, None, :]                  # [B, 1, T]
        x = history.transpose(1, 2) * m
        x = torch.nn.functional.silu(self.conv1(x)) * m
        x = torch.nn.functional.silu(self.conv2(x)) * m
        out = self.proj(x.flatten(1))
        any_ = mask.any(dim=1, keepdim=True)
        return torch.where(any_, out, self.empty.expand_as(out))


class EntityEncoder(nn.Module):
    """Shared MLP per visible dragon, attention-pooled to 64."""

    def __init__(self, in_dim: int = N_ENTITY_FEATURES, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, out_dim), nn.SiLU())
        self.score = nn.Linear(out_dim, 1)
        self.empty = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.empty, std=0.02)
        self.out_dim = out_dim

    def forward(self, entities: Tensor, mask: Tensor) -> Tensor:
        enc = self.net(entities)                                 # [B, E, D]
        scores = self.score(enc).squeeze(-1).masked_fill(~mask, -3.0e4)
        attn = torch.softmax(scores, dim=-1) * mask.to(enc.dtype)
        pooled = (attn[..., None] * enc).sum(dim=1)
        any_ = mask.any(dim=1, keepdim=True)
        return torch.where(any_, pooled, self.empty.expand_as(pooled))


class InboxEncoder(nn.Module):
    """Every received uint32 as four raw bytes with byte-position embeddings,
    an MLP per word, streaming mean and max pooling plus count features."""

    def __init__(self, embed: int = 8, hidden: int = 32, out_dim: int = 32):
        super().__init__()
        self.byte = nn.Embedding(256, embed)
        self.position = nn.Parameter(torch.zeros(4, embed))
        nn.init.normal_(self.position, std=0.02)
        self.word = nn.Sequential(nn.Linear(4 * embed, hidden), nn.SiLU())
        self.proj = nn.Sequential(nn.Linear(2 * hidden + 3, out_dim), nn.SiLU())
        self.out_dim = out_dim

    def forward(self, words: Tensor, mask: Tensor) -> Tensor:
        """words [B, M, 4] long in 0..255, mask [B, M]. M may be zero."""
        B, M = mask.shape
        if M == 0:
            words = words.new_zeros(B, 1, 4)
            mask = mask.new_zeros(B, 1)
        enc = self.word((self.byte(words) + self.position).flatten(2))   # [B, M, H]
        w = mask.to(enc.dtype)[..., None]
        count = w.sum(dim=1)
        mean = (enc * w).sum(dim=1) / count.clamp(min=1.0)
        peak = enc.masked_fill(~mask[..., None], -3.0e4).amax(dim=1)
        peak = torch.where(count > 0, peak, torch.zeros_like(peak))
        extra = torch.cat([count / 8.0, torch.log1p(count), (count == 0).to(enc.dtype)], dim=-1)
        return self.proj(torch.cat([mean, peak, extra], dim=-1))
