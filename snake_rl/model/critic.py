"""The centralised team critic -- training only, never a policy input.

It reads the whole board and every dragon on it, which no actor may see, and
returns one value per team per timestep. Keep it in the learner: nothing here
may be serialised into an observation.

The dragon set is pooled permutation-invariantly, so the order dragons happen
to be listed in cannot change the value, and the number of them can vary as
they split and die.

Both team values come from the *same* weights evaluated from each team's point
of view: the team-specific map channels are swapped and the own/enemy flags
flipped. That makes `V(s)[A] == V(swap(s))[B]` true by construction, which is
the symmetry the game actually has.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn


@dataclass
class CriticState:
    """Full state, as the learner reconstructs it. Not an actor input.

    `map_grid` is the whole board, `dragons` are per-dragon features padded to a
    fixed N with `dragon_mask`, and `dragon_team` is 0 or 1 per dragon.
    """
    map_grid: Tensor          # [B, C, H, W]
    dragons: Tensor           # [B, N, F]
    dragon_team: Tensor       # [B, N] long, 0 or 1
    dragon_mask: Tensor       # [B, N] bool

    def swap_teams(self, channel_pairs: tuple[tuple[int, int], ...]) -> "CriticState":
        grid = self.map_grid.clone()
        for a, b in channel_pairs:
            grid[:, [a, b]] = grid[:, [b, a]]
        return replace(self, map_grid=grid, dragon_team=1 - self.dragon_team)


class TeamCritic(nn.Module):
    def __init__(self, map_channels: int, dragon_features: int,
                 team_channel_pairs: tuple[tuple[int, int], ...] = (),
                 hidden: int = 256, pooled_grid: int = 4, zero_sum: bool = False):
        super().__init__()
        self.team_channel_pairs = tuple(team_channel_pairs)
        self.zero_sum = zero_sum

        self.map_body = nn.Sequential(
            nn.Conv2d(map_channels, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(pooled_grid),        # boards vary from 7x7 to 64x64
            nn.Flatten(),
        )
        self.map_head = nn.Sequential(
            nn.Linear(64 * pooled_grid * pooled_grid, hidden), nn.SiLU())

        # +2 for the own/enemy flag attached to each dragon
        self.dragon_encoder = nn.Sequential(
            nn.Linear(dragon_features + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.set_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())

        self.value = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.value[-1].weight, 1.0)
        nn.init.zeros_(self.value[-1].bias)

    def _pool_dragons(self, state: CriticState) -> Tensor:
        own = torch.nn.functional.one_hot(state.dragon_team.clamp(0, 1), 2)
        feats = torch.cat([state.dragons, own.to(state.dragons.dtype)], dim=-1)
        encoded = self.dragon_encoder(feats)                       # [B, N, H]

        mask = state.dragon_mask.unsqueeze(-1).to(encoded.dtype)
        total = (encoded * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        mean = total / count
        very_negative = torch.finfo(encoded.dtype).min
        peak = encoded.masked_fill(~state.dragon_mask.unsqueeze(-1), very_negative)
        peak = peak.max(dim=1).values
        peak = torch.where(state.dragon_mask.any(dim=1, keepdim=True),
                           peak, torch.zeros_like(peak))
        return self.set_head(torch.cat([mean, peak], dim=-1))

    def value_for(self, state: CriticState) -> Tensor:
        """Value for team 0 of this state, i.e. from the perspective it is in."""
        spatial = self.map_head(self.map_body(state.map_grid))
        return self.value(torch.cat([spatial, self._pool_dragons(state)], dim=-1)).squeeze(-1)

    def forward(self, state: CriticState) -> Tensor:
        """[B, 2] -- the value for team A and for team B."""
        a = self.value_for(state)
        b = self.value_for(state.swap_teams(self.team_channel_pairs))
        if self.zero_sum:
            half = (a - b) * 0.5
            return torch.stack([half, -half], dim=-1)
        return torch.stack([a, b], dim=-1)
