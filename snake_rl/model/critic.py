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


# ===========================================================================
# dragons-spatial-v1: the team-event critic (spec section 7)
# ===========================================================================
#
# Training only. It consumes the exact native state at an event boundary --
# the instant before a living dragon acts -- plus the round/turn phase and
# detached summaries of the acting dragons' private state. Nothing here is
# ever an actor input; the actor has no argument it could arrive through.

EV_C_PEARL = 0
EV_C_COUNTDOWN = 1
EV_C_NEVER = 2
EV_C_KELP = 3          # 4, N E S W
EV_C_PORTAL = 7        # 4
EV_C_A_HEAD = 11
EV_C_A_BODY = 12
EV_C_B_HEAD = 13
EV_C_B_BODY = 14
EV_C_FACING = 15       # 4
EV_C_NEXT = 19         # head of the dragon about to act
EV_N_MAP_CHANNELS = 20
EV_TEAM_PAIRS = ((EV_C_A_HEAD, EV_C_B_HEAD), (EV_C_A_BODY, EV_C_B_BODY))

# per-dragon features: sin/cos x,y (4), heading (4), length/area, log length,
# id rank, is-next-actor, coverage (seen/area), known-body fraction, |h| mean,
# |h| std, born round / 500
EV_N_DRAGON_FEATURES = 16
# per-team scalars, given for the perspective team then the other:
# max length/area, total/area, units/limit, opponent policy id / 16
EV_N_TEAM_FEATURES = 4
# phase: round/500, remaining/500, next actor is team 0
EV_N_PHASE = 3


@dataclass
class EventState:
    map_grid: Tensor          # [B, 20, H, W]
    dragons: Tensor           # [B, N, 16]
    dragon_team: Tensor       # [B, N] long
    dragon_mask: Tensor       # [B, N] bool
    team_feats: Tensor        # [B, 2, 4]  (team A, team B)
    phase: Tensor             # [B, 3]

    def swap_teams(self) -> "EventState":
        grid = self.map_grid.clone()
        for a, b in EV_TEAM_PAIRS:
            grid[:, [a, b]] = grid[:, [b, a]]
        phase = self.phase.clone()
        phase[:, 2] = 1.0 - phase[:, 2]
        return EventState(grid, self.dragons, 1 - self.dragon_team, self.dragon_mask,
                          self.team_feats.flip(1), phase)


class EventCritic(nn.Module):
    """Two team values from shared weights, one per perspective."""

    def __init__(self, hidden: int = 128, pooled_grid: int = 4, zero_sum: bool = False):
        super().__init__()
        self.zero_sum = zero_sum
        self.map_body = nn.Sequential(
            nn.Conv2d(EV_N_MAP_CHANNELS, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(pooled_grid), nn.Flatten())
        self.map_head = nn.Sequential(nn.Linear(64 * pooled_grid * pooled_grid, hidden), nn.SiLU())
        self.entity = nn.Sequential(nn.Linear(EV_N_DRAGON_FEATURES + 2, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden), nn.SiLU())
        # own team and enemy team pooled separately: mean, max, count each
        self.pool_head = nn.Sequential(nn.Linear(2 * (2 * hidden + 1), hidden), nn.SiLU())
        self.value = nn.Sequential(
            nn.Linear(hidden + hidden + 2 * EV_N_TEAM_FEATURES + EV_N_PHASE, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.value[-1].weight, 1.0)
        nn.init.zeros_(self.value[-1].bias)

    def _pool(self, enc: Tensor, mask: Tensor) -> Tensor:
        w = mask.to(enc.dtype)[..., None]
        count = w.sum(1)
        mean = (enc * w).sum(1) / count.clamp(min=1.0)
        peak = enc.masked_fill(~mask[..., None], -1e9).amax(1)
        peak = torch.where(count > 0, peak, torch.zeros_like(peak))
        return torch.cat([mean, peak, count / 64.0], -1)

    def value_for(self, s: EventState) -> Tensor:
        own = torch.nn.functional.one_hot(s.dragon_team.clamp(0, 1), 2).to(s.dragons.dtype)
        enc = self.entity(torch.cat([s.dragons, own], -1))
        mine = self._pool(enc, s.dragon_mask & (s.dragon_team == 0))
        theirs = self._pool(enc, s.dragon_mask & (s.dragon_team == 1))
        pooled = self.pool_head(torch.cat([mine, theirs], -1))
        spatial = self.map_head(self.map_body(s.map_grid))
        x = torch.cat([spatial, pooled, s.team_feats.flatten(1), s.phase], -1)
        return self.value(x).squeeze(-1)

    def forward(self, s: EventState) -> Tensor:
        a = self.value_for(s)
        b = self.value_for(s.swap_teams())
        if self.zero_sum:
            half = (a - b) * 0.5
            return torch.stack([half, -half], -1)
        return torch.stack([a, b], -1)
