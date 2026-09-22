"""The team-event critic (spec section 7) -- training only, never an actor input.

It consumes the exact native state at an event boundary -- the instant before
a living dragon acts -- plus the round/turn phase and detached summaries of
the acting dragons' private state. The actor has no argument it could arrive
through.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

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
