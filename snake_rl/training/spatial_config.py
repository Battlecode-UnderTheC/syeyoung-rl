"""Configuration for dragons-spatial-v1 training (spec section 9, 11).

The values are the spec's initial configuration: tunable, not claimed optimal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..algo.ppo import SpatialPPOConfig
from ..model.actor import SpatialActorConfig


@dataclass
class SpatialTrainConfig:
    schema: str = "dragons-spatial-v1"
    run_name: str = "spatial-v1"
    out_dir: str = "runs"
    seed: int = 42
    device: str = "cpu"
    iterations: int = 1000
    save_every: int = 10
    resume: str = ""
    wandb_project: str = "snake-marl"
    wandb_mode: str = "disabled"

    actor: SpatialActorConfig = field(default_factory=SpatialActorConfig)
    ppo: SpatialPPOConfig = field(default_factory=SpatialPPOConfig)
    learning_rate: float = 3.0e-4

    # rollout: complete matches with the behaviour policy frozen until the update
    arenas: int = 32
    complete_matches: bool = True
    snapshot_every: int = 16
    max_rounds: int = 500

    # environment curriculum (11): vary everything the rules allow
    board_choices: tuple[int, ...] = (11, 13, 15, 19, 23)
    kelp_range: tuple[float, float] = (0.0, 0.08)
    portal_pairs_choices: tuple[int, ...] = (0, 0, 1, 2)
    length_choices: tuple[int, ...] = (3, 4, 6, 8)
    dragons_per_team_choices: tuple[int, ...] = (1, 1, 2, 3)
    pearl_gap_choices: tuple[tuple[int, int], ...] = ((4, 12), (8, 20), (15, 40), (20, 60))

    # historical-opponent league (11)
    recent_fraction: float = 0.5        # of non-self-play matches: recent snapshots
    self_play_fraction: float = 0.5     # matches where both teams are the learner
    league_floor: float = 0.1           # minimum share for the oldest opponents
    snapshot_interval: int = 5          # add a frozen copy every N iterations
    max_league: int = 32
    scripted_eval_matches: int = 4      # baseline evaluation only, never a teacher

    def as_dict(self) -> dict:
        return asdict(self)
