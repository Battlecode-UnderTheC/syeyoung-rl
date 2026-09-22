"""Training configuration. Phase 2 of the curriculum by default: 1v1, no
splitting, no communication, terminal reward only."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    # run
    run_name: str = "phase2-1v1"
    wandb_project: str = "snake-marl"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    out_dir: str = "runs"
    seed: int = 42
    device: str = "cuda:0"
    iterations: int = 2000
    save_every: int = 25
    resume: str = ""

    # rollout
    workers: int = 24
    matches_per_worker: int = 1
    # wasmtime pins one host-callback closure per match in a process-global
    # slab, so a worker's memory grows ~1 MB per match and no GC can reclaim
    # it. Recycling the process is the only reliable bound.
    worker_recycle_matches: int = 300
    result_timeout_s: float = 900.0
    board: int = 15
    board_choices: tuple[int, ...] = (15, 19, 23, 27)
    kelp_density: float = 0.04
    kelp_range: tuple[float, float] = (0.0, 0.08)
    portal_pairs: int = 0
    dragons_per_team: int = 1
    # Sparse and varied. Food on every tile removes any reason to hunt for it
    # and makes growth fast enough to guarantee self-trapping.
    pearl_gap: tuple[int, int] = (6, 14)
    pearl_gap_choices: tuple[tuple[int, int], ...] = (
        (20, 60), (40, 120), (60, 200), (15, 40))
    scripted_opponent_probability: float = 0.25

    # action space / model
    max_sprint: int = 3
    split_sizes: tuple[int, ...] = ()
    enable_sonar: bool = False
    hidden: int = 256

    # ppo
    gamma: float = 0.997
    gae_lambda: float = 0.95
    learning_rate: float = 3.0e-4
    epochs: int = 3
    sequence_length: int = 64
    burn_in: int = 16
    sequences_per_minibatch: int = 64
    critic_samples_per_minibatch: int = 1024
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip_norm: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)
