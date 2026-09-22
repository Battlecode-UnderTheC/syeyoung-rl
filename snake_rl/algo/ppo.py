"""Recurrent multi-agent PPO.

Two things this gets deliberately right, because the spec is emphatic and both
are easy to get subtly wrong:

* the **actor** loss is averaged over active dragon-steps, so a team that splits
  into more dragons does not get a bigger gradient just for existing;
* the **value** loss is computed once per team-timestep, not once per dragon
  alive at that timestep.

Sequences are contiguous slices of a single dragon's life, replayed from the
hidden state the rollout recorded at the slice start. Optional burn-in steps
warm the GRU without contributing loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model.actor import Actor, ActorObs, JointAction
from ..model.critic import CriticState, TeamCritic


@dataclass
class PPOConfig:
    clip_epsilon: float = 0.2
    epochs: int = 3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip_norm: float = 0.5
    burn_in: int = 16
    normalize_advantages: bool = True
    value_clip: float | None = 0.2


@dataclass
class ActorBatch:
    """Contiguous per-dragon sequences. Time-major fields are [B, T, ...]."""
    obs: ActorObs
    hidden: Tensor            # [B, H] state at the first step of the slice
    actions: JointAction      # fields [B, T] (and [B, T, S] for symbols)
    old_log_prob: Tensor      # [B, T]
    advantages: Tensor        # [B, T] team advantage of the round each step is in
    valid: Tensor             # [B, T] bool, False for padding


@dataclass
class CriticBatch:
    """One entry per team-timestep, never per dragon."""
    state: CriticState
    team: Tensor              # [B] long, which team's value is supervised
    returns: Tensor           # [B]
    old_values: Tensor        # [B]


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(values.dtype)
    total = weight.sum()
    if total == 0:
        return values.sum() * 0.0
    return (values * weight).sum() / total


def _split_burn_in(batch: ActorBatch, burn_in: int) -> tuple[ActorBatch, int]:
    """Mark the first `burn_in` steps invalid for the loss; they still run so the
    GRU state is warm by the time the scored steps begin."""
    if burn_in <= 0:
        return batch, 0
    steps = batch.obs.grid.shape[1]
    burn_in = min(burn_in, max(steps - 1, 0))
    if burn_in == 0:
        return batch, 0
    valid = batch.valid.clone()
    valid[:, :burn_in] = False
    return ActorBatch(batch.obs, batch.hidden, batch.actions, batch.old_log_prob,
                      batch.advantages, valid), burn_in


def ppo_update(actor: Actor, critic: TeamCritic, optimizer: torch.optim.Optimizer,
               actor_batch: ActorBatch, critic_batch: CriticBatch,
               config: PPOConfig) -> dict[str, float]:
    """One PPO update. Returns metrics for logging."""
    metrics: dict[str, float] = {}
    batch, _ = _split_burn_in(actor_batch, config.burn_in)

    advantages = batch.advantages
    if config.normalize_advantages:
        weight = batch.valid.to(advantages.dtype)
        count = weight.sum().clamp(min=1.0)
        mean = (advantages * weight).sum() / count
        var = (((advantages - mean) ** 2) * weight).sum() / count
        advantages = (advantages - mean) / (var.sqrt() + 1e-8)

    for epoch in range(config.epochs):
        log_prob, entropy, _ = actor.evaluate(batch.obs, batch.hidden, batch.actions)
        ratio = torch.exp(log_prob - batch.old_log_prob)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon,
                              1.0 + config.clip_epsilon) * advantages
        # averaged over active dragon-steps, so splitting does not scale the gradient
        actor_loss = -_masked_mean(torch.min(unclipped, clipped), batch.valid)
        entropy_bonus = _masked_mean(entropy, batch.valid)

        # one value per team-timestep
        values = critic(critic_batch.state)
        values = values.gather(-1, critic_batch.team.unsqueeze(-1)).squeeze(-1)
        if config.value_clip is None:
            value_loss = F.mse_loss(values, critic_batch.returns)
        else:
            clipped_value = critic_batch.old_values + torch.clamp(
                values - critic_batch.old_values, -config.value_clip, config.value_clip)
            value_loss = torch.max((values - critic_batch.returns) ** 2,
                                   (clipped_value - critic_batch.returns) ** 2).mean()

        loss = (actor_loss + config.value_coef * value_loss
                - config.entropy_coef * entropy_bonus)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            config.grad_clip_norm)
        optimizer.step()

        with torch.no_grad():
            approx_kl = _masked_mean((ratio - 1.0) - torch.log(ratio), batch.valid)
            clip_fraction = _masked_mean(
                ((ratio - 1.0).abs() > config.clip_epsilon).float(), batch.valid)
        metrics = {
            "loss/total": loss.detach().item(),
            "loss/actor": actor_loss.detach().item(),
            "loss/value": value_loss.detach().item(),
            "loss/entropy": entropy_bonus.detach().item(),
            "ppo/approx_kl": approx_kl.item(),
            "ppo/clip_fraction": clip_fraction.item(),
            "ppo/grad_norm": float(grad_norm),
            "ppo/epochs_done": epoch + 1,
        }
    return metrics
