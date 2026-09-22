"""Tests for the credit assignment rules the spec is strict about."""
from __future__ import annotations

import numpy as np
import pytest

from snake_rl.algo import team_gae


def test_terminal_reward_propagates_back_through_the_whole_match():
    steps = 20
    rewards = np.zeros(steps, dtype=np.float32)
    rewards[-1] = 1.0
    values = np.zeros(steps, dtype=np.float32)
    adv, ret = team_gae(rewards, values, gamma=0.999, lam=0.95, terminal=True)
    assert adv[-1] == pytest.approx(1.0)
    assert np.all(adv > 0), "every earlier round shares the credit for the win"
    assert np.all(np.diff(adv) > 0), "later rounds are credited more"
    assert np.allclose(ret, adv)


def test_a_dragon_dying_does_not_reset_the_trace():
    """There is no per-dragon done flag anywhere in this computation: the only
    thing that ends a trace is the match ending."""
    rewards = np.zeros(10, dtype=np.float32)
    rewards[-1] = -1.0
    values = np.linspace(0.5, -0.5, 10).astype(np.float32)
    adv, _ = team_gae(rewards, values, 0.99, 0.95, terminal=True)
    assert np.all(np.isfinite(adv))
    assert adv[0] != 0.0, "an early action still feels a loss that came later"


def test_truncation_bootstraps_where_termination_does_not():
    rewards = np.zeros(5, dtype=np.float32)
    values = np.zeros(5, dtype=np.float32)
    ended, _ = team_gae(rewards, values, 0.99, 0.95, terminal=True)
    cut, _ = team_gae(rewards, values, 0.99, 0.95, terminal=False, bootstrap_value=1.0)
    assert np.allclose(ended, 0.0)
    assert cut[-1] == pytest.approx(0.99), "the cut-off future has to come from the critic"


def test_gae_matches_the_explicit_discounted_sum():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=12).astype(np.float32)
    values = rng.normal(size=12).astype(np.float32)
    gamma, lam = 0.97, 0.9
    adv, _ = team_gae(rewards, values, gamma, lam, terminal=True)

    expected = np.zeros(12, dtype=np.float64)
    for t in range(12):
        total = 0.0
        for k in range(t, 12):
            nxt = values[k + 1] if k + 1 < 12 else 0.0
            delta = rewards[k] + gamma * nxt - values[k]
            total += (gamma * lam) ** (k - t) * delta
        expected[t] = total
    assert np.allclose(adv, expected, atol=1e-5)


# -- PPO ------------------------------------------------------------------

def _tiny_batches(device="cpu", sequences=4, steps=8, actions=12):
    import torch
    from snake_rl.algo.ppo import ActorBatch, CriticBatch
    from snake_rl.model.actor import ActorObs, JointAction
    from snake_rl.model.critic import CriticState
    g = torch.Generator().manual_seed(0)
    obs = ActorObs(
        grid=torch.rand(sequences, steps, 21, 7, 7, generator=g),
        scalars=torch.rand(sequences, steps, 12, generator=g),
        messages=torch.randint(0, 16, (sequences, steps, 4, 4), generator=g),
        message_mask=torch.rand(sequences, steps, 4, generator=g) > 0.5,
        action_mask=torch.ones(sequences, steps, actions, dtype=torch.bool),
    )
    actor_batch = ActorBatch(
        obs=obs, hidden=torch.zeros(sequences, 256),
        actions=JointAction(
            torch.randint(0, actions, (sequences, steps), generator=g),
            torch.rand(sequences, steps, generator=g) > 0.5,
            torch.randint(0, 16, (sequences, steps, 4), generator=g)),
        old_log_prob=torch.full((sequences, steps), -2.5),
        advantages=torch.randn(sequences, steps, generator=g),
        valid=torch.ones(sequences, steps, dtype=torch.bool),
    )
    critic_batch = CriticBatch(
        state=CriticState(torch.rand(16, 9, 13, 13, generator=g),
                          torch.rand(16, 8, 4, generator=g),
                          torch.randint(0, 2, (16, 8), generator=g),
                          torch.ones(16, 8, dtype=torch.bool)),
        team=torch.randint(0, 2, (16,), generator=g),
        returns=torch.randn(16, generator=g),
        old_values=torch.zeros(16),
    )
    return actor_batch, critic_batch


def test_ppo_update_is_finite_and_actually_moves_the_policy():
    import torch
    from snake_rl.algo.ppo import PPOConfig, ppo_update
    from snake_rl.model.actor import Actor
    from snake_rl.model.critic import TeamCritic
    torch.manual_seed(0)
    actor, critic = Actor(12), TeamCritic(9, 4, team_channel_pairs=((4, 6),))
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), 1e-3)
    before = [p.detach().clone() for p in actor.parameters()]

    actor_batch, critic_batch = _tiny_batches()
    metrics = ppo_update(actor, critic, optimizer, actor_batch, critic_batch,
                         PPOConfig(epochs=2, burn_in=2))

    assert all(np.isfinite(v) for v in metrics.values()), metrics
    moved = [not torch.equal(a, b) for a, b in zip(before, actor.parameters())]
    assert any(moved), "the update did not change the policy"
    assert metrics["ppo/approx_kl"] > 0, "the second epoch must see a changed policy"


def test_burn_in_steps_carry_no_loss():
    import torch
    from snake_rl.algo.ppo import _split_burn_in
    actor_batch, _ = _tiny_batches(steps=8)
    trimmed, burn = _split_burn_in(actor_batch, 3)
    assert burn == 3
    assert not trimmed.valid[:, :3].any() and trimmed.valid[:, 3:].all()
    assert actor_batch.valid.all(), "the original batch must not be mutated"
