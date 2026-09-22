"""Model tests, following the acceptance list in the spec.

The themes: the actor is strictly decentralised, the conditional heads are
consistent between acting and evaluating, masked actions are unreachable, and
the critic is permutation invariant over a variable dragon set.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from snake_rl.env import ActionSpace, MatchRunner, mapgen
from snake_rl.model import Actor, ActorObs, CriticState, TeamCritic, TorchPolicy
from snake_rl.model.actor import NEG_INF


def make_obs(batch=6, actions=47, channels=None, scalars=None, slots=8, symbols=4,
             seed=0, all_legal=True):
    from snake_rl.env.observation import N_CHANNELS, N_SCALARS
    channels = N_CHANNELS if channels is None else channels
    scalars = N_SCALARS if scalars is None else scalars
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(batch, actions, dtype=torch.bool)
    if not all_legal:
        mask = torch.rand(batch, actions, generator=g) > 0.5
        mask[:, 0] = True
    return ActorObs(
        grid=torch.randn(batch, channels, 7, 7, generator=g),
        scalars=torch.randn(batch, scalars, generator=g),
        messages=torch.randint(0, 16, (batch, slots, symbols), generator=g),
        message_mask=torch.rand(batch, slots, generator=g) > 0.5,
        action_mask=mask,
    )


@pytest.fixture(scope="module")
def actor() -> Actor:
    torch.manual_seed(0)
    return Actor(47)


# -- decentralisation ------------------------------------------------------

def test_a_dragon_is_unaffected_by_the_other_dragons_in_the_batch(actor):
    """Shared weights are not shared state: permuting the batch permutes the
    outputs and nothing else."""
    obs = make_obs(seed=1)
    hidden = torch.randn(6, actor.hidden)
    heads, new_hidden = actor.step(obs, hidden)

    order = torch.tensor([3, 1, 5, 0, 4, 2])
    shuffled = ActorObs(obs.grid[order], obs.scalars[order], obs.messages[order],
                        obs.message_mask[order], obs.action_mask[order])
    heads2, new_hidden2 = actor.step(shuffled, hidden[order])

    assert torch.allclose(heads2.action_logits, heads.action_logits[order], atol=1e-6)
    assert torch.allclose(new_hidden2, new_hidden[order], atol=1e-6)


def test_the_actor_takes_no_global_state(actor):
    """There is deliberately no argument a full state could arrive through."""
    import inspect
    fields = set(ActorObs.__dataclass_fields__)
    assert fields == {"grid", "scalars", "messages", "message_mask", "action_mask"}
    params = set(inspect.signature(actor.step).parameters)
    assert params == {"obs", "hidden"}


def test_a_fresh_dragon_starts_from_zero_memory(actor):
    assert torch.count_nonzero(actor.initial_state(4)) == 0


# -- masking ---------------------------------------------------------------

def test_masked_actions_get_no_probability(actor):
    obs = make_obs(seed=2, all_legal=False)
    heads, _ = actor.step(obs, actor.initial_state(6))
    probs = heads.action_dist.probs
    assert torch.all(probs[~obs.action_mask] < 1e-12)
    assert torch.allclose(probs.sum(-1), torch.ones(6), atol=1e-5)

    for _ in range(20):
        sampled = heads.sample().action
        assert obs.action_mask[torch.arange(6), sampled].all()


def test_a_boxed_in_dragon_still_produces_a_finite_distribution(actor):
    """Nothing legal is a real state in this game; it must not produce NaNs."""
    obs = make_obs(batch=2, seed=3)
    obs.action_mask[:] = False
    heads, _ = actor.step(obs, actor.initial_state(2))
    assert torch.isfinite(heads.action_logits).all()
    assert torch.isfinite(heads.entropy()).all()
    assert torch.isfinite(heads.log_prob(heads.sample())).all()


# -- conditional heads -----------------------------------------------------

def test_the_payload_contributes_nothing_when_transmit_is_off(actor):
    obs = make_obs(batch=4, seed=4)
    heads, _ = actor.step(obs, actor.initial_state(4))
    joint = heads.sample()

    joint.transmit[:] = False
    quiet = heads.log_prob(joint)
    changed = type(joint)(joint.action, joint.transmit,
                          (joint.symbols + 1) % actor.symbol_vocab)
    assert torch.allclose(quiet, heads.log_prob(changed), atol=1e-6), \
        "a silent dragon's payload must not enter the joint log-prob"

    joint.transmit[:] = True
    assert not torch.allclose(heads.log_prob(joint), heads.log_prob(changed))


def test_the_fast_sampling_path_scores_exactly_like_the_distributions(actor):
    """The rollout stores these log-probs and PPO recomputes them with the
    distribution path, so the two must agree to the bit that matters."""
    obs = make_obs(batch=64, seed=21, all_legal=False)
    heads, _ = actor.step(obs, torch.randn(64, actor.hidden))
    torch.manual_seed(0)
    joint, fast = heads.sample_with_log_prob()
    assert torch.allclose(fast, heads.log_prob(joint), atol=1e-6)
    assert obs.action_mask[torch.arange(64), joint.action].all()

    joint_d, fast_d = heads.sample_with_log_prob(deterministic=True)
    assert torch.equal(joint_d.action, heads.action_logits.argmax(-1))
    assert torch.allclose(fast_d, heads.log_prob(joint_d), atol=1e-6)


def test_fast_sampling_matches_the_categorical_distribution(actor):
    """Gumbel-max has to reproduce the actual probabilities, not just be legal."""
    logits = torch.tensor([[2.0, 0.0, -1.0, 1.0]])
    heads = type(actor._heads(torch.zeros(1, actor.hidden), None))(
        logits, torch.zeros(1), torch.zeros(1, 1, 4))
    torch.manual_seed(0)
    counts = torch.zeros(4)
    for _ in range(20000):
        counts[heads.sample_with_log_prob()[0].action] += 1
    empirical = counts / counts.sum()
    assert torch.allclose(empirical, torch.softmax(logits[0], -1), atol=0.02)


def test_entropy_counts_the_payload_in_proportion_to_transmitting(actor):
    obs = make_obs(batch=4, seed=5)
    heads, _ = actor.step(obs, actor.initial_state(4))
    p = torch.sigmoid(heads.transmit_logit)
    expected = (heads.action_dist.entropy() + heads.transmit_dist.entropy()
                + p * heads.symbol_dist.entropy().sum(-1))
    assert torch.allclose(heads.entropy(), expected, atol=1e-6)


def test_evaluate_reproduces_the_log_prob_the_rollout_stored(actor):
    """PPO's ratio is only meaningful if old and new use the same head structure."""
    obs = make_obs(batch=3, seed=6)
    hidden = torch.randn(3, actor.hidden)
    heads, _ = actor.step(obs, hidden)
    joint = heads.sample()
    stored = heads.log_prob(joint)

    seq = ActorObs(obs.grid.unsqueeze(1), obs.scalars.unsqueeze(1),
                   obs.messages.unsqueeze(1), obs.message_mask.unsqueeze(1),
                   obs.action_mask.unsqueeze(1))
    # the sequence path wants the actions shaped like the observations
    seq_joint = type(joint)(joint.action.unsqueeze(1), joint.transmit.unsqueeze(1),
                            joint.symbols.unsqueeze(1))
    recomputed, entropy, _ = actor.evaluate(seq, hidden, seq_joint)
    assert recomputed.shape == (3, 1)
    assert torch.allclose(recomputed.squeeze(1), stored, atol=1e-6)
    assert torch.isfinite(entropy).all()


# -- recurrence ------------------------------------------------------------

def test_stepping_and_unrolling_agree(actor):
    """The rollout path and the training path must be the same function."""
    steps = 5
    obs = [make_obs(batch=2, seed=10 + i) for i in range(steps)]
    hidden = torch.zeros(2, actor.hidden)
    stepped = []
    for o in obs:
        heads, hidden = actor.step(o, hidden)
        stepped.append(heads.action_logits)
    stepped = torch.stack(stepped, dim=1)

    seq = ActorObs(torch.stack([o.grid for o in obs], 1),
                   torch.stack([o.scalars for o in obs], 1),
                   torch.stack([o.messages for o in obs], 1),
                   torch.stack([o.message_mask for o in obs], 1),
                   torch.stack([o.action_mask for o in obs], 1))
    unrolled, final = actor.unroll(seq, torch.zeros(2, actor.hidden))
    assert torch.allclose(unrolled.action_logits, stepped, atol=1e-5)
    assert torch.allclose(final, hidden, atol=1e-5)


def test_gradients_reach_every_part_of_the_actor(actor):
    obs = make_obs(batch=4, seed=7)
    heads, _ = actor.step(obs, actor.initial_state(4))
    joint = heads.sample()
    joint.transmit[:] = True                       # exercise the payload heads too
    value, _ = actor.value(obs, actor.initial_state(4))
    loss = (-(heads.log_prob(joint).mean()) - 0.01 * heads.entropy().mean()
            + value.pow(2).mean())
    actor.zero_grad()
    loss.backward()
    missing = [n for n, p in actor.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    assert torch.isfinite(loss)


def test_saving_and_loading_reproduces_inference(tmp_path, actor):
    obs = make_obs(batch=2, seed=8)
    hidden = torch.randn(2, actor.hidden)
    before, _ = actor.step(obs, hidden)

    path = tmp_path / "actor.pt"
    torch.save(actor.state_dict(), path)
    clone = Actor(47)
    clone.load_state_dict(torch.load(path))
    clone.eval()
    after, _ = clone.step(obs, hidden)
    assert torch.allclose(before.action_logits, after.action_logits, atol=1e-6)


# -- critic ----------------------------------------------------------------

def _critic_state(batch=3, dragons=5, channels=10, size=13, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(batch, dragons, dtype=torch.bool)
    mask[:, -1] = False                             # a padded slot
    return CriticState(
        map_grid=torch.randn(batch, channels, size, size, generator=g),
        dragons=torch.randn(batch, dragons, 4, generator=g),
        dragon_team=torch.randint(0, 2, (batch, dragons), generator=g),
        dragon_mask=mask,
    )


@pytest.fixture(scope="module")
def critic() -> TeamCritic:
    torch.manual_seed(0)
    return TeamCritic(10, 4, team_channel_pairs=((2, 3), (4, 5)))


def test_critic_is_permutation_invariant_over_dragons(critic):
    state = _critic_state(seed=1)
    base = critic(state)
    order = torch.tensor([2, 0, 1, 4, 3])
    shuffled = CriticState(state.map_grid, state.dragons[:, order],
                           state.dragon_team[:, order], state.dragon_mask[:, order])
    assert torch.allclose(base, critic(shuffled), atol=1e-5)


def test_critic_ignores_padded_dragons(critic):
    state = _critic_state(seed=2)
    noisy = CriticState(state.map_grid, state.dragons.clone(),
                        state.dragon_team.clone(), state.dragon_mask)
    noisy.dragons[:, -1] = 99.0                     # the masked-out slot
    noisy.dragon_team[:, -1] = 1 - noisy.dragon_team[:, -1]
    assert torch.allclose(critic(state), critic(noisy), atol=1e-5)


def test_critic_values_both_teams_with_the_same_weights(critic):
    state = _critic_state(seed=3)
    values = critic(state)
    swapped = critic(state.swap_teams(critic.team_channel_pairs))
    assert torch.allclose(values[:, 0], swapped[:, 1], atol=1e-5)
    assert torch.allclose(values[:, 1], swapped[:, 0], atol=1e-5)


def test_zero_sum_critic_is_antisymmetric():
    torch.manual_seed(0)
    critic = TeamCritic(10, 4, team_channel_pairs=((2, 3),), zero_sum=True)
    values = critic(_critic_state(seed=4))
    assert torch.allclose(values.sum(-1), torch.zeros(values.shape[0]), atol=1e-6)


def test_critic_handles_a_variable_number_of_dragons(critic):
    for n in (1, 3, 12):
        state = _critic_state(dragons=n, seed=5)
        assert critic(state).shape == (3, 2)


# -- end to end ------------------------------------------------------------

def test_the_actor_can_play_a_whole_match_legally(engine):
    """Strict mode raises if the actor ever picks a masked action while a legal
    one was available, so finishing the match is the assertion."""
    torch.manual_seed(0)
    space = ActionSpace()
    runner = MatchRunner(engine, space=space, strict=True)
    policy = TorchPolicy(Actor(space.n))

    trapped: set[tuple[int, int]] = set()
    inner = policy.__call__

    def watched(obs):
        if not obs.action_mask.any():
            trapped.add((obs.dragon_id, obs.round))
        return inner(obs)

    spec = mapgen.open_board(15, 15, pearl_gap=(6, 14))
    outcome = runner.run(spec.to_bytes(), watched,
                         on_spawn=policy.on_spawn, on_death=policy.on_death)

    assert outcome.turns > 0
    assert set(policy.hidden) <= set(outcome.final_lengths), \
        "a dead dragon's memory must not survive it"
    # an untrained actor splits a lot, so dragons do get boxed in; what must not
    # happen is an invalid action while something legal was on offer
    rejected = [d for d in outcome.deaths if d[2] == "A"]
    assert all((did, rnd) in trapped for did, rnd, _ in rejected), \
        f"an avoidable invalid action got through: {rejected}"


def test_sonar_can_be_switched_off_entirely():
    """Early curriculum stages train without communication; the heads must then
    contribute nothing rather than sit there as a no-op action."""
    torch.manual_seed(0)
    quiet = Actor(47, enable_sonar=False)
    obs = make_obs(batch=8, seed=30)
    heads, _ = quiet.step(obs, quiet.initial_state(8))
    joint, logp = heads.sample_with_log_prob()
    assert not joint.transmit.any()
    assert torch.allclose(logp, heads.action_dist.log_prob(joint.action), atol=1e-6)
    assert torch.allclose(heads.entropy(), heads.action_dist.entropy(), atol=1e-6)
    assert torch.allclose(logp, heads.log_prob(joint), atol=1e-6)


def test_scripted_actor_matches_the_eager_one(tmp_path):
    """The native trainer runs the scripted graph, so it must be the same
    function as the eager model the tests above pin down."""
    import torch
    from snake_rl.model import ScriptableActor, export

    torch.manual_seed(0)
    actor = Actor(47).eval()
    obs = make_obs(batch=5, seed=41, all_legal=False)
    hidden = torch.randn(5, actor.hidden)
    with torch.no_grad():
        heads, new_hidden = actor.step(obs, hidden)

    scripted = torch.jit.script(ScriptableActor(actor).eval())
    with torch.no_grad():
        logits, transmit, symbols, hidden_out, _value = scripted(
            obs.grid, obs.scalars, obs.messages, obs.message_mask,
            obs.action_mask, hidden)

    assert torch.allclose(logits, heads.action_logits, atol=1e-6)
    assert torch.allclose(transmit, heads.transmit_logit, atol=1e-6)
    assert torch.allclose(symbols, heads.symbol_logits, atol=1e-6)
    assert torch.allclose(hidden_out, new_hidden, atol=1e-6)
    with torch.no_grad():
        expected_value, _ = actor.value(obs, hidden)
    assert torch.allclose(scripted(obs.grid, obs.scalars, obs.messages,
                                   obs.message_mask, obs.action_mask, hidden)[4],
                          expected_value, atol=1e-6)

    path = export(actor, tmp_path / "actor.pt")
    reloaded = torch.jit.load(str(path))
    with torch.no_grad():
        again = reloaded(obs.grid, obs.scalars, obs.messages, obs.message_mask,
                         obs.action_mask, hidden)
    assert torch.allclose(again[0], heads.action_logits, atol=1e-6)


def test_scripted_critic_matches_the_eager_one(tmp_path):
    """Spec section 4's critic, exported the same way as the actor so the native
    trainer runs the very same graph."""
    import torch
    from snake_rl.model import ScriptableCritic, export_critic

    torch.manual_seed(0)
    critic = TeamCritic(10, 4, team_channel_pairs=((2, 3), (4, 5)))
    state = _critic_state(seed=11)
    with torch.no_grad():
        expected = critic(state)

    scripted = torch.jit.script(ScriptableCritic(critic).eval())
    with torch.no_grad():
        got = scripted(state.map_grid, state.dragons, state.dragon_team,
                       state.dragon_mask)
    assert torch.allclose(got, expected, atol=1e-5), (got[:2], expected[:2])

    path = export_critic(critic, tmp_path / "critic.ts")
    reloaded = torch.jit.load(str(path))
    with torch.no_grad():
        again = reloaded(state.map_grid, state.dragons, state.dragon_team,
                         state.dragon_mask)
    assert torch.allclose(again, expected, atol=1e-5)
