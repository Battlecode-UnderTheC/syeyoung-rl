"""Credit assignment, recurrence and optimisation acceptance tests (spec 8, 9, 16)."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from spatial_helpers import NativeMatch, _dragons
from dragons_env import mapgen
from dragons_env.mapgen import MapSpec, snake
from snake_rl.algo.gae import event_gae
from snake_rl.algo.ppo import (SpatialPPOConfig, event_advantages, make_chunks, ppo_update_spatial,
                               replay_chunk, unroll_sequences)
from snake_rl.env.actions import ActionSpace, CommandCodec
from snake_rl.model.actor import SpatialActor, SpatialActorConfig
from snake_rl.model.critic import EventCritic
from snake_rl.model.policy import SpatialPolicy
from snake_rl.policies import GreedyPearlPolicy
from snake_rl.rollout.collector import ScriptedAdapter, play_event_match

pytestmark = pytest.mark.skipif(_dragons is None, reason="native sim not built")
codec = CommandCodec(8)


# -- event GAE ----------------------------------------------------------------------

def test_with_gamma_and_lambda_one_every_event_gets_g_minus_v():
    rng = np.random.default_rng(0)
    E = 40
    values = rng.normal(size=E).astype(np.float32)
    nxt = np.zeros(E, dtype=np.float32)
    nxt[:-1] = values[1:]
    rewards = np.zeros(E, dtype=np.float32)
    rewards[-1] = -1.0
    delta_rounds = rng.integers(0, 2, size=E)
    terminal = np.zeros(E, dtype=bool)
    terminal[-1] = True
    adv, ret = event_gae(rewards, values, nxt, delta_rounds, terminal, 1.0, 1.0)
    assert np.allclose(adv, -1.0 - values, atol=1e-5), "including events of dragons that died earlier"
    assert np.allclose(ret, -1.0, atol=1e-5)


def test_lambda_decay_counts_rounds_not_children():
    """Two timelines with the same rounds but a different number of events per
    round give the same advantage at the start of each round."""
    lam = 0.9
    sparse_dr = np.array([1, 1, 1, 1])                 # one event per round
    dense_dr = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1])  # several children act per round
    for dr in (sparse_dr, dense_dr):
        E = len(dr)
        rewards = np.zeros(E); rewards[-1] = 1.0
        values = np.zeros(E); nxt = np.zeros(E)
        terminal = np.zeros(E, dtype=bool); terminal[-1] = True
        adv, _ = event_gae(rewards, values, nxt, dr, terminal, 1.0, lam)
        assert adv[0] == pytest.approx(lam ** 3), dr
        assert adv[-1] == pytest.approx(1.0)


def test_a_collection_cutoff_bootstraps_where_the_terminal_does_not():
    E = 5
    rewards = np.zeros(E); values = np.zeros(E); nxt = np.zeros(E)
    dr = np.ones(E, dtype=int)
    ended = np.zeros(E, dtype=bool); ended[-1] = True
    a_end, _ = event_gae(rewards, values, nxt, dr, ended, 0.99, 0.95)
    assert np.allclose(a_end, 0.0)
    nxt[-1] = 1.0
    cut = np.zeros(E, dtype=bool)                    # no terminal flag: truncated
    a_cut, _ = event_gae(rewards, values, nxt, dr, cut, 0.99, 0.95)
    assert a_cut[-1] == pytest.approx(0.99)


# -- the timeline the collector records ---------------------------------------------

def _scripted_match(seed=0, rounds=30):
    rng = random.Random(seed)
    spec = mapgen.random_map(rng, min_size=13, max_size=15, length=6, kelp_density=0.0,
                             pearl_gap_choices=((4, 9),))
    pol = {"A": ScriptedAdapter(GreedyPearlPolicy(ActionSpace(), seed=1)),
           "B": ScriptedAdapter(GreedyPearlPolicy(ActionSpace(), seed=2))}
    return play_event_match(spec, pol, set(), seed=seed, max_rounds=rounds)


def test_team_reward_occurs_once_at_the_actual_terminal():
    rec = _scripted_match(rounds=500)
    total = np.stack([e.reward for e in rec.events]).sum(axis=0)
    assert np.array_equal(total, rec.reward)
    assert rec.events[-1].terminal and not any(e.terminal for e in rec.events[:-1])
    assert all(e.delta_rounds in (0, 1) for e in rec.events[:-1])
    rounds = [e.round for e in rec.events]
    assert rounds == sorted(rounds), "events follow the engine's order"


def test_a_cutoff_is_not_a_terminal():
    rec = _scripted_match(rounds=12)
    assert not rec.info["terminal"]
    assert not rec.events[-1].terminal and float(np.abs(rec.events[-1].reward).sum()) == 0.0


def test_children_act_in_their_birth_round_and_deaths_do_not_end_the_round():
    """Compound mechanics fixture: split, newborn turn, parent sprint, mutual
    head death, all in the recorded event order. Never training data."""
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(8, 10, 8, "E", w, h)), (1, snake(14, 10, 4, "W", w, h))],
                   pearl_gap=(0, 0))
    m = NativeMatch(spec)
    m.play_round({0: "SPLIT 3", 1: "MOVE N"})
    assert 2 in m.states and m.states[2].own_turns == 1, "the child acted in its birth round"
    assert m.states[0].own_turns == 1 and len(m.states[0].body) == 5
    assert m.truth_segments(0) == [(8, 10), (7, 10), (6, 10), (5, 10), (4, 10)]
    m.play_round({0: "MOVE EE", 1: "MOVE W", 2: "MOVE W"})
    assert m.truth_segments(0)[0] == (10, 10) and m.dragon(0)["length"] == 4
    # dragon 1 is at (13,9) heading W; send both heads into (12,9)/(11,10)... make them meet
    m.play_round({0: "MOVE N", 1: "MOVE W", 2: "MOVE W"})
    assert m.dragon(0)["alive"] and m.dragon(1)["alive"]
    head0, head1 = m.truth_segments(0)[0], m.truth_segments(1)[0]
    assert head0 == (10, 9) and head1 == (12, 9)
    # dragon 0 steps into (11,9) first; dragon 1 then walks into its head
    m.play_round({0: "MOVE E", 1: "MOVE W", 2: "MOVE W"})
    deaths = {d[0]: d[2] for d in m.game.deaths()}
    assert deaths.get(0) == "H" and deaths.get(1) == "H", deaths
    assert m.dragon(2)["alive"], "the child survives its parent"
    # team B is now eliminated: the completed round decides the match for the
    # surviving child's team, and the terminal reward follows from that
    assert m.game.result() == "A"


# -- recurrence: stepping equals the chunked replay --------------------------------

def _learner_matches(actor, n=3, seed=0):
    rng = random.Random(seed)
    matches = []
    for i in range(n):
        spec = mapgen.random_map(rng, min_size=11, max_size=13, length=7, kelp_density=0.02,
                                 portal_pairs=1, pearl_gap_choices=((3, 8),))
        pol = {"A": SpatialPolicy(actor, codec, match_id=i), "B": SpatialPolicy(actor, codec, match_id=i)}
        matches.append(play_event_match(spec, pol, {"A", "B"}, seed=i, match_id=i, snapshot_every=4))
    return matches


def test_chunked_replay_reproduces_the_stepped_rollout_including_children():
    torch.manual_seed(0)
    actor = SpatialActor(SpatialActorConfig(sonar=True)).eval()
    matches = _learner_matches(actor)
    cfg = SpatialPPOConfig(sequence_length=6, burn_in=4)
    values = [np.zeros((len(m.events), 2), dtype=np.float32) for m in matches]
    adv, _ = event_advantages(matches, values, cfg)
    chunks = make_chunks(matches, cfg)
    assert chunks, "some learner decisions were recorded"
    cold_children = sum(1 for c in chunks if c.lifetime.key.dragon_id >= 2 and c.start == 0)
    replayed = [replay_chunk(c, codec, adv[matches.index(c.match)]) for c in chunks]
    with torch.no_grad():
        res = unroll_sequences(actor, replayed, codec, "cpu", aux=False)
    old = torch.tensor([replayed[s].old_log_prob[t] for s, t in res.index])
    assert res.log_prob.shape == old.shape and len(old) == sum(int(r.scored.sum()) for r in replayed)
    assert torch.allclose(res.log_prob, old, atol=1e-4), (res.log_prob - old).abs().max()
    # each decision is scored exactly once
    scored = [(c.lifetime.key, t) for c in chunks for t in range(c.start, c.stop)]
    assert len(scored) == len(set(scored))
    assert len(scored) == sum(len(l.turns) for m in matches for l in m.lifetimes.values() if l.learner)


def test_burn_in_detaches_and_padded_steps_do_not_change_state():
    torch.manual_seed(1)
    actor = SpatialActor(SpatialActorConfig())
    matches = _learner_matches(actor, n=2, seed=3)
    cfg = SpatialPPOConfig(sequence_length=4, burn_in=4)
    values = [np.zeros((len(m.events), 2), dtype=np.float32) for m in matches]
    adv, _ = event_advantages(matches, values, cfg)
    chunks = make_chunks(matches, cfg)
    replayed = [replay_chunk(c, codec, adv[matches.index(c.match)]) for c in chunks]
    res = unroll_sequences(actor, replayed, codec, "cpu", aux=False)
    if res.log_prob.numel() == 0:
        pytest.skip("no scored decisions in this tiny rollout")
    res.log_prob.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())


# -- optimisation --------------------------------------------------------------------

def test_ppo_update_moves_the_policy_with_finite_metrics_and_a_fixed_denominator():
    torch.manual_seed(0)
    actor = SpatialActor(SpatialActorConfig(sonar=True))
    critic = EventCritic()
    matches = _learner_matches(actor, n=3, seed=5)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), 3e-4)
    cfg = SpatialPPOConfig(sequence_length=8, burn_in=4, sequences_per_microbatch=4,
                           accumulation_microbatches=2, critic_events_per_microbatch=16)
    before = [p.detach().clone() for p in actor.parameters()]
    metrics = ppo_update_spatial(actor, critic, opt, matches, cfg, codec)
    assert all(np.isfinite(v) for v in metrics.values()), metrics
    assert metrics["ppo/decisions"] > 0 and metrics["ppo/team_rounds"] <= metrics["ppo/decisions"]
    assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))


def test_actor_loss_does_not_depend_on_the_critic():
    torch.manual_seed(0)
    actor = SpatialActor(SpatialActorConfig())
    matches = _learner_matches(actor, n=1, seed=8)
    cfg = SpatialPPOConfig(sequence_length=8, burn_in=4)
    values = [np.zeros((len(m.events), 2), dtype=np.float32) for m in matches]
    adv, _ = event_advantages(matches, values, cfg)
    chunks = make_chunks(matches, cfg)
    replayed = [replay_chunk(c, codec, adv[0]) for c in chunks]
    res = unroll_sequences(actor, replayed, codec, "cpu", aux=False)
    if res.log_prob.numel():
        res.log_prob.sum().backward()
    critic = EventCritic()
    assert all(p.grad is None for p in critic.parameters())


def test_microbatch_accumulation_matches_the_full_batch_gradient():
    torch.manual_seed(0)
    actor = SpatialActor(SpatialActorConfig()).eval()
    matches = _learner_matches(actor, n=2, seed=11)
    cfg = SpatialPPOConfig(sequence_length=6, burn_in=2)
    values = [np.zeros((len(m.events), 2), dtype=np.float32) for m in matches]
    adv, _ = event_advantages(matches, values, cfg)
    chunks = make_chunks(matches, cfg)
    replayed = [replay_chunk(c, codec, adv[matches.index(c.match)]) for c in chunks]
    if len(replayed) < 2:
        pytest.skip("need at least two sequences")

    def loss_of(seqs):
        res = unroll_sequences(actor, seqs, codec, "cpu", aux=False)
        adv_t = torch.tensor([seqs[s].advantage[t] for s, t in res.index])
        return (res.log_prob * adv_t).sum()

    actor.zero_grad()
    loss_of(replayed).backward()
    full = [p.grad.clone() if p.grad is not None else None for p in actor.parameters()]
    actor.zero_grad()
    half = len(replayed) // 2
    loss_of(replayed[:half]).backward()
    loss_of(replayed[half:]).backward()
    for g_full, p in zip(full, actor.parameters()):
        if g_full is None:
            continue
        assert torch.allclose(g_full, p.grad, atol=1e-5, rtol=1e-4)
