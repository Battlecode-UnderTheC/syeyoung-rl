"""Spec section 12: acceptance tests and runtime gates (protocol fixtures live
in dragons-env: tests/test_sonar_fixtures.py and the lockstep suite)."""

from __future__ import annotations

import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dragons_env import Game, mapgen
from dragons_rl import _native, codec, collect, features, model, ppo



def small_maps(n, seed=0, max_side=10):
    r = random.Random(seed)
    return [mapgen.random_map(r, max_side=max_side, unit_limit=None).to_text() for _ in range(n)]


@pytest.fixture(scope="module")
def nets():
    a = model.init_actor(jax.random.PRNGKey(3))
    c = model.init_critic(jax.random.PRNGKey(4))
    return a, c


@pytest.fixture(scope="module")
def collected(nets):
    a, c = nets
    setups = [{"map": t, "policy": [0, 0], "record": [True, True]} for t in small_maps(6, seed=1)]
    col = collect.play(setups, collect.stack_policies([a], 2), c, threads=2)
    return col.harvest()


# ---- 12.1 protocol -----------------------------------------------------------------------------

def test_uint64_words_parse_without_precision_loss():
    g = Game(small_maps(1)[0])
    init = g.init_block(g.actor)
    block = g.round_block().replace("NUM_MSGS 0\n", "NUM_MSGS 2\n0\n18446744073709551615\n", 1)
    parsed = _native.parse_round(init, block)
    assert parsed["inbox"] == [0, 2**64 - 1]


# ---- 12.2 privacy / 12.3 geometry / 12.4 lifecycle, checked against the true game ------------------

def play_random_with_facts(text, turns, seed):
    """Plays `turns` dragon-turns with random safe-ish commands, keeping a
    Lifetime per dragon from its own blocks only; yields (game-before, dragon, lifetime)."""
    g = Game(text)
    rng = random.Random(seed)
    lives, inits = {}, {}
    for _ in range(turns):
        if g.done:
            return
        d = g.actor
        if d not in lives:
            inits[d] = g.init_block(d)
            lives[d] = _native.Lifetime(inits[d])
        block = g.round_block()
        assert lives[d].observe(inits[d], block)
        assert not lives[d].observe(inits[d], block)            # repeated input is idempotent
        yield g, d, lives[d], inits[d], block
        me = g.dragons()[d]
        kind = 1 if me["length"] >= 4 and rng.random() < 0.1 else 0
        n = rng.choice([1, 1, 1, 2, 3])
        steps = [rng.randrange(4) for _ in range(n)]
        cmd = codec.command_arrays(1)
        cmd["kind"][0] = kind
        cmd["nsteps"][0] = n if kind == 0 else 0
        cmd["dirs"][0, :n] = steps
        cmd["split"][0] = 2
        lives[d].commit(cmd)
        g.act(kind, steps if kind == 0 else [], 2, [None] * 4, None, 3)
        for dead in g.deaths():
            lives.pop(dead[0], None)                           # closed, never reused
        g.clear_deaths()


@pytest.mark.parametrize("seed", range(8))
def test_known_facts_are_true(seed):
    """every known body entry, every known J and every recorded edge agree with the engine"""
    text = small_maps(1, seed=seed, max_side=14)[0]
    truth = Game(text)
    W, H = truth.width, truth.height
    step = np.asarray(_native.step_table(text))            # the engine's own transitions
    checked = 0
    for g, d, life, init, block in play_random_with_facts(text, 400, seed):
        body = [x + y * W for x, y in g.dragons()[d]["body"]]
        mine = life.body()
        assert len(mine) == len(body)
        assert mine[0] == body[0]
        for k, t in enumerate(mine):
            if t >= 0:
                assert t == body[k], (seed, d, k)
                checked += 1
        for t in range(W * H):
            for dir_ in range(4):
                j = life.dest(t, dir_)
                if j >= 0:
                    assert j == step[t * 4 + dir_], (t, dir_)
    assert checked > 0


def test_privacy_unseen_state_changes_nothing():
    """two boards that differ only far from a dragon give identical facts/tensors while it cannot see the difference"""
    base = mapgen.MapSpec(24, 24, dragons=[(0, [(3, 3), (3, 4), (3, 5)]), (1, [(20, 20), (20, 21), (20, 22)])])
    other = mapgen.MapSpec(24, 24, dragons=list(base.dragons), gaps={(15, 15): (5, 9)}, kelp_h={(14, 14)})
    outs = []
    for spec in (base, other):
        g = Game(spec.to_text())
        life = _native.Lifetime(g.init_block(0))
        block = g.round_block()
        life.observe(g.init_block(0), block)
        buf = codec.actor_buffers(1, 1024, 4, 4)
        life.encode(g.init_block(0), block, buf)
        outs.append(buf)
    for k in outs[0]:
        assert np.array_equal(outs[0][k], outs[1][k]), k


def test_received_words_do_not_touch_facts():
    g = Game(small_maps(1, seed=5)[0])
    init, block = g.init_block(g.actor), g.round_block()
    noisy = block.replace("NUM_MSGS 0\n", "NUM_MSGS 3\n1\n2\n3\n", 1)
    bufs = []
    for b in (block, noisy):
        life = _native.Lifetime(init)
        life.observe(init, b)
        buf = codec.actor_buffers(1, 256, 4, 4)
        life.encode(init, b, buf)
        bufs.append(buf)
    for k in ("tile", "dest", "entity", "segment"):
        assert np.array_equal(bufs[0][k], bufs[1][k]), k


# ---- 12.5 network contract --------------------------------------------------------------------------

def test_widths_and_parameter_counts(nets, collected):
    a, c = nets
    assert model.count(a) == 530_115 and model.count(c) == 83_457
    rep = _native.Replay([collected], 1)
    buf = codec.actor_buffers(1, 128, 4, 4)
    rep.encode_chunk(np.array([0], np.int32), np.array([0], np.int32), 1, buf)
    inp = features.actor_inputs({k: jnp.asarray(buf[k][0]) for k in codec.ACTOR_KEYS})
    assert inp["X"].shape[-1] == 56 and inp["U"].shape[-1] == 20 and inp["S"].shape == (32,)
    assert inp["D"].shape[-1] == 12 and inp["C_prev"].shape == (36,)


def test_padding_does_not_change_outputs(nets, collected):
    a, _ = nets
    rep = _native.Replay([collected], 1)
    outs = []
    for n_pad, k_pad, e_pad, s_pad in ((128, 4, 4, 8), (512, 16, 8, 32)):
        rep.reset()
        buf = codec.actor_buffers(1, n_pad, k_pad, e_pad)
        rep.encode_chunk(np.array([0], np.int32), np.array([0], np.int32), 1, buf)
        raw = {k: jnp.asarray(buf[k][0]) for k in codec.ACTOR_KEYS}
        u = jnp.asarray(np.linspace(0.05, 0.95, codec.UNIFORMS))
        cmd, logp, h, _ = model.act(a, raw, jnp.zeros(192, jnp.float32), u, s_pad)
        outs.append((jax.tree.map(np.asarray, cmd), float(logp), np.asarray(h)))
    assert all(np.array_equal(outs[0][0][k], outs[1][0][k]) for k in outs[0][0])
    assert abs(outs[0][1] - outs[1][1]) < 1e-5
    assert np.allclose(outs[0][2], outs[1][2], atol=1e-5)


# ---- 12.6 actions ---------------------------------------------------------------------------------------

def test_sampled_and_teacher_forced_logp_agree(collected, nets):
    a, _ = nets
    rep = _native.Replay([collected], 1)
    D = rep.decisions()
    forced = codec.from_native(D)
    life = D["lifetime"]
    rep.reset()
    worst = 0.0
    n = rep.lifetimes()["decisions"]
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    for i in range(rep.num_lifetimes):
        buf = codec.actor_buffers(int(n[i]), 128, 16, 8)
        rep.encode_chunk(np.array([i], np.int32), np.array([0], np.int32), int(n[i]), buf)
        h = jnp.zeros(192, jnp.float32)
        for t in range(int(n[i])):
            row = start[i] + t
            raw = {k: jnp.asarray(buf[k][t]) for k in codec.ACTOR_KEYS}
            f = {k: jnp.asarray(v[row]) for k, v in forced.items()}
            logp, _, _, h = model.score(a, raw, h, f, 64)
            worst = max(worst, abs(float(logp) - float(D["logp"][row])))
            assert np.allclose(np.asarray(h), D["h"][row], atol=1e-4), "replay from birth must reproduce the rollout h"
    assert worst < 1e-5, worst
    assert life.size > 0


def test_action_space_rules(nets):
    a, _ = nets
    # a long dragon with room to split: every size 2..L-2 is available; STOP never first; <= 8 tokens
    spec = mapgen.MapSpec(20, 20, dragons=[(0, [(10, y) for y in range(5, 17)]), (1, [(0, 0), (1, 0), (2, 0)])])
    g = Game(spec.to_text())
    life = _native.Lifetime(g.init_block(0))
    life.observe(g.init_block(0), g.round_block())
    buf = codec.actor_buffers(1, 512, 4, 4)
    life.encode(g.init_block(0), g.round_block(), buf)
    raw = {k: jnp.asarray(buf[k][0]) for k in codec.ACTOR_KEYS}
    inp = features.actor_inputs(raw)
    h, z = model.encode(a, inp, jnp.zeros(192, jnp.float32))
    _, smask = model.split_features(h, inp["L"], inp["A"], 16)
    assert int(smask.sum()) == 12 - 3                                    # n = 2..L-2
    rng = np.random.default_rng(0)
    for _ in range(40):
        u = jnp.asarray(rng.random(codec.UNIFORMS))
        cmd, logp, _, _, draws = model.decode(a, h, z, inp, 16, uniforms=u)
        if int(cmd["kind"]) == 0:
            assert 1 <= int(cmd["nsteps"]) <= 8
        f = jax.tree.map(jnp.asarray, cmd)
        _, logp2, _, _, _ = model.decode(a, h, z, inp, 16, forced=f)
        assert abs(float(logp) - float(logp2)) < 1e-5
    # fatal directions are not masked, and zero differs from silence
    assert codec.to_native({"kind": np.array([0]), "nsteps": np.array([1]), "dirs": np.zeros((1, 8)),
                            "split": np.array([0]), "sonar": np.array([[1, 0, 1, 257 - 1]])})["sonar_mask"][0] == 0b1101


# ---- 12.7 returns / 12.8 replay / 12.9 optimisation ---------------------------------------------------------

def test_advantages_use_the_final_team_result(collected):
    sh = ppo.make_shards([collected], jax.devices()[:1], 1)[0][0]
    life = sh.decisions["lifetime"]
    for i in range(0, len(life), 7):
        m = sh.lifetimes["match"][life[i]]
        team = sh.lifetimes["team"][life[i]]
        ev = sh.full.replay.events(m)
        G = ppo.returns(ev["winner"], team)
        assert abs(sh.adv[i] - (G - ev["values"][sh.decisions["event"][i], team])) < 1e-6


def test_gradients_reach_every_module(collected, nets):
    a, c = nets
    sh = ppo.make_shards([collected], jax.devices()[:1], 1)[0][0]
    grads, logp_new, _ = ppo.actor_epoch(sh, a, 512.0 * 6)
    assert np.allclose(logp_new, sh.decisions["logp"], atol=1e-5)
    finite = all(np.isfinite(g).all() for g in jax.tree.leaves(grads))
    assert finite
    reached = {jax.tree_util.keystr(k): float(np.abs(g).sum()) for k, g in jax.tree_util.tree_leaves_with_path(grads)}
    dead = [k for k, v in reached.items() if v == 0.0]
    # split parameters get gradient only if a split was taken in the fixture batch
    took_split = (sh.decisions["kind"] == 1).any()
    dead = [k for k in dead if took_split or "split" not in k]
    # an ECHOES field counts at most four pings, so it is one byte and the echo
    # GRU only ever steps from h = 0: its recurrent weights cannot get gradient
    dead = [k for k in dead if k != "['echo']['gru']['w_h']"]
    # the split scorer's output bias shifts every candidate equally: softmax ignores it
    dead = [k for k in dead if k != "['split']['l2']['b']"]
    assert not dead, dead
    cg, loss = ppo.critic_epoch(sh, c, float(len(sh.critic_rows)))
    assert np.isfinite(loss) and all(np.isfinite(g).all() for g in jax.tree.leaves(cg))


# ---- 12.10 deterministic seeded sampling ----------------------------------------------------------------------

def test_sharding_does_not_change_the_update(collected, nets, monkeypatch):
    """the full-batch gradient is the same however lifetimes and events are dealt to devices"""
    monkeypatch.setattr(ppo, "CRITIC_FRACTION", 1.0)      # the same events on both sides
    a, c = nets
    dev = jax.devices()[0]
    one = ppo.make_shards([collected], [dev], 1)[0]
    three = ppo.make_shards([collected], [dev, dev, dev], 3)[0]
    assert len(three) > 1
    for shards in (one, three):
        assert sum(len(s.decisions["logp"]) for s in shards) == len(one[0].decisions["logp"])
    def grads(shards):
        g = ppo._sum([ppo.actor_epoch(s, a, 512.0 * 6)[0] for s in shards if len(s.decisions["logp"])])
        cg = ppo._sum([r for r in (ppo.critic_epoch(s, c, 1000.0)[0] for s in shards) if r is not None])
        return g, cg
    (g1, c1), (g3, c3) = grads(one), grads(three)
    for x, y in zip(jax.tree.leaves(g1) + jax.tree.leaves(c1), jax.tree.leaves(g3) + jax.tree.leaves(c3)):
        assert np.allclose(x, y, atol=1e-6, rtol=1e-4)
