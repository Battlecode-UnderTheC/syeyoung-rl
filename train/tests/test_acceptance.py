"""Spec section 12: acceptance tests and runtime gates (protocol fixtures live
in dragons-env: tests/test_sonar_fixtures.py and the lockstep suite)."""

from __future__ import annotations

import random

import jax
import jax.numpy as jnp
from jax import lax
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
        buf = codec.actor_buffers(1, 4, 4)
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
        buf = codec.actor_buffers(1, 4, 4)
        life.encode(init, b, buf)
        bufs.append(buf)
    for k in ("win", "wdest", "grid", "entity", "segment"):
        assert np.array_equal(bufs[0][k], bufs[1][k]), k


# ---- 12.5 network contract --------------------------------------------------------------------------

def test_widths_and_parameter_counts(nets, collected):
    a, c = nets
    assert model.count(a) == model.ACTOR_PARAMETERS and model.count(c) == model.CRITIC_PARAMETERS
    rep = _native.Replay([collected], 1)
    buf = codec.actor_buffers(1, 4, 4)
    rep.encode_chunk(np.array([0], np.int32), np.array([0], np.int32), 1, buf)
    inp = features.actor_inputs({k: jnp.asarray(buf[k][0]) for k in codec.ACTOR_KEYS})
    assert inp["X"].shape[-1] == 56 and inp["U"].shape[-1] == 20 and inp["S"].shape == (32,)
    assert inp["D"](0).shape == (4, 12) and inp["C_prev"].shape == (36,)


def test_padding_does_not_change_outputs(nets, collected):
    a, _ = nets
    rep = _native.Replay([collected], 1)
    outs = []
    for k_pad, e_pad, s_pad in ((4, 4, 8), (16, 8, 32)):
        rep.reset()
        buf = codec.actor_buffers(1, k_pad, e_pad)
        rep.encode_chunk(np.array([0], np.int32), np.array([0], np.int32), 1, buf)
        raw = {k: jnp.asarray(buf[k][0]) for k in codec.ACTOR_KEYS}
        u = jnp.asarray(np.linspace(0.05, 0.95, codec.UNIFORMS))
        cmd, logp, h, _ = model.act(a, raw, jnp.zeros(192, jnp.float32), u, s_pad)
        outs.append((jax.tree.map(np.asarray, cmd), float(logp), np.asarray(h)))
    assert all(np.array_equal(outs[0][0][k], outs[1][0][k]) for k in outs[0][0])
    assert abs(outs[0][1] - outs[1][1]) < 1e-5
    assert np.allclose(outs[0][2], outs[1][2], atol=1e-5)


def test_critic_runs_every_few_turns_and_values_are_reused(nets):
    """V_old: fresh at most every CRITIC_EVERY dragon-turns of a match, the cached
    value (negated for the other team) in between; always finite"""
    a, c = nets
    setups = [{"map": t, "policy": [0, 0], "record": [True, True]} for t in small_maps(4, seed=7)]
    g = collect.Group(setups, 2)
    dev = jax.devices()[0]
    stack = jax.device_put(collect.stack_policies([a], 2), dev)
    key, ran, rows = np.random.default_rng(0), 0, 0
    while not g.col.done():
        key, _ = collect._dispatch(g, stack, c, dev, key, 128)
        ran += sum(len(call[3]) for call in g.inflight)
        rows += sum(call[1] for call in g.inflight)
        collect._finish(g)
    assert 0 < ran <= rows // collect.CRITIC_EVERY + 2 * len(setups)
    rep = _native.Replay([g.col.harvest()], 1)
    for m in range(rep.num_matches):
        v = rep.events(m)["values"]
        assert np.isfinite(v).all() and np.allclose(v[:, 0], -v[:, 1])


def test_shared_and_gathered_policy_steps_agree(nets):
    """a learner row gives the same command, logp and h whether its call shares
    the weights or gathers them per row"""
    a, c = nets
    setups = [{"map": t, "policy": [0, 0], "record": [True, False]} for t in small_maps(3, seed=9)]
    g = collect.Group(setups, 2)
    dev = jax.devices()[0]
    stack = jax.device_put(collect.stack_policies([a], 2), dev)
    key = np.random.default_rng(0)
    key, _ = collect._dispatch(g, stack, c, dev, key, 128)
    for matches, B, out, learn, values, (abuf, slot), last, _ in g.inflight:
        _, _, seed, s_pad, _, shared = last
        assert shared
        args = jax.device_put((slot, {k: abuf[k] for k in codec.ACTOR_KEYS}, abuf["h"]), dev)
        other = collect._step(g, stack, False, *args, seed, s_pad)
        for x, y in zip(jax.tree.leaves(out), jax.tree.leaves(other)):
            assert np.allclose(np.asarray(x)[:B], np.asarray(y)[:B], atol=1e-5)
    collect._finish(g)


def test_packs_survive_serialization(collected, nets):
    """a pack as bytes (another process) replays to the same training inputs"""
    back = _native.Pack.deserialize(collected.serialize())
    # a worker's collector keeps no observations; its bytes rebuild the same pack
    a0, c0 = nets
    setups = [{"map": t, "policy": [0, 0], "record": [True, True]} for t in small_maps(6, seed=1)]
    col = collect.play(setups, collect.stack_policies([a0], 2), c0, threads=2, keep_observations=False)
    raw = col.harvest()
    lean = _native.Replay([_native.Pack.deserialize(raw.serialize())], 1)
    assert len(lean.decisions()["logp"]) == raw.decisions() > 0
    n = lean.lifetimes()["decisions"]
    buf = codec.actor_buffers(int(n[0]), 16, 8)
    lean.encode_chunk(np.array([0], np.int32), np.array([0], np.int32), int(n[0]), buf)
    a, b = _native.Replay([collected], 1), _native.Replay([back], 1)
    for k, v in a.lifetimes().items():
        assert np.array_equal(v, b.lifetimes()[k]), k
    for k, v in a.decisions().items():
        assert np.array_equal(v, b.decisions()[k], equal_nan=True), k
    for m in range(a.num_matches):
        for k, v in a.events(m).items():
            assert np.array_equal(np.asarray(v), np.asarray(b.events(m)[k]), equal_nan=True), (m, k)
        ev = np.arange(len(a.events(m)["dragon"]), dtype=np.int32)
        ca, cb = codec.critic_buffers(len(ev), 64), codec.critic_buffers(len(ev), 64)
        a.encode_events(m, ev, ca)
        b.encode_events(m, ev, cb)
        assert all(np.array_equal(ca[k], cb[k]) for k in ca)
    n = a.lifetimes()["decisions"]
    for i in range(a.num_lifetimes):
        fa, fb = codec.actor_buffers(int(n[i]), 16, 8), codec.actor_buffers(int(n[i]), 16, 8)
        a.encode_chunk(np.array([i], np.int32), np.array([0], np.int32), int(n[i]), fa)
        b.encode_chunk(np.array([i], np.int32), np.array([0], np.int32), int(n[i]), fb)
        assert all(np.array_equal(fa[k], fb[k]) for k in fa)
    assert np.array_equal(collected.results()["winner"], back.results()["winner"])


def test_critic_cells_cover_the_board(collected):
    """the critic's 8x8 cells: every tile once, one head per living dragon"""
    rep = _native.Replay([collected], 1)
    CC, CG = features.CC, _native.CRITIC_GLOBAL
    for m in range(rep.num_matches):
        ev = np.arange(len(rep.events(m)["dragon"]), dtype=np.int32)
        buf = codec.critic_buffers(len(ev), 64)
        rep.encode_events(m, ev, buf)
        cells, q = buf["cells"], buf["global"]
        area = q[:, CG["WIDTH"]] * q[:, CG["HEIGHT"]]
        assert np.array_equal(cells[:, :, CC["TILES"]].sum(1), area)
        heads = cells[:, :, CC["HEAD_A"]].sum(1) + cells[:, :, CC["HEAD_B"]].sum(1)
        assert np.array_equal(heads, q[:, CG["ALIVE"]])


def test_call_rows_shrink_with_width():
    """a call's rows x split width stay bounded; pads are capped"""
    assert codec.rows_for_width(512, 32, 16) == 512 and codec.rows_for_width(512, 32, 64) == 512
    assert codec.rows_for_width(512, 512, 16) == 128 and codec.rows_for_width(8, 512, 64) == 4
    assert codec.split_pad(35) == 32 and codec.split_pad(36) == codec.split_pad(5000) == codec.MAX_SPLIT
    assert codec.inbox_pad(300) == _native.FIELDS["MAX_INBOX"] and codec.echo_pad(300) == _native.FIELDS["MAX_ECHO"]


def test_split_scorer_equals_the_mlp_over_concatenated_features():
    key = jax.random.PRNGKey(11)
    p = model.init_actor(key)["split"]
    h = jax.random.normal(key, (192,))
    fast, mask = model.split_logits_of(p, h, 40, 400, 64)
    x, _ = model.split_candidates(40, 400, 64)
    slow = model.Score(p, jnp.concatenate([jnp.broadcast_to(h, (64, 192)), x], -1))[:, 0]
    assert np.allclose(np.asarray(fast), np.asarray(slow), atol=1e-4) and int(mask.sum()) == 40 - 3


def test_native_step_loop_plays_and_harvests(nets):
    """the workers' native loop: refills from the queue, every row a match, packs
    that replay into training inputs with finite V_old on recorded decisions"""
    from dragons_rl import actors
    a, c = nets
    maps = small_maps(12, seed=21)
    M, Mc = 6, 4
    col = _native.Collector([actors.new_match(maps[i], 2, 1) for i in range(M)], 2, False)
    col.set_matchmaking(0.5, 7)
    col.push_maps(maps[M:])
    buf = codec.actor_buffers(M, codec.FIELDS["MAX_INBOX"], codec.FIELDS["MAX_ECHO"])
    cbuf = codec.critic_buffers(Mc, 128)
    opp, crow, cteam = np.zeros(M, np.int32), np.zeros(Mc, np.int32), np.zeros(Mc, np.int32)
    other = model.init_actor(jax.random.PRNGKey(99))
    packs, saw_opp = [], False
    for step in range(300):
        n = col.step_encode(buf, opp, cbuf, Mc, collect.CRITIC_EVERY, crow, cteam)
        assert 0 <= n <= Mc and set(np.unique(opp)) <= {-1, 0, 1}
        saw_opp |= bool((opp == 1).any())
        out = collect.native_step(a, other, c, {k: buf[k] for k in codec.ACTOR_KEYS}, buf["h"], opp, cbuf, cteam,
                                  step, codec.MAX_SPLIT)
        cmd, logp, h, v = jax.device_get(out)
        col.step_apply(cmd["kind"], cmd["nsteps"], cmd["dirs"], cmd["split"], cmd["sonar"], logp, h, v)
        if step % 16 == 15:
            pk = col.harvest()
            if pk.num_matches:
                packs.append(_native.Pack.deserialize(pk.serialize()))
    assert saw_opp and packs and col.turns > 300 and col.nonfinite == 0
    rep = _native.Replay(packs, 1)
    D = rep.decisions()
    assert len(D["logp"]) > 0 and np.isfinite(D["logp"]).all()
    for m in range(rep.num_matches):
        ev = rep.events(m)
        rec = np.asarray(ev["values"])
        assert np.isfinite(rec).any()


def test_conv_matches_lax_convolution():
    key = jax.random.PRNGKey(3)
    for (h, w, cin, cout, stride) in ((7, 7, 56, 32, 1), (15, 15, 32, 32, 2), (8, 8, 32, 32, 2), (7, 7, 32, 32, 1)):
        k1, k2, k3, key = jax.random.split(key, 4)
        x = jax.random.normal(k1, (h, w, cin))
        p = {"w": jax.random.normal(k2, (cout, cin, 3, 3)), "b": jax.random.normal(k3, (cout,))}
        ref = lax.conv_general_dilated(x[None], jnp.transpose(p["w"], (2, 3, 1, 0)), (stride, stride), "SAME",
                                       dimension_numbers=("NHWC", "HWIO", "NHWC"))[0] + p["b"]
        got = model.Conv(p, x, stride)
        assert got.shape == ref.shape and np.allclose(np.asarray(got), np.asarray(ref), atol=1e-3), (h, stride)


def test_half_precision_collection_path_is_close(nets, collected):
    """collection runs the tile MLPs in fp16: h, logp and V stay close to the fp32 path"""
    a, c = nets
    rep = _native.Replay([collected], 1)
    n = int(rep.lifetimes()["decisions"][0])
    buf = codec.actor_buffers(n, 16, 8)
    rep.encode_chunk(np.zeros(n, np.int32), np.arange(n, dtype=np.int32), 1, buf)
    for i in range(n):
        raw = {k: jnp.asarray(buf[k][i]) for k in codec.ACTOR_KEYS}
        u = jnp.asarray(np.linspace(0.05, 0.95, codec.UNIFORMS))
        h0 = jnp.asarray(buf["h"][i])
        full = model.act(a, raw, h0, u, 32)
        half = model.act(a, raw, h0, u, 32, half=True)
        assert np.all(np.isfinite(np.asarray(half[2])))
        assert np.allclose(np.asarray(full[2]), np.asarray(half[2]), atol=2e-2)
        cmd_same = all(np.array_equal(np.asarray(full[0][k]), np.asarray(half[0][k])) for k in full[0])
        if cmd_same:
            assert abs(float(full[1]) - float(half[1])) < 2e-2


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
        buf = codec.actor_buffers(int(n[i]), 16, 8)
        rep.encode_chunk(np.array([i], np.int32), np.array([0], np.int32), int(n[i]), buf)
        h = jnp.zeros(192, jnp.float32)
        for t in range(int(n[i])):
            row = start[i] + t
            raw = {k: jnp.asarray(buf[k][t]) for k in codec.ACTOR_KEYS}
            f = {k: jnp.asarray(v[row]) for k, v in forced.items()}
            logp, _, _, h = model.score(a, raw, h, f, 64)
            worst = max(worst, abs(float(logp) - float(D["logp"][row])))
            # the rollout ran the tile MLPs in fp16, the replay in fp32
            if (t + 1) % ppo.CHUNK == 0:                          # the rollout keeps h at chunk ends
                assert np.allclose(np.asarray(h), D["h"][row], atol=5e-3), "replay from birth must reproduce the rollout h"
    assert worst < 5e-3, worst
    assert life.size > 0


def test_action_space_rules(nets):
    a, _ = nets
    # a long dragon with room to split: every size 2..L-2 is available; STOP never first; <= 8 tokens
    spec = mapgen.MapSpec(20, 20, dragons=[(0, [(10, y) for y in range(5, 17)]), (1, [(0, 0), (1, 0), (2, 0)])])
    g = Game(spec.to_text())
    life = _native.Lifetime(g.init_block(0))
    life.observe(g.init_block(0), g.round_block())
    buf = codec.actor_buffers(1, 4, 4)
    life.encode(g.init_block(0), g.round_block(), buf)
    raw = {k: jnp.asarray(buf[k][0]) for k in codec.ACTOR_KEYS}
    inp = features.actor_inputs(raw)
    h, z = model.encode(a, inp, jnp.zeros(192, jnp.float32))
    _, smask = model.split_candidates(inp["L"], inp["A"], 16)
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
        penalty = ppo.DEATH_PENALTY * (sh.lifetimes["death"][life[i]] in ppo.SELF_INFLICTED)
        assert abs(sh.adv[i] - (G - ev["values"][sh.decisions["event"][i], team] - penalty)) < 1e-6


def test_gradients_reach_every_module(collected, nets):
    a, c = nets
    sh = ppo.make_shards([collected], jax.devices()[:1], 1)[0][0]
    grads, logp_new, _ = ppo.actor_epoch(sh, a, 512.0 * 6)
    assert np.allclose(logp_new, sh.decisions["logp"], atol=5e-3)      # collection ran the tile MLPs in fp16
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


@pytest.mark.parametrize("seed", range(6))
def test_incremental_tables_equal_the_reference(seed):
    """the per-lifetime tile tables, updated only where facts changed, equal a from-scratch encoding"""
    text = small_maps(1, seed=100 + seed, max_side=20)[0]
    checked = 0
    for g, d, life, init, block in play_random_with_facts(text, 300, seed):
        n = g.width * g.height
        fast = np.zeros((n, _native.FIELDS["TILE"]), np.int32), np.zeros((n, 4), np.int32)
        ref = np.zeros((n, _native.FIELDS["TILE"]), np.int32), np.zeros((n, 4), np.int32)
        life.tables(init, block, *fast)
        life.tables(init, block, *ref, reference=True)
        assert np.array_equal(fast[0], ref[0]) and np.array_equal(fast[1], ref[1]), seed
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("seed", range(3))
def test_window_and_grid_follow_the_tables(seed):
    """simple1's inputs are the whole-board tables seen from the head: window cells hold
    the records at their offsets (toroidal), window J leads to the cell holding J's
    tile (or -2 outside), and the grid covers every tile once"""
    text = small_maps(1, seed=200 + seed, max_side=40)[0]
    S, TF, WIN = _native.STATUS, _native.FIELDS["TILE"], _native.FIELDS["WIN"]
    checked = 0
    for g, d, life, init, block in play_random_with_facts(text, 200, seed):
        W, H, n = g.width, g.height, g.width * g.height
        tile, dest = np.zeros((n, TF), np.int32), np.zeros((n, 4), np.int32)
        life.tables(init, block, tile, dest)
        buf = codec.actor_buffers(1, 16, 8)
        life.encode(init, block, buf)
        hx, hy = int(buf["status"][0, S["HEAD_X"]]), int(buf["status"][0, S["HEAD_Y"]])
        win, wdest, grid = buf["win"][0], buf["wdest"][0], buf["grid"][0]
        at = lambda c: ((hy + c // WIN - WIN // 2) % H) * W + (hx + c % WIN - WIN // 2) % W
        for c in range(WIN * WIN):
            t = at(c)
            assert np.array_equal(win[c], tile[t]), (seed, c)
            for k in range(4):
                if dest[t, k] < 0:
                    assert wdest[c, k] == -1
                elif wdest[c, k] >= 0:
                    assert at(int(wdest[c, k])) == dest[t, k]
                else:
                    assert wdest[c, k] == -2
        assert grid[:, 0].sum() == n
        checked += 1
    assert checked > 0


def test_incremental_tables_over_long_lifetimes():
    """many turns of the collector's own play (the trained-or-not policy), every learner decision checked"""
    import jax
    a = model.init_actor(jax.random.PRNGKey(5)); c = model.init_critic(jax.random.PRNGKey(6))
    setups = [{"map": t, "policy": [0, 0], "record": [True, True]} for t in small_maps(4, seed=7, max_side=16)]
    pack = collect.play(setups, collect.stack_policies([a]), c, threads=2).harvest()
    rep = _native.Replay([pack], 1)
    n = rep.lifetimes()["decisions"]
    assert n.sum() > 0
    for i in range(rep.num_lifetimes):
        L = int(n[i])
        tiles = int(rep.lifetimes()["tiles"][i])
        buf = codec.actor_buffers(L, 64, 8)
        rep.encode_chunk(np.array([i], np.int32), np.array([0], np.int32), L, buf)
        # the same decisions re-encoded from scratch by a second replay's reference path
        ref = codec.actor_buffers(L, 64, 8)
        rep2 = _native.Replay([pack], 1)
        rep2.encode_chunk_reference(np.array([i], np.int32), np.array([0], np.int32), L, ref)
        for k in ("win", "wdest", "grid", "entity", "segment", "status"):
            assert np.array_equal(buf[k], ref[k]), (i, k)
