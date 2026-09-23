"""The model's inputs and its two networks, against the real environment."""

import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dragons_env import SCALARS, VecEnv, mapgen, maps
from dragons_model import features as F
from dragons_model import model as M

L = F.layout(16, True)


def env_records(texts, n=16, steps=0, seed=0):
    env = VecEnv(texts, n, seed=seed, threads=1, state=True)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        env.step(np.zeros(n, np.uint8), rng.integers(0, 4, (n, 1)).astype(np.uint8), np.ones(n, np.int32),
                 np.zeros(n, np.int32), np.full(n, -1, np.int64))
    return env, np.array(env.buffers()["records"])


def test_decode_shapes_and_self():
    env, rec = env_records(list(maps.contest_maps().values()), steps=30)
    x = F.decode(jnp.asarray(rec), L)
    b = rec.shape[0]
    assert x.window.shape == (b, 49, F.WINDOW_CHANNELS) and x.scalars.shape == (b, F.NUM_SCALARS)
    assert x.mask.shape == (b, F.NUM_ACTIONS) and bool(x.mask.any(1).all())
    # the head is the dragon itself, on the centre tile
    assert bool((x.window[:, F.CENTER, 4] == 1).all())
    assert int(x.write_cell.max()) < F.MAP_CELLS ** 2 and int(x.crop_cell.max()) < F.MAP_CELLS ** 2
    s = F.decode_state(jnp.asarray(rec), L)
    assert s.grid.shape == (b, 8, 16, 16) and bool(jnp.isfinite(s.scalars).all())


def replay(text, history, extra):
    """A fresh single-env VecEnv on `text`, the history replayed, then `extra`; the deaths of the last step."""
    env = VecEnv([text], 1, seed=0, threads=1)
    for kind, steps, split in history + [extra]:
        st = np.zeros((1, 2), np.uint8)
        st[0, :len(steps)] = steps
        env.step(np.array([kind], np.uint8), st, np.array([len(steps)], np.int32), np.array([split], np.int32),
                 np.array([-1], np.int64))
    return env.deaths()


@pytest.mark.parametrize("seed", range(3))
def test_masked_moves_are_certain_death(seed):
    rng = random.Random(seed)
    text = mapgen.random_map(rng, max_side=12, kelp=0.15, portal_pairs=2, dragons_per_team=(2, 3), length=(3, 8)).to_text()
    env = VecEnv([text], 1, seed=0, threads=1)
    history, checked = [], 0
    for turn in range(60):
        rec = np.array(env.buffers()["records"])
        x = F.decode(jnp.asarray(rec), L)
        mask = np.array(x.mask[0])
        me = int(rec.view(np.int32)[0, SCALARS["ID"]])
        length = int(env.buffers()["scalars"][0, SCALARS["LENGTH"]])
        if not mask[:4].all() or not mask[4:20].all():
            fallback = not mask[4:].any() and mask[:4].all()
            win = np.array(x.window[0])
            for a in range(20):
                if mask[a] or fallback:
                    continue
                if a >= 4:
                    # two conservative rules, not certain death (MODEL.md): a first step through a
                    # portal, and a second step onto the dragon's own body (its tail may move away)
                    first, second = (a - 4) // 4, (a - 4) % 4
                    r1, c1 = 3 + F.DR[first], 3 + F.DC[first]
                    t2 = (r1 + F.DR[second]) * 7 + c1 + F.DC[second]
                    if win[F.CENTER, 17 + first] or win[t2, 4] or length < 3:
                        continue
                    if not mask[first]:
                        continue          # the first step alone is already certain death: checked above
                kind, steps, split = F.action_command(a, length)
                assert me in replay(text, history, (kind, steps, split))["id"], \
                    f"masked action {a} did not kill dragon {me} at turn {turn}"
                checked += 1
        allowed = [a for a in range(4) if mask[a]] or [0]
        move = (0, [rng.choice(allowed)], 0)
        history.append(move)
        st = np.zeros((1, 2), np.uint8)
        st[0, 0] = move[1][0]
        env.step(np.zeros(1, np.uint8), st, np.ones(1, np.int32), np.zeros(1, np.int32), np.full(1, -1, np.int64))
        if env.buffers()["done"][0]:
            break
    assert checked > 0


def test_actor_and_critic_step():
    env, rec = env_records(list(maps.contest_maps().values()), n=8, steps=10)
    x = F.decode(jnp.asarray(rec), L)
    sc = env.buffers()["scalars"]

    def pool_bins(i):
        # bin of every cell of env i's board (64 for cells off the board)
        w, h_ = int(sc[i, SCALARS["WIDTH"]]), int(sc[i, SCALARS["HEIGHT"]])
        wc, hc = (w + 1) // 2, (h_ + 1) // 2
        cy, cx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
        bins_ = (cy * 8 // hc) * 8 + cx * 8 // wc
        return np.where((cy < hc) & (cx < wc), bins_, 64).reshape(-1)
    s = F.decode_state(jnp.asarray(rec), L)
    pa, pc = M.init_actor(jax.random.PRNGKey(0)), M.init_critic(jax.random.PRNGKey(1))
    h = jnp.zeros((8, M.HIDDEN))
    m = (jax.random.normal(jax.random.PRNGKey(2), (8, M.CELLS, M.MAP_CHANNELS)) * 0.1).astype(jnp.float16)
    bins = jnp.stack([jax.ops.segment_sum(m[i].astype(jnp.float32), np.array(pool_bins(i)), 65)[:64] for i in range(8)])
    logits, h2, m2, bins2 = M.actor_step(pa, x, h, m, bins)
    assert logits.shape == (8, 24) and h2.shape == h.shape and m2.shape == m.shape and m2.dtype == m.dtype
    assert bool(jnp.isfinite(logits).all() & jnp.isfinite(m2).all())
    # only the cells under the window change, and the bin sums stay the sums of the map
    changed = np.array((m2 != m).any(-1))
    for i in range(8):
        assert set(np.nonzero(changed[i])[0]) <= set(np.array(x.write_cell[i]).tolist())
        want = jax.ops.segment_sum(m2[i].astype(jnp.float32), np.array(pool_bins(i)), 65)[:64]
        assert np.allclose(np.array(bins2[i]), np.array(want), atol=1e-3)
    lp = M.masked_log_probs(logits, x.mask)
    assert np.allclose(np.exp(np.array(lp)).sum(1), 1, atol=1e-5)
    assert bool((jnp.where(x.mask, 0, jnp.exp(lp)) == 0).all())
    v = M.critic_value(pc, s, h2)
    assert v.shape == (8,)
    # gradients reach the map write through a second step
    def loss(p):
        _, hh, mm, bb = M.actor_step(p, x, h, m, bins)
        lg, _, _, _ = M.actor_step(p, x, hh, mm, bb)
        return lg.sum()
    g = jax.grad(loss)(pa)
    assert float(jnp.abs(g["write_content"]["w"]).sum()) > 0
