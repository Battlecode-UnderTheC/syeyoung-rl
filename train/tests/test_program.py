"""The exported programs' semantics: act keeps per-slot state, gather reads it
back, and train_step replays a chunk to exactly the log-probs act sampled."""

import collections

import jax
import jax.numpy as jnp
import numpy as np

from dragons_env import SCALARS, VecEnv, maps
from dragons_model import features as F
from dragons_model import program as P

CFG = P.Config(num_envs=8, pool=128, gather=8, chunk=12, minibatch=4)


def rollout(steps=40, seed=0):
    env = VecEnv(list(maps.contest_maps().values()), CFG.num_envs, seed=seed, threads=1, state=True)
    buf = env.buffers()
    act = jax.jit(P.make_act(CFG))
    gather = jax.jit(P.make_gather(CFG))
    params = P.init_params(jax.random.PRNGKey(0))
    pool_h = jnp.zeros((CFG.pool, 512))
    pool_m = jnp.zeros((CFG.pool, 1024, 16), P.MAP_DTYPE)
    pool_b = jnp.zeros((CFG.pool, 64, 16))
    key = jnp.array([0, 1], jnp.uint32)
    slots, next_slot = {}, 0
    game = [0] * CFG.num_envs
    traj = collections.defaultdict(list)       # (env, game, id) -> [(record, action, logp, start h, start m)]
    for _ in range(steps):
        rec = np.array(buf["records"])
        sc = buf["scalars"]
        s_idx = np.zeros(CFG.num_envs, np.int32)
        fresh = np.zeros(CFG.num_envs, bool)
        for i in range(CFG.num_envs):
            k = (i, game[i], int(sc[i, SCALARS["ID"]]))
            if k not in slots:
                slots[k] = next_slot
                next_slot += 1
                fresh[i] = True
            s_idx[i] = slots[k]
        h0, m0, b0 = gather(pool_h, pool_m, pool_b, jnp.asarray(s_idx), jnp.asarray(fresh))
        a, lp, v, pool_h, pool_m, pool_b, key = act(params, jnp.asarray(rec), jnp.asarray(s_idx), jnp.asarray(fresh),
                                                    pool_h, pool_m, pool_b, key)
        a = np.array(a)
        kind, st, ns, sp = (np.zeros(CFG.num_envs, np.uint8), np.zeros((CFG.num_envs, 2), np.uint8),
                            np.zeros(CFG.num_envs, np.int32), np.zeros(CFG.num_envs, np.int32))
        for i in range(CFG.num_envs):
            k = (i, game[i], int(sc[i, SCALARS["ID"]]))
            traj[k].append((rec[i], int(a[i]), float(lp[i]), np.array(h0[i]), np.array(m0[i]), np.array(b0[i])))
            kd, steps_, split = F.action_command(int(a[i]), int(sc[i, SCALARS["LENGTH"]]))
            kind[i], ns[i], sp[i] = kd, len(steps_), split
            st[i, :len(steps_)] = steps_
        env.step(kind, st, ns, sp, np.full(CFG.num_envs, -1, np.int64))
        for i in np.nonzero(buf["done"])[0]:
            game[i] += 1
    return params, traj


def test_replay_matches_sampled_log_probs():
    params, traj = rollout()
    long = [t for t in traj.values() if len(t) >= 3][:CFG.minibatch]
    assert len(long) == CFG.minibatch
    T, B, R = CFG.chunk, CFG.minibatch, CFG.layout()["bytes"]
    rec = np.zeros((T, B, R), np.uint8)
    valid = np.zeros((T, B), bool)
    act_ = np.zeros((T, B), np.int32)
    lp = np.zeros((T, B), np.float32)
    for b, t in enumerate(long):
        for s, (r, a, l, *_) in enumerate(t[:T]):
            rec[s, b], valid[s, b], act_[s, b], lp[s, b] = r, True, a, l
    h0 = np.stack([t[0][3] for t in long])
    m0 = np.stack([t[0][4] for t in long])
    b0 = np.stack([t[0][5] for t in long])
    train_step, adam = P.make_train_step(CFG)
    zeros = np.zeros((T, B), np.float32)
    f = lambda v: jnp.float32(v)
    new_params, _, metrics = jax.jit(train_step)(params, adam.init(params), rec, valid, act_, lp, zeros, zeros, h0, m0, b0,
                                                 f(0.0), f(0.2), f(0.01), f(0.5), f(1.0))
    m = dict(zip(P.METRICS, np.array(metrics)))
    assert m["samples"] == valid.sum()
    assert abs(m["approx_kl"]) < 1e-6 and m["clip_fraction"] == 0, m
    # lr 0 leaves the parameters alone
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree_util.tree_leaves(params),
                                                           jax.tree_util.tree_leaves(new_params)))


def test_train_step_moves_toward_advantage():
    params, traj = rollout(steps=24, seed=1)
    long = [t for t in traj.values() if len(t) >= 2][:CFG.minibatch]
    T, B, R = CFG.chunk, CFG.minibatch, CFG.layout()["bytes"]
    rec = np.zeros((T, B, R), np.uint8); valid = np.zeros((T, B), bool)
    act_ = np.zeros((T, B), np.int32); lp = np.zeros((T, B), np.float32)
    for b, t in enumerate(long):
        for s, (r, a, l, *_) in enumerate(t[:T]):
            rec[s, b], valid[s, b], act_[s, b], lp[s, b] = r, True, a, l
    h0 = np.stack([t[0][3] for t in long]); m0 = np.stack([t[0][4] for t in long]); b0 = np.stack([t[0][5] for t in long])
    adv = np.where(valid, 1.0, 0.0).astype(np.float32)     # every taken action was good
    train_step, adam = P.make_train_step(CFG)
    step = jax.jit(train_step)
    f = lambda v: jnp.float32(v)
    opt = adam.init(params)
    p = params
    for _ in range(3):
        p, opt, metrics = step(p, opt, rec, valid, act_, lp, adv, adv, h0, m0, b0, f(1e-3), f(0.2), f(0.0), f(0.5), f(1.0))
    m = dict(zip(P.METRICS, np.array(metrics)))
    # the taken actions became more likely: the surrogate (-policy_loss) rose above 1
    assert -m["policy_loss"] > 1.0 and m["approx_kl"] > 0, m
