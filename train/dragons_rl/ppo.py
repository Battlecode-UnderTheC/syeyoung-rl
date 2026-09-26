"""PPO over harvested matches: complete-match returns, chunked recurrent
replay, clipped policy loss with entropy terms, a critic on learner events.

Throughput choices (deviations from the v1 spec, by decision):
  * a chunk of CHUNK decisions starts from the h recorded during collection
    (stored-state replay), so every chunk of every lifetime trains at once in
    big device calls; gradients still flow through the CHUNK steps;
  * the critic is evaluated and trained only at learner decisions, from the
    learner's perspective (the opponent's value is its negative);
  * EPOCHS passes, one Adam step each, KL early stop as before.

Facts are rebuilt on the host from birth (cheap), so nothing but raw
observations is stored. Work is dealt to devices by cost; gradients are summed
on the host with full-batch denominators.
"""

from __future__ import annotations

import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import _native, codec, features, model

CHUNK = _native.FIELDS["CHUNK"]      # decisions per replayed chunk (the collector keeps h at chunk ends)
EPOCHS = 1
MINIBATCHES = 4               # Adam steps per epoch: random lifetime groups (and critic event groups)
# Self-inflicted deaths (kelp, own body, an unaffordable sprint) cost the dying dragon's
# decisions this much on top of the team result; head-to-head and enemy-body deaths can be
# tactics and cost nothing extra. A penalty, not a mask: a sacrifice stays possible.
DEATH_PENALTY = 0.3
SELF_INFLICTED = (0, 1, 4)    # engine causes: wall, self, no valid action
CRITIC_FRACTION = 0.25        # of learner events the critic trains on each update
CLIP = (0.9, 1.1)
ENTROPY = (0.005, 0.001)            # physical, sonar
SCALE = 512.0
KL_LIMIT = 0.02
GRAD_NORM = 0.5
ADAM = dict(b1=0.9, b2=0.999, eps=1e-5)
LR = 3e-4
# rows (x CHUNK steps) of one actor call: simple1's rows are fixed-size (the window
# and the grid), so a call is cut by rows alone
ACTOR_ROWS = 512
CRITIC_ROWS = 4096            # events of one critic call


# ---- optimiser ---------------------------------------------------------------------------

def adam_init(params):
    zeros = jax.tree.map(lambda p: np.zeros(p.shape, np.float32), params)
    return {"m": zeros, "v": jax.tree.map(np.copy, zeros), "t": 0}


def clip_by_norm(grads, limit: float):
    norm = math.sqrt(sum(float(np.sum(np.square(g, dtype=np.float64))) for g in jax.tree.leaves(grads)))
    scale = min(1.0, limit / norm) if norm > 0 else 1.0
    return jax.tree.map(lambda g: (g * scale).astype(np.float32), grads), norm


def adam_step(params, grads, state, rate: float):
    t = state["t"] + 1
    b1, b2, eps = ADAM["b1"], ADAM["b2"], ADAM["eps"]
    m = jax.tree.map(lambda m, g: (b1 * m + (1 - b1) * g).astype(np.float32), state["m"], grads)
    v = jax.tree.map(lambda v, g: (b2 * v + (1 - b2) * g * g).astype(np.float32), state["v"], grads)
    c1, c2 = 1 - b1 ** t, 1 - b2 ** t
    new = jax.tree.map(lambda p, m, v: (np.asarray(p) - rate * (m / c1) / (np.sqrt(v / c2) + eps)).astype(np.float32),
                       params, m, v)
    return new, {"m": m, "v": v, "t": t}


# ---- device programs -------------------------------------------------------------------------

# half: the tile MLP as in collection (first layer fused with the features, second in
# fp16); the loss is scaled so fp16 cotangents stay in range, and the gradient unscaled
LOSS_SCALE = 1024.0


@partial(jax.jit, static_argnames=("s_pad", "half"))
def actor_chunk(params, raw, h0, reset, forced, mask, logp_old, adv, denom, s_pad, half=False):
    """Gradient of one call's share of the actor loss. raw etc. are [rows, T, ...].
    A row packs consecutive chunks of several lifetimes: where reset[r, t], h
    restarts from the stored state h0[r, t] (a chunk's first decision)."""
    tm = lambda x: jnp.swapaxes(x, 0, 1)

    def loss_fn(params):
        # nothing before the GRU reads h: encode every decision at once, then run
        # only the GRU through time, then decode every decision at once
        R, T = mask.shape
        flat = jax.tree.map(lambda x: x.reshape(R * T, *x.shape[2:]), raw)
        # rematerialised in the backward pass: without, a call's activations outgrow the learner's GPU share
        enc = jax.checkpoint(lambda r: model.encode_fused(params, features.actor_inputs(r), half))
        fused, z_tile = jax.vmap(enc)(flat)
        fused_t = tm(fused.reshape(R, T, -1))

        def step(h, x):
            f, m, rs, h_start = x
            h = jnp.where(rs[:, None], h_start, h)
            h_new = model.GRU(params["gru"], f, h)
            return jnp.where(m[:, None], h_new, h), h_new

        _, h_new = lax.scan(step, jnp.zeros_like(h0[:, 0]), (fused_t, tm(mask), tm(reset), tm(h0)))
        h_new = tm(h_new).reshape(R * T, -1)

        @jax.checkpoint
        def dec(r, h, z, f):
            _, lp, hp, hs, _ = model.decode(params, h, z, features.actor_inputs(r), s_pad, forced=f)
            return lp, hp, hs

        logp, hp, hs = jax.vmap(dec)(flat, h_new, z_tile, jax.tree.map(lambda x: x.reshape(R * T, *x.shape[2:]), forced))
        logp, hp, hs = logp.reshape(R, T), hp.reshape(R, T), hs.reshape(R, T)
        # padded or excluded positions never reach an exp, so no inf * 0 in the backward pass
        ratio = jnp.exp(jnp.where(mask, logp - logp_old, 0.0))
        surrogate = jnp.minimum(ratio * adv, jnp.clip(ratio, *CLIP) * adv)
        per = jnp.where(mask, -surrogate - (ENTROPY[0] * jnp.where(mask, hp, 0.0) + ENTROPY[1] * jnp.where(mask, hs, 0.0)),
                        0.0)
        stats = {"clipped": jnp.sum(jnp.where(mask, jnp.abs(ratio - 1.0) > CLIP[1] - 1.0, False)),
                 "h_phys": jnp.sum(jnp.where(mask, hp, 0.0)), "h_sonar": jnp.sum(jnp.where(mask, hs, 0.0))}
        scale = LOSS_SCALE if half else 1.0
        return jnp.sum(per) / denom * scale, (logp, stats)

    grads, (logp, stats) = jax.grad(loss_fn, has_aux=True)(params)
    if half:
        grads = jax.tree.map(lambda g: g / LOSS_SCALE, grads)
    return grads, logp, stats


@jax.jit
def critic_chunk(params, raw, team, target, mask, denom):
    """Gradient of one call's share of the critic loss (learner perspective)."""
    def loss_fn(params):
        v = jax.vmap(lambda r, t: model.value(params, r, t))(raw, team)
        return jnp.sum(jnp.where(mask, (v - target) ** 2, 0.0)) / denom
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return grads, loss


# ---- the batch and its shards ------------------------------------------------------------------

def returns(winner: int, team: int) -> float:
    """G for `team`: +1 win, 0 draw, -1 loss"""
    return 0.0 if winner < 0 else (1.0 if winner == team else -1.0)


@dataclass
class Batch:
    """The full batch's per-decision quantities, computed once."""
    replay: object

    def build(self):
        r = self.replay
        L = r.lifetimes()
        D = r.decisions()
        self.n = L["decisions"].astype(np.int64)
        self.start = np.concatenate([[0], np.cumsum(self.n)[:-1]]).astype(np.int64)
        self.life_tiles = L["tiles"]
        events = [r.events(m) for m in range(r.num_matches)]
        life = D["lifetime"]
        self.match = L["match"][life]
        self.team = L["team"][life]
        self.event = D["event"]
        self.tiles = D["tiles"]
        winner = np.array([ev["winner"] for ev in events], np.int64)
        G = np.where(winner[self.match] < 0, 0.0, np.where(winner[self.match] == self.team, 1.0, -1.0))
        offset = np.concatenate([[0], np.cumsum([len(ev["dragon"]) for ev in events])[:-1]]).astype(np.int64)
        values = np.concatenate([ev["values"] for ev in events]) if events else np.zeros((0, 2), np.float32)
        v_old = values[offset[self.match] + self.event, self.team]
        self.target = G.astype(np.float32)
        died = np.isin(L["death"], SELF_INFLICTED)[life]
        self.adv = (G - v_old - DEATH_PENALTY * died).astype(np.float32)
        h = D["h"]
        first = D["step"] == 0
        self.h_before = np.zeros_like(h)
        self.h_before[~first] = h[np.nonzero(~first)[0] - 1]
        self.matches = len(np.unique(self.match))
        self.unit_limit = max(ev["unit_limit"] for ev in events) if events else 64
        self.logp_old = D["logp"]
        return self


@dataclass
class Shard:
    """One device's share of the batch: some learner lifetimes (actor) and
    some learner events (critic)."""
    packs: list
    device: object
    keep: list = field(default_factory=list)          # recorded-lifetime indices
    critic_rows: list = field(default_factory=list)   # decision rows of the full batch, for the critic

    def prepare(self, threads: int, full: Batch):
        self.full = full
        self.replay = _native.Replay(self.packs, threads, self.keep)
        self.lifetimes = self.replay.lifetimes()
        self.decisions = self.replay.decisions()
        n = self.lifetimes["decisions"]
        self.start = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
        rows = (np.concatenate([full.start[k] + np.arange(full.n[k]) for k in self.keep]) if self.keep
                else np.zeros(0, np.int64))
        self.adv = full.adv[rows]
        self.h_before = full.h_before[rows]
        self.forced = codec.from_native(self.decisions)
        self.skip = ~np.isfinite(self.adv)                           # excluded from the actor loss
        self.adv = np.where(self.skip, 0.0, self.adv).astype(np.float32)
        return self

    @property
    def num_decisions(self) -> int:
        return len(self.decisions["logp"])


def make_shards(packs, devices, threads: int):
    """Deal lifetimes (actor) and whole matches' learner events (critic) to
    devices, largest first, by cost (decisions, weighted by board)."""
    full = Batch(_native.Replay(packs, threads)).build()
    D = len(devices)
    keep = [[] for _ in range(D)]
    load = np.zeros(D)
    # a decision costs about the same on any board; big boards only somewhat more
    cost = full.n                              # simple1: a decision costs the same on any board
    for i in np.argsort(-cost):
        d = int(np.argmin(load))
        load[d] += cost[i]
        keep[d].append(int(i))
    by_match: dict[int, list] = {}
    for row, m in enumerate(full.match):
        by_match.setdefault(int(m), []).append(row)
    cload = np.zeros(D)
    crows = [[] for _ in range(D)]
    rng = np.random.default_rng()
    by_match = {m: [r for r in rows if rng.random() < CRITIC_FRACTION] for m, rows in by_match.items()}
    by_match = {m: rows for m, rows in by_match.items() if rows}
    for m, rows in sorted(by_match.items(), key=lambda kv: -len(kv[1])):
        d = int(np.argmin(cload))
        cload[d] += len(rows)
        crows[d] += rows
    per = max(1, threads // D)
    shards = [Shard(packs, devices[d], sorted(keep[d]), crows[d]).prepare(per, full) for d in range(D)]
    return shards, full


# ---- epochs ------------------------------------------------------------------------------------

def _pack(segments, T):
    """first-fit decreasing: segments (lifetime, start, length <= T) into rows of T steps"""
    free = []                         # per row, steps left
    rows = []
    by_free = [[] for _ in range(T + 1)]
    for seg in sorted(segments, key=lambda s: -s[2]):
        need = seg[2]
        r = None
        for f in range(need, T + 1):
            while by_free[f] and free[by_free[f][-1]] != f:
                by_free[f].pop()
            if by_free[f]:
                r = by_free[f].pop()
                break
        if r is None:
            r = len(rows)
            rows.append([])
            free.append(T)
        rows[r].append(seg)
        free[r] -= need
        by_free[free[r]].append(r)
    return rows


def _calls(shard: Shard, subset=None):
    """Every chunk of every lifetime (each starts from its stored h, so chunks are
    independent), packed into rows and cut into calls of ACTOR_ROWS rows.
    `subset`: only these lifetimes."""
    n = shard.lifetimes["decisions"]
    if subset is not None:
        keep = np.zeros(len(n), bool)
        keep[subset] = True
        n = np.where(keep, n, 0)
    segs = [(int(i), start, int(min(CHUNK, n[i] - start))) for i in np.nonzero(n > 0)[0]
            for start in range(0, int(n[i]), CHUNK)]
    rows = _pack(segs, CHUNK)
    for k in range(0, len(rows), ACTOR_ROWS):
        yield rows[k:k + ACTOR_ROWS]


def _rows(n: int, full: int) -> int:
    """rows of a call: full / 8, / 4, / 2 or full, so few compiled shapes"""
    return min(full, max(full // 8, 1 << (n - 1).bit_length()))


_SMALL = ("entity", "segment", "status", "prev_dirs", "prev_words", "inbox", "echo", "echo_len")
_add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))


class _Buffers:
    """two sets of host buffers per call shape: one call in flight while the next is encoded"""

    def __init__(self):
        self.sets, self.turn = {}, {}

    def get(self, rows, k_pad, e_pad):
        key = (rows, k_pad, e_pad)
        if key not in self.sets:
            self.sets[key] = [codec.actor_buffers(rows, k_pad, e_pad) for _ in range(2)]
            self.turn[key] = 0
        self.turn[key] ^= 1
        return self.sets[key][self.turn[key]]


_dumped = set()


def actor_epoch(shard: Shard, params, denom: float, subset=None):
    """Every learner decision of the shard (or of the lifetimes in `subset`) once:
    summed gradient, logp_new (NaN elsewhere), stats."""
    dev = shard.device
    p = jax.device_put(params, dev)
    shard.replay.reset()
    D = shard.decisions
    logp_new = np.full(len(D["logp"]), np.nan, np.float32)
    grads = None                                  # summed on the device, fetched once
    stats = {"clipped": 0.0, "h_phys": 0.0, "h_sonar": 0.0}
    prof = {"host_s": 0.0, "wait_s": 0.0, "calls": 0, "rows": 0, "slots": 0,
            "pack_s": 0.0, "encode_s": 0.0, "put_s": 0.0, "call_s": 0.0}
    if not hasattr(shard, "buffers"):
        shard.buffers = _Buffers()
    pending = []

    def drain(keep: int):
        while len(pending) > keep:
            t_w = time.perf_counter()
            logp, st, dec, valid = pending.pop(0)
            logp = np.asarray(logp)
            logp_new[dec[valid]] = logp[valid]
            for k in stats:
                stats[k] += float(st[k])
            prof["wait_s"] += time.perf_counter() - t_w

    T = CHUNK
    calls = _calls(shard, subset)
    while True:
        t_h = time.perf_counter()
        try:
            rows = next(calls)
        except StopIteration:
            break
        R = _rows(len(rows), ACTOR_ROWS)
        which = np.full((R, T), -1, np.int32)
        step = np.zeros((R, T), np.int32)
        reset = np.zeros((R, T), bool)
        for r, segs in enumerate(rows):
            t = 0
            for life, start, length in segs:
                which[r, t:t + length] = life
                step[r, t:t + length] = np.arange(start, start + length)
                reset[r, t] = True
                t += length
        valid = which >= 0
        dec = np.where(valid, shard.start[np.maximum(which, 0)] + step, 0)      # shard decision rows
        k_pad = codec.inbox_pad(int(D["inbox"][dec[valid]].max()))
        e_pad = codec.echo_pad(int(D["echo_bytes"][dec[valid]].max()))
        buf = shard.buffers.get(R * T, k_pad, e_pad)
        t_e = time.perf_counter()
        prof["pack_s"] += t_e - t_h
        shard.replay.encode_list(which.reshape(-1), step.reshape(-1), buf)     # straight into the packed layout
        prof["encode_s"] += time.perf_counter() - t_e
        flat_valid = valid.reshape(-1)
        first = int(np.argmax(flat_valid))
        pad = np.nonzero(~flat_valid)[0]
        if len(pad):                               # padding: a real decision's small arrays (masked out);
            for k in _SMALL:                       # its tile arrays hold whatever finite rows were there
                buf[k][pad] = buf[k][first]
        raw = {k: buf[k].reshape(R, T, *buf[k].shape[1:]) for k in codec.ACTOR_KEYS}
        dec_p = np.where(valid, dec, dec.reshape(-1)[first])
        forced = {k: v[dec_p] for k, v in shard.forced.items()}
        s_pad = codec.split_pad(int(D["length"][dec[valid]].max()))
        use = valid & ~shard.skip[dec_p]
        t_p = time.perf_counter()
        args = jax.device_put((raw, shard.h_before[dec_p], reset, forced, use, D["logp"][dec_p], shard.adv[dec_p]), dev)
        t_c = time.perf_counter()
        prof["put_s"] += t_c - t_p
        if os.environ.get("DUMP_ACTOR") and R not in _dumped:          # inputs of one call per row count, for profiling
            _dumped.add(R)
            with open(f"{os.environ['DUMP_ACTOR']}/actor_{R}.pkl", "wb") as f:
                pickle.dump((params, (raw, shard.h_before[dec_p], reset, forced, use, D["logp"][dec_p],
                                      shard.adv[dec_p]), denom, s_pad), f)
        g, logp, st = actor_chunk(p, *args, denom, s_pad)
        prof["call_s"] += time.perf_counter() - t_c
        grads = g if grads is None else _add(grads, g)
        pending.append((logp, st, dec, valid))
        prof["host_s"] += time.perf_counter() - t_h
        prof["calls"] += 1
        prof["rows"] += int(valid.sum())
        prof["slots"] += R * T
        drain(1)                              # one call in flight while the host prepares the next
    drain(0)
    stats["prof"] = prof
    grads = jax.tree.map(np.asarray, grads) if grads is not None else None
    return grads, logp_new, stats


def critic_epoch(shard: Shard, params, denom: float, rows=None):
    """Every learner event of the shard (or `rows` of it): summed gradient and summed squared error."""
    full = shard.full
    rows_used = shard.critic_rows if rows is None else rows
    if not len(rows_used):
        return None, 0.0
    dev = shard.device
    p = jax.device_put(params, dev)
    put = lambda x: jax.device_put(x, dev)
    d_pad = codec.pow2(2 * full.unit_limit, 8)
    by: dict[int, list] = {}
    for r in rows_used:
        by.setdefault(int(full.match[r]), []).append(int(r))
    grads, loss = None, 0.0
    pending = []

    def drain(keep: int):
        nonlocal grads, loss
        while len(pending) > keep:
            g, l = pending.pop(0)
            g = jax.tree.map(np.asarray, g)
            grads = g if grads is None else jax.tree.map(np.add, grads, g)
            loss += float(l)

    # simple1: rows are fixed-size (the board as cells), so a call is cut by rows alone
    cap = CRITIC_ROWS
    items = []
    for m, rows in by.items():
        rows = sorted(rows, key=lambda r: full.event[r])
        items += [(m, rows[i:i + cap]) for i in range(0, len(rows), cap)]
    batches, cur, used = [], [], 0
    for m, rows in items:
        if used + len(rows) > cap and cur:
            batches.append(cur)
            cur, used = [], 0
        cur.append((m, rows))
        used += len(rows)
    if cur:
        batches.append(cur)
    for batch in batches:
        total = sum(len(r) for _, r in batch)
        rows_n = _rows(total, cap)
        buf = codec.critic_buffers(rows_n, d_pad)
        team = np.zeros(rows_n, np.int32)
        target = np.zeros(rows_n, np.float32)
        at = 0
        for m, rows in batch:
            view = {k: v[at:at + len(rows)] for k, v in buf.items()}
            full.replay.encode_events(m, np.asarray([full.event[r] for r in rows], np.int32), view)
            team[at:at + len(rows)] = full.team[rows]
            target[at:at + len(rows)] = full.target[rows]
            at += len(rows)
        for a in buf.values():
            a[at:] = a[0]
        mask = np.arange(rows_n) < at
        pending.append(critic_chunk(p, {k: put(v) for k, v in buf.items()}, put(team), put(target), put(mask),
                                    denom))
        drain(1)
    drain(0)
    return grads, loss


def kl_estimate(logp_new, logp_old) -> float:
    x = logp_new.astype(np.float64) - logp_old.astype(np.float64)
    return float(np.mean(np.exp(x) - 1.0 - x))


_background = ThreadPoolExecutor(1)        # the critic's minibatch, while the actor's runs


def _per_device(shards, fn):
    with ThreadPoolExecutor(len(shards)) as pool:
        return list(pool.map(fn, shards))


def _sum(trees):
    trees = [t for t in trees if t is not None]
    out = trees[0]
    for t in trees[1:]:
        out = jax.tree.map(np.add, out, t)
    return out


def update(actor, critic, actor_opt, critic_opt, shards, full: Batch, rate: float = LR, log=print):
    """One PPO update over a harvested batch: EPOCHS passes of MINIBATCHES Adam
    steps each (random lifetime groups for the actor, random event groups for the
    critic), each loss a mean over its minibatch. Returns new weights, optimiser
    states and statistics."""
    decisions = sum(s.num_decisions for s in shards)
    stats = {"lr": rate, "decisions": decisions, "matches": full.matches, "actor_steps": 0, "critic_steps": 0,
             "kl": [], "stopped": False, "nonfinite_advantages": int(sum(s.skip.sum() for s in shards)), "excluded": 0}
    if stats["nonfinite_advantages"]:
        log(f"{stats['nonfinite_advantages']} decisions with a nonfinite V_old excluded")
    rng = np.random.default_rng()
    K = MINIBATCHES
    timing = {"actor_s": 0.0, "actor_host_s": 0.0, "actor_wait_s": 0.0, "critic_s": 0.0, "actor_calls": 0,
              "tile_steps": 0}
    h_sum = {"h_phys": 0.0, "h_sonar": 0.0}
    clipped = 0.0
    vmse = 0.0
    actor_on = True
    for epoch in range(EPOCHS):
        life_groups = [np.array_split(rng.permutation(len(sh.lifetimes["decisions"])), K) for sh in shards]
        event_groups = [np.array_split(rng.permutation(np.asarray(sh.critic_rows, np.int64)), K) for sh in shards]
        for k in range(K):
            # the critic's minibatch runs alongside the actor's (independent losses)
            rows_k = [g[k] for g in event_groups]
            m_k = sum(len(r) for r in rows_k)
            critic_job = None
            if m_k:
                critic_now = critic
                t_c = time.perf_counter()
                critic_job = _background.submit(_per_device, list(zip(shards, rows_k)),
                                                lambda x: critic_epoch(x[0], critic_now, float(m_k), x[1]))
            if actor_on:
                subsets = [g[k] for g in life_groups]
                n_k = sum(int(sh.lifetimes["decisions"][sub].sum()) for sh, sub in zip(shards, subsets))
                if n_k:
                    run = lambda: _per_device(list(zip(shards, subsets)),
                                              lambda x: actor_epoch(x[0], actor, float(n_k), x[1]))
                    t_a = time.perf_counter()
                    results = run()
                    timing["actor_s"] += time.perf_counter() - t_a
                    for r in results:
                        pr = r[2]["prof"]
                        timing["actor_host_s"] += pr["host_s"] / len(results)
                        timing["actor_wait_s"] += pr["wait_s"] / len(results)
                        timing["actor_calls"] += pr["calls"]
                        timing["tile_steps"] += pr["slots"]
                        for key in ("pack_s", "encode_s", "put_s", "call_s"):
                            timing[key] = timing.get(key, 0.0) + pr[key] / len(results)
                    bad_any = False
                    for sh, r in zip(shards, results):
                        lp_new, lp_old = r[1], sh.decisions["logp"]
                        have = ~np.isnan(lp_new)
                        bad = have & (~np.isfinite(lp_new) | ~np.isfinite(lp_old)) & ~sh.skip
                        if bad.any():
                            at = np.nonzero(bad)[0]
                            log(f"{len(at)} nonfinite decisions excluded; first: " + "; ".join(
                                f"logp_old {lp_old[i]:.3g} new {lp_new[i]:.3g} kind/steps/split/len/tiles "
                                f"{[int(sh.decisions[f][i]) for f in ('kind', 'nsteps', 'split', 'length', 'tiles')]}"
                                for i in at[:5]))
                            sh.skip |= bad
                            stats["excluded"] += int(len(at))
                            bad_any = True
                    if bad_any:
                        results = run()
                    lp_new = np.concatenate([r[1] for r in results])
                    lp_old = np.concatenate([sh.decisions["logp"] for sh in shards])
                    ok = np.isfinite(lp_new) & np.isfinite(lp_old)
                    kl = kl_estimate(lp_new[ok], lp_old[ok])
                    stats["kl"].append(kl)
                    for key in h_sum:
                        h_sum[key] += sum(r[2][key] for r in results)
                    clipped += sum(r[2]["clipped"] for r in results)
                    grads = _sum([r[0] for r in results])
                    if not np.isfinite(kl) or not all(np.isfinite(g).all() for g in jax.tree.leaves(grads)):
                        raise FloatingPointError(f"nonfinite actor update at epoch {epoch} minibatch {k} (kl {kl})")
                    if kl > KL_LIMIT:
                        actor_on = False
                        stats["stopped"] = True
                    else:
                        grads, norm = clip_by_norm(grads, GRAD_NORM)
                        stats["actor_grad_norm"] = norm
                        actor, actor_opt = adam_step(actor, grads, actor_opt, rate)
                        stats["actor_steps"] += 1
            if critic_job is not None:
                cres = critic_job.result()
                timing["critic_s"] += time.perf_counter() - t_c
                cgrads = _sum([r[0] for r in cres])
                vmse += sum(r[1] for r in cres) * m_k
                if not all(np.isfinite(g).all() for g in jax.tree.leaves(cgrads)):
                    raise FloatingPointError(f"nonfinite critic gradient at epoch {epoch} minibatch {k}")
                cgrads, cnorm = clip_by_norm(cgrads, GRAD_NORM)
                stats["critic_grad_norm"] = cnorm
                critic, critic_opt = adam_step(critic, cgrads, critic_opt, rate)
                stats["critic_steps"] += 1
        log(f"epoch {epoch}: {stats['actor_steps']} actor steps, {stats['critic_steps']} critic steps, "
            f"kl {max(stats['kl']) if stats['kl'] else 0:.4f}")
    stats.update({key: v / max(1, decisions) for key, v in h_sum.items()})
    stats["timing"] = timing
    log(f"  training: actor {timing['actor_s']:.1f}s (host {timing['actor_host_s']:.1f} = pack {timing.get('pack_s', 0):.1f} "
        f"encode {timing.get('encode_s', 0):.1f} put {timing.get('put_s', 0):.1f} call {timing.get('call_s', 0):.1f}; "
        f"wait {timing['actor_wait_s']:.1f}, "
        f"{timing['actor_calls']} calls, {timing['tile_steps'] / max(1, decisions):.2f} slots/decision), "
        f"critic {timing['critic_s']:.1f}s")
    stats["clip_fraction"] = clipped / max(1, decisions)
    stats["value_mse"] = vmse / max(1, sum(len(s.critic_rows) for s in shards))
    return actor, critic, actor_opt, critic_opt, stats
