"""Spec section 9.2-9.4: complete-return advantages, recurrent replay from
birth, full-batch losses, the KL rule and one Adam step per network per epoch.

Work is split into shards, one per device: a shard is the matches that device
collected, and every lifetime stays on its shard so its chunks run in order.
Gradients are summed over shards on the host; the loss denominators are the
full batch's (512 M for the actor, |T| for the critic), so the sum is the
full-batch gradient. A microbatch is four 64-decision chunks of independent
lifetimes; a device call runs several microbatches, which changes nothing but
speed. Nothing is ever dropped to fit memory: a call just holds fewer chunks.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import _native, codec, model

CHUNK = 64
MICROBATCH = 4
CLIP = (0.9, 1.1)
ENTROPY = (0.005, 0.001)            # physical, sonar
SCALE = 512.0
KL_LIMIT = 0.02
GRAD_NORM = 0.5
ADAM = dict(b1=0.9, b2=0.999, eps=1e-5)
UPDATES = 2000
LR = (3e-4, 3e-5)
# tile-chunk budget of one device call: chunks per call = BUDGET / n_pad, in whole microbatches
ACTOR_BUDGET = 64 * 1024
CRITIC_BUDGET = 32 * 1024


def lr(u: int) -> float:
    return LR[0] + (LR[1] - LR[0]) * u / (UPDATES - 1)


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

@partial(jax.jit, static_argnames=("s_pad",))
def actor_chunk(params, raw, h0, forced, mask, logp_old, adv, denom, s_pad):
    """Gradient of one call's share of the actor loss. raw etc. are [rows, CHUNK, ...];
    h0 [rows, 192] is the carried (detached) state. Returns (grads, h_final, logp_new, stats)."""
    tm = lambda x: jnp.swapaxes(x, 0, 1)
    raw_t = jax.tree.map(tm, raw)
    forced_t = jax.tree.map(tm, forced)
    mask_t = tm(mask)

    def loss_fn(params):
        @jax.checkpoint
        def step(h, xs):
            r, f, m = xs
            logp, hp, hs, h_new = jax.vmap(model.score, in_axes=(None, 0, 0, 0, None))(params, r, h, f, s_pad)
            return jnp.where(m[:, None], h_new, h), (logp, hp, hs)

        h_final, (logp, hp, hs) = lax.scan(step, h0, (raw_t, forced_t, mask_t))
        logp, hp, hs = tm(logp), tm(hp), tm(hs)
        ratio = jnp.exp(logp - logp_old)
        surrogate = jnp.minimum(ratio * adv, jnp.clip(ratio, *CLIP) * adv)
        per = jnp.where(mask, -surrogate - (ENTROPY[0] * hp + ENTROPY[1] * hs), 0.0)
        stats = {"clipped": jnp.sum(jnp.where(mask, jnp.abs(ratio - 1.0) > CLIP[1] - 1.0, False)),
                 "h_phys": jnp.sum(jnp.where(mask, hp, 0.0)), "h_sonar": jnp.sum(jnp.where(mask, hs, 0.0))}
        return jnp.sum(per) / denom, (h_final, logp, stats)

    grads, (h_final, logp, stats) = jax.grad(loss_fn, has_aux=True)(params)
    return grads, lax.stop_gradient(h_final), logp, stats


@jax.jit
def critic_chunk(params, raw, target, mask, denom):
    """Gradient of one call's share of the critic loss; target [rows, 2] is G per team."""
    def loss_fn(params):
        v = jax.vmap(lambda r: jnp.stack([model.value(params, r, 0), model.value(params, r, 1)]))(raw)
        return jnp.sum(jnp.where(mask[:, None], (v - target) ** 2, 0.0)) / denom
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return grads, loss


# ---- a shard ---------------------------------------------------------------------------------

@dataclass
class Shard:
    """One device's share of one collector: some of its learner lifetimes
    (replayed from birth) and some ranges of its events (for the critic)."""
    collector: object
    device: object
    keep: list = field(default_factory=list)       # recorded-lifetime indices
    ranges: list = field(default_factory=list)     # (match, first event, count)
    replay: object = None
    decisions: dict = field(default_factory=dict)
    lifetimes: dict = field(default_factory=dict)
    start: np.ndarray = None           # first decision row of each lifetime
    adv: np.ndarray = None             # per decision
    events: list = field(default_factory=list)   # every match of the collector: values, winner, tiles

    def prepare(self, threads: int):
        self.replay = _native.Replay(self.collector, threads, self.keep)
        self.lifetimes = self.replay.lifetimes()
        self.decisions = self.replay.decisions()
        n = self.lifetimes["decisions"]
        self.start = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
        self.events = [self.replay.events(m) for m in range(self.replay.num_matches)]
        for ev in self.events:
            if not ev["finished"]:
                raise RuntimeError("an unfinished match reached training")
        life = self.decisions["lifetime"]
        match = self.lifetimes["match"][life]
        team = self.lifetimes["team"][life]
        winner = np.array([ev["winner"] for ev in self.events], np.int64)
        G = np.where(winner[match] < 0, 0.0, np.where(winner[match] == team, 1.0, -1.0))
        offset = np.concatenate([[0], np.cumsum([len(ev["dragon"]) for ev in self.events])[:-1]]).astype(np.int64)
        values = np.concatenate([ev["values"] for ev in self.events]) if self.events else np.zeros((0, 2), np.float32)
        v_old = values[offset[match] + self.decisions["event"], team]
        self.adv = (G - v_old).astype(np.float32)
        self.forced = codec.from_native(self.decisions)
        return self

    @property
    def num_events(self) -> int:
        return sum(c for _, _, c in self.ranges)


EVENT_PIECE = 1024


def make_shards(collectors, devices, threads: int) -> list[Shard]:
    """Deals every learner lifetime and every event range to the devices,
    greedily by cost (decisions x board bucket), largest first, so one huge
    match cannot leave the other devices idle."""
    D = len(devices)
    items = []
    for ci, c in enumerate(collectors):
        r = c.recorded_sizes()
        items += [(int(n) * codec.tile_bucket(int(t)), ci, li) for li, (n, t) in enumerate(zip(r["decisions"], r["tiles"]))]
        res = c.results()
        for m, (E, t) in enumerate(zip(res["events"], res["tiles"])):
            for s0 in range(0, int(E), EVENT_PIECE):
                items.append((min(EVENT_PIECE, int(E) - s0) * codec.tile_bucket(int(t)), ci, (m, s0, min(EVENT_PIECE, int(E) - s0))))
    load = np.zeros(D)
    parts: dict = {}
    for cost, ci, what in sorted(items, key=lambda x: -x[0]):
        d = int(np.argmin(load))
        load[d] += cost
        keep, ranges = parts.setdefault((ci, d), ([], []))
        (ranges if isinstance(what, tuple) else keep).append(what)
    per = max(1, threads // max(1, len(parts)))
    return [Shard(collectors[ci], devices[d], sorted(keep), sorted(ranges)).prepare(per)
            for (ci, d), (keep, ranges) in sorted(parts.items())]


def returns(winner: int, team: int) -> float:
    """G for `team`: +1 win, 0 draw, -1 loss (gamma = 1, no other reward)"""
    return 0.0 if winner < 0 else (1.0 if winner == team else -1.0)


def _chunk_calls(shard: Shard, wave: int):
    """the lifetimes with a chunk `wave`, grouped by board bucket and cut into calls"""
    n = shard.lifetimes["decisions"]
    tiles = shard.decisions["tiles"][shard.start]
    have = np.nonzero(n > wave * CHUNK)[0]
    groups: dict[int, list[int]] = {}
    for i in have:
        groups.setdefault(codec.tile_bucket(int(tiles[i])), []).append(int(i))
    for n_pad, lives in groups.items():
        per = MICROBATCH
        while per * 4 <= ACTOR_BUDGET // n_pad:          # the largest bucket size within the budget
            per *= 4
        for k in range(0, len(lives), per):
            yield n_pad, lives[k:k + per]


def actor_epoch(shard: Shard, params, denom: float):
    """One pass over every learner decision of the shard: summed gradient,
    logp_new per decision and statistics."""
    dev = shard.device
    p = jax.device_put(params, dev)
    shard.replay.reset()
    D = shard.decisions
    n = shard.lifetimes["decisions"]
    H = np.zeros((len(n), codec.HIDDEN), np.float32)
    logp_new = np.zeros(len(D["logp"]), np.float32)
    grads = None
    stats = {"clipped": 0.0, "h_phys": 0.0, "h_sonar": 0.0}
    waves = int(math.ceil(n.max() / CHUNK)) if len(n) else 0
    for wave in range(waves):
        for n_pad, lives in _chunk_calls(shard, wave):
            rows = codec.geometric(len(lives), MICROBATCH, 4)
            lives_a = np.asarray(lives)
            count = np.minimum(n[lives_a] - wave * CHUNK, CHUNK)
            # the scan only runs as far as this call's longest chunk (a coarse bucket, at most
            # CHUNK); steps past a lifetime's chunk are masked either way, so this is only speed
            T = min(CHUNK, codec.geometric(int(count.max()), 4, 4))
            idx = shard.start[lives_a][:, None] + wave * CHUNK + np.arange(T)[None, :]    # decision rows
            valid = np.arange(T)[None, :] < count[:, None]
            idx = np.where(valid, idx, shard.start[lives_a][:, None])
            k_pad = codec.inbox_pad(int(D["inbox"][idx][valid].max()))
            e_pad = codec.echo_pad(int(D["echo_bytes"][idx][valid].max()))
            s_pad = codec.split_pad(int(D["length"][idx][valid].max()))
            buf = codec.actor_buffers(rows * T, n_pad, k_pad, e_pad)
            shard.replay.encode_chunk(lives_a.astype(np.int32), np.full(len(lives), wave * CHUNK, np.int32), T, buf)
            pad = rows - len(lives)
            def rows_of(a):
                a = a.reshape(rows, T, *a.shape[1:])
                return a
            raw = {k: rows_of(buf[k]) for k in codec.ACTOR_KEYS}
            # padding (steps past a lifetime's end, rows past the last lifetime) copies
            # a real decision and is masked out, so it never computes a NaN to backpropagate
            for k in raw:
                for r_, c_ in enumerate(count):
                    raw[k][r_, c_:] = raw[k][r_, 0]
                raw[k][len(lives):] = raw[k][0]
            mask = np.zeros((rows, T), bool)
            mask[:len(lives)] = valid
            full_idx = np.zeros((rows, T), np.int64)
            full_idx[:len(lives)] = idx
            full_idx[len(lives):] = idx[0]
            forced = {k: v[full_idx] for k, v in shard.forced.items()}
            h0 = np.zeros((rows, codec.HIDDEN), np.float32)
            h0[:len(lives)] = H[lives_a]
            put = lambda x: jax.device_put(x, dev)
            g, h_final, logp, st = actor_chunk(
                p, {k: put(v) for k, v in raw.items()}, put(h0), {k: put(v) for k, v in forced.items()}, put(mask),
                put(D["logp"][full_idx]), put(shard.adv[full_idx]), denom, s_pad)
            H[lives_a] = np.asarray(h_final)[:len(lives)]
            logp = np.asarray(logp)
            logp_new[idx[valid]] = logp[:len(lives)][valid]
            g = jax.tree.map(np.asarray, g)
            grads = g if grads is None else jax.tree.map(np.add, grads, g)
            for k in stats:
                stats[k] += float(st[k])
    return grads, logp_new, stats


def critic_epoch(shard: Shard, params, denom: float):
    """Every team-event of the shard, packed across matches into fixed-size calls per board bucket."""
    dev = shard.device
    p = jax.device_put(params, dev)
    d_pad = codec.pow2(2 * max(e["unit_limit"] for e in shard.events), 8) if shard.events else 8
    ranges: dict[int, list] = {}
    for m, first, count in shard.ranges:
        ranges.setdefault(codec.tile_bucket(int(shard.events[m]["tiles"])), []).append((m, first, count))
    grads, loss = None, 0.0
    put = lambda x: jax.device_put(x, dev)
    if not shard.ranges:
        return None, 0.0
    for n_pad, todo in ranges.items():
        total = sum(c for _, _, c in todo)
        cap = max(16, CRITIC_BUDGET // n_pad)
        rows = min(codec.geometric(total, 16, 4), codec.geometric(cap, 16, 4))
        todo = list(todo)
        while todo:
            buf = codec.critic_buffers(rows, n_pad, d_pad)
            target = np.zeros((rows, 2), np.float32)
            used = 0
            while todo and used < rows:
                m, start, count = todo.pop(0)
                take = min(count, rows - used)
                view = {k: v[used:used + take] for k, v in buf.items()}
                shard.replay.encode_events(m, start, take, view)
                w = shard.events[m]["winner"]
                target[used:used + take] = [returns(w, 0), returns(w, 1)]
                used += take
                if take < count:
                    todo.insert(0, (m, start + take, count - take))
            for a in buf.values():
                a[used:] = a[0]
            mask = np.arange(rows) < used
            g, l = critic_chunk(p, {k: put(v) for k, v in buf.items()}, put(target), put(mask), denom)
            g = jax.tree.map(np.asarray, g)
            grads = g if grads is None else jax.tree.map(np.add, grads, g)
            loss += float(l)
    return grads, loss


def kl_estimate(logp_new, logp_old) -> float:
    x = logp_new.astype(np.float64) - logp_old.astype(np.float64)
    return float(np.mean(np.exp(x) - 1.0 - x))


def _per_device(shards, fn):
    """fn(shard) for every shard, the shards of one device in turn, devices in parallel;
    results in shard order"""
    by_dev: dict = {}
    for i, sh in enumerate(shards):
        by_dev.setdefault(id(sh.device), []).append(i)
    out = [None] * len(shards)

    def run(idx):
        for i in idx:
            out[i] = fn(shards[i])

    with ThreadPoolExecutor(len(by_dev)) as pool:
        list(pool.map(run, by_dev.values()))
    return out


def update(actor, critic, actor_opt, critic_opt, shards: list[Shard], u: int, matches: int, log=print):
    """Section 9.4 for one collected batch. Returns new weights, optimiser
    states and the logged statistics."""
    denom_actor = SCALE * matches
    denom_critic = 2.0 * sum(s.num_events for s in shards)
    decisions = sum(len(s.decisions["logp"]) for s in shards)
    logp_old = np.concatenate([s.decisions["logp"] for s in shards])
    rate = lr(u)
    stats = {"lr": rate, "decisions": decisions, "events": denom_critic / 2, "actor_steps": 0, "kl": [], "stopped": False}
    actor_on = True
    with ThreadPoolExecutor(2) as outer:
        for epoch in range(3):
            critic_now = critic
            future_c = outer.submit(_per_device, shards, lambda s: critic_epoch(s, critic_now, denom_critic))
            if actor_on and decisions:
                results = _per_device(shards, lambda s: actor_epoch(s, actor, denom_actor))
                grads = _sum([r[0] for r in results if r[0] is not None])
                logp_new = np.concatenate([r[1] for r in results])
                kl = kl_estimate(logp_new, logp_old)
                stats["kl"].append(kl)
                if epoch == 0:
                    ent = {k: sum(r[2][k] for r in results) / decisions for k in ("h_phys", "h_sonar")}
                    stats.update(ent)
                stats[f"clip_fraction_{epoch}"] = sum(r[2]["clipped"] for r in results) / decisions
                if not np.isfinite(kl) or not all(np.isfinite(g).all() for g in jax.tree.leaves(grads)):
                    raise FloatingPointError(f"nonfinite actor update at update {u} epoch {epoch} (kl {kl})")
                if kl > KL_LIMIT:
                    actor_on = False
                    stats["stopped"] = True
                else:
                    grads, norm = clip_by_norm(grads, GRAD_NORM)
                    stats[f"actor_grad_norm_{epoch}"] = norm
                    actor, actor_opt = adam_step(actor, grads, actor_opt, rate)
                    stats["actor_steps"] += 1
            cres = future_c.result()
            log(f"epoch {epoch}: {stats['actor_steps']} actor steps, "
                f"kl {stats['kl'][-1] if stats['kl'] else 0:.4f}")
            cgrads = _sum([r[0] for r in cres if r[0] is not None])
            closs = sum(r[1] for r in cres)
            stats[f"value_mse_{epoch}"] = closs
            if not all(np.isfinite(g).all() for g in jax.tree.leaves(cgrads)):
                raise FloatingPointError(f"nonfinite critic gradient at update {u} epoch {epoch}")
            cgrads, cnorm = clip_by_norm(cgrads, GRAD_NORM)
            stats[f"critic_grad_norm_{epoch}"] = cnorm
            critic, critic_opt = adam_step(critic, cgrads, critic_opt, rate)
    return actor, critic, actor_opt, critic_opt, stats


def _sum(trees):
    out = trees[0]
    for t in trees[1:]:
        out = jax.tree.map(np.add, out, t)
    return out
