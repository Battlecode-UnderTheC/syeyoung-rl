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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import _native, codec, model

CHUNK = 16
EPOCHS = 2
CLIP = (0.9, 1.1)
ENTROPY = (0.005, 0.001)            # physical, sonar
SCALE = 512.0
KL_LIMIT = 0.02
GRAD_NORM = 0.5
ADAM = dict(b1=0.9, b2=0.999, eps=1e-5)
LR = 3e-4
# rows x n_pad of one device call. Memory scales with rows x board per scan
# step (steps are rematerialised), not with the chunk length.
ACTOR_BUDGET = 1 << 17
CRITIC_BUDGET = 1 << 17


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
    """Gradient of one call's share of the actor loss. raw etc. are [rows, T, ...];
    h0 [rows, 192] the stored state before each chunk."""
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

        _, (logp, hp, hs) = lax.scan(step, h0, (raw_t, forced_t, mask_t))
        logp, hp, hs = tm(logp), tm(hp), tm(hs)
        ratio = jnp.exp(logp - logp_old)
        surrogate = jnp.minimum(ratio * adv, jnp.clip(ratio, *CLIP) * adv)
        per = jnp.where(mask, -surrogate - (ENTROPY[0] * hp + ENTROPY[1] * hs), 0.0)
        stats = {"clipped": jnp.sum(jnp.where(mask, jnp.abs(ratio - 1.0) > CLIP[1] - 1.0, False)),
                 "h_phys": jnp.sum(jnp.where(mask, hp, 0.0)), "h_sonar": jnp.sum(jnp.where(mask, hs, 0.0))}
        return jnp.sum(per) / denom, (logp, stats)

    grads, (logp, stats) = jax.grad(loss_fn, has_aux=True)(params)
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
        self.adv = (G - v_old).astype(np.float32)
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
        return self

    @property
    def num_decisions(self) -> int:
        return len(self.decisions["logp"])


def make_shards(packs, devices, threads: int):
    """Deal lifetimes (actor) and whole matches' learner events (critic) to
    devices, largest first, by cost (decisions x board bucket)."""
    full = Batch(_native.Replay(packs, threads)).build()
    D = len(devices)
    keep = [[] for _ in range(D)]
    load = np.zeros(D)
    cost = full.n * np.array([codec.tile_bucket(int(t)) for t in full.life_tiles], np.int64)
    for i in np.argsort(-cost):
        d = int(np.argmin(load))
        load[d] += cost[i]
        keep[d].append(int(i))
    by_match: dict[int, list] = {}
    for row, m in enumerate(full.match):
        by_match.setdefault(int(m), []).append(row)
    cload = np.zeros(D)
    crows = [[] for _ in range(D)]
    for m, rows in sorted(by_match.items(), key=lambda kv: -len(kv[1])):
        d = int(np.argmin(cload))
        cload[d] += len(rows) * codec.tile_bucket(int(full.tiles[rows[0]]))
        crows[d] += rows
    per = max(1, threads // D)
    shards = [Shard(packs, devices[d], sorted(keep[d]), crows[d]).prepare(per, full) for d in range(D)]
    return shards, full


# ---- epochs ------------------------------------------------------------------------------------

def _calls(shard: Shard, wave: int):
    n = shard.lifetimes["decisions"]
    tiles = shard.lifetimes["tiles"]
    have = np.nonzero(n > wave * CHUNK)[0]
    groups: dict[int, list[int]] = {}
    for i in have:
        groups.setdefault(codec.tile_bucket(int(tiles[i])), []).append(int(i))
    for n_pad, lives in groups.items():
        per = max(4, ACTOR_BUDGET // n_pad)
        per = 1 << (per.bit_length() - 1)
        for k in range(0, len(lives), per):
            yield n_pad, lives[k:k + per]


def actor_epoch(shard: Shard, params, denom: float):
    """Every learner decision of the shard once: summed gradient, logp_new, stats."""
    dev = shard.device
    p = jax.device_put(params, dev)
    put = lambda x: jax.device_put(x, dev)
    shard.replay.reset()
    D = shard.decisions
    n = shard.lifetimes["decisions"]
    logp_new = np.zeros(len(D["logp"]), np.float32)
    grads = None
    stats = {"clipped": 0.0, "h_phys": 0.0, "h_sonar": 0.0}
    waves = int(math.ceil(n.max() / CHUNK)) if len(n) else 0
    pending = []
    prof = {"host_s": 0.0, "wait_s": 0.0, "calls": 0, "rows": 0, "slots": 0}

    def drain(keep: int):
        nonlocal grads
        while len(pending) > keep:
            t_w = time.perf_counter()
            g, logp, st, idx, valid, nl = pending.pop(0)
            logp = np.asarray(logp)
            logp_new[idx[valid]] = logp[:nl][valid]
            g = jax.tree.map(np.asarray, g)
            grads = g if grads is None else jax.tree.map(np.add, grads, g)
            for k in stats:
                stats[k] += float(st[k])
            prof["wait_s"] += time.perf_counter() - t_w

    for wave in range(waves):
        for n_pad, lives in _calls(shard, wave):
            t_h = time.perf_counter()
            lives_a = np.asarray(lives)
            rows = codec.geometric(len(lives), 4, 4)
            count = np.minimum(n[lives_a] - wave * CHUNK, CHUNK)
            T = min(CHUNK, codec.geometric(int(count.max()), 4, 2))
            idx = shard.start[lives_a][:, None] + wave * CHUNK + np.arange(T)[None, :]
            valid = np.arange(T)[None, :] < count[:, None]
            idx = np.where(valid, idx, shard.start[lives_a][:, None] + wave * CHUNK)
            buf = codec.actor_buffers(rows * T, n_pad, codec.inbox_pad(int(D["inbox"][idx][valid].max())),
                                      codec.echo_pad(int(D["echo_bytes"][idx][valid].max())))
            shard.replay.encode_chunk(lives_a.astype(np.int32), np.full(len(lives), wave * CHUNK, np.int32), T, buf)
            mask = np.zeros((rows, T), bool)
            mask[:len(lives)] = valid
            # padding (steps past a chunk's end, rows past the last lifetime) copies a
            # real decision and is masked out: take index 0 of row 0 there
            src_r = np.where(mask, np.arange(rows)[:, None], 0)
            src_t = np.where(mask, np.arange(T)[None, :], 0)
            flat = (src_r * T + src_t).reshape(-1)
            raw = {k: buf[k][flat].reshape(rows, T, *buf[k].shape[1:]) for k in codec.ACTOR_KEYS}
            full_idx = np.zeros((rows, T), np.int64)
            full_idx[:len(lives)] = idx
            full_idx[len(lives):] = idx[0]
            forced = {k: v[full_idx] for k, v in shard.forced.items()}
            h0 = shard.h_before[full_idx[:, 0]]
            s_pad = codec.split_pad(int(D["length"][idx][valid].max()))
            g, logp, st = actor_chunk(p, {k: put(v) for k, v in raw.items()}, put(h0),
                                      {k: put(v) for k, v in forced.items()}, put(mask),
                                      put(D["logp"][full_idx]), put(shard.adv[full_idx]), denom, s_pad)
            pending.append((g, logp, st, idx, valid, len(lives)))
            prof["host_s"] += time.perf_counter() - t_h
            prof["calls"] += 1
            prof["rows"] += int(valid.sum())
            prof["slots"] += rows * T * n_pad
            drain(1)                              # one call in flight while the host encodes the next
    drain(0)
    stats["prof"] = prof
    return grads, logp_new, stats


def critic_epoch(shard: Shard, params, denom: float):
    """Every learner event of the shard: summed gradient and summed squared error."""
    full = shard.full
    if not shard.critic_rows:
        return None, 0.0
    dev = shard.device
    p = jax.device_put(params, dev)
    put = lambda x: jax.device_put(x, dev)
    d_pad = codec.pow2(2 * full.unit_limit, 8)
    by: dict[int, dict[int, list]] = {}
    for r in shard.critic_rows:
        by.setdefault(codec.tile_bucket(int(full.tiles[r])), {}).setdefault(int(full.match[r]), []).append(int(r))
    grads, loss = None, 0.0
    pending = []

    def drain(keep: int):
        nonlocal grads, loss
        while len(pending) > keep:
            g, l = pending.pop(0)
            g = jax.tree.map(np.asarray, g)
            grads = g if grads is None else jax.tree.map(np.add, grads, g)
            loss += float(l)

    for n_pad, matches in by.items():
        cap = max(16, CRITIC_BUDGET // n_pad)
        items = []
        for m, rows in matches.items():
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
            rows_n = codec.geometric(total, 16, 4)
            buf = codec.critic_buffers(rows_n, n_pad, d_pad)
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
    """One PPO update over a harvested batch. Returns new weights, optimiser
    states and statistics."""
    denom_actor = SCALE * max(1, full.matches)
    decisions = sum(s.num_decisions for s in shards)
    denom_critic = float(max(1, decisions))
    logp_old = np.concatenate([s.decisions["logp"] for s in shards])
    stats = {"lr": rate, "decisions": decisions, "matches": full.matches, "actor_steps": 0, "kl": [], "stopped": False}
    actor_on = True
    with ThreadPoolExecutor(1) as outer:
        for epoch in range(EPOCHS):
            critic_now = critic
            future_c = outer.submit(_per_device, shards, lambda s: critic_epoch(s, critic_now, denom_critic))
            if actor_on and decisions:
                results = _per_device(shards, lambda s: actor_epoch(s, actor, denom_actor))
                grads = _sum([r[0] for r in results])
                logp_new = np.concatenate([r[1] for r in results])
                kl = kl_estimate(logp_new, logp_old)
                stats["kl"].append(kl)
                if epoch == 0:
                    stats.update({k: sum(r[2][k] for r in results) / decisions for k in ("h_phys", "h_sonar")})
                for i, r in enumerate(results):
                    pr = r[2]["prof"]
                    log(f"  actor shard {i}: {pr['calls']} calls, {pr['rows']} decisions, host {pr['host_s']:.0f}s "
                        f"wait {pr['wait_s']:.0f}s, fill {pr['rows'] * 1.0 / max(1, pr['calls']):.0f} decisions/call")
                stats[f"clip_fraction_{epoch}"] = sum(r[2]["clipped"] for r in results) / decisions
                if not np.isfinite(kl) or not all(np.isfinite(g).all() for g in jax.tree.leaves(grads)):
                    raise FloatingPointError(f"nonfinite actor update at epoch {epoch} (kl {kl})")
                if kl > KL_LIMIT:
                    actor_on = False
                    stats["stopped"] = True
                else:
                    grads, norm = clip_by_norm(grads, GRAD_NORM)
                    stats[f"actor_grad_norm_{epoch}"] = norm
                    actor, actor_opt = adam_step(actor, grads, actor_opt, rate)
                    stats["actor_steps"] += 1
            cres = future_c.result()
            cgrads = _sum([r[0] for r in cres])
            stats[f"value_mse_{epoch}"] = sum(r[1] for r in cres)
            if not all(np.isfinite(g).all() for g in jax.tree.leaves(cgrads)):
                raise FloatingPointError(f"nonfinite critic gradient at epoch {epoch}")
            cgrads, cnorm = clip_by_norm(cgrads, GRAD_NORM)
            stats[f"critic_grad_norm_{epoch}"] = cnorm
            critic, critic_opt = adam_step(critic, cgrads, critic_opt, rate)
            log(f"epoch {epoch}: {stats['actor_steps']} actor steps, kl {stats['kl'][-1] if stats['kl'] else 0:.4f}")
    return actor, critic, actor_opt, critic_opt, stats
