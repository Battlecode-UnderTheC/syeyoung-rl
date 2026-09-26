"""Collection: many concurrent matches per GPU, big batched calls.

A match advances one dragon-turn per step (turns are sequential by the
rules), so throughput comes from running many matches at once: every step
batches the waiting decision of every live match on the device. Slots are
refilled the moment a match ends, so there is no long tail. Each group of
matches has its own thread, so one group's host work overlaps another's
device work.

Policies are a stack of actor weights on the device; slot k is row k and each
example uses its own slot. The critic runs only on recorded (learner)
decisions, from the actor's team's perspective (the other team's value is its
negative: the game is zero-sum).
"""

from __future__ import annotations

import threading
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import _native, codec, model

SLOTS = 34                     # stack rows: 0 the learner, the rest frozen opponents
# simple1: an actor row is fixed-size (the window and the grid), so actor calls are
# cut by rows only; the critic (the whole board) is cut by tile-rows
ACTOR_ROWS = 2048
COLLECT_BUDGET = 1 << 19        # critic tile-rows per call
# Every distinct call shape is a compile (seconds to a minute, per worker process),
# so collection uses few: rows 32 or 512 (more only on huge groups), inbox pads
# of 128 or 1024 words, echo pads of 32 bytes.
ROW_BUCKETS = (32, 512, 2048, 8192)
# Rows of a call whose policies differ per row: each row gathers its own copy of
# the weights (MBs), so these calls stay small. The learner's rows (most of
# them) share one copy.
GATHERED_ROWS = 256
# V_old is only PPO's baseline: any value computed before the action keeps the
# policy gradient unbiased. A match's critic runs at most every CRITIC_EVERY of
# its dragon-turns; between, a learner decision takes the latest value (negated
# for the other team). The critic's own training replays events, not these.
CRITIC_EVERY = 8


def stack_policies(params_list, slots: int = SLOTS):
    """slots rows of actor weights (the list, then copies of its first entry)"""
    padded = list(params_list) + [params_list[0]] * (slots - len(params_list))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *padded)


@partial(jax.jit, static_argnames=("s_pad",))
def policy_step(stack, slot, raw, h, seed, s_pad):
    """rows whose policies differ: each gathers its own weights from the stack.
    seed: a host-drawn int32, so no device-side key splitting per call"""
    uniforms = jax.random.uniform(jax.random.PRNGKey(seed), (slot.shape[0], codec.UNIFORMS), jnp.float32)

    def one(slot, raw, h, u):
        params = jax.tree.map(lambda a: a[slot], stack)
        return model.act(params, raw, h, u, s_pad, half=True)
    return jax.vmap(one)(slot, raw, h, uniforms)


@partial(jax.jit, static_argnames=("s_pad",))
def policy_step_shared(params, raw, h, seed, s_pad):
    """rows that all play `params` (no per-row weights)"""
    uniforms = jax.random.uniform(jax.random.PRNGKey(seed), (h.shape[0], codec.UNIFORMS), jnp.float32)
    return jax.vmap(lambda r, hh, u: model.act(params, r, hh, u, s_pad, half=True))(raw, h, uniforms)


_learner_of = jax.jit(lambda stack: jax.tree.map(lambda a: a[0], stack))


def _step(group, stack, shared, slot_d, raw_d, h_d, seed, s_pad):
    if shared:
        if group.learner_of is not stack:                  # slot 0's weights, sliced once per stack
            group.learner, group.learner_of = _learner_of(stack), stack
        return policy_step_shared(group.learner, raw_d, h_d, seed, s_pad)
    return policy_step(stack, slot_d, raw_d, h_d, seed, s_pad)


@jax.jit
def critic_step(params, raw, team):
    return jax.vmap(lambda r, t: model.value(params, r, t, half=True))(raw, team)


def rows_for(n: int, cap: int) -> int:
    """a call's padded rows: a ROW_BUCKETS size, at most cap (a power of two >= n)"""
    for b in ROW_BUCKETS:
        if n <= b:
            return max(n, min(b, cap))
    return codec.pow2(n, ROW_BUCKETS[-1])


# The critic's per-tile arrays stay zero in padding rows (VALID = 0 masks every tile);
# every other array copies a real row, so nothing divides by a zero board.
_TILE_ARRAYS = ("board_tile", "board_dest", "grid", "dragons", "summary")


def pad_rows(buf: dict, used: int) -> dict:
    """make rows past `used` harmless copies of row 0 in their small fields"""
    for k, a in buf.items():
        if k not in _TILE_ARRAYS:
            a[used:] = a[0]
    return buf


class Group:
    """One Collector, its in-flight device work, and its host buffers (reused:
    a group's previous call has finished before its next dispatch)."""

    def __init__(self, setups, threads, keep_observations: bool = True):
        self.col = _native.Collector(setups, threads, keep_observations)
        self.inflight = None
        self.buffers = {}
        self.learner, self.learner_of = None, None
        M = self.col.num_matches
        self.v_value = np.zeros(M, np.float32)   # the match's latest V_old, from team v_team's perspective
        self.v_team = np.zeros(M, np.int32)
        self.v_age = np.full(M, CRITIC_EVERY, np.int64)   # its dragon-turns since

    def reset(self, m: int, setup: dict):
        """a new match in slot m (finished)"""
        self.col.reset(m, setup)
        self.v_age[m] = CRITIC_EVERY

    def actor_buffers(self, rows, k_pad, e_pad, part):
        key = ("a", rows, k_pad, e_pad, part)
        if key not in self.buffers:
            self.buffers[key] = codec.actor_buffers(rows, k_pad, e_pad)
        return self.buffers[key]

    def critic_buffers(self, rows, n_pad, d_pad, part):
        key = ("c", rows, n_pad, d_pad, part)
        if key not in self.buffers:
            self.buffers[key] = codec.critic_buffers(rows, n_pad, d_pad)
        return self.buffers[key]


def _dispatch(group: Group, stack, critic, device, key, d_pad=None):
    """prepare, encode and launch every waiting decision of the group (async).
    `key` is a numpy Generator for the per-call sampling seeds."""
    col = group.col
    pend = col.prepare()
    put = lambda x: jax.device_put(x, device)
    calls = []
    if len(pend["match"]):
        idx_all = np.arange(len(pend["match"]))
        learner = pend["policy"] == 0
        work = []
        for shared, sub_idx in ((True, idx_all[learner]), (False, idx_all[~learner])):
            per = ACTOR_ROWS if shared else GATHERED_ROWS
            work += [(sub_idx[k:k + per], shared, per) for k in range(0, len(sub_idx), per)]
        for part, (idx, shared, per) in enumerate(work):
            B, rows = len(idx), rows_for(len(idx), per)
            matches = pend["match"][idx]
            abuf = group.actor_buffers(rows, max(128, codec.inbox_pad(int(pend["inbox"][idx].max()))),
                                       max(32, codec.echo_pad(int(pend["echo_bytes"][idx].max()))), part)
            col.encode_actors(matches, abuf)
            pad_rows(abuf, B)
            slot = np.zeros(rows, np.int32)
            slot[:B] = pend["policy"][idx]
            sub = int(key.integers(0, 2**31 - 1))
            s_pad = codec.split_pad(int(pend["length"][idx].max()))
            slot_d, raw_d, h_d = put((slot, {k: abuf[k] for k in codec.ACTOR_KEYS}, abuf["h"]))   # one transfer call
            out = _step(group, stack, shared, slot_d, raw_d, h_d, sub, s_pad)
            recorded = np.nonzero(pend["record"][idx])[0]
            learn = recorded[group.v_age[matches[recorded]] >= CRITIC_EVERY]      # the critic runs on these
            # the critic reads the whole board: its rows are grouped by board size
            values = []
            by: dict[int, list[int]] = {}
            for r in learn:
                by.setdefault(codec.board_bucket(int(pend["tiles"][idx[r]])), []).append(int(r))
            for c, (n_pad, rs) in enumerate(sorted(by.items())):
                cap = 1 << (max(8, COLLECT_BUDGET // n_pad).bit_length() - 1)
                for k in range(0, len(rs), cap):
                    li = np.asarray(rs[k:k + cap])
                    crows = rows_for(len(li), cap)
                    d_need = codec.geometric(int(pend["alive"][idx][li].max()), 16, 8)   # dragons alive: 16 or 128
                    cbuf = group.critic_buffers(crows, n_pad, d_need, (part, c, k))
                    col.encode_critics(matches[li], cbuf)
                    pad_rows(cbuf, len(li))
                    team = np.zeros(crows, np.int32)
                    team[:len(li)] = pend["team"][idx][li]
                    values.append((li, critic_step(critic, *put((cbuf, team)))))
            calls.append((matches, B, out, learn, values, (abuf, slot),
                          (device, stack, sub, s_pad, B, shared), (recorded, pend["team"][idx])))
    group.inflight = calls
    return key, int(len(pend["match"]))


DUMP_DIR = None            # set by the trainer: where to save inputs that produce nonfinite outputs
_dumps = [0]


def _dump(inputs, bad_actor):
    """save the inputs of one step with nonfinite outputs (a few per run), for offline reproduction"""
    import pathlib
    if DUMP_DIR is None or _dumps[0] >= 5:
        return
    _dumps[0] += 1
    abuf, slot = inputs
    out = {f"actor/{k}": v for k, v in abuf.items()}
    out["slot"] = slot
    out["bad_actor"] = bad_actor
    path = pathlib.Path(DUMP_DIR)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f"nonfinite-{int(time.time())}-{_dumps[0]}.npz", **out)


_nonfinite = [0]           # rows recomputed so far (for the log)


def _recompute(group, inputs, last, command, lp, h, bad_actor):
    """rerun a call whose outputs had nonfinite rows; take the rerun's rows where it is finite,
    and zero h where it is still not"""
    abuf, slot = inputs
    dev, stack, key, s_pad, B, shared = last
    slot_d, raw_d, h_d = jax.device_put((slot, {k: abuf[k] for k in codec.ACTOR_KEYS}, abuf["h"]), dev)
    out = _step(group, stack, shared, slot_d, raw_d, h_d, key, s_pad)
    c2 = {k: np.asarray(x)[:B] for k, x in out[0].items()}
    lp2 = np.asarray(out[1])[:B]
    h2 = np.asarray(out[2])[:B]
    ok2 = np.isfinite(h2).all(1) & np.isfinite(lp2)
    fix = bad_actor & ok2
    for k in command:
        command[k][fix] = c2[k][fix]
    lp = np.where(fix, lp2, lp)
    h = np.where(fix[:, None], h2, h)
    still = bad_actor & ~ok2
    h[still] = 0.0
    lp = np.where(still, 0.0, lp)
    return command, lp, h


def _finish(group: Group):
    """wait for the group's results and play them"""
    for matches, B, out, learn, values, inputs, last, (recorded, team) in group.inflight or []:
        command, logp, h_new, _ = out
        command = {k: np.array(v)[:B] for k, v in command.items()}          # writable copies
        v = np.full(B, np.nan, np.float32)
        for li, value in values:
            v[li] = np.asarray(value)[:len(li)]      # a nonfinite V_old stays: PPO excludes that decision
        h_np = np.array(h_new)[:B]
        lp_np = np.array(logp)[:B]
        bad_actor = ~np.isfinite(h_np).all(1) | ~np.isfinite(lp_np)
        if bad_actor.any():
            _dump(inputs, bad_actor)
            # recompute once: a nonfinite h would poison that dragon for the rest of its life
            command, lp_np, h_np = _recompute(group, inputs, last, command, lp_np, h_np, bad_actor)
            _nonfinite[0] += int(bad_actor.sum())
        # latest values: fresh where the critic ran, the match's cached one elsewhere
        ran = matches[learn]
        group.v_value[ran], group.v_team[ran], group.v_age[ran] = v[learn], team[learn], 0
        rest = np.setdiff1d(recorded, learn, assume_unique=True)
        m_rest = matches[rest]
        v[rest] = np.where(group.v_team[m_rest] == team[rest], group.v_value[m_rest], -group.v_value[m_rest])
        group.v_age[matches] += 1
        group.col.apply(matches, codec.to_native(command), np.asarray(lp_np, np.float32)[:B],
                        np.asarray(h_np, np.float32)[:B], v)
    group.inflight = None


class Actors:
    """One thread per (device, group), each playing its Collector of matches
    continuously; a group's host work (prepare, encode, apply) overlaps the
    other groups' device work. new_match() -> setup dict. Finished matches are
    harvested into Packs (take())."""

    def __init__(self, devices, slots_per_device: int, groups: int, threads_per_device: int, new_match, log=print,
                 max_pending: int = 1 << 62):
        self.devices = devices
        self.max_pending = max_pending     # backpressure: stop stepping while this many learner decisions wait
        self.new_match = new_match
        self.log = log
        self.lock = threading.Lock()
        self.packs = []
        self.decisions = 0                 # recorded, harvested
        self.turns = 0                     # every dragon-turn played
        self.running = threading.Event()
        self.stop = False
        self.weights = None
        self.version = 0
        per = max(1, slots_per_device // groups)
        self.workers = [(dev, Group([new_match() for _ in range(per)], max(1, threads_per_device // groups)))
                        for dev in devices for _ in range(groups)]
        self.paused = [threading.Event() for _ in self.workers]
        self.threads = [threading.Thread(target=self._run, args=(i,), daemon=True) for i in range(len(self.workers))]
        self.error = None

    def set_weights(self, policies, critic):
        """new stack (list of actor weights, slot order) and critic; applied at each group's next step"""
        with self.lock:
            self.weights = (list(policies), critic)
            self.version += 1

    def start(self):
        self.running.set()
        for t in self.threads:
            t.start()

    def pause(self):
        """stop stepping (in-flight work finished, finished matches harvested) and wait for every group"""
        self.running.clear()
        for e in self.paused:
            e.wait()
        if self.error:
            raise RuntimeError("an actor thread died") from self.error

    def resume(self):
        for e in self.paused:
            e.clear()
        self.running.set()

    def take(self):
        with self.lock:
            packs, self.packs = self.packs, []
            n, self.decisions = self.decisions, 0
        return packs, n

    def _run(self, i):
        try:
            dev, g = self.workers[i]
            key = np.random.default_rng()
            version, stack, critic = -1, None, None
            step = 0
            while not self.stop:
                if not self.running.is_set():
                    self._harvest(g)
                    self.paused[i].set()
                    self.running.wait()
                    continue
                if self.decisions >= self.max_pending:
                    time.sleep(0.05)
                    continue
                if self.version != version:
                    with self.lock:
                        policies, critic_w = self.weights
                        version = self.version
                    stack = jax.device_put(stack_policies(policies), dev)
                    critic = jax.device_put(critic_w, dev)
                key, n = _dispatch(g, stack, critic, dev, key, 128)
                _finish(g)
                self._refill(g)
                with self.lock:
                    self.turns += n
                step += 1
                if step % 16 == 0:
                    self._harvest(g)
        except BaseException as e:           # surfaced to the trainer by pause()
            self.error = e
            self.log(f"actor thread {i} died: {e!r}")
            for e_ in self.paused:
                e_.set()
            raise

    def _refill(self, g: Group):
        col = g.col
        for m in range(col.num_matches):
            if col.finished(m):
                g.reset(m, self.new_match())

    def _harvest(self, g: Group):
        pack = g.col.harvest()
        if pack.num_matches:
            n = pack.decisions()
            with self.lock:
                self.packs.append(pack)
                self.decisions += n


def play(setups: list[dict], stack, critic_params, *, device=None, threads: int = 1, log=None,
         keep_observations: bool = True):
    """Plays every setup to its end (evaluation): all matches at once, batched."""
    device = device or jax.devices()[0]
    stack = jax.device_put(stack, device)
    critic_params = jax.device_put(critic_params, device)
    g = Group(setups, threads, keep_observations)
    key = np.random.default_rng()
    while not g.col.done():
        key, _ = _dispatch(g, stack, critic_params, device, key, 128)
        _finish(g)
    return g.col


def play_parallel(setups: list[dict], policies: list, critic_params, devices: list, threads: int, log=None):
    """play() over devices, matches dealt round-robin. Returns one finished
    Collector per device and each device's positions into setups."""
    from concurrent.futures import ThreadPoolExecutor
    if len(policies) > SLOTS:
        raise ValueError(f"at most {SLOTS} policies in one collection")
    parts = [(dev, list(range(d, len(setups), len(devices)))) for d, dev in enumerate(devices)]
    parts = [(dev, p) for dev, p in parts if p]

    def run(item):
        dev, positions = item
        return play([setups[i] for i in positions], stack_policies(policies), critic_params, device=dev,
                    threads=max(1, threads // len(parts)), log=log)

    with ThreadPoolExecutor(len(parts)) as pool:
        collectors = list(pool.map(run, parts))
    return collectors, [p for _, p in parts]
