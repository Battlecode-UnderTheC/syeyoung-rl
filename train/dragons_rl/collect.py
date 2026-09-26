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
# tile-decisions of one collection call (rows x board bucket): bounds device memory;
# a bigger bucket is split into several calls, never truncated
COLLECT_BUDGET = 1 << 20
ROW_BUCKETS = (8, 32, 128, 512, 2048, 8192)


def stack_policies(params_list, slots: int = SLOTS):
    """slots rows of actor weights (the list, then copies of its first entry)"""
    padded = list(params_list) + [params_list[0]] * (slots - len(params_list))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *padded)


def gather_tiles(pool, pages, tiles, n_pad: int):
    """rows' tile records from the device pool: tile [R, n_pad, 16] and dest [R, n_pad, 4],
    padding (past the board, or pages -1) as the codec writes it: records 0, J -1"""
    t = jnp.arange(n_pad, dtype=jnp.int32)
    page = pages[:, t // codec.PAGE_TILES]                                    # [R, n_pad]
    ok = (page >= 0) & (t[None, :] < tiles[:, None])
    rec = pool[jnp.where(ok, page * codec.PAGE_TILES + t % codec.PAGE_TILES, 0)].astype(jnp.int32)
    tile = jnp.where(ok[..., None], rec[..., :codec.FIELDS["TILE"]], 0)
    dest = jnp.where(ok[..., None], rec[..., codec.FIELDS["TILE"]:], -1)
    return tile, dest


@partial(jax.jit, donate_argnums=(0,))
def apply_delta(pool, index, rows):
    """changed records into the pool (index past the pool: padding, dropped)"""
    return pool.at[index].set(rows, mode="drop")


@jax.jit
def grow_pool(pool, new):
    return new.at[:pool.shape[0]].set(pool)


class Pool:
    """One Collector's device-resident tile records (see the native PagePool)."""

    def __init__(self, device):
        self.device = device
        self.rows = jax.device_put(np.zeros((1024 * codec.PAGE_TILES, codec.RESIDENT_FIELDS), np.int16), device)

    def ensure(self, pages: int):
        need = pages * codec.PAGE_TILES
        if need > self.rows.shape[0]:
            size = codec.pow2(need, self.rows.shape[0])
            new = jax.device_put(np.zeros((size, codec.RESIDENT_FIELDS), np.int16), self.device)
            self.rows = grow_pool(self.rows, new)

    def update(self, index: np.ndarray, rows: np.ndarray):
        if not len(index):
            return
        K = codec.pow2(len(index), 1024)
        ip = np.full(K, self.rows.shape[0], np.int32)
        ip[:len(index)] = index
        rp = np.zeros((K, codec.RESIDENT_FIELDS), np.int16)
        rp[:len(index)] = rows
        self.rows = apply_delta(self.rows, *jax.device_put((ip, rp), self.device))


@partial(jax.jit, static_argnames=("s_pad", "n_pad"))
def policy_step(stack, slot, raw, h, seed, pool, pages, tiles, s_pad, n_pad):
    """seed: a host-drawn int32, so no device-side key splitting per call. raw: the
    small arrays; the tile records come from the pool."""
    uniforms = jax.random.uniform(jax.random.PRNGKey(seed), (slot.shape[0], codec.UNIFORMS), jnp.float32)
    raw = dict(raw)
    raw["tile"], raw["dest"] = gather_tiles(pool, pages, tiles, n_pad)

    def one(slot, raw, h, u):
        params = jax.tree.map(lambda a: a[slot], stack)
        return model.act(params, raw, h, u, s_pad, half=True)
    return jax.vmap(one)(slot, raw, h, uniforms)


@jax.jit
def critic_step(params, raw, team):
    return jax.vmap(lambda r, t: model.value(params, r, t, half=True))(raw, team)


def rows_for(n: int) -> int:
    for b in ROW_BUCKETS:
        if n <= b:
            return b
    return codec.pow2(n, ROW_BUCKETS[-1])


# Per-tile arrays stay zero in padding rows: VALID = 0 masks every tile (and J = -1 follows).
# Only the small per-decision arrays need a real row, so nothing divides by a zero board.
_TILE_ARRAYS = ("tile", "dest", "board_tile", "board_dest", "grid", "dragons", "summary", "pages", "tiles")


def pad_rows(buf: dict, used: int) -> dict:
    """make rows past `used` harmless copies of row 0 in their small fields"""
    for k, a in buf.items():
        if k not in _TILE_ARRAYS:
            a[used:] = a[0]
    return buf


class Group:
    """One Collector, its in-flight device work, and its host buffers (reused:
    a group's previous call has finished before its next dispatch)."""

    def __init__(self, setups, threads):
        self.col = _native.Collector(setups, threads)
        self.inflight = None
        self.buffers = {}
        self.pool = None                     # device-resident tile records, made on first dispatch

    def actor_buffers(self, rows, k_pad, e_pad, part):
        key = ("a", rows, k_pad, e_pad, part)
        if key not in self.buffers:
            self.buffers[key] = codec.actor_small_buffers(rows, k_pad, e_pad)
        buf = self.buffers[key]
        buf["pages"].fill(-1)
        buf["tiles"].fill(0)
        return buf

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
    if group.pool is None:
        group.pool = Pool(device)
    group.pool.ensure(col.page_count)
    calls = []
    if len(pend["match"]):
        buckets: dict[int, list[int]] = {}
        for i, tiles in enumerate(pend["tiles"]):
            buckets.setdefault(codec.board_bucket(int(tiles)), []).append(i)
        work = []
        for n_pad, idx in buckets.items():
            per = max(8, COLLECT_BUDGET // n_pad)
            work += [(n_pad, np.asarray(idx[k:k + per])) for k in range(0, len(idx), per)]
        for part, (n_pad, idx) in enumerate(work):
            B, rows = len(idx), rows_for(len(idx))
            matches = pend["match"][idx]
            abuf = group.actor_buffers(rows, codec.inbox_pad(int(pend["inbox"][idx].max())),
                                       codec.echo_pad(int(pend["echo_bytes"][idx].max())), part)
            small = {k: abuf[k] for k in codec.ACTOR_SMALL_KEYS + ("h",)}
            index, delta = col.encode_actors_resident(matches, small, abuf["pages"], abuf["tiles"])
            group.pool.update(index, delta)
            pad_rows(abuf, B)
            slot = np.zeros(rows, np.int32)
            slot[:B] = pend["policy"][idx]
            sub = int(key.integers(0, 2**31 - 1))
            s_pad = codec.split_pad(int(pend["length"][idx].max()))
            slot_d, raw_d, h_d, pages_d, tiles_d = put((slot, {k: abuf[k] for k in codec.ACTOR_SMALL_KEYS}, abuf["h"],
                                                        abuf["pages"], abuf["tiles"]))            # one transfer call
            out = policy_step(stack, slot_d, raw_d, h_d, sub, group.pool.rows, pages_d, tiles_d, s_pad, n_pad)
            learn = np.nonzero(pend["record"][idx])[0]
            value = None
            if len(learn):
                crows = rows_for(len(learn))
                d_need = codec.geometric(int(pend["alive"][idx][learn].max()), 16, 8)   # dragons alive: 16 or 128
                cbuf = group.critic_buffers(crows, n_pad, d_need, part)
                col.encode_critics(matches[learn], cbuf)
                pad_rows(cbuf, len(learn))
                team = np.zeros(crows, np.int32)
                team[:len(learn)] = pend["team"][idx][learn]
                cb_d, team_d = put((cbuf, team))
                value = critic_step(critic, cb_d, team_d)
            calls.append((matches, B, out, learn, value, (abuf, cbuf if len(learn) else None, slot),
                          (device, stack, critic, sub, s_pad, B, n_pad), pend["team"][idx][learn]))
    group.inflight = calls
    return key, int(len(pend["match"]))


DUMP_DIR = None            # set by the trainer: where to save inputs that produce nonfinite outputs
_dumps = [0]


def _dump(inputs, bad_actor, bad_critic, learn):
    """save the inputs of one step with nonfinite outputs (a few per run), for offline reproduction"""
    import pathlib
    if DUMP_DIR is None or _dumps[0] >= 5:
        return
    _dumps[0] += 1
    abuf, cbuf, slot = inputs
    out = {f"actor/{k}": v for k, v in abuf.items()}
    out["slot"] = slot
    out["bad_actor"] = bad_actor
    if cbuf is not None:
        out.update({f"critic/{k}": v for k, v in cbuf.items()})
        out["bad_critic"] = bad_critic
        out["learn"] = learn
    path = pathlib.Path(DUMP_DIR)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f"nonfinite-{int(time.time())}-{_dumps[0]}.npz", **out)


_nonfinite = [0]           # rows recomputed so far (for the log)


def _recompute(group, inputs, command, lp, h, v, learn, bad_actor, bad_critic):
    """rerun a call whose outputs had nonfinite rows; take the rerun's rows where it is finite,
    and zero h / drop V_old where it is still not"""
    abuf, cbuf, slot = inputs
    dev, stack, critic, key, s_pad, B, n_pad = group.last
    put = lambda x: jax.device_put(x, dev)
    # the pool still holds these lifetimes' records: pages are only freed in apply
    slot_d, raw_d, h_d, pages_d, tiles_d = put((slot, {k: abuf[k] for k in codec.ACTOR_SMALL_KEYS}, abuf["h"],
                                                abuf["pages"], abuf["tiles"]))
    out = policy_step(stack, slot_d, raw_d, h_d, key, group.pool.rows, pages_d, tiles_d, s_pad, n_pad)
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
    if bad_critic.any() and cbuf is not None:
        team = np.zeros(len(cbuf["grid"]), np.int32)
        team[:len(learn)] = group.last_team
        v2 = np.asarray(critic_step(critic, {k: put(x) for k, x in cbuf.items()}, put(team)))[:len(learn)]
        vl = v[learn]
        vl = np.where(bad_critic & np.isfinite(v2), v2, vl)
        v[learn] = vl
    return command, lp, h, v


def _finish(group: Group):
    """wait for the group's results and play them"""
    for matches, B, out, learn, value, inputs, last, last_team in group.inflight or []:
        group.last, group.last_team = last, last_team
        command, logp, h_new, _ = out
        command = {k: np.array(v)[:B] for k, v in command.items()}          # writable copies
        v = np.full(B, np.nan, np.float32)
        if value is not None:
            v[learn] = np.asarray(value)[:len(learn)]
        h_np = np.array(h_new)[:B]
        lp_np = np.array(logp)[:B]
        bad_actor = ~np.isfinite(h_np).all(1) | ~np.isfinite(lp_np)
        bad_critic = ~np.isfinite(v[learn]) if len(learn) else np.zeros(0, bool)
        if bad_actor.any() or bad_critic.any():
            _dump(inputs, bad_actor, bad_critic, learn)
            # recompute once: a nonfinite h would poison that dragon for the rest of its life
            command, lp_np, h_np, v = _recompute(group, inputs, command, lp_np, h_np, v, learn, bad_actor, bad_critic)
            _nonfinite[0] += int(bad_actor.sum() + bad_critic.sum())
        logp = lp_np
        group.col.apply(matches, codec.to_native(command), np.asarray(logp, np.float32)[:B],
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
                col.reset(m, self.new_match())

    def _harvest(self, g: Group):
        pack = g.col.harvest()
        if pack.num_matches:
            n = pack.decisions()
            with self.lock:
                self.packs.append(pack)
                self.decisions += n


def play(setups: list[dict], stack, critic_params, *, device=None, threads: int = 1, log=None):
    """Plays every setup to its end (evaluation): all matches at once, batched."""
    device = device or jax.devices()[0]
    stack = jax.device_put(stack, device)
    critic_params = jax.device_put(critic_params, device)
    g = Group(setups, threads)
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
