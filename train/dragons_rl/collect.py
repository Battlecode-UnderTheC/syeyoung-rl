"""Collection: many concurrent matches per GPU, big batched calls.

A match advances one dragon-turn per step (turns are sequential by the
rules), so throughput comes from running many matches at once: every step
batches the waiting decision of every live match on the device. Slots are
refilled the moment a match ends, so there is no long tail. Two groups of
matches alternate, so the host prepares one group while the GPU runs the
other.

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
ROW_BUCKETS = (8, 32, 128, 512, 2048, 8192)


def stack_policies(params_list, slots: int = SLOTS):
    """slots rows of actor weights (the list, then copies of its first entry)"""
    padded = list(params_list) + [params_list[0]] * (slots - len(params_list))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *padded)


@partial(jax.jit, static_argnames=("s_pad",))
def policy_step(stack, slot, raw, h, key, s_pad):
    uniforms = jax.random.uniform(key, (slot.shape[0], codec.UNIFORMS), jnp.float32)

    def one(slot, raw, h, u):
        params = jax.tree.map(lambda a: a[slot], stack)
        return model.act(params, raw, h, u, s_pad)
    return jax.vmap(one)(slot, raw, h, uniforms)


@jax.jit
def critic_step(params, raw, team):
    return jax.vmap(lambda r, t: model.value(params, r, t))(raw, team)


def rows_for(n: int) -> int:
    for b in ROW_BUCKETS:
        if n <= b:
            return b
    return codec.pow2(n, ROW_BUCKETS[-1])


def pad_rows(buf: dict, used: int) -> dict:
    """fill unused rows with row 0 (a real decision), so padding computes finite values"""
    for a in buf.values():
        a[used:] = a[0]
    return buf


class Group:
    """One Collector and its in-flight device work."""

    def __init__(self, setups, threads):
        self.col = _native.Collector(setups, threads)
        self.inflight = None


def _dispatch(group: Group, stack, critic, device, key, d_pad):
    """prepare, encode and launch every waiting decision of the group (async)"""
    col = group.col
    pend = col.prepare()
    put = lambda x: jax.device_put(x, device)
    calls = []
    if len(pend["match"]):
        buckets: dict[int, list[int]] = {}
        for i, tiles in enumerate(pend["tiles"]):
            buckets.setdefault(codec.tile_bucket(int(tiles)), []).append(i)
        for n_pad, idx in buckets.items():
            idx = np.asarray(idx)
            B, rows = len(idx), rows_for(len(idx))
            matches = pend["match"][idx]
            abuf = codec.actor_buffers(rows, n_pad, codec.inbox_pad(int(pend["inbox"][idx].max())),
                                       codec.echo_pad(int(pend["echo_bytes"][idx].max())))
            col.encode_actors(matches, abuf)
            pad_rows(abuf, B)
            slot = np.zeros(rows, np.int32)
            slot[:B] = pend["policy"][idx]
            key, sub = jax.random.split(key)
            out = policy_step(stack, put(slot), {k: put(abuf[k]) for k in codec.ACTOR_KEYS}, put(abuf["h"]), sub,
                              codec.split_pad(int(pend["length"][idx].max())))
            learn = np.nonzero(pend["record"][idx])[0]
            value = None
            if len(learn):
                crows = rows_for(len(learn))
                cbuf = codec.critic_buffers(crows, n_pad, d_pad)
                col.encode_critics(matches[learn], cbuf)
                pad_rows(cbuf, len(learn))
                team = np.zeros(crows, np.int32)
                team[:len(learn)] = pend["team"][idx][learn]
                value = critic_step(critic, {k: put(v) for k, v in cbuf.items()}, put(team))
            calls.append((matches, B, out, learn, value))
    group.inflight = calls
    return key, int(len(pend["match"]))


def _finish(group: Group):
    """wait for the group's results and play them"""
    for matches, B, out, learn, value in group.inflight or []:
        command, logp, h_new, _ = out
        command = {k: np.asarray(v)[:B] for k, v in command.items()}
        v = np.full(B, np.nan, np.float32)
        if value is not None:
            v[learn] = np.asarray(value)[:len(learn)]
        group.col.apply(matches, codec.to_native(command), np.asarray(logp, np.float32)[:B],
                        np.asarray(h_new, np.float32)[:B], v)
    group.inflight = None


class Actors:
    """One thread per device, each with `groups` Collectors of matches, playing
    continuously. new_match() -> setup dict. Finished matches are harvested
    into Packs (take())."""

    def __init__(self, devices, slots_per_device: int, groups: int, threads_per_device: int, new_match, log=print):
        self.devices = devices
        self.new_match = new_match
        self.log = log
        self.lock = threading.Lock()
        self.packs = []
        self.decisions = 0                 # recorded, harvested
        self.turns = 0                     # every dragon-turn played
        self.running = threading.Event()
        self.stop = False
        self.paused = [threading.Event() for _ in devices]
        self.weights = None
        self.version = 0
        per = max(1, slots_per_device // groups)
        self.groups = [[Group([new_match() for _ in range(per)], max(1, threads_per_device // groups))
                        for _ in range(groups)] for _ in devices]
        self.threads = [threading.Thread(target=self._run, args=(i,), daemon=True) for i in range(len(devices))]
        self.error = None

    def set_weights(self, policies, critic):
        """new stack (list of actor weights, slot order) and critic; applied at each device's next step"""
        with self.lock:
            self.weights = (list(policies), critic)
            self.version += 1

    def start(self):
        self.running.set()
        for t in self.threads:
            t.start()

    def pause(self):
        """stop stepping (in-flight work finished, finished matches harvested) and wait for every device"""
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
            dev = self.devices[i]
            groups = self.groups[i]
            key = jax.random.PRNGKey(int(time.time() * 1e6) % (2**31) + i)
            version, stack, critic = -1, None, None
            step = 0
            while not self.stop:
                if not self.running.is_set():
                    for g in groups:
                        if g.inflight is not None:
                            _finish(g)
                            self._refill(g)
                    self._harvest(groups)
                    self.paused[i].set()
                    self.running.wait()
                    continue
                if self.version != version:
                    with self.lock:
                        policies, critic_w = self.weights
                        version = self.version
                    stack = jax.device_put(stack_policies(policies), dev)
                    critic = jax.device_put(critic_w, dev)
                for g in groups:
                    if g.inflight is not None:
                        _finish(g)
                        self._refill(g)
                    key, n = _dispatch(g, stack, critic, dev, key, 128)
                    with self.lock:
                        self.turns += n
                step += 1
                if step % 16 == 0:
                    self._harvest(groups)
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

    def _harvest(self, groups):
        for g in groups:
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
    key = jax.random.PRNGKey(int(time.time() * 1e6) % (2**31))
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
