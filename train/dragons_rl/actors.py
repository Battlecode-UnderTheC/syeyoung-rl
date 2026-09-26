"""Collection in worker processes: no GIL between the matches' Python glue.

Each worker is a spawned process with its own JAX client on one GPU (a small
share of its memory) and one Group of matches, played continuously as in
collect.Actors. The learner sends weights (its own parameters, the critic and
the league snapshots' paths, which workers load and cache); workers send back
finished matches as serialized Packs (see the native SerializePack), which the
learner rebuilds by replay in a thread pool. Same interface as collect.Actors.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pathlib
import queue
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def opponent(n_slots: int, recent_n: int) -> int:
    """the league's policy slot for a new match's opponent: 50% the learner itself
    (both sides then train), 25% a recent snapshot, 25% an older one"""
    u = random.random()
    if u < 0.5 or n_slots <= 1:
        return 0
    older = n_slots - 1 - recent_n
    if u < 0.75 or older == 0:
        return 1 + random.randrange(recent_n)
    return 1 + recent_n + random.randrange(older)


def new_match(map_text: str, n_slots: int, recent_n: int) -> dict:
    side = random.randrange(2)
    policy = [0, 0]
    policy[1 - side] = opponent(n_slots, recent_n)
    record = [False, False]
    record[side] = True
    record[1 - side] = policy[1 - side] == 0          # self-play: both sides are the learner
    return {"map": map_text, "policy": policy, "record": record}


class _WorkerMaps:
    """fresh fair maps from the learner's generator queue, contest maps now and then,
    recent ones reused when the generators fall behind"""

    def __init__(self, q, contest: list, contest_share: float):
        self.q, self.contest, self.share = q, contest, contest_share
        self.recent = [q.get() for _ in range(16)]

    def next(self) -> str:
        if random.random() < self.share:
            return random.choice(self.contest)
        try:
            text = self.q.get_nowait()
            self.recent[random.randrange(len(self.recent))] = text
            return text
        except queue.Empty:
            return random.choice(self.recent)


def pick_snapshot(paths: list[str], recent_n: int) -> str | None:
    """this update's league opponent: a recent snapshot or an older one, evenly; none yet: the learner"""
    if not paths:
        return None
    older = len(paths) - recent_n
    if older == 0 or random.random() < 0.5:
        return paths[random.randrange(recent_n)]
    return paths[recent_n + random.randrange(older)]


def _worker(i: int, gpu: int, mem_fraction: float, slots: int, threads: int, inbox, outbox, running, pending,
            max_pending: int, maps_q, contest: list, contest_share: float, cache_dir: str | None):
    """The native step loop: the Collector refills, observes, encodes and applies
    (C++); Python moves one buffer to the GPU and back per step."""
    parent = os.getppid()                                   # the trainer: gone means stop
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="
    try:
        open("/proc/self/oom_score_adj", "w").write("500")   # under memory pressure, a worker goes before the learner
    except OSError:
        pass
    try:
        import jax
        from . import _native, codec, collect, league
        if cache_dir:                                           # compiled programs shared with the learner and restarts
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        dev = jax.devices()[0]
        random.seed(os.getpid() ^ int(time.time()))
        maps = _WorkerMaps(maps_q, contest, contest_share)
        msg = inbox.get()                                       # the first weights
        cache: dict[str, dict] = {}
        nets = {}

        def load(msg):
            _, actor, critic_w, paths, recent_n = msg
            path = pick_snapshot(paths, recent_n)
            if path is not None and path not in cache:
                cache.clear()
                cache[path] = league.load_params(pathlib.Path(path), actor)
            nets["learner"] = jax.device_put(actor, dev)
            nets["opponent"] = jax.device_put(cache[path], dev) if path is not None else nets["learner"]
            nets["critic"] = jax.device_put(critic_w, dev)

        load(msg)
        M = slots
        Mo = M // 4 + 16                                        # opponent rows per step (more wait a step)
        Mc = max(8, M // 4)                                     # critic rows per step
        # packs leave serialized, without observations (rebuilt by replay): don't keep them here
        col = _native.Collector([new_match(maps.next(), 2, 1) for _ in range(M)], threads, False)
        col.set_matchmaking(0.5, random.getrandbits(63))
        lbuf = codec.actor_buffers(M, codec.FIELDS["MAX_INBOX"], codec.FIELDS["MAX_ECHO"])
        obuf = codec.actor_buffers(Mo, codec.FIELDS["MAX_INBOX"], codec.FIELDS["MAX_ECHO"])
        cbuf = codec.critic_buffers(Mc, 128)
        crow, cteam = np.zeros(Mc, np.int32), np.zeros(Mc, np.int32)
        seeds = np.random.default_rng()
        step, turns_sent = 0, 0
        t_log, s_log, n_log = time.time(), 0, col.turns
        spent = {"encode": 0.0, "gpu": 0.0, "apply": 0.0, "idle": 0.0}
        while os.getppid() == parent:
            if time.time() - t_log > 30:
                dt = time.time() - t_log
                outbox.put(("log", f"worker {i} (gpu {gpu}): {(step - s_log) / dt:.1f} steps/s, "
                                   f"{(col.turns - n_log) / dt:.0f} decisions/s; s/s "
                                   + " ".join(f"{k} {v / dt:.2f}" for k, v in spent.items())
                                   + f"; nonfinite rows {col.nonfinite}", 0, 0))
                t_log, s_log, n_log = time.time(), step, col.turns
                spent = dict.fromkeys(spent, 0.0)
            try:
                while True:
                    msg = inbox.get_nowait()
                    if msg[0] == "stop":
                        return
                    load(msg)
            except queue.Empty:
                pass
            if not running.is_set() or pending.value >= max_pending:
                time.sleep(0.05)
                spent["idle"] += 0.05
                continue
            if col.queued_maps < M // 4:
                col.push_maps([maps.next() for _ in range(M // 2)])
            t0 = time.perf_counter()
            col.step_encode(lbuf, obuf, cbuf, collect.CRITIC_EVERY, crow, cteam)
            t1 = time.perf_counter()
            args = jax.device_put(({k: lbuf[k] for k in codec.ACTOR_KEYS}, lbuf["h"],
                                   {k: obuf[k] for k in codec.ACTOR_KEYS}, obuf["h"], cbuf, cteam), dev)
            out = collect.native_step(nets["learner"], nets["opponent"], nets["critic"], *args,
                                      int(seeds.integers(0, 2**31 - 1)), codec.MAX_SPLIT)
            command, logp, h, value = jax.device_get(out)
            t2 = time.perf_counter()
            col.step_apply(command["kind"], command["nsteps"], command["dirs"], command["split"], command["sonar"],
                           logp, h, value)
            t3 = time.perf_counter()
            spent["encode"] += t1 - t0
            spent["gpu"] += t2 - t1
            spent["apply"] += t3 - t2
            step += 1
            if step % 16 == 0:
                pack = col.harvest()
                if pack.num_matches:
                    n_dec = pack.decisions()
                    with pending.get_lock():
                        pending.value += n_dec
                    outbox.put(("pack", pack.serialize(), n_dec, col.turns - turns_sent))
                    turns_sent = col.turns
    except BaseException:
        outbox.put(("error", f"worker {i}: " + traceback.format_exc(), 0, 0))


class ActorProcesses:
    """collect.Actors with each group in its own process. `workers_per_gpu` groups
    of `slots` matches on each of `gpus` (indices into the visible devices)."""

    def __init__(self, gpus: list[int], workers_per_gpu: int, slots: int, threads: int, maps_q, contest: list,
                 contest_share: float, log=print, max_pending: int = 1 << 62, mem_fraction: float = 0.2,
                 cache_dir: str | None = None):
        self.log = log
        ctx = mp.get_context("spawn")
        self.running = ctx.Event()
        self.pending = ctx.Value("q", 0)
        self.outbox = ctx.Queue()
        self.inboxes = []
        self.procs = []
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        ids = visible.split(",") if visible else [str(g) for g in gpus]
        for gpu in gpus:
            for _ in range(workers_per_gpu):
                inbox = ctx.Queue()
                p = ctx.Process(target=_worker, daemon=True,
                                args=(len(self.procs), int(ids[gpu]), mem_fraction, slots, threads, inbox, self.outbox,
                                      self.running, self.pending, max_pending, maps_q, contest, contest_share,
                                      cache_dir))
                self.inboxes.append(inbox)
                self.procs.append(p)
        self.lock = threading.Lock()
        self.packs = []
        self.decisions = 0
        self.turns = 0
        self.error = None
        self.stop = False
        self._pool = ThreadPoolExecutor(max(2, threads // 2))
        self._rebuilding = []
        self.rebuild_s, self.rebuilt = 0.0, 0                # thread-seconds rebuilding packs, their decisions
        self._reader = threading.Thread(target=self._read, daemon=True)

    def set_weights(self, actor, critic, league_paths: list[str], recent_n: int):
        """the learner's weights and the league (snapshot paths, the first recent_n recent)"""
        msg = ("weights", _numpy(actor), _numpy(critic), list(league_paths), recent_n)
        for q in self.inboxes:
            q.put(msg)

    def start(self):
        for p in self.procs:
            p.start()
        self._reader.start()
        self.running.set()

    def pause(self):
        """stop stepping; workers finish their in-flight step (results arrive as packs later)"""
        self.running.clear()
        time.sleep(0.5)
        if self.error:
            raise RuntimeError("an actor process died") from self.error

    def resume(self):
        self.running.set()

    def take(self):
        with self.lock:
            done = [f for f in self._rebuilding if f.done()]
            self._rebuilding = [f for f in self._rebuilding if not f.done()]
        packs = [f.result() for f in done]
        n = sum(p.decisions() for p in packs)
        with self.pending.get_lock():
            self.pending.value -= n
        return packs, n

    def _read(self):
        from . import _native
        while not self.stop:
            try:
                kind, payload, n, turns = self.outbox.get(timeout=1.0)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in self.procs) and self.error is None:
                    self.error = RuntimeError("an actor process exited")
                continue
            if kind == "log":
                self.log(payload)
                continue
            if kind == "error":
                self.log(payload)
                self.error = RuntimeError(payload)
                continue
            f = self._pool.submit(self._rebuild, payload)
            with self.lock:
                self._rebuilding.append(f)
                self.turns += turns

    def _rebuild(self, payload):
        t = time.perf_counter()
        pack = __import__("dragons_rl._native", fromlist=["Pack"]).Pack.deserialize(payload)
        with self.lock:
            self.rebuild_s += time.perf_counter() - t
            self.rebuilt += pack.decisions()
        return pack

    def close(self):
        self.stop = True
        for q in self.inboxes:
            q.put(("stop",))


def _numpy(tree):
    import jax
    return jax.tree.map(np.asarray, tree)
