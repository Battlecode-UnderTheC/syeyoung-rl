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


def _worker(i: int, gpu: int, mem_fraction: float, slots: int, threads: int, inbox, outbox, running, pending,
            max_pending: int, maps_q, contest: list, contest_share: float, cache_dir: str | None):
    parent = os.getppid()                                   # the trainer: gone means stop
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    # collection is host-bound: autotuning its many call shapes costs minutes and buys little
    os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="
    try:
        open("/proc/self/oom_score_adj", "w").write("500")   # under memory pressure, a worker goes before the learner
    except OSError:
        pass
    try:
        import jax
        from . import collect, league
        if cache_dir:                                           # compiled programs shared with the learner and restarts
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        dev = jax.devices()[0]
        random.seed(os.getpid() ^ int(time.time()))
        maps = _WorkerMaps(maps_q, contest, contest_share)
        msg = inbox.get()                                       # the first weights
        cache: dict[str, dict] = {}
        n_slots = recent_n = 1
        stack = critic = None

        def load(msg):
            nonlocal n_slots, recent_n, stack, critic
            _, actor, critic_w, paths, recent_n = msg
            for p in list(cache):
                if p not in paths:
                    del cache[p]
            for p in paths:
                if p not in cache:
                    cache[p] = league.load_params(pathlib.Path(p), actor)
            policies = [actor] + [cache[p] for p in paths]
            n_slots = len(policies)
            stack = jax.device_put(collect.stack_policies(policies), dev)
            critic = jax.device_put(critic_w, dev)

        load(msg)
        g = collect.Group([new_match(maps.next(), n_slots, recent_n) for _ in range(slots)], threads)
        key = np.random.default_rng()
        step, turns = 0, 0
        t_log, n_log, s_log, spent = time.time(), 0, 0, {"finish": 0.0, "dispatch": 0.0, "idle": 0.0}
        while os.getppid() == parent:
            if time.time() - t_log > 30:
                dt = time.time() - t_log
                outbox.put(("log", f"worker {i} (gpu {gpu}): {(step - s_log) / dt:.1f} steps/s, "
                                   f"{n_log / dt:.0f} decisions/s; s/s in dispatch {spent['dispatch'] / dt:.2f}, "
                                   f"finish {spent['finish'] / dt:.2f}, idle {spent['idle'] / dt:.2f}", 0, 0))
                t_log, n_log, s_log = time.time(), 0, step
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
                if g.inflight is not None:
                    collect._finish(g)
                time.sleep(0.05)
                spent["idle"] += 0.05
                continue
            t0 = time.time()
            if g.inflight is not None:
                collect._finish(g)
                for m in range(g.col.num_matches):
                    if g.col.finished(m):
                        g.reset(m, new_match(maps.next(), n_slots, recent_n))
            t1 = time.time()
            key, n = collect._dispatch(g, stack, critic, dev, key, 128)
            spent["finish"] += t1 - t0
            spent["dispatch"] += time.time() - t1
            turns += n
            n_log += n
            step += 1
            if step % 16 == 0:
                pack = g.col.harvest()
                if pack.num_matches:
                    n_dec = pack.decisions()
                    with pending.get_lock():
                        pending.value += n_dec
                    outbox.put(("pack", pack.serialize(), n_dec, turns))
                    turns = 0
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
            f = self._pool.submit(_native.Pack.deserialize, payload)
            with self.lock:
                self._rebuilding.append(f)
                self.turns += turns

    def close(self):
        self.stop = True
        for q in self.inboxes:
            q.put(("stop",))


def _numpy(tree):
    import jax
    return jax.tree.map(np.asarray, tree)
