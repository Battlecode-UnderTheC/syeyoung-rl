"""The training loop, built for throughput.

    python -m dragons_rl.train --run runs/fast0

Actor threads (one per GPU) play many matches at once, continuously, on fresh
fair maps (plus the contest maps), against the learner itself or frozen
league snapshots. Every BATCH learner decisions from finished matches, the
actors pause, the learner trains on them (ppo.update), and the actors resume
with the new weights. Matches in progress simply continue, so no update waits
for a straggler. Validation against fixed held-out maps and opponents runs
every VALIDATE_EVERY updates.

Model and codec are the v1 spec's; the loop, batching, sampling and critic
usage are throughput deviations (see ppo.py and collect.py).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import pickle
import queue
import random
import threading
import time

# before jax starts: the training calls are sized for most of the card
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import jax
import numpy as np

from . import collect, evaluate, league, manifest, model, ppo

BATCH = 65536                 # learner decisions per update
SNAPSHOT_EVERY = 20
VALIDATE_EVERY = 100
CONTEST_SHARE = 0.1
DEATH_CAUSES = ("wall", "self", "other_body", "head_to_head", "no_valid_action")


# ---- fresh maps, generated in other processes ---------------------------------------------------

def _gen_maps(q, seed: int):
    from . import fairmaps
    rng = random.Random(seed)
    i = 0
    while True:
        q.put(fairmaps.fair_map(rng, 7, 64, f"fresh-{seed}-{i}"))
        i += 1


class Maps:
    """fresh fair maps from worker processes; recent ones are reused if the workers fall behind"""

    def __init__(self, workers: int = 4):
        from dragons_env import maps as contest
        self.contest = list(contest.contest_maps().values())
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=256)
        self.procs = [ctx.Process(target=_gen_maps, args=(self.q, random.randrange(2**31)), daemon=True)
                      for _ in range(workers)]
        for p in self.procs:
            p.start()
        self.recent = [self.q.get() for _ in range(64)]
        self.lock = threading.Lock()

    def next(self) -> str:
        if random.random() < CONTEST_SHARE:
            return random.choice(self.contest)
        try:
            text = self.q.get_nowait()
            with self.lock:
                self.recent[random.randrange(len(self.recent))] = text
            return text
        except queue.Empty:
            with self.lock:
                return random.choice(self.recent)


# ---- the league ------------------------------------------------------------------------------------

class League:
    """slot 0 is the learner; slots 1.. hold frozen snapshots: the 8 most recent
    and a random sample of older ones, redrawn after every update"""

    def __init__(self, history: list):
        self.history = history          # [{"path", "update"}]
        self.slots = []
        self.recent_n = 0

    def refresh(self, template):
        recent = self.history[-8:]
        older = self.history[:-8]
        older = random.sample(older, min(len(older), collect.SLOTS - 1 - len(recent)))
        self.slots = recent + older
        self.recent_n = len(recent)
        return [league.load_params(pathlib.Path(h["path"]), template) for h in self.slots]

    def opponent(self) -> int:
        """20% the learner itself, 40% a recent snapshot, 40% an older one"""
        u = random.random()
        if u < 0.2 or not self.slots:
            return 0
        if u < 0.6 or len(self.slots) == self.recent_n:
            return 1 + random.randrange(self.recent_n)
        return 1 + self.recent_n + random.randrange(len(self.slots) - self.recent_n)


# ---- the run -------------------------------------------------------------------------------------------

class Run:
    def __init__(self, path: pathlib.Path, log):
        self.path = path
        self.log = log
        (path / "history").mkdir(parents=True, exist_ok=True)
        (path / "ckpt").mkdir(exist_ok=True)
        self.metrics = open(path / "metrics.jsonl", "a")

    def create(self):
        actor = jax.tree.map(np.asarray, model.init_actor(jax.random.PRNGKey(random.randrange(2**31))))
        critic = jax.tree.map(np.asarray, model.init_critic(jax.random.PRNGKey(random.randrange(2**31))))
        s0 = self.path / "history" / "S0.npz"
        league.save_params(s0, actor)
        entries = manifest.build_pool(self.path / "maps")
        opponents = [manifest.opponent("greedy-v1"), manifest.opponent("random-v1"), manifest.opponent("dragons-v1", s0)]
        manifest.write(self.path, 0, entries, opponents, opponents, "throughput-run")
        self.state = {"update": 0, "actor": actor, "critic": critic, "actor_opt": ppo.adam_init(actor),
                      "critic_opt": ppo.adam_init(critic), "history": [{"path": str(s0), "update": 0}],
                      "validation": [], "incumbent": {"update": 0, "path": str(s0), "scores": None},
                      "decisions": 0, "turns": 0}
        self.save()

    def resume(self) -> bool:
        ckpts = sorted(self.path.glob("ckpt/*.pkl"), key=lambda p: int(p.stem))
        if not ckpts:
            return False
        with open(ckpts[-1], "rb") as f:
            self.state = pickle.load(f)
        return True

    def save(self):
        path = self.path / "ckpt" / f"{self.state['update']}.pkl"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        for old in sorted(self.path.glob("ckpt/*.pkl"), key=lambda p: int(p.stem))[:-2]:
            old.unlink()

    def record(self, row: dict):
        self.metrics.write(json.dumps(row, default=float) + "\n")
        self.metrics.flush()

    def validate(self, devices, threads):
        s = self.state
        m = manifest.load(self.path)
        maps = manifest.split(m, "validation")
        opps = m["validation_opponents"]
        weights = {o["sha256"]: league.load_params(pathlib.Path(o["artifact_path"]), s["actor"])
                   for o in opps if o["adapter_id"] == "dragons-v1"}
        run = lambda params: evaluate.scores(params, weights, s["critic"], maps, opps, evaluate.VALIDATION_SEEDS,
                                             devices, threads)
        inc = s["incumbent"]
        if inc["scores"] is None:
            inc["scores"] = run(league.load_params(pathlib.Path(inc["path"]), s["actor"]))
        cand = run(s["actor"])
        result = evaluate.compare(cand, inc["scores"])
        result["update"] = s["update"]
        if result["promote"]:
            path = self.path / "history" / f"incumbent-{s['update']}.npz"
            league.save_params(path, s["actor"])
            s["incumbent"] = {"update": s["update"], "path": str(path), "scores": cand}
        s["validation"].append(result)
        self.log(f"validation at {s['update']}: candidate {result['candidate_score']:.3f} vs incumbent "
                 f"{result['incumbent_score']:.3f} (update {inc['update']}), diff CI [{result['ci_low']:.3f}, "
                 f"{result['ci_high']:.3f}] over {result['groups']} groups -> {'promote' if result['promote'] else 'keep'}")
        return result


def rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def learner_results(packs):
    """W/D/L, rounds and death causes from the learner's side of every harvested match"""
    w = d = l = 0
    rounds = []
    causes = np.zeros((2, 5), np.int64)
    for p in packs:
        r = p.results()
        for winner, rnd, rec, dead in zip(r["winner"], r["rounds"], r["record"], r["deaths"]):
            if not rec.any():
                continue
            side = int(np.argmax(rec))
            w += int(winner == side)
            d += int(winner < 0)
            l += int(winner >= 0 and winner != side)
            rounds.append(int(rnd))
            causes[0] += dead[side]
            causes[1] += dead[1 - side]
    return w, d, l, rounds, causes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=pathlib.Path, required=True)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--slots", type=int, default=512, help="concurrent matches per GPU")
    p.add_argument("--groups", type=int, default=2)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--devices", type=str, default="")
    p.add_argument("--updates", type=int, default=10**9)
    args = p.parse_args()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    # checkpoints are only written at update boundaries (atomically), so a stop
    # can exit at once instead of waiting on the actor threads and map workers
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    jax.config.update("jax_compilation_cache_dir", str(args.run.resolve() / "jax_cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    devs = jax.devices()
    if args.devices:
        devs = [devs[int(i)] for i in args.devices.split(",")]
    run = Run(args.run, log)
    if not run.resume():
        run.create()
        log(f"created {args.run}")
    s = run.state
    collect.DUMP_DIR = str(run.path / "nonfinite")
    maps = Maps()
    lg = League(s["history"])

    def new_match():
        side = random.randrange(2)
        policy = [0, 0]
        policy[1 - side] = lg.opponent()
        record = [False, False]
        record[side] = True
        return {"map": maps.next(), "policy": policy, "record": record}

    opponents = lg.refresh(s["actor"])
    actors = collect.Actors(devs, args.slots, args.groups, max(1, args.threads // len(devs)), new_match, log)
    actors.set_weights([s["actor"]] + opponents, s["critic"])
    actors.start()
    log(f"update {s['update']}: {len(devs)} devices x {args.slots} matches, batch {args.batch}")
    while s["update"] < args.updates:
        u = s["update"]
        t0 = time.time()
        packs, have = [], 0
        t_log, turns_log = time.time(), 0
        while have < args.batch:
            time.sleep(1.0)
            if actors.error:
                raise RuntimeError("an actor thread died") from actors.error
            got, n = actors.take()
            packs += got
            have += n
            if time.time() - t_log > 30:
                with actors.lock:
                    turns_now = actors.turns
                log(f"collecting: {have} learner decisions harvested, "
                    f"{(turns_now - turns_log) / (time.time() - t_log):.0f} turns/s, rss {rss_gb():.1f} GB")
                t_log, turns_log = time.time(), turns_now
        t_collect = time.time() - t0
        # the actors keep playing (with the previous weights) while the learner trains
        with actors.lock:
            turns, actors.turns = actors.turns, 0
        t1 = time.time()
        shards, full = ppo.make_shards(packs, devs, args.threads)
        actor, critic, aopt, copt, stats = ppo.update(s["actor"], s["critic"], s["actor_opt"], s["critic_opt"],
                                                      shards, full, log=log)
        t_train = time.time() - t1
        s.update(update=u + 1, actor=actor, critic=critic, actor_opt=aopt, critic_opt=copt)
        if (u + 1) % SNAPSHOT_EVERY == 0:
            path = run.path / "history" / f"S{u + 1}.npz"
            league.save_params(path, actor)
            s["history"].append({"path": str(path), "update": u + 1})
        opponents = lg.refresh(actor)
        actors.set_weights([actor] + opponents, critic)
        row_val = None
        if (u + 1) % VALIDATE_EVERY == 0:
            actors.pause()
            row_val = run.validate(devs, args.threads)
            actors.resume()
        run.save()
        wall = time.time() - t0
        w, d, l, rounds, causes = learner_results(packs)
        dec = stats["decisions"]
        s["decisions"] += dec
        s["turns"] += turns
        row = {"update": u, "time": time.time(), "collect_s": t_collect, "train_s": t_train, "wall_s": wall,
               "learner_decisions_per_s": dec / wall, "turns_per_s": turns / max(t_collect, 1e-9),
               "collect_turns_per_s": turns / max(t_collect, 1e-9), "train_decisions_per_s": dec / max(t_train, 1e-9),
               "win": w, "draw": d, "loss": l, "score": (w + 0.5 * d) / max(1, w + d + l),
               "mean_rounds": float(np.mean(rounds)) if rounds else 0.0,
               **{f"deaths_per100/{c}": 100.0 * float(causes[0, k]) / max(1, dec) for k, c in enumerate(DEATH_CAUSES)},
               "total_decisions": s["decisions"], "total_turns": s["turns"], **stats}
        if row_val:
            row["validation"] = row_val
        run.record(row)
        log(f"update {u}: {dec} decisions from {stats['matches']} matches, W/D/L {w}/{d}/{l}, "
            f"rounds {row['mean_rounds']:.1f}, kl max {max(stats['kl'] or [0]):.4f}, {stats['actor_steps']} steps, "
            f"H {stats.get('h_phys', 0):.3f}/{stats.get('h_sonar', 0):.3f}, vmse {stats.get('value_mse', 0):.3f}, "
            f"{row['learner_decisions_per_s']:.0f} learner decisions/s ({turns / wall:.0f} turns/s; "
            f"collect {t_collect:.0f}s at {row['collect_turns_per_s']:.0f} turns/s, train {t_train:.0f}s at "
            f"{row['train_decisions_per_s']:.0f}/s)")
        log(f"update {u}: host memory {rss_gb():.1f} GB")
        if rss_gb() > 40:                    # compiled programs pile up: reload them from the disk cache
            jax.clear_caches()
            log(f"update {u}: cleared compiled caches, host memory {rss_gb():.1f} GB")
        log(f"update {u}: deaths (learner) " + " ".join(f"{c} {int(n)}" for c, n in zip(DEATH_CAUSES, causes[0])))
    actors.stop = True


if __name__ == "__main__":
    main()
