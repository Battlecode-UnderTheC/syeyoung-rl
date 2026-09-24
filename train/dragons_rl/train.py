"""Spec section 10.4: the reference loop.

    python -m dragons_rl.train --run runs/seed0 --seed 0 --fixtures FIXTURES.json

creates the run (actor, critic, S0, the frozen run_manifest.json) or resumes
it from its latest completed-update checkpoint, then runs updates to 2,000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time

import jax
import numpy as np

from . import _native, codec, collect, evaluate, league, manifest, model, ppo

MATCHES = 32
SNAPSHOT_EVERY = 20
VALIDATE_EVERY = 100


def codec_hash() -> str:
    """what a checkpoint must agree on: the codec's field layout and the model's fixed choices"""
    spec = {"fields": _native.FIELDS, "tile": _native.TILE, "entity": _native.ENTITY, "status": _native.STATUS,
            "critic_dragon": _native.CRITIC_DRAGON, "critic_global": _native.CRITIC_GLOBAL,
            "max_tokens": model.MAX_TOKENS, "sonar_symbols": model.SONAR_SYMBOLS, "hidden": model.HIDDEN,
            "actor_parameters": model.ACTOR_PARAMETERS, "critic_parameters": model.CRITIC_PARAMETERS,
            "schema": manifest.SCHEMA}
    return manifest.sha256_text(json.dumps(spec, sort_keys=True))


class Run:
    def __init__(self, path: pathlib.Path, log):
        self.path = path
        self.log = log
        (path / "history").mkdir(parents=True, exist_ok=True)
        (path / "ckpt").mkdir(exist_ok=True)
        self.metrics = open(path / "metrics.jsonl", "a")

    # ---- state ---------------------------------------------------------------------------------

    def create(self, seed: int, fixtures_hash: str):
        root = np.random.Generator(np.random.PCG64(seed))
        keys = root.integers(0, 2**63, 2)
        actor = jax.tree.map(np.asarray, model.init_actor(jax.random.PRNGKey(int(keys[0]))))
        critic = jax.tree.map(np.asarray, model.init_critic(jax.random.PRNGKey(int(keys[1]))))
        assert model.count(actor) == model.ACTOR_PARAMETERS and model.count(critic) == model.CRITIC_PARAMETERS
        s0 = self.path / "history" / "S0.npz"
        s0_sha = league.save_params(s0, actor)
        entries = manifest.build_pool(self.path / "maps")
        opponents = [manifest.opponent("greedy-v1"), manifest.opponent("random-v1"),
                     manifest.opponent("dragons-v1", s0)]
        m = manifest.write(self.path, seed, entries, opponents, opponents, fixtures_hash)
        self.state = {
            "update": 0, "actor": actor, "critic": critic,
            "actor_opt": ppo.adam_init(actor), "critic_opt": ppo.adam_init(critic),
            "rng": root.bit_generator.state, "history": [{"path": str(s0), "sha256": s0_sha, "update": 0}],
            "manifest": manifest.digest(m), "codec": codec_hash(),
            "validation": [], "incumbent": {"update": 0, "path": str(s0), "sha256": s0_sha, "scores": None},
        }
        self.save()

    def resume(self):
        ckpts = sorted(self.path.glob("ckpt/*.pkl"), key=lambda p: int(p.stem))
        if not ckpts:
            return False
        with open(ckpts[-1], "rb") as f:
            self.state = pickle.load(f)
        if self.state["codec"] != codec_hash():
            raise ValueError("the checkpoint's schema, dimensions, move cap or sonar vocabulary differ from this code")
        return True

    def save(self):
        u = self.state["update"]
        path = self.path / "ckpt" / f"{u}.pkl"
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

    # ---- one update --------------------------------------------------------------------------------

    def select(self, m: dict, rng: np.random.Generator):
        """32 matches: maps by training weight, 16 learner-A and 16 learner-B
        (shuffled), each with a frozen opponent from the league"""
        train = manifest.split(m, "train")
        w = np.array([e["weight"] for e in train], np.float64)
        maps = [train[i] for i in rng.choice(len(train), MATCHES, p=w / w.sum())]
        sides = np.array([0] * (MATCHES // 2) + [1] * (MATCHES // 2))
        rng.shuffle(sides)
        opponents = [league.choose_opponent(rng, self.state["history"]) for _ in range(MATCHES)]
        return maps, sides, opponents

    def update(self, m: dict, devices, threads: int):
        s = self.state
        u = s["update"]
        rng = np.random.Generator(np.random.PCG64(0))
        rng.bit_generator.state = s["rng"]
        maps, sides, opponents = self.select(m, rng)
        behaviour = s["actor"]
        policies, slot_of = [behaviour], {}
        for o in opponents:
            if o is not league.CURRENT and o["sha256"] not in slot_of:
                slot_of[o["sha256"]] = len(policies)
                policies.append(league.load_params(o["path"], behaviour, o["sha256"]))
        setups = []
        for e, side, o in zip(maps, sides, opponents):
            policy = [0, 0]
            policy[1 - side] = 0 if o is league.CURRENT else slot_of[o["sha256"]]
            record = [False, False]
            record[side] = True
            setups.append({"map": open(e["path"]).read(), "policy": policy, "record": record})
        t0 = time.time()
        collectors, parts = collect.play_parallel(setups, policies, s["critic"], m["training_seed"], u,
                                                  list(range(MATCHES)), devices, threads, log=self.log)
        t_collect = time.time() - t0
        shards = [ppo.Shard(c, d).prepare(max(1, threads // len(collectors))) for c, d in zip(collectors, devices)]
        t1 = time.time()
        actor, critic, aopt, copt, stats = ppo.update(s["actor"], s["critic"], s["actor_opt"], s["critic_opt"], shards,
                                                      u, MATCHES, log=self.log)
        t_train = time.time() - t1
        # results from the learner's side
        wins = draws = losses = 0
        rounds = []
        for c, positions in zip(collectors, parts):
            r = c.results()
            for w, rnd, i in zip(r["winner"], r["rounds"], positions):
                side = sides[i]
                wins += int(w == side)
                draws += int(w < 0)
                losses += int(w >= 0 and w != side)
                rounds.append(int(rnd))
        acts = np.concatenate([sh.decisions["kind"] for sh in shards]) if shards else np.zeros(0)
        steps = np.concatenate([sh.decisions["nsteps"] for sh in shards]) if shards else np.zeros(0)
        masks = np.concatenate([sh.decisions["sonar_mask"] for sh in shards]) if shards else np.zeros(0)
        row = {"update": u, "time": time.time(), "collect_s": t_collect, "train_s": t_train,
               "win": wins, "draw": draws, "loss": losses, "mean_rounds": float(np.mean(rounds)),
               "split_fraction": float(np.mean(acts == 1)) if len(acts) else 0.0,
               "mean_move_steps": float(steps[acts == 0].mean()) if (acts == 0).any() else 0.0,
               "sonar_per_direction": [float(np.mean((masks >> d) & 1)) for d in range(4)] if len(masks) else [],
               "opponents_current": sum(o is league.CURRENT for o in opponents), **stats}
        s.update(update=u + 1, actor=actor, critic=critic, actor_opt=aopt, critic_opt=copt, rng=rng.bit_generator.state)
        if (u + 1) % SNAPSHOT_EVERY == 0:
            path = self.path / "history" / f"S{u + 1}.npz"
            s["history"].append({"path": str(path), "sha256": league.save_params(path, actor), "update": u + 1})
        if (u + 1) % VALIDATE_EVERY == 0:
            row["validation"] = self.validate(m, devices, threads)
        self.save()
        self.record(row)
        self.log(f"update {u}: W/D/L {wins}/{draws}/{losses}, {stats['decisions']} decisions, "
                 f"{stats['events']:.0f} events, kl {['%.4f' % k for k in stats['kl']]}, "
                 f"H {stats.get('h_phys', 0):.3f}/{stats.get('h_sonar', 0):.3f}, "
                 f"value mse {stats.get('value_mse_2', 0):.3f}, collect {t_collect:.0f}s train {t_train:.0f}s")

    # ---- validation (section 11) ---------------------------------------------------------------

    def validate(self, m: dict, devices, threads: int) -> dict:
        s = self.state
        maps = manifest.split(m, "validation")
        opps = m["validation_opponents"]
        weights = {o["sha256"]: league.load_params(o["artifact_path"], s["actor"], o["sha256"])
                   for o in opps if o["adapter_id"] == "dragons-v1"}
        run = lambda params: evaluate.scores(params, weights, s["critic"], maps, opps, evaluate.VALIDATION_SEEDS,
                                             devices, threads)
        inc = s["incumbent"]
        if inc["scores"] is None:
            inc["scores"] = run(league.load_params(inc["path"], s["actor"], inc["sha256"]))
        cand = run(s["actor"])
        result = evaluate.compare(cand, inc["scores"])
        result["update"] = s["update"]
        if result["promote"]:
            path = self.path / "history" / f"incumbent-{s['update']}.npz"
            s["incumbent"] = {"update": s["update"], "path": str(path), "sha256": league.save_params(path, s["actor"]),
                              "scores": cand}
        s["validation"].append(result)
        self.log(f"validation at {s['update']}: candidate {result['candidate_score']:.3f} vs incumbent "
                 f"{result['incumbent_score']:.3f} (update {inc['update']}), diff CI [{result['ci_low']:.3f}, "
                 f"{result['ci_high']:.3f}] over {result['groups']} groups -> {'promote' if result['promote'] else 'keep'}")
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=pathlib.Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixtures", type=pathlib.Path, help="protocol fixture results (JSON) for a new run")
    p.add_argument("--updates", type=int, default=ppo.UPDATES)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--devices", type=str, default="", help="comma-separated device indices (default: all)")
    args = p.parse_args()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    # compiled programs survive restarts (many shapes: board, inbox and split buckets)
    jax.config.update("jax_compilation_cache_dir", str(args.run.resolve() / "jax_cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    devs = jax.devices()
    if args.devices:
        devs = [devs[int(i)] for i in args.devices.split(",")]
    run = Run(args.run, log)
    if not run.resume():
        if not args.fixtures:
            sys.exit("a new run needs --fixtures (the protocol fixture results)")
        fixtures = args.fixtures.read_bytes()
        if not json.loads(fixtures).get("passed"):
            sys.exit("the protocol fixtures did not pass")
        run.create(args.seed, hashlib.sha256(fixtures).hexdigest())
        log(f"created {args.run}: seed {args.seed}, S0 saved, manifest frozen")
    m = manifest.load(args.run)
    if manifest.digest(m) != run.state["manifest"]:
        raise ValueError("the run manifest changed after it was frozen")
    log(f"update {run.state['update']} of {args.updates} on {[str(d) for d in devs]}")
    while run.state["update"] < args.updates:
        run.update(m, devs, args.threads)


if __name__ == "__main__":
    main()
