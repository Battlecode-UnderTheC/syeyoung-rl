"""Self-play PPO for dragons-spatial-v1 on the native sim (spec sections 9, 11, 12).

Collection plays complete matches with the behaviour weights frozen, against a
mixture of the current policy (self-play) and frozen historical snapshots; the
update then runs the recurrent joint-command PPO of `snake_rl.algo.ppo`.
Scripted baselines are evaluated separately and never enter the buffer.

    PYTHONPATH=.:cpp:dragons-env python -m snake_rl.training.spatial_train --iterations 5

Matches are played in worker processes (fork), each with its own copy of the
weights; the learner process owns the optimiser.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import pathlib
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from ..algo.ppo import SpatialPPOConfig, ppo_update_spatial
from ..env import mapgen
from ..env.actions import ActionSpace, CommandCodec
from ..env.schema import ACTION_CODEC_VERSION, SCHEMA_VERSION, check_schema, schema_hash
from ..model.actor import SpatialActor, SpatialActorConfig
from ..model.critic import EventCritic
from ..model.policy import SpatialPolicy
from ..policies import GreedyPearlPolicy
from ..rollout.collector import MatchRecord, ScriptedAdapter, play_event_match
from .spatial_config import SpatialTrainConfig


# -- curriculum ---------------------------------------------------------------------

def sample_map(cfg: SpatialTrainConfig, rng: random.Random):
    size = rng.choice(cfg.board_choices)
    return mapgen.random_map(
        rng, min_size=size, max_size=size,
        dragons_per_team=rng.choice(cfg.dragons_per_team_choices),
        length=rng.choice(cfg.length_choices),
        kelp_density=rng.uniform(*cfg.kelp_range),
        portal_pairs=rng.choice(cfg.portal_pairs_choices),
        pearl_gap_choices=cfg.pearl_gap_choices)


# -- league ---------------------------------------------------------------------------

class League:
    """Frozen historical opponents. Opponent 0 is always 'current self'."""

    def __init__(self, cfg: SpatialTrainConfig):
        self.cfg = cfg
        self.snapshots: list[tuple[int, dict]] = []      # (iteration, state_dict)
        self.results: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

    def add(self, iteration: int, actor: SpatialActor) -> None:
        self.snapshots.append((iteration, {k: v.detach().cpu().numpy().copy()
                                           for k, v in actor.state_dict().items()}))
        if len(self.snapshots) > self.cfg.max_league:
            # keep the oldest few (they reveal forgotten weaknesses), thin the middle
            keep = self.snapshots[:4] + self.snapshots[4:-4][1::2] + self.snapshots[-4:]
            self.snapshots = keep

    def pick(self, rng: random.Random) -> tuple[int, dict | None]:
        """(opponent id, weights). Id 0 is self-play, i+1 is snapshot i."""
        if not self.snapshots or rng.random() < self.cfg.self_play_fraction:
            return 0, None
        n = len(self.snapshots)
        recent = max(1, n // 4)
        u = rng.random()
        if u < self.cfg.recent_fraction:
            i = rng.randrange(n - recent, n)
        elif u < self.cfg.recent_fraction + self.cfg.league_floor:
            i = rng.randrange(0, min(4, n))             # the floor for the oldest
        else:
            i = rng.randrange(n)
        return i + 1, self.snapshots[i][1]

    def record(self, opponent: int, learner_result: float) -> None:
        self.results[opponent][{1.0: "win", -1.0: "loss"}.get(learner_result, "draw")] += 1

    def state(self) -> dict:
        return {"snapshots": self.snapshots, "results": {k: dict(v) for k, v in self.results.items()}}

    def load(self, state: dict) -> None:
        self.snapshots = list(state["snapshots"])
        for k, v in state["results"].items():
            self.results[k] = collections.Counter(v)


# -- workers ----------------------------------------------------------------------------

def _play(args) -> MatchRecord:
    (cfg_dict, weights, opponent_weights, opponent_id, spec_text, seed, match_id, learner_team,
     scripted) = args
    torch.set_num_threads(1)
    cfg = config_from_dict(cfg_dict)
    codec = CommandCodec(cfg.actor.max_move_steps)
    actor = SpatialActor(cfg.actor).eval()
    actor.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    spec = spec_from_text(spec_text)
    policies: dict[str, object] = {}
    learner_teams: set[str] = set()
    for team in "AB":
        if scripted and team != learner_team:
            policies[team] = ScriptedAdapter(GreedyPearlPolicy(ActionSpace(), seed=seed))
        elif opponent_weights is None or team == learner_team:
            policies[team] = SpatialPolicy(actor, codec, match_id=match_id)
            learner_teams.add(team)
        else:
            frozen = SpatialActor(cfg.actor).eval()
            frozen.load_state_dict({k: torch.from_numpy(v) for k, v in opponent_weights.items()})
            policies[team] = SpatialPolicy(frozen, codec, match_id=match_id)
    if scripted:
        learner_teams = {learner_team}
    with torch.no_grad():
        rec = play_event_match(spec, policies, learner_teams, seed=seed, match_id=match_id,
                               snapshot_every=cfg.snapshot_every, max_rounds=cfg.max_rounds,
                               opponent_ids={t: (0 if t in learner_teams else opponent_id) for t in "AB"})
    rec.info["opponent"] = opponent_id
    rec.info["scripted"] = scripted
    rec.info["learner_team"] = learner_team
    return rec


def spec_from_text(text: str):
    """Rebuild a MapSpec from its map text (workers only need the text)."""
    from dragons_env.mapgen import MapSpec
    lines = text.splitlines()
    w, h = map(int, lines[0].split()[1:3])
    spec = MapSpec(w, h, [])
    spec.tile_gaps = {}
    for line in lines:
        parts = line.split()
        if parts[0] == "MAP_NAME":
            spec.name = " ".join(parts[1:])
        elif parts[0] == "UNIT_LIMIT":
            spec.unit_limit = int(parts[1])
        elif parts[0] == "TILE":
            spec.tile_gaps[(int(parts[1]), int(parts[2]))] = (int(parts[3]), int(parts[4]))
        elif parts[0] == "EDGE":
            if parts[2] == "1":
                spec.kelp.append(int(parts[1]))
            elif parts[2] == "2":
                spec.portals.append((int(parts[1]), int(parts[3])))
        elif parts[0] == "DRAGON":
            n = int(parts[2])
            coords = list(map(int, parts[3:3 + 2 * n]))
            spec.dragons.append((int(parts[1]), list(zip(coords[0::2], coords[1::2]))))
    return spec


def config_from_dict(d: dict) -> SpatialTrainConfig:
    out = dict(d)
    out["actor"] = SpatialActorConfig(**d["actor"])
    out["ppo"] = SpatialPPOConfig(**d["ppo"])
    for k in ("board_choices", "portal_pairs_choices", "length_choices", "dragons_per_team_choices"):
        out[k] = tuple(d[k])
    out["kelp_range"] = tuple(d["kelp_range"])
    out["pearl_gap_choices"] = tuple(tuple(x) for x in d["pearl_gap_choices"])
    out["actor"].spatial_widths = tuple(out["actor"].spatial_widths)
    return SpatialTrainConfig(**out)


# -- learner ----------------------------------------------------------------------------

def save_checkpoint(path: pathlib.Path, actor, critic, optimizer, cfg, iteration, league,
                    rng_state, best: bool = False) -> None:
    torch.save({
        "schema_version": SCHEMA_VERSION, "schema_hash": schema_hash(),
        "action_codec_version": ACTION_CODEC_VERSION,
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(), "config": cfg.as_dict(),
        "iteration": iteration, "league": league.state(), "rng": rng_state,
        "torch_rng": torch.get_rng_state(), "best": best,
    }, path)


def load_checkpoint(path: str, actor, critic, optimizer, league) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    check_schema(state.get("schema_version", "missing"), state.get("schema_hash"))
    if state.get("action_codec_version") != ACTION_CODEC_VERSION:
        raise ValueError("action codec version mismatch")
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    optimizer.load_state_dict(state["optimizer"])
    league.load(state["league"])
    torch.set_rng_state(state["torch_rng"])
    return int(state["iteration"]) + 1


def raise_descriptor_limit(target: int = 65536) -> None:
    """A forked pool with dozens of workers exchanging match records needs
    more descriptors than the usual 1024 soft limit; raise it up to the hard one."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        wanted = min(target, hard) if hard != resource.RLIM_INFINITY else target
        if soft < wanted:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    except (ImportError, ValueError, OSError):
        pass


def main(cfg: SpatialTrainConfig, workers: int = 4) -> None:
    raise_descriptor_limit()
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    codec = CommandCodec(cfg.actor.max_move_steps)
    actor = SpatialActor(cfg.actor).to(device)
    critic = EventCritic().to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                                 lr=cfg.learning_rate)
    league = League(cfg)
    run_dir = pathlib.Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    start = load_checkpoint(cfg.resume, actor, critic, optimizer, league) if cfg.resume else 0
    print(f"actor params {sum(p.numel() for p in actor.parameters()):,} "
          f"critic params {sum(p.numel() for p in critic.parameters()):,} "
          f"schema {SCHEMA_VERSION}/{schema_hash()}", flush=True)

    ctx = mp.get_context("fork")
    pool = ctx.Pool(workers) if workers > 1 else None
    best_score = -1e9
    try:
        for iteration in range(start, cfg.iterations):
            # numpy, not tensors: torch shares tensors with forked workers through
            # file descriptors, and arenas x parameters of them exhausts the limit
            weights = {k: v.detach().cpu().numpy() for k, v in actor.state_dict().items()}
            cfg_dict = cfg.as_dict()
            tasks = []
            for a in range(cfg.arenas):
                opponent_id, opp = league.pick(rng)
                tasks.append((cfg_dict, weights, opp, opponent_id, sample_map(cfg, rng).to_text(),
                              rng.randrange(2 ** 31), iteration * 100000 + a, rng.choice("AB"), False))
            for a in range(cfg.scripted_eval_matches):
                tasks.append((cfg_dict, weights, None, -1, sample_map(cfg, rng).to_text(),
                              rng.randrange(2 ** 31), iteration * 100000 + 90000 + a,
                              rng.choice("AB"), True))
            t0 = time.time()
            records = pool.map(_play, tasks) if pool else [_play(t) for t in tasks]
            collect_s = time.time() - t0

            train = [r for r in records if not r.info["scripted"]]
            evals = [r for r in records if r.info["scripted"]]
            t0 = time.time()
            actor.train()
            metrics = ppo_update_spatial(actor, critic, optimizer, train, cfg.ppo, codec, device, np_rng)
            actor.eval()
            learn_s = time.time() - t0

            # diagnostics (spec 17): behaviour statistics, never rewards
            deaths: collections.Counter = collections.Counter()
            splits = moves = steps = events = 0
            for r in train:
                deaths.update(r.info["deaths"])
                events += r.info["events"]
                for l in r.lifetimes.values():
                    if not l.learner:
                        continue
                    for t in l.turns:
                        if t.command.top == 1:
                            splits += 1
                        else:
                            moves += 1
                            steps += len(t.command.dirs)
                if r.info["opponent"] > 0:
                    team = r.info["learner_team"]
                    league.record(r.info["opponent"],
                                  1.0 if r.result == team else (-1.0 if r.result in "AB" else 0.0))
            scripted_score = [1.0 if r.result == r.info["learner_team"]
                              else (-1.0 if r.result in "AB" else 0.0) for r in evals]
            log = {
                "iteration": iteration, "time/collect_s": collect_s, "time/learn_s": learn_s,
                "match/rounds": float(np.mean([r.info["rounds"] for r in train])),
                "match/events": events,
                "behaviour/split_fraction": splits / max(splits + moves, 1),
                "behaviour/mean_move_steps": steps / max(moves, 1),
                **{f"death/{k}": v / max(len(train), 1) for k, v in deaths.items()},
                **metrics,
            }
            if scripted_score:
                log["eval/vs_scripted"] = float(np.mean(scripted_score))
            print(f"iter {iteration:4d} rounds {log['match/rounds']:5.1f} events {events:6d} "
                  f"split {log['behaviour/split_fraction']:.2f} "
                  f"steps {log['behaviour/mean_move_steps']:.2f} "
                  f"kl {metrics.get('ppo/approx_kl', float('nan')):.4f} "
                  f"vs_scripted {log.get('eval/vs_scripted', float('nan')):+.2f} "
                  f"collect {collect_s:5.1f}s learn {learn_s:5.1f}s", flush=True)

            if iteration % cfg.snapshot_interval == 0:
                league.add(iteration, actor)
            if iteration % cfg.save_every == 0:
                save_checkpoint(run_dir / f"checkpoint-{iteration:06d}.pt", actor, critic, optimizer,
                                cfg, iteration, league, rng.getstate())
                save_checkpoint(run_dir / "latest.pt", actor, critic, optimizer, cfg, iteration,
                                league, rng.getstate())
            score = log.get("eval/vs_scripted", -1e9)
            if score > best_score:
                best_score = score
                save_checkpoint(run_dir / "best-held-out.pt", actor, critic, optimizer, cfg,
                                iteration, league, rng.getstate(), best=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def parse_args() -> tuple[SpatialTrainConfig, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=SpatialTrainConfig.iterations)
    parser.add_argument("--arenas", type=int, default=SpatialTrainConfig.arenas)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SpatialTrainConfig.seed)
    parser.add_argument("--run-name", type=str, default=SpatialTrainConfig.run_name)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--sonar", action="store_true")
    parser.add_argument("--max-move-steps", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    cfg = SpatialTrainConfig(iterations=args.iterations, arenas=args.arenas, seed=args.seed,
                             run_name=args.run_name, resume=args.resume, device=args.device,
                             max_rounds=args.max_rounds)
    cfg.actor = dataclasses.replace(cfg.actor, sonar=args.sonar, max_move_steps=args.max_move_steps)
    return cfg, args.workers


if __name__ == "__main__":
    config, n_workers = parse_args()
    main(config, n_workers)
