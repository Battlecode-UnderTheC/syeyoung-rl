"""Self-play PPO on the real engine.

Workers are processes: the engine callback holds the GIL, so threads anti-scale
and only processes give real parallelism. Each worker keeps its own CPU copy of
the actor, plays whole matches (the engine cannot be stepped or reset, so a
match is the unit of collection) and ships back trajectories.

The learner holds the critic on the GPU, computes one advantage per team per
round, and hands that same advantage to every dragon of that team that acted in
the round -- so an action is credited with the team's whole future, including
what happens after that dragon is dead.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from ..algo.gae import team_gae
from ..algo.ppo import ActorBatch, CriticBatch, PPOConfig, ppo_update
from ..env import ActionSpace, MatchRunner, WasmEngine, mapgen
from ..env.global_state import N_DRAGON_FEATURES, N_MAP_CHANNELS, TEAM_CHANNEL_PAIRS
from ..model.actor import Actor, ActorObs, JointAction
from ..model.critic import CriticState, TeamCritic
from ..policies import GreedyPearlPolicy
from ..rollout.collector import Trajectory, play_match
from .config import TrainConfig


def build_actor(config: TrainConfig, space: ActionSpace) -> Actor:
    return Actor(space.n, hidden=config.hidden, enable_sonar=config.enable_sonar)


def build_critic(config: TrainConfig) -> TeamCritic:
    return TeamCritic(N_MAP_CHANNELS, N_DRAGON_FEATURES,
                      team_channel_pairs=TEAM_CHANNEL_PAIRS, hidden=config.hidden)


def sample_map(config: TrainConfig, rng: random.Random):
    """A training board.

    Three things here are deliberate. Board size varies, so nothing can be
    memorised about one geometry. Pearl rates vary and are *sparse*: with food
    on every tile a policy never needs to seek it and grows fast enough to seal
    itself in, which is exactly what the first runs did. And because the engine
    seeds its pearl RNG from a constant, changing the gap range is what
    re-rolls the whole countdown field -- two maps with identical TILE lines
    get identical pearls no matter what else differs.
    """
    size = rng.choice(config.board_choices) if config.board_choices else config.board
    spec = mapgen.random_map(
        rng, min_size=size, max_size=size,
        dragons_per_team=config.dragons_per_team,
        kelp_density=rng.uniform(*config.kelp_range),
        portal_pairs=config.portal_pairs,
        pearl_gap_choices=config.pearl_gap_choices)
    return spec


# -- worker ---------------------------------------------------------------

def rollout_worker(rank: int, config: TrainConfig, task_queue, result_queue,
                   generation: int = 0) -> None:
    """Plays matches until told to stop, or until it has played enough that its
    leaked wasm memory is worth throwing the whole process away for."""
    import resource

    torch.set_num_threads(1)
    rng = random.Random(config.seed * 1000 + rank * 97 + generation)
    space = ActionSpace(max_sprint=config.max_sprint, split_sizes=config.split_sizes)
    runner = MatchRunner(WasmEngine(), space=space)
    actor = build_actor(config, space).eval()
    scripted = GreedyPearlPolicy(space, seed=rank)

    played = 0
    while True:
        task = task_queue.get()
        if task is None:
            return
        state_dict, iteration = task
        actor.load_state_dict(state_dict)

        batch: list[Trajectory] = []
        for _ in range(config.matches_per_worker):
            spec = sample_map(config, rng)
            if rng.random() < config.scripted_opponent_probability:
                learner_team = rng.choice(["A", "B"])
                other = "B" if learner_team == "A" else "A"
                actors = {learner_team: actor, other: scripted}
                learner_teams = {learner_team}
            else:
                actors = {"A": actor, "B": actor}
                learner_teams = {"A", "B"}
            with torch.no_grad():
                batch.append(play_match(runner, spec, actors, learner_teams))
            played += 1

        # ru_maxrss is kilobytes on Linux and bytes on macOS
        divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
        retiring = played >= config.worker_recycle_matches
        result_queue.put((rank, iteration, batch, rss_mb, retiring))
        if retiring:
            return


# -- batching -------------------------------------------------------------

@torch.no_grad()
def critic_values(critic: TeamCritic, trajectories: list[Trajectory], device,
                  chunk: int = 512) -> list[np.ndarray]:
    """Value of every round of every match, for both teams."""
    out = []
    for traj in trajectories:
        values = np.zeros((len(traj.rounds), 2), dtype=np.float32)
        for start in range(0, len(traj.rounds), chunk):
            stop = min(start + chunk, len(traj.rounds))
            state = CriticState(
                map_grid=torch.as_tensor(traj.map_grid[start:stop]).float().to(device),
                dragons=torch.as_tensor(traj.dragon_feats[start:stop]).to(device),
                dragon_team=torch.as_tensor(traj.dragon_team[start:stop]).to(device),
                dragon_mask=torch.as_tensor(traj.dragon_mask[start:stop]).to(device),
            )
            values[start:stop] = critic(state).float().cpu().numpy()
        out.append(values)
    return out


def advantages_per_round(trajectories: list[Trajectory], values: list[np.ndarray],
                         config: TrainConfig) -> tuple[list[np.ndarray], list[np.ndarray]]:
    advantages, returns = [], []
    for traj, value in zip(trajectories, values):
        steps = len(traj.rounds)
        adv = np.zeros((steps, 2), dtype=np.float32)
        ret = np.zeros((steps, 2), dtype=np.float32)
        for team in (0, 1):
            rewards = np.zeros(steps, dtype=np.float32)
            rewards[-1] = traj.reward[team]           # terminal only
            a, r = team_gae(rewards, value[:, team], config.gamma, config.gae_lambda,
                            terminal=True)
            adv[:, team], ret[:, team] = a, r
        advantages.append(adv)
        returns.append(ret)
    return advantages, returns


def make_sequences(trajectories: list[Trajectory], advantages: list[np.ndarray],
                   config: TrainConfig) -> list[dict]:
    """Contiguous slices of one dragon's life, never crossing a lifetime."""
    chunks: list[dict] = []
    length = config.sequence_length
    for traj, adv in zip(trajectories, advantages):
        if len(traj) == 0:
            continue
        round_of = {int(r): i for i, r in enumerate(traj.rounds)}
        order = np.lexsort((traj.round, traj.dragon))
        by_dragon: dict[int, list[int]] = collections.defaultdict(list)
        for i in order:
            by_dragon[int(traj.dragon[i])].append(int(i))
        for steps in by_dragon.values():
            for start in range(0, len(steps), length):
                idx = np.array(steps[start:start + length])
                rows = [round_of.get(int(traj.round[i]), 0) for i in idx]
                teams = traj.team[idx]
                chunks.append({
                    "traj": traj, "idx": idx,
                    "advantage": adv[rows, teams].astype(np.float32),
                })
    return chunks


def assemble_actor_batch(chunks: list[dict], config: TrainConfig, device,
                         n_actions: int) -> ActorBatch:
    size = len(chunks)
    steps = config.sequence_length
    first = chunks[0]["traj"]
    grid = np.zeros((size, steps, *first.grid.shape[1:]), dtype=np.float32)
    scalars = np.zeros((size, steps, first.scalars.shape[1]), dtype=np.float32)
    messages = np.zeros((size, steps, *first.messages.shape[1:]), dtype=np.int64)
    message_mask = np.zeros((size, steps, first.message_mask.shape[1]), dtype=bool)
    action_mask = np.zeros((size, steps, n_actions), dtype=bool)
    action = np.zeros((size, steps), dtype=np.int64)
    transmit = np.zeros((size, steps), dtype=bool)
    symbols = np.zeros((size, steps, first.symbols.shape[1]), dtype=np.int64)
    log_prob = np.zeros((size, steps), dtype=np.float32)
    advantage = np.zeros((size, steps), dtype=np.float32)
    valid = np.zeros((size, steps), dtype=bool)
    hidden = np.zeros((size, config.hidden), dtype=np.float32)

    for row, chunk in enumerate(chunks):
        traj, idx = chunk["traj"], chunk["idx"]
        n = len(idx)
        grid[row, :n] = traj.grid[idx].astype(np.float32) / 255.0
        scalars[row, :n] = traj.scalars[idx]
        messages[row, :n] = traj.messages[idx]
        message_mask[row, :n] = traj.message_mask[idx]
        action_mask[row, :n] = traj.action_mask[idx]
        action[row, :n] = traj.action[idx]
        transmit[row, :n] = traj.transmit[idx]
        symbols[row, :n] = traj.symbols[idx]
        log_prob[row, :n] = traj.log_prob[idx]
        advantage[row, :n] = chunk["advantage"]
        valid[row, :n] = True
        hidden[row] = traj.hidden[idx[0]].astype(np.float32)

    t = lambda x, dtype=None: torch.as_tensor(x, device=device, dtype=dtype)
    return ActorBatch(
        obs=ActorObs(t(grid), t(scalars), t(messages), t(message_mask), t(action_mask)),
        hidden=t(hidden),
        actions=JointAction(t(action), t(transmit), t(symbols)),
        old_log_prob=t(log_prob), advantages=t(advantage), valid=t(valid),
    )


def assemble_critic_batch(trajectories: list[Trajectory], returns: list[np.ndarray],
                          values: list[np.ndarray], count: int, rng: np.random.Generator,
                          device) -> CriticBatch:
    pool = [(m, r, team) for m, traj in enumerate(trajectories)
            for r in range(len(traj.rounds)) for team in (0, 1)]
    picked = [pool[i] for i in rng.choice(len(pool), size=min(count, len(pool)),
                                          replace=False)]
    maps = np.stack([trajectories[m].map_grid[r] for m, r, _ in picked]).astype(np.float32)
    feats = np.stack([trajectories[m].dragon_feats[r] for m, r, _ in picked])
    teams_in = np.stack([trajectories[m].dragon_team[r] for m, r, _ in picked])
    masks = np.stack([trajectories[m].dragon_mask[r] for m, r, _ in picked])
    target = np.array([returns[m][r, team] for m, r, team in picked], dtype=np.float32)
    old = np.array([values[m][r, team] for m, r, team in picked], dtype=np.float32)
    team_id = np.array([team for _, _, team in picked], dtype=np.int64)

    t = lambda x: torch.as_tensor(x, device=device)
    return CriticBatch(
        state=CriticState(t(maps), t(feats), t(teams_in), t(masks)),
        team=t(team_id), returns=t(target), old_values=t(old),
    )


# -- learner --------------------------------------------------------------

def main(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    space = ActionSpace(max_sprint=config.max_sprint, split_sizes=config.split_sizes)
    actor = build_actor(config, space).to(device)
    critic = build_critic(config).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=config.learning_rate)
    ppo_config = PPOConfig(
        clip_epsilon=config.clip_epsilon, epochs=1, value_coef=config.value_coef,
        entropy_coef=config.entropy_coef, grad_clip_norm=config.grad_clip_norm,
        burn_in=config.burn_in)

    run_dir = pathlib.Path(config.out_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity,
                               name=config.run_name, config=config.as_dict(),
                               mode=config.wandb_mode, dir=str(run_dir))
    except Exception as error:      # training must not die because logging did
        print(f"wandb disabled: {error}", flush=True)

    start_iteration = 0
    if config.resume:
        state = torch.load(config.resume, map_location=device, weights_only=False)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        optimizer.load_state_dict(state["optimizer"])
        start_iteration = int(state.get("iteration", 0)) + 1
        print(f"resumed from {config.resume} at iteration {start_iteration}", flush=True)

    ctx = mp.get_context("fork")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    generation = [0] * config.workers
    workers: list = [None] * config.workers

    def spawn(rank: int) -> None:
        process = ctx.Process(target=rollout_worker,
                              args=(rank, config, task_queue, result_queue,
                                    generation[rank]), daemon=True)
        process.start()
        workers[rank] = process

    for rank in range(config.workers):
        spawn(rank)

    total_steps = 0
    worker_restarts = 0
    started = time.time()
    try:
        for iteration in range(start_iteration, config.iterations):
            weights = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
            for _ in range(config.workers):
                task_queue.put((weights, iteration))

            collect_start = time.time()
            trajectories: list[Trajectory] = []
            retired: list[int] = []
            peak_worker_mb = 0.0
            for _ in range(config.workers):
                try:
                    rank, _, batch, rss_mb, retiring = result_queue.get(
                        timeout=config.result_timeout_s)
                except Exception:
                    dead = [r for r, w in enumerate(workers)
                            if w is not None and not w.is_alive()]
                    print(f"worker(s) {dead} stopped responding; restarting", flush=True)
                    for r in dead:
                        generation[r] += 1
                        spawn(r)
                        worker_restarts += 1
                    break
                trajectories.extend(batch)
                peak_worker_mb = max(peak_worker_mb, rss_mb)
                if retiring:
                    retired.append(rank)

            for rank in retired:
                workers[rank].join(timeout=30)
                generation[rank] += 1
                spawn(rank)
                worker_restarts += 1
            collect_seconds = time.time() - collect_start

            steps = sum(len(t) for t in trajectories)
            total_steps += steps
            if steps == 0:
                continue

            learn_start = time.time()
            values = critic_values(critic, trajectories, device)
            advantages, returns = advantages_per_round(trajectories, values, config)
            chunks = make_sequences(trajectories, advantages, config)

            metrics: dict[str, float] = {}
            for _ in range(config.epochs):
                order = rng.permutation(len(chunks))
                for start in range(0, len(order), config.sequences_per_minibatch):
                    picked = [chunks[i] for i in order[start:start + config.sequences_per_minibatch]]
                    if not picked:
                        continue
                    actor_batch = assemble_actor_batch(picked, config, device, space.n)
                    critic_batch = assemble_critic_batch(
                        trajectories, returns, values,
                        config.critic_samples_per_minibatch, rng, device)
                    metrics = ppo_update(actor, critic, optimizer, actor_batch,
                                         critic_batch, ppo_config)
            learn_seconds = time.time() - learn_start

            deaths: collections.Counter = collections.Counter()
            wins = collections.Counter()
            lengths = []
            scripted = []
            for traj in trajectories:
                deaths.update(traj.info["deaths"])
                wins[traj.info["winner"] or "draw"] += 1
                lengths += [traj.info["a_length"], traj.info["b_length"]]
                if traj.info.get("learner_result") is not None:
                    scripted.append(traj.info["learner_result"])

            elapsed = time.time() - started
            log = {
                "iteration": iteration,
                "steps/total": total_steps,
                "steps/per_iteration": steps,
                "steps/per_second": steps / max(collect_seconds + learn_seconds, 1e-6),
                "time/collect_s": collect_seconds,
                "time/learn_s": learn_seconds,
                "time/elapsed_h": elapsed / 3600.0,
                "match/rounds": float(np.mean([t.info["rounds"] for t in trajectories])),
                "match/total_length": float(np.mean(lengths)),
                "match/draws": wins["draw"] / max(len(trajectories), 1),
                "match/sequences": len(chunks),
                "worker/peak_rss_mb": peak_worker_mb,
                "worker/restarts": worker_restarts,
                **{f"death/{k}": v / max(len(trajectories), 1) for k, v in deaths.items()},
                **metrics,
            }
            if scripted:
                arr = np.array(scripted, dtype=np.float32)
                log["vs_scripted/win_rate"] = float((arr > 0).mean())
                log["vs_scripted/loss_rate"] = float((arr < 0).mean())
                log["vs_scripted/draw_rate"] = float((arr == 0).mean())
                log["vs_scripted/score"] = float(arr.mean())
                log["vs_scripted/matches"] = float(len(arr))
            if wandb_run is not None:
                wandb_run.log(log, step=iteration)
            print(f"iter {iteration:5d} steps {total_steps:>10,} "
                  f"{log['steps/per_second']:7.0f}/s rounds {log['match/rounds']:5.1f} "
                  f"len {log['match/total_length']:5.1f} "
                  f"loss {metrics.get('loss/total', float('nan')):+.4f} "
                  f"kl {metrics.get('ppo/approx_kl', float('nan')):.4f} "
                  f"vs_scripted {log.get('vs_scripted/win_rate', float('nan')):.2f} "
                  f"rss {peak_worker_mb:5.0f}MB", flush=True)

            if iteration % config.save_every == 0:
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": dataclasses.asdict(config), "iteration": iteration},
                           run_dir / f"checkpoint-{iteration:06d}.pt")
    finally:
        for _ in range(config.workers):
            task_queue.put(None)
        for worker in workers:
            if worker is None:
                continue
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
        if wandb_run is not None:
            wandb_run.finish()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(TrainConfig):
        if field.type in ("tuple[int, ...]", "tuple[int, int]", "str | None"):
            continue
        parser.add_argument(f"--{field.name.replace('_', '-')}",
                            type=type(field.default), default=field.default)
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in vars(args).items() if k in known})


if __name__ == "__main__":
    main(parse_args())
