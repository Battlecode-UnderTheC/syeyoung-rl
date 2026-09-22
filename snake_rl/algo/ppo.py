"""Recurrent PPO over joint commands (spec sections 8-10)."""

from __future__ import annotations

import collections as _collections
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from dragons_env.protocol import parse_turn
from ..env.actions import CommandCodec
from ..env.memory import LifetimeState
from ..env.observation import prepare
from ..env.schema import F as _F, TILES as _TILES, VISION as _VISION
from ..model.actor import SpatialActor, SpatialInputs, StoredCommand
from ..model.critic import EventCritic, EventState
from .gae import event_gae


@dataclass
class SpatialPPOConfig:
    clip: float = 0.1
    epochs: int = 3
    sequence_length: int = 64
    burn_in: int = 16
    sequences_per_microbatch: int = 8
    accumulation_microbatches: int = 8
    value_coef: float = 0.5
    entropy_surrogate_coef: float = 0.005
    sonar_entropy_coef: float = 0.005
    grad_norm: float = 0.5
    target_kl: float = 0.02
    gamma_round: float = 1.0
    lambda_round: float = 0.99
    aux_coefficient: float = 0.05
    aux_enabled: bool = True
    normalize_advantages: bool = True
    critic_events_per_microbatch: int = 256


# -- critic values and event advantages ------------------------------------------

def _event_state_batch(events, device) -> EventState:
    n = max(e.dragons.shape[0] for e in events)
    B = len(events)
    H, W = events[0].map_grid.shape[1:]
    grid = np.stack([e.map_grid for e in events]).astype(np.float32)
    dragons = np.zeros((B, n, events[0].dragons.shape[1]), dtype=np.float32)
    team = np.zeros((B, n), dtype=np.int64)
    mask = np.zeros((B, n), dtype=bool)
    for i, e in enumerate(events):
        k = e.dragons.shape[0]
        dragons[i, :k] = e.dragons
        team[i, :k] = e.dragon_team
        mask[i, :k] = e.dragon_team.shape[0] > 0
        # a padded placeholder row carries no dragon
        if e.dragons.shape[0] == 1 and not e.dragons[0].any():
            mask[i, 0] = False
    t = lambda x: torch.as_tensor(x, device=device)
    return EventState(t(grid), t(dragons), t(team), t(mask),
                      t(np.stack([e.team_feats for e in events])),
                      t(np.stack([e.phase for e in events])))


@torch.no_grad()
def event_values(critic: EventCritic, matches, device, chunk: int = 256) -> list[np.ndarray]:
    """V_old for every event of every match, both teams: list of [E, 2]."""
    out = []
    for match in matches:
        values = np.zeros((len(match.events), 2), dtype=np.float32)
        # batch by board size
        groups: dict[tuple, list[int]] = _collections.defaultdict(list)
        for i, e in enumerate(match.events):
            groups[e.map_grid.shape[1:]].append(i)
        for idx in groups.values():
            for start in range(0, len(idx), chunk):
                picked = idx[start:start + chunk]
                state = _event_state_batch([match.events[i] for i in picked], device)
                values[picked] = critic(state).float().cpu().numpy()
        out.append(values)
    return out


def event_advantages(matches, values: list[np.ndarray], cfg: SpatialPPOConfig
                     ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    advantages, returns = [], []
    for match, value in zip(matches, values):
        E = len(match.events)
        adv = np.zeros((E, 2), dtype=np.float32)
        ret = np.zeros((E, 2), dtype=np.float32)
        delta_rounds = np.array([e.delta_rounds for e in match.events])
        terminal = np.array([e.terminal for e in match.events])
        for team in (0, 1):
            rewards = np.array([e.reward[team] for e in match.events], dtype=np.float32)
            nxt = np.zeros(E, dtype=np.float32)
            nxt[:-1] = value[1:, team]
            a, r = event_gae(rewards, value[:, team], nxt, delta_rounds, terminal,
                             cfg.gamma_round, cfg.lambda_round)
            adv[:, team], ret[:, team] = a, r
        advantages.append(adv)
        returns.append(ret)
    return advantages, returns


# -- sequences ---------------------------------------------------------------------

@dataclass
class Chunk:
    match: object
    lifetime: object
    burn_start: int
    start: int
    stop: int


def make_chunks(matches, cfg: SpatialPPOConfig) -> list[Chunk]:
    """Contiguous per-lifetime slices. Each learner decision is scored in
    exactly one chunk; burn-in replays the preceding turns from the snapshot
    at or before them. Newborn prefixes are included in full."""
    chunks = []
    L, burn = cfg.sequence_length, cfg.burn_in
    for match in matches:
        for rec in match.lifetimes.values():
            if not rec.learner or not rec.turns:
                continue
            for start in range(0, len(rec.turns), L):
                burn_start = max(start - burn, 0)
                # walk back to the nearest snapshot at or before the burn-in start
                while burn_start not in rec.snapshots:
                    burn_start -= 1
                    if burn_start < 0:
                        raise RuntimeError("no snapshot before the sequence start")
                chunks.append(Chunk(match, rec, burn_start, start, min(start + L, len(rec.turns))))
    return chunks


@dataclass
class Replayed:
    """One chunk replayed into actor tensors."""
    chunk: Chunk
    prepared: list                # PreparedObservation per replayed turn
    stored: list                  # StoredCommand (single-row) per replayed turn
    scored: np.ndarray            # [T] bool, False during burn-in
    old_log_prob: np.ndarray      # [T]
    advantage: np.ndarray         # [T]
    team_round: list              # [T] (team, round) keys
    M0: np.ndarray
    h0: np.ndarray
    aux: dict                     # per-turn label arrays
    blocks: list


def _enemy_target(block_now, block_next, dragon_id: int, team: str) -> int:
    """Window cell of the nearest visible enemy head at the next own turn,
    or 49 if that dragon's head is not observed then; -1 if none now."""
    heads = [(abs(p.x - block_now.head_x) + abs(p.y - block_now.head_y), p.dragon_id)
             for p in block_now.parts if p.is_head and p.team != team]
    if not heads:
        return -1
    _, target = min(heads)
    if block_next is None:
        return _TILES
    for p in block_next.parts:
        if p.is_head and p.dragon_id == target:
            i = block_next.window_index(p.x, p.y)
            return _TILES if i is None else i
    return _TILES


def replay_chunk(chunk: Chunk, codec: CommandCodec, advantages: np.ndarray) -> Replayed:
    rec, match = chunk.lifetime, chunk.match
    snap, M0, h0 = rec.snapshots[chunk.burn_start]
    state = LifetimeState.from_snapshot(snap)
    prepared, stored, blocks = [], [], []
    turns = rec.turns[chunk.burn_start:chunk.stop]
    for turn in turns:
        block = parse_turn(turn.block, rec.width, rec.height)
        obs = prepare(state, block, codec.max_steps)
        prepared.append(obs)
        blocks.append(block)
        cmd = turn.command
        if cmd.top == 0 and cmd.move is None:
            cmd.move = codec.trace_move(obs, state, cmd.dirs)
        if cmd.top == 1 and cmd.split_index < 0:
            sizes, feats = codec.split_table(obs)
            cmd.split_index = int(np.flatnonzero(sizes == cmd.split_n)[0])
        stored.append(cmd)
        if cmd.top == 0:
            state.commit_command(block, "move", cmd.dirs)
        else:
            state.commit_command(block, "split", split_n=cmd.split_n)
    T = len(turns)
    scored = np.zeros(T, dtype=bool)
    scored[chunk.start - chunk.burn_start:] = True
    team_idx = 0 if rec.key.team == "A" else 1
    adv = np.array([advantages[t.event, team_idx] for t in turns], dtype=np.float32)
    old = np.array([t.log_prob for t in turns], dtype=np.float32)
    keys = [(rec.key.team, t.round) for t in turns]

    # auxiliary labels from observations only (spec 10)
    following = rec.turns[chunk.burn_start + 1:chunk.stop + 1]
    enemy = np.full(T, -1, dtype=np.int64)
    alive = np.zeros(T, dtype=np.float32)
    alive_mask = np.zeros(T, dtype=bool)
    delta = np.zeros(T, dtype=np.int64)
    delta_mask = np.zeros(T, dtype=bool)
    for i, block in enumerate(blocks):
        nxt = parse_turn(following[i].block, rec.width, rec.height) if i < len(following) else None
        enemy[i] = _enemy_target(block, nxt, rec.key.dragon_id, rec.key.team)
        if nxt is not None:
            alive[i], alive_mask[i] = 1.0, True
            delta[i] = int(np.clip(nxt.length - block.length, -8, 8)) + 8
            delta_mask[i] = True
        elif rec.death is not None:
            alive[i], alive_mask[i] = 0.0, True
        # else: censored by the end of collection
    return Replayed(chunk, prepared, stored, scored, old, adv, keys, M0, h0,
                    {"enemy": enemy, "alive": alive, "alive_mask": alive_mask,
                     "delta": delta, "delta_mask": delta_mask}, blocks)


# -- recurrent unroll --------------------------------------------------------------

@dataclass
class UnrollResult:
    log_prob: Tensor          # [N] scored decisions, in order of `index`
    entropy: Tensor           # [N]
    sonar_entropy: Tensor     # [N]
    index: list               # [(sequence, t)] for every scored decision
    aux_loss: Tensor          # scalar, already normalised per component
    aux_metrics: dict


def unroll_sequences(actor: SpatialActor, seqs: list[Replayed], codec: CommandCodec, device,
                     aux: bool = True) -> UnrollResult:
    """Replay burn-in (no gradient, detached at its end) then the scored steps,
    grouping sequences by board size so torus wrapping stays exact."""
    groups: dict[tuple, list[int]] = _collections.defaultdict(list)
    for i, s in enumerate(seqs):
        groups[(s.prepared[0].height, s.prepared[0].width)].append(i)
    log_probs, ents, sents, index = [], [], [], []
    aux_terms = _collections.defaultdict(list)
    for (H, W), members in groups.items():
        B = len(members)
        M = torch.as_tensor(np.stack([seqs[i].M0 for i in members]), device=device)
        h = torch.as_tensor(np.stack([seqs[i].h0 for i in members]), device=device)
        T = max(len(seqs[i].prepared) for i in members)
        burn_end = [int(np.argmax(seqs[i].scored)) for i in members]
        for t in range(T):
            rows = [r for r, i in enumerate(members) if t < len(seqs[i].prepared)]
            if not rows:
                break
            items = [seqs[members[r]].prepared[t] for r in rows]
            x = SpatialInputs.from_prepared(items, codec, device)
            rows_t = torch.tensor(rows, device=device)
            burn = [r for r in rows if t < burn_end[r]]
            score = [r for r in rows if t >= burn_end[r]]
            ctx, M_new, h_new = actor.encode_and_update(x, M[rows_t], h[rows_t])
            M = M.index_put((rows_t,), M_new) if M.requires_grad or M_new.requires_grad else M.clone().index_put_((rows_t,), M_new)
            h = h.index_put((rows_t,), h_new)
            if score:
                sel = torch.tensor([rows.index(r) for r in score], device=device)
                st = StoredCommand.from_commands([seqs[members[r]].stored[t] for r in score], codec, device)
                sub_x = x.select(sel)
                sub_ctx = type(ctx)(ctx.h[sel], ctx.fused[sel], ctx.memory_visible_old[sel])
                lp, ent, sent = actor.decode_evaluate(sub_ctx, sub_x, st)
                log_probs.append(lp)
                ents.append(ent)
                sents.append(sent)
                index += [(members[r], t) for r in score]
                if aux:
                    _aux_losses(actor, sub_ctx, sub_x, st, [seqs[members[r]] for r in score], t, aux_terms)
            # burn-in ends: cut the gradient into the scored unroll
            done = [r for r in burn if t == burn_end[r] - 1]
            if done:
                d = torch.tensor(done, device=device)
                M = M.index_put((d,), M[d].detach())
                h = h.index_put((d,), h[d].detach())
    lp = torch.cat(log_probs) if log_probs else torch.zeros(0, device=device)
    ent = torch.cat(ents) if ents else torch.zeros(0, device=device)
    sent = torch.cat(sents) if sents else torch.zeros(0, device=device)
    aux_loss = torch.zeros((), device=device)
    metrics = {}
    for name, terms in aux_terms.items():
        losses = torch.stack([l for l, _ in terms])
        weights = torch.stack([w for _, w in terms])
        total = (losses * weights).sum() / weights.sum().clamp(min=1.0)
        aux_loss = aux_loss + total
        metrics[f"aux/{name}"] = float(total.detach())
    return UnrollResult(lp, ent, sent, index, aux_loss, metrics)


def _aux_losses(actor, ctx, x, stored, seqs, t, out) -> None:
    preds = actor.aux_predictions(ctx, x, stored)
    # 1. pearl revisit: score the pre-update forecast against the arriving
    #    observation, balanced between present and absent pearls
    label = x.facts[torch.arange(x.batch, device=x.facts.device)[:, None], _F.PEARL,
                    x.visible_xy[..., 1], x.visible_xy[..., 0]]
    seen_before = x.visible_prev[..., _F.SEEN] > 0
    logit = preds["pearl_logit"]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, label, reduction="none")
    pos, neg = seen_before & (label > 0), seen_before & (label == 0)
    for name, m in (("pearl_pos", pos), ("pearl_neg", neg)):
        w = m.float().sum()
        out[name].append(((bce * m.float()).sum() / w.clamp(min=1.0), w.clamp(max=1.0)))
    # 2. next observed enemy head, with a not-observed category
    enemy = torch.tensor([s.aux["enemy"][t] for s in seqs], device=logit.device)
    valid = enemy >= 0
    if valid.any():
        ce = torch.nn.functional.cross_entropy(preds["enemy_logits"][valid], enemy[valid])
        out["enemy"].append((ce, valid.float().sum().clamp(max=1.0)))
    # 3. next-own-turn alive / length change (censored when collection ended)
    alive = torch.tensor([s.aux["alive"][t] for s in seqs], device=logit.device)
    amask = torch.tensor([s.aux["alive_mask"][t] for s in seqs], device=logit.device)
    if amask.any():
        b = torch.nn.functional.binary_cross_entropy_with_logits(preds["alive_logit"][amask], alive[amask])
        out["alive"].append((b, amask.float().sum().clamp(max=1.0)))
    delta = torch.tensor([s.aux["delta"][t] for s in seqs], device=logit.device)
    dmask = torch.tensor([s.aux["delta_mask"][t] for s in seqs], device=logit.device)
    if dmask.any():
        ce = torch.nn.functional.cross_entropy(preds["delta_logits"][dmask], delta[dmask])
        out["delta"].append((ce, dmask.float().sum().clamp(max=1.0)))


# -- the update ------------------------------------------------------------------------

def ppo_update_spatial(actor: SpatialActor, critic: EventCritic, optimizer, matches,
                       cfg: SpatialPPOConfig, codec: CommandCodec, device="cpu",
                       rng: np.random.Generator | None = None) -> dict[str, float]:
    """One PPO update over complete matches collected with frozen weights."""
    rng = rng or np.random.default_rng(0)
    values = event_values(critic, matches, device)
    advantages, returns = event_advantages(matches, values, cfg)
    chunks = make_chunks(matches, cfg)
    if not chunks:
        return {"ppo/decisions": 0.0}
    replayed = [replay_chunk(c, codec, advantages[matches.index(c.match)]) for c in chunks]

    # fixed normalisation over the whole update: valid learner samples only
    all_adv = np.concatenate([r.advantage[r.scored] for r in replayed])
    team_rounds = {k for r in replayed for k, s in zip(r.team_round, r.scored) if s}
    denominator = float(max(len(team_rounds), 1))
    mean, std = float(all_adv.mean()), float(all_adv.std() + 1e-8)
    if cfg.normalize_advantages:
        for r in replayed:
            r.advantage = (r.advantage - mean) / std

    # value targets over unique team-events
    event_pool = [(m, e, team) for m, match in enumerate(matches)
                  for e in range(len(match.events)) for team in (0, 1)]
    params = [p for g in optimizer.param_groups for p in g["params"]]
    metrics: dict[str, float] = {}
    total_decisions = int(sum(r.scored.sum() for r in replayed))
    epochs_done = 0
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(replayed))
        kls, clips, losses, ents, grad_norms, vlosses = [], [], [], [], [], []
        micro = cfg.sequences_per_microbatch
        batches = [order[i:i + micro] for i in range(0, len(order), micro)]
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for bi, picked in enumerate(batches):
            seqs = [replayed[i] for i in picked]
            res = unroll_sequences(actor, seqs, codec, device, aux=cfg.aux_enabled)
            old = torch.tensor([seqs[s].old_log_prob[t] for s, t in res.index], device=device)
            adv = torch.tensor([seqs[s].advantage[t] for s, t in res.index], device=device)
            ratio = torch.exp(res.log_prob - old)
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)
            policy_loss = -surrogate.sum() / denominator
            entropy_term = (cfg.entropy_surrogate_coef * res.entropy.sum()
                            + cfg.sonar_entropy_coef * res.sonar_entropy.sum()) / denominator
            # critic: a sample of unique team-events, once each
            pick = rng.choice(len(event_pool), size=min(cfg.critic_events_per_microbatch, len(event_pool)), replace=False)
            groups: dict[tuple, list[int]] = _collections.defaultdict(list)
            for i in pick:
                m, e, team = event_pool[i]
                groups[matches[m].events[e].map_grid.shape[1:]].append(i)
            value_loss = torch.zeros((), device=device)
            for members in groups.values():
                evs = [matches[event_pool[i][0]].events[event_pool[i][1]] for i in members]
                teams = torch.tensor([event_pool[i][2] for i in members], device=device)
                target = torch.tensor([returns[event_pool[i][0]][event_pool[i][1], event_pool[i][2]]
                                       for i in members], device=device)
                v = critic(_event_state_batch(evs, device)).gather(1, teams[:, None]).squeeze(1)
                value_loss = value_loss + ((v - target) ** 2).sum()
            value_loss = value_loss / max(len(pick), 1)
            loss = (policy_loss - entropy_term + cfg.value_coef * value_loss / cfg.accumulation_microbatches
                    + cfg.aux_coefficient * res.aux_loss / cfg.accumulation_microbatches)
            loss.backward()
            pending += 1
            with torch.no_grad():
                kls.append(float(((ratio - 1) - torch.log(ratio)).sum()))
                clips.append(float(((ratio - 1).abs() > cfg.clip).float().sum()))
                losses.append(float(policy_loss))
                ents.append(float(res.entropy.sum()))
                vlosses.append(float(value_loss))
            if pending == cfg.accumulation_microbatches or bi == len(batches) - 1:
                grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_norm)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            for k, v in res.aux_metrics.items():
                metrics[k] = v
        epochs_done = epoch + 1
        approx_kl = sum(kls) / max(total_decisions, 1)
        metrics.update({
            "loss/policy": sum(losses), "loss/value": float(np.mean(vlosses)),
            "loss/entropy_surrogate": sum(ents) / max(total_decisions, 1),
            "ppo/approx_kl": approx_kl,
            "ppo/clip_fraction": sum(clips) / max(total_decisions, 1),
            "ppo/grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        })
        if approx_kl > cfg.target_kl:
            break
    metrics.update({
        "ppo/epochs_done": float(epochs_done),
        "ppo/decisions": float(total_decisions),
        "ppo/team_rounds": denominator,
        "ppo/sequences": float(len(replayed)),
        "adv/mean": mean, "adv/std": std,
    })
    return metrics
