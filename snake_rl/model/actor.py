"""The spatial-memory actor (spec sections 5, 6, 10 and 13).

Persistent state is explicit: a learned map M [B, 16, H, W] and a tactical
GRU state h [B, 128], both owned by the caller per lifetime. The interfaces
are the ones section 13 names:

    encode_and_update(inputs, M_old, h_old) -> ctx, M_new, h_new     (pure)
    decode_sample(ctx, inputs, prepared, states, codec)               (rollout)
    decode_evaluate(ctx, inputs, stored) -> joint_logp, entropy_proxy (PPO)

`decode_sample` and `decode_evaluate` share every module and every mask, so
the log probability a rollout stores is the one PPO recomputes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

NEG_INF = -3.0e4      # representable in fp16, scriptable


def _gumbel_argmax(logits: Tensor) -> Tensor:
    """Exact categorical sampling without building a distribution object."""
    noise = torch.rand_like(logits).clamp_(min=torch.finfo(logits.dtype).tiny)
    return (logits - torch.log(-torch.log(noise))).argmax(dim=-1)

from ..env.actions import Command, CommandCodec, MoveTrace
from ..env.schema import (N_PREFIX_FEATURES, N_SONAR_CHOICES, N_TOKENS, SONAR_SILENT,
                          TOKEN_STOP, TOP_MOVE, TOP_SPLIT, VISION)
from .encoder import (EntityEncoder, HistoryEncoder, InboxEncoder, LocalEncoder,
                      StatusEncoder)
from .spatial_memory import (GlobalReads, LocalMapRead, MultiscaleReader, SparseWriter)

BOS_TOKEN = N_TOKENS          # decoder start symbol


@dataclass
class SpatialInputs:
    """One batch of prepared observations on one board size."""
    local: Tensor
    facts: Tensor
    board_valid: Tensor
    visible_xy: Tensor
    visible_prev: Tensor
    visible_age: Tensor
    status: Tensor
    history: Tensor
    history_mask: Tensor
    entities: Tensor
    entity_mask: Tensor
    portal_pairs: Tensor
    pair_mask: Tensor
    portal_axis: Tensor
    portal_unpaired: Tensor
    unpaired_mask: Tensor
    inbox: Tensor
    inbox_mask: Tensor
    head_xy: Tensor
    split_legal: Tensor
    split_sizes: Tensor
    split_feats: Tensor
    split_mask: Tensor
    length: Tensor

    @property
    def batch(self) -> int:
        return self.local.shape[0]

    @staticmethod
    def from_prepared(items: Sequence, codec: CommandCodec, device="cpu") -> "SpatialInputs":
        H, W = items[0].height, items[0].width
        if any(o.height != H or o.width != W for o in items):
            raise ValueError("a batch must share one board size")
        B = len(items)
        t = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)

        def pad_stack(arrays, fill=0, dtype=None):
            n = max((a.shape[0] for a in arrays), default=0)
            first = arrays[0]
            out = np.full((B, n) + first.shape[1:], fill, dtype=dtype or first.dtype)
            mask = np.zeros((B, n), dtype=bool)
            for i, a in enumerate(arrays):
                out[i, :a.shape[0]] = a
                mask[i, :a.shape[0]] = True
            return out, mask

        pairs, pair_mask = pad_stack([o.portal_pairs for o in items])
        axis, _ = pad_stack([o.portal_axis for o in items])
        loose, loose_mask = pad_stack([o.portal_unpaired for o in items])
        inbox, inbox_mask = pad_stack([o.inbox for o in items])
        tables = [codec.split_table(o) for o in items]
        sizes, split_mask = pad_stack([n for n, _ in tables], dtype=np.int64)
        feats, _ = pad_stack([f for _, f in tables], dtype=np.float32)
        return SpatialInputs(
            local=t(np.stack([o.local for o in items])),
            facts=t(np.stack([o.facts for o in items])),
            board_valid=t(np.stack([o.board_valid for o in items])),
            visible_xy=t(np.stack([o.visible_xy for o in items]), torch.long),
            visible_prev=t(np.stack([o.visible_prev for o in items])),
            visible_age=t(np.stack([o.visible_age_before for o in items])),
            status=t(np.stack([o.status for o in items])),
            history=t(np.stack([o.history for o in items])),
            history_mask=t(np.stack([o.history_mask for o in items])),
            entities=t(np.stack([o.entities for o in items])),
            entity_mask=t(np.stack([o.entity_mask for o in items])),
            portal_pairs=t(pairs, torch.long), pair_mask=t(pair_mask),
            portal_axis=t(axis, torch.long),
            portal_unpaired=t(loose, torch.long), unpaired_mask=t(loose_mask),
            inbox=t(inbox, torch.long), inbox_mask=t(inbox_mask),
            head_xy=t(np.array([o.head for o in items]), torch.long),
            split_legal=t(np.array([o.split_legal for o in items])),
            split_sizes=t(sizes, torch.long), split_feats=t(feats), split_mask=t(split_mask),
            length=t(np.array([o.length for o in items]), torch.long),
        )

    def select(self, rows: Tensor) -> "SpatialInputs":
        return SpatialInputs(**{k: v[rows] for k, v in self.__dict__.items()})


@dataclass
class StoredCommand:
    """The lossless record PPO teacher-forces (spec 6.4 / 12.1)."""
    top: Tensor            # [B] long
    tokens: Tensor         # [B, K] long
    active: Tensor         # [B, K] bool
    masks: Tensor          # [B, K, 5] bool
    prefix: Tensor         # [B, K, N_PREFIX] float
    split_index: Tensor    # [B] long, index into the candidate table
    sonar: Tensor          # [B] long, 0 = silent

    @staticmethod
    def from_commands(cmds: Sequence[Command], codec: CommandCodec, device="cpu") -> "StoredCommand":
        K = codec.max_steps
        B = len(cmds)
        traces = [c.move if c.move is not None else codec.empty_trace() for c in cmds]
        t = lambda x, dt=None: torch.as_tensor(np.stack(x), dtype=dt, device=device)
        return StoredCommand(
            top=torch.as_tensor([c.top for c in cmds], dtype=torch.long, device=device),
            tokens=t([tr.tokens for tr in traces], torch.long),
            active=t([tr.active for tr in traces], torch.bool),
            masks=t([tr.masks for tr in traces], torch.bool),
            prefix=t([tr.prefix for tr in traces], torch.float32),
            split_index=torch.as_tensor([max(c.split_index, 0) for c in cmds], dtype=torch.long, device=device),
            sonar=torch.as_tensor([c.sonar for c in cmds], dtype=torch.long, device=device),
        )

    def select(self, rows: Tensor) -> "StoredCommand":
        return StoredCommand(**{k: v[rows] for k, v in self.__dict__.items()})


@dataclass
class Context:
    h: Tensor                 # [B, 128] the new tactical state driving all heads
    fused: Tensor             # [B, 128] pre-GRU fused features
    memory_visible_old: Tensor    # [B, 49, 16] learned memory at visible cells before the write


@dataclass
class SpatialActorConfig:
    latent_channels: int = 16
    tactical_hidden: int = 128
    decoder_hidden: int = 64
    spatial_widths: tuple[int, int, int] = (16, 24, 32)
    spatial_iterations: int = 2
    global_reads: int = 4
    read_width: int = 32
    history_turns: int = 32
    max_move_steps: int = 8
    sonar: bool = False


class SpatialActor(nn.Module):
    def __init__(self, config: SpatialActorConfig | None = None):
        super().__init__()
        cfg = config or SpatialActorConfig()
        self.config = cfg
        self.K = cfg.max_move_steps
        self.latent = cfg.latent_channels
        self.hidden = cfg.tactical_hidden
        H = self.hidden

        self.local_enc = LocalEncoder(channels=32, out_dim=128)
        self.status_enc = StatusEncoder(out_dim=64, small=16)
        self.history_enc = HistoryEncoder(turns=cfg.history_turns, out_dim=32)
        self.entity_enc = EntityEncoder(out_dim=64)
        self.inbox_enc = InboxEncoder(out_dim=32) if cfg.sonar else None
        self.writer = SparseWriter(32, cfg.latent_channels, 16)
        self.reader = MultiscaleReader(42 + cfg.latent_channels, cfg.spatial_widths,
                                       cfg.spatial_iterations, cond_dim=H + 64)
        self.map_local = LocalMapRead(cfg.spatial_widths[0], 128)
        self.global_reads = GlobalReads(cfg.spatial_widths[2], cfg.global_reads,
                                        cfg.read_width, query_in=128 + H + 64)
        fuse_in = 128 + 128 + cfg.global_reads * cfg.read_width + 64 + 64 + 32
        if cfg.sonar:
            fuse_in += 32
        self.fuse = nn.Sequential(nn.Linear(fuse_in, H), nn.SiLU())
        self.gru = nn.GRUCell(H, H)

        # heads
        self.top_head = nn.Linear(H, 2)
        self.dec_init = nn.Linear(H, cfg.decoder_hidden)
        self.tok_embed = nn.Embedding(N_TOKENS + 1, 16)
        self.prefix_proj = nn.Sequential(nn.Linear(N_PREFIX_FEATURES, 32), nn.SiLU())
        self.h_proj = nn.Sequential(nn.Linear(H, 32), nn.SiLU())
        self.dec_cell = nn.GRUCell(16 + 32 + 32, cfg.decoder_hidden)
        self.tok_head = nn.Linear(cfg.decoder_hidden, N_TOKENS)
        self.split_mlp = nn.Sequential(nn.Linear(H + 4, 64), nn.SiLU(), nn.Linear(64, 1))
        self.cmd_embed = nn.Sequential(nn.Linear(2 + 4 + 1 + 1, 32), nn.SiLU())
        self.sonar_head = nn.Linear(H + 32, N_SONAR_CHOICES) if cfg.sonar else None

        # auxiliary heads (spec 10): representation only, never a reward
        self.aux_pearl = nn.Sequential(nn.Linear(cfg.latent_channels + 42 + 1, 32), nn.SiLU(),
                                       nn.Linear(32, 1))
        self.aux_enemy = nn.Linear(H, VISION * VISION + 1)
        self.aux_own = nn.Sequential(nn.Linear(H + 32, 64), nn.SiLU(), nn.Linear(64, 1 + 17))

        for head in (self.top_head, self.tok_head, self.split_mlp[-1]):
            nn.init.orthogonal_(head.weight, 0.01)
            nn.init.zeros_(head.bias)
        if self.sonar_head is not None:
            nn.init.orthogonal_(self.sonar_head.weight, 0.01)
            nn.init.zeros_(self.sonar_head.bias)

    # -- state -----------------------------------------------------------------

    def initial_state(self, batch: int, height: int, width: int, device=None
                      ) -> tuple[Tensor, Tensor]:
        return (torch.zeros(batch, self.latent, height, width, device=device),
                torch.zeros(batch, self.hidden, device=device))

    # -- encoding (pure) -------------------------------------------------------

    def encode_and_update(self, x: SpatialInputs, M_old: Tensor, h_old: Tensor
                          ) -> tuple[Context, Tensor, Tensor]:
        B = x.batch
        cells, local128 = self.local_enc(x.local)
        status64, status16 = self.status_enc(x.status)
        b = torch.arange(B, device=M_old.device)[:, None].expand(B, x.visible_xy.shape[1])
        M_vis_old = M_old[b, :, x.visible_xy[..., 1], x.visible_xy[..., 0]]
        M_new = self.writer(M_old, cells, x.visible_prev, x.visible_age, status16, x.visible_xy)

        cond = torch.cat([h_old, status64], dim=-1)
        fine, coarse, v_coarse = self.reader(x.facts, M_new, x.board_valid, cond,
                                             x.portal_pairs, x.pair_mask, x.portal_axis,
                                             x.portal_unpaired, x.unpaired_mask)
        map_local = self.map_local(fine, x.head_xy)
        global_ = self.global_reads(coarse, v_coarse, x.facts, x.head_xy,
                                    torch.cat([local128, h_old, status64], dim=-1))
        parts = [local128, map_local, global_, self.entity_enc(x.entities, x.entity_mask),
                 status64, self.history_enc(x.history, x.history_mask)]
        if self.inbox_enc is not None:
            parts.append(self.inbox_enc(x.inbox, x.inbox_mask))
        fused = self.fuse(torch.cat(parts, dim=-1))
        h_new = self.gru(fused, h_old)
        return Context(h_new, fused, M_vis_old), M_new, h_new

    # -- heads -----------------------------------------------------------------

    def top_logits(self, h: Tensor, split_legal: Tensor) -> Tensor:
        logits = self.top_head(h)
        return logits.masked_fill(~split_legal[:, None] & (torch.arange(2, device=h.device) == TOP_SPLIT)[None], NEG_INF)

    def split_logits(self, h: Tensor, feats: Tensor, mask: Tensor) -> Tensor:
        C = feats.shape[1]
        if C == 0:
            return h.new_full((h.shape[0], 1), 0.0)
        hh = h[:, None, :].expand(-1, C, -1)
        logits = self.split_mlp(torch.cat([hh, feats], dim=-1)).squeeze(-1)
        usable = mask | ~mask.any(-1, keepdim=True)
        return logits.masked_fill(~usable, NEG_INF)

    def move_init(self, h: Tensor) -> Tensor:
        return torch.tanh(self.dec_init(h))

    def move_step(self, dec: Tensor, h: Tensor, prev_tok: Tensor, prefix: Tensor,
                  mask: Tensor) -> tuple[Tensor, Tensor]:
        """One decoder step. Rows whose mask is empty get an unmasked row so
        the softmax stays finite; such rows are never active."""
        inp = torch.cat([self.tok_embed(prev_tok), self.prefix_proj(prefix), self.h_proj(h)], -1)
        dec = self.dec_cell(inp, dec)
        logits = self.tok_head(dec)
        usable = mask | ~mask.any(-1, keepdim=True)
        return logits.masked_fill(~usable, NEG_INF), dec

    def command_embedding(self, stored: StoredCommand, x: SpatialInputs) -> Tensor:
        real = (stored.tokens != TOKEN_STOP) & (torch.cumsum((stored.tokens == TOKEN_STOP).long(), 1) == 0)
        counts = torch.zeros(stored.top.shape[0], N_TOKENS, device=stored.top.device)
        counts.scatter_add_(1, stored.tokens, real.float())
        steps = real.float().sum(1, keepdim=True) / self.K
        top = torch.nn.functional.one_hot(stored.top, 2).float()
        frac = x.split_feats[torch.arange(x.batch, device=stored.top.device),
                             stored.split_index.clamp(max=max(x.split_feats.shape[1] - 1, 0)), 0:1] \
            if x.split_feats.shape[1] else torch.zeros_like(steps)
        frac = frac * (stored.top == TOP_SPLIT).float()[:, None]
        return self.cmd_embed(torch.cat([top, counts[:, :4] / self.K, steps, frac], -1))

    def sonar_logits(self, h: Tensor, cmd: Tensor) -> Tensor:
        return self.sonar_head(torch.cat([h, cmd], -1))

    # -- teacher-forced evaluation --------------------------------------------

    def decode_evaluate(self, ctx: Context, x: SpatialInputs, stored: StoredCommand
                        ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (joint_logp [B], entropy_proxy [B], sonar_entropy [B])."""
        h = ctx.h
        B = h.shape[0]
        top = torch.log_softmax(self.top_logits(h, x.split_legal), -1)
        top_p = top.exp()
        lp = top.gather(1, stored.top[:, None]).squeeze(1)
        ent = -(top_p * top).sum(-1)
        is_move = (stored.top == TOP_MOVE).float()

        dec = self.move_init(h)
        prev = torch.full((B,), BOS_TOKEN, dtype=torch.long, device=h.device)
        move_lp = torch.zeros(B, device=h.device)
        move_ent = torch.zeros(B, device=h.device)
        for j in range(self.K):
            logits, dec = self.move_step(dec, h, prev, stored.prefix[:, j], stored.masks[:, j])
            logp = torch.log_softmax(logits, -1)
            act = stored.active[:, j].float()
            move_lp = move_lp + logp.gather(1, stored.tokens[:, j:j + 1]).squeeze(1) * act
            move_ent = move_ent - (logp.exp() * logp).sum(-1) * act
            prev = stored.tokens[:, j]
        split = torch.log_softmax(self.split_logits(h, x.split_feats, x.split_mask), -1)
        idx = stored.split_index.clamp(max=split.shape[1] - 1)
        split_lp = split.gather(1, idx[:, None]).squeeze(1)
        split_ent = -(split.exp() * split).sum(-1)

        lp = lp + is_move * move_lp + (1 - is_move) * split_lp
        ent = ent + is_move * move_ent / self.K + (1 - is_move) * split_ent
        sonar_ent = torch.zeros(B, device=h.device)
        if self.sonar_head is not None:
            s = torch.log_softmax(self.sonar_logits(h, self.command_embedding(stored, x)), -1)
            lp = lp + s.gather(1, stored.sonar[:, None]).squeeze(1)
            sonar_ent = -(s.exp() * s).sum(-1)
        return lp, ent, sonar_ent

    # -- sampling ----------------------------------------------------------------

    @torch.no_grad()
    def decode_sample(self, ctx: Context, x: SpatialInputs, prepared: Sequence, states: Sequence,
                      codec: CommandCodec, deterministic: bool = False
                      ) -> tuple[list[Command], Tensor, StoredCommand]:
        """Sample one command per row, running the private prefix simulator
        for the move tokens. Returns commands, joint log-probs and the stored
        record `decode_evaluate` reproduces."""
        h = ctx.h
        B = h.shape[0]
        dev = h.device
        K = self.K
        pick = (lambda lg: lg.argmax(-1)) if deterministic else _gumbel_argmax

        top_logits = self.top_logits(h, x.split_legal)
        top = pick(top_logits)
        lp = torch.log_softmax(top_logits, -1).gather(1, top[:, None]).squeeze(1)

        tokens = torch.full((B, K), TOKEN_STOP, dtype=torch.long, device=dev)
        active = torch.zeros(B, K, dtype=torch.bool, device=dev)
        masks = torch.zeros(B, K, N_TOKENS, dtype=torch.bool, device=dev)
        prefix = torch.zeros(B, K, N_PREFIX_FEATURES, device=dev)
        sims = [codec.begin_move(prepared[i], states[i]) if top[i] == TOP_MOVE else None
                for i in range(B)]
        running = [i for i in range(B) if sims[i] is not None]
        dec = self.move_init(h)
        prev = torch.full((B,), BOS_TOKEN, dtype=torch.long, device=dev)
        for j in range(K):
            if not running:
                break
            rows = torch.tensor(running, device=dev)
            for i in running:
                masks[i, j] = torch.as_tensor(sims[i].mask(), device=dev)
                prefix[i, j] = torch.as_tensor(sims[i].features(), device=dev)
            logits, dec_rows = self.move_step(dec[rows], h[rows], prev[rows],
                                              prefix[rows, j], masks[rows, j])
            dec[rows] = dec_rows
            logp = torch.log_softmax(logits, -1)
            sampled = pick(logits)
            n_allowed = masks[rows, j].sum(-1)
            forced = masks[rows, j].float().argmax(-1)
            tok = torch.where(n_allowed > 1, sampled, forced)
            tokens[rows, j] = tok
            active[rows, j] = n_allowed > 1
            lp[rows] = lp[rows] + logp.gather(1, tok[:, None]).squeeze(1) * (n_allowed > 1).float()
            prev[rows] = tok
            still = []
            for r, i in enumerate(running):
                t = int(tok[r])
                if t != TOKEN_STOP:
                    sims[i].advance(t)
                    still.append(i)
            running = still

        split_logits = self.split_logits(h, x.split_feats, x.split_mask)
        split_idx = pick(split_logits)
        split_lp = torch.log_softmax(split_logits, -1).gather(1, split_idx[:, None]).squeeze(1)
        is_split = top == TOP_SPLIT
        lp = lp + split_lp * is_split.float()

        stored = StoredCommand(top, tokens, active, masks, prefix,
                               torch.where(is_split, split_idx, torch.zeros_like(split_idx)),
                               torch.zeros(B, dtype=torch.long, device=dev))
        if self.sonar_head is not None:
            s_logits = self.sonar_logits(h, self.command_embedding(stored, x))
            sonar = pick(s_logits)
            lp = lp + torch.log_softmax(s_logits, -1).gather(1, sonar[:, None]).squeeze(1)
            stored.sonar = sonar

        cmds = []
        for i in range(B):
            if bool(is_split[i]):
                sizes, feats = codec.split_table(prepared[i])
                k = int(split_idx[i])
                cmds.append(Command(TOP_SPLIT, split_n=int(sizes[k]), sonar=int(stored.sonar[i]),
                                    split_candidates=sizes, split_features=feats, split_index=k))
            else:
                trace = MoveTrace(tokens[i].cpu().numpy(), active[i].cpu().numpy(),
                                  masks[i].cpu().numpy(), prefix[i].cpu().numpy())
                cmds.append(Command(TOP_MOVE, dirs=codec.dirs_of(trace), sonar=int(stored.sonar[i]),
                                    move=trace))
        return cmds, lp, stored

    # -- auxiliary predictions (spec 10) ---------------------------------------

    def aux_predictions(self, ctx: Context, x: SpatialInputs, stored: StoredCommand
                        ) -> dict[str, Tensor]:
        """Forecasts made before the target observations are consumed."""
        pearl_in = torch.cat([ctx.memory_visible_old, x.visible_prev, x.visible_age[..., None]], -1)
        own = self.aux_own(torch.cat([ctx.h, self.command_embedding(stored, x)], -1))
        return {
            "pearl_logit": self.aux_pearl(pearl_in).squeeze(-1),      # [B, 49]
            "enemy_logits": self.aux_enemy(ctx.h),                     # [B, 50]
            "alive_logit": own[:, 0],                                  # [B]
            "delta_logits": own[:, 1:],                                # [B, 17] length change -8..+8
        }
