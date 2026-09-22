"""Export the actor as TorchScript so the native trainer runs the same graph.

Rather than reimplement the network in C++ and risk it drifting from the Python
one, the model is defined once here and scripted. The C++ trainer loads the
scripted module, runs it, and optimises its parameters directly -- there is only
ever one definition of the policy.

The scripted wrapper is plain tensors in, plain tensors out: TorchScript cannot
take the dataclasses the eager path uses.
"""

from __future__ import annotations

import pathlib

import torch
from torch import Tensor, nn

from .actor import NEG_INF, Actor
from .critic import CriticState, TeamCritic


class ScriptableActor(nn.Module):
    """One timestep of the actor, as tensors."""

    def __init__(self, actor: Actor):
        super().__init__()
        self.spatial = actor.spatial
        self.scalar = actor.scalar
        self.message = actor.message
        self.fuse = actor.fuse
        # A single timestep through cuDNN's RNN path costs a weight compaction
        # per call and is the wrong primitive for seq_len 1, so the gate maths
        # is done directly on the same parameters. test_model.py checks this
        # against nn.GRU.
        self.weight_ih = actor.gru.weight_ih_l0
        self.weight_hh = actor.gru.weight_hh_l0
        self.bias_ih = actor.gru.bias_ih_l0
        self.bias_hh = actor.gru.bias_hh_l0
        self.action_head = actor.action_head
        self.value_head = actor.value_head
        self.transmit_head = actor.transmit_head
        self.symbol_head = actor.symbol_head
        self.n_symbols = int(actor.n_symbols)
        self.symbol_vocab = int(actor.symbol_vocab)
        self.hidden_size = int(actor.hidden)
        self.sonar = bool(actor.enable_sonar)
        self.neg_inf = float(NEG_INF)   # TorchScript cannot close over globals

    def forward(self, grid: Tensor, scalars: Tensor, messages: Tensor,
                message_mask: Tensor, action_mask: Tensor, hidden: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        features = self.fuse(torch.cat(
            [self.spatial(grid), self.scalar(scalars),
             self.message(messages, message_mask)], dim=-1))

        # one GRU step, matching torch.nn.GRU's gate order (r, z, n)
        gates_input = torch.addmm(self.bias_ih, features, self.weight_ih.t())
        gates_hidden = torch.addmm(self.bias_hh, hidden, self.weight_hh.t())
        size = self.hidden_size
        reset = torch.sigmoid(gates_input[:, :size] + gates_hidden[:, :size])
        update = torch.sigmoid(gates_input[:, size:2 * size] +
                               gates_hidden[:, size:2 * size])
        candidate = torch.tanh(gates_input[:, 2 * size:] +
                               reset * gates_hidden[:, 2 * size:])
        state = (1.0 - update) * candidate + update * hidden
        new_hidden = state

        logits = self.action_head(state)
        # a boxed-in dragon has nothing legal; leave its row alone rather than
        # emit a row of -inf that would sample NaNs
        usable = action_mask | torch.logical_not(action_mask.any(-1, keepdim=True))
        logits = logits.masked_fill(torch.logical_not(usable), self.neg_inf)

        symbols = self.symbol_head(state).view(
            state.size(0), self.n_symbols, self.symbol_vocab)
        return (logits, self.transmit_head(state).squeeze(-1), symbols, new_hidden,
                self.value_head(state).squeeze(-1))


class ScriptableCritic(nn.Module):
    """The team critic as tensors, for the native trainer.

    Spec section 4's baseline: full state in, one value per player out. The
    actor never sees any of this -- it is loaded by the learner only.
    """

    def __init__(self, critic: TeamCritic):
        super().__init__()
        self.map_body = critic.map_body
        self.map_head = critic.map_head
        self.dragon_encoder = critic.dragon_encoder
        self.set_head = critic.set_head
        self.value = critic.value
        self.pairs_a = [a for a, _ in critic.team_channel_pairs]
        self.pairs_b = [b for _, b in critic.team_channel_pairs]
        self.zero_sum = bool(critic.zero_sum)
        # torch.finfo is not scriptable; any value below every real activation
        # does the same job for a masked max.
        self.very_negative = -1.0e30

    def _side(self, map_grid: Tensor, dragons: Tensor, team: Tensor,
              mask: Tensor) -> Tensor:
        own = torch.nn.functional.one_hot(team.clamp(0, 1), 2).to(dragons.dtype)
        encoded = self.dragon_encoder(torch.cat([dragons, own], dim=-1))
        weight = mask.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        peak = encoded.masked_fill(torch.logical_not(mask.unsqueeze(-1)),
                                   self.very_negative)
        peak = peak.max(dim=1).values
        peak = torch.where(mask.any(dim=1, keepdim=True), peak, torch.zeros_like(peak))
        pooled = self.set_head(torch.cat([mean, peak], dim=-1))
        spatial = self.map_head(self.map_body(map_grid))
        return self.value(torch.cat([spatial, pooled], dim=-1)).squeeze(-1)

    def forward(self, map_grid: Tensor, dragons: Tensor, team: Tensor,
                mask: Tensor) -> Tensor:
        """[B, 2] -- the value for team A and for team B."""
        a = self._side(map_grid, dragons, team, mask)
        # the other side's view: swap the team-specific map channels and flip
        # the own/enemy flags, so both values come from the same weights
        flipped = map_grid.clone()
        for i in range(len(self.pairs_a)):
            first, second = self.pairs_a[i], self.pairs_b[i]
            swap = flipped[:, first].clone()
            flipped[:, first] = flipped[:, second]
            flipped[:, second] = swap
        b = self._side(flipped, dragons, 1 - team, mask)
        if self.zero_sum:
            half = (a - b) * 0.5
            return torch.stack([half, -half], dim=-1)
        return torch.stack([a, b], dim=-1)


def export_critic(critic: TeamCritic, path: str | pathlib.Path) -> pathlib.Path:
    module = ScriptableCritic(critic).eval()
    scripted = torch.jit.script(module)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(path))
    return path


def export(actor: Actor, path: str | pathlib.Path) -> pathlib.Path:
    """Script the actor and write it where the native trainer can load it."""
    module = ScriptableActor(actor).eval()
    scripted = torch.jit.script(module)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(path))
    return path


# ===========================================================================
# dragons-spatial-v1 export (spec section 13)
# ===========================================================================
#
# Two scripted entry points, plain tensors in and out: the neural encoding
# (one turn: M_old, h_old -> h_new, M_new) and the decoder evaluation (top,
# one move token step, split scores, sonar). Sampling is the same decoder step
# driven by the runtime's private prefix simulator, which is deterministic
# bookkeeping and stays outside the graph. Schema and dimension metadata ride
# along in `spatial_export_metadata`.

import json as _json

from ..env.schema import (ACTION_CODEC_VERSION, N_PREFIX_FEATURES, N_TOKENS, SCHEMA_VERSION,
                          schema_hash)
from .actor import BOS_TOKEN, SpatialActor


class ScriptableSpatialEncoder(nn.Module):
    def __init__(self, actor: SpatialActor):
        super().__init__()
        self.local_enc = actor.local_enc
        self.status_enc = actor.status_enc
        self.history_enc = actor.history_enc
        self.entity_enc = actor.entity_enc
        self.writer = actor.writer
        self.reader = actor.reader
        self.map_local = actor.map_local
        self.global_reads = actor.global_reads
        self.fuse = actor.fuse
        self.gru = actor.gru
        self.sonar = actor.inbox_enc is not None
        self.inbox_enc = actor.inbox_enc if actor.inbox_enc is not None else nn.Identity()

    def forward(self, local: Tensor, facts: Tensor, board_valid: Tensor, visible_xy: Tensor,
                visible_prev: Tensor, visible_age: Tensor, status: Tensor, history: Tensor,
                history_mask: Tensor, entities: Tensor, entity_mask: Tensor,
                portal_pairs: Tensor, pair_mask: Tensor, portal_axis: Tensor,
                portal_unpaired: Tensor, unpaired_mask: Tensor, inbox: Tensor,
                inbox_mask: Tensor, head_xy: Tensor, M_old: Tensor, h_old: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
        B = local.shape[0]
        cells, local128 = self.local_enc(local)
        status64, status16 = self.status_enc(status)
        M_new = self.writer(M_old, cells, visible_prev, visible_age, status16, visible_xy)
        cond = torch.cat([h_old, status64], dim=-1)
        fine, coarse, v_coarse = self.reader(facts, M_new, board_valid, cond, portal_pairs,
                                             pair_mask, portal_axis, portal_unpaired, unpaired_mask)
        map_local = self.map_local(fine, head_xy)
        global_ = self.global_reads(coarse, v_coarse, facts, head_xy,
                                    torch.cat([local128, h_old, status64], dim=-1))
        parts = [local128, map_local, global_, self.entity_enc(entities, entity_mask), status64,
                 self.history_enc(history, history_mask)]
        if self.sonar:
            parts.append(self.inbox_enc(inbox, inbox_mask))
        fused = self.fuse(torch.cat(parts, dim=-1))
        h_new = self.gru(fused, h_old)
        return h_new, M_new, fused


class ScriptableSpatialDecoder(nn.Module):
    """Top-level, split, sonar logits and one teacher-forced move step."""

    def __init__(self, actor: SpatialActor):
        super().__init__()
        self.top_head = actor.top_head
        self.dec_init = actor.dec_init
        self.tok_embed = actor.tok_embed
        self.prefix_proj = actor.prefix_proj
        self.h_proj = actor.h_proj
        self.dec_cell = actor.dec_cell
        self.tok_head = actor.tok_head
        self.split_mlp = actor.split_mlp
        self.cmd_embed = actor.cmd_embed
        self.sonar = actor.sonar_head is not None
        self.sonar_head = actor.sonar_head if actor.sonar_head is not None else nn.Identity()
        self.neg_inf = float(NEG_INF)
        self.K = int(actor.K)

    def top_logits(self, h: Tensor, split_legal: Tensor) -> Tensor:
        logits = self.top_head(h)
        deny = torch.logical_not(split_legal)[:, None] & (torch.arange(2, device=h.device) == 1)[None]
        return logits.masked_fill(deny, self.neg_inf)

    def move_init(self, h: Tensor) -> Tensor:
        return torch.tanh(self.dec_init(h))

    def move_step(self, dec: Tensor, h: Tensor, prev_tok: Tensor, prefix: Tensor, mask: Tensor
                  ) -> tuple[Tensor, Tensor]:
        inp = torch.cat([self.tok_embed(prev_tok), self.prefix_proj(prefix), self.h_proj(h)], -1)
        dec = self.dec_cell(inp, dec)
        logits = self.tok_head(dec)
        usable = mask | torch.logical_not(mask.any(-1, keepdim=True))
        return logits.masked_fill(torch.logical_not(usable), self.neg_inf), dec

    def split_logits(self, h: Tensor, feats: Tensor, mask: Tensor) -> Tensor:
        C = feats.shape[1]
        hh = h[:, None, :].expand(-1, C, -1)
        logits = self.split_mlp(torch.cat([hh, feats], dim=-1)).squeeze(-1)
        usable = mask | torch.logical_not(mask.any(-1, keepdim=True))
        return logits.masked_fill(torch.logical_not(usable), self.neg_inf)

    def sonar_logits(self, h: Tensor, cmd_features: Tensor) -> Tensor:
        """cmd_features [B, 8]: top one-hot(2), direction counts/K (4), steps/K, split fraction."""
        return self.sonar_head(torch.cat([h, self.cmd_embed(cmd_features)], -1))

    def forward(self, h: Tensor, split_legal: Tensor, dec: Tensor, prev_tok: Tensor,
                prefix: Tensor, mask: Tensor, split_feats: Tensor, split_mask: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        top = self.top_logits(h, split_legal)
        tok, dec_new = self.move_step(dec, h, prev_tok, prefix, mask)
        split = self.split_logits(h, split_feats, split_mask)
        return top, tok, dec_new, split


def spatial_export_metadata(actor: SpatialActor) -> dict:
    cfg = actor.config
    return {
        "schema_version": SCHEMA_VERSION, "schema_hash": schema_hash(),
        "action_codec_version": ACTION_CODEC_VERSION,
        "latent_channels": cfg.latent_channels, "tactical_hidden": cfg.tactical_hidden,
        "decoder_hidden": cfg.decoder_hidden, "spatial_widths": list(cfg.spatial_widths),
        "spatial_iterations": cfg.spatial_iterations, "global_reads": cfg.global_reads,
        "read_width": cfg.read_width, "history_turns": cfg.history_turns,
        "max_move_steps": cfg.max_move_steps, "sonar": cfg.sonar,
        "n_tokens": N_TOKENS, "bos_token": BOS_TOKEN, "n_prefix_features": N_PREFIX_FEATURES,
        "parameters": sum(p.numel() for p in actor.parameters()),
    }


def export_spatial(actor: SpatialActor, directory: str | pathlib.Path) -> dict[str, pathlib.Path]:
    """Script both entry points and write them with their metadata."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    encoder = torch.jit.script(ScriptableSpatialEncoder(actor).eval())
    torch.jit.save(encoder, str(directory / "spatial_encoder.ts"))
    out["encoder"] = directory / "spatial_encoder.ts"
    decoder = torch.jit.script(ScriptableSpatialDecoder(actor).eval())
    torch.jit.save(decoder, str(directory / "spatial_decoder.ts"))
    out["decoder"] = directory / "spatial_decoder.ts"
    (directory / "spatial_metadata.json").write_text(_json.dumps(spatial_export_metadata(actor), indent=2))
    out["metadata"] = directory / "spatial_metadata.json"
    return out
