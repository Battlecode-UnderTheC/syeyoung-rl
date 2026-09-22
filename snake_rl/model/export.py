"""TorchScript export of the spatial actor (spec section 13).

Two scripted entry points, plain tensors in and out: the neural encoding
(one turn: M_old, h_old -> h_new, M_new) and the decoder evaluation (top, one
move token step, split scores, sonar). Sampling is the same decoder step
driven by the runtime's private prefix simulator, which is deterministic
bookkeeping and stays outside the graph.
"""

from __future__ import annotations

import json as _json
import pathlib

import torch
from torch import Tensor, nn

from ..env.schema import (ACTION_CODEC_VERSION, N_PREFIX_FEATURES, N_TOKENS, SCHEMA_VERSION,
                          schema_hash)
from .actor import BOS_TOKEN, NEG_INF, SpatialActor


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
