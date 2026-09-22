"""Learned spatial memory: sparse writes, multiscale recurrent reads (spec 5.1-5.3).

Persistent learned state is exactly two tensors per lifetime: the 16-channel
map `M` and the 128-wide tactical state `h`. Everything in here is either a
write into `M` at the coordinates the dragon actually observed this turn, or a
read of `M` concatenated with the factual map. The coarse ConvGRU is working
memory *within* a decision, re-initialised from the current coarse features
every turn; it is not a third persistent map.

Boards are tori, so every spatial convolution pads circularly at the board's
real size. A batch must share one (H, W): the caller groups by dimensions,
which keeps the wrapping exact instead of wrapping a padded 64x64 tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn
from torch import Tensor, nn

from ..env.schema import F, N_FACT_CHANNELS, VISION, VISION_RADIUS

# TorchScript cannot read module globals or IntEnum members inside the graph,
# so the factual channel indices used below are literals, pinned here to the
# schema so a registry change cannot silently drift from them.
assert (int(F.SEEN), int(F.VISIBLE), int(F.AGE_SEEN), int(F.PEARL)) == (0, 1, 2, 3)
assert (int(F.OCC), int(F.KELP), int(F.PORTAL), int(F.BODY_KNOWN)) == (6, 21, 25, 38)
assert VISION_RADIUS == 3


# -- torus helpers -----------------------------------------------------------

def circular_conv(x: Tensor, weight: Tensor, bias: Tensor | None, groups: int = 1) -> Tensor:
    k = weight.shape[-1]
    p = k // 2
    if p:
        x = Fn.pad(x, (p, p, p, p), mode="circular")
    return Fn.conv2d(x, weight, bias, groups=groups)


class SeparableConv(nn.Module):
    """Depthwise 3x3 (circular) followed by a pointwise 1x1."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.depth = nn.Conv2d(cin, cin, 3, groups=cin, padding=0)
        self.point = nn.Conv2d(cin, cout, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.point(circular_conv(x, self.depth.weight, self.depth.bias, groups=x.shape[1]))


class ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = SeparableConv(c, c)

    def forward(self, x: Tensor) -> Tensor:
        return x + Fn.silu(self.conv(x))


def masked_downsample(x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Area-average by 2 with ceil sizes; partial cells average only their valid
    tiles. `valid` is [B, 1, H, W] float. Returns (x_half, valid_half)."""
    B, C, H, W = x.shape
    ph, pw = H % 2, W % 2
    if ph or pw:
        x = Fn.pad(x, (0, pw, 0, ph))
        valid = Fn.pad(valid, (0, pw, 0, ph))
    total = Fn.avg_pool2d(x * valid, 2)
    count = Fn.avg_pool2d(valid, 2)
    return total / count.clamp(min=0.25), (count > 0).to(x.dtype)


def upsample_to(x: Tensor, H: int, W: int) -> Tensor:
    return Fn.interpolate(x, scale_factor=2.0, mode="nearest")[:, :, :H, :W]


def crop_window(x: Tensor, head_xy: Tensor) -> Tensor:
    """[B, C, 7, 7] wrapped crop of x centred on each row's head (x, y)."""
    B, C, H, W = x.shape
    off = torch.arange(-3, 4, device=x.device)      # VISION_RADIUS literal for TorchScript
    ys = (head_xy[:, 1:2] + off[None, :]) % H          # [B, 7]
    xs = (head_xy[:, 0:1] + off[None, :]) % W
    b = torch.arange(B, device=x.device)[:, None, None]
    rows = x[b, :, ys[:, :, None], xs[:, None, :]]      # [B, 7, 7, C]
    return rows.permute(0, 3, 1, 2)


# -- 5.1 sparse learned writes ---------------------------------------------

class SparseWriter(nn.Module):
    """One GRUCell per observed cell, addressed by its absolute coordinate.

    Input per cell: the local encoder's feature for that window tile, the
    cell's factual features *before* this turn's write, its pre-write age and a
    16-d projection of the status vector, so the writer can tell a revisit from
    continuous visibility. Unobserved cells are untouched.
    """

    def __init__(self, local_channels: int = 32, latent: int = 16, status_dim: int = 16,
                 fact_channels: int = N_FACT_CHANNELS):
        super().__init__()
        self.latent = latent
        self.status_proj = nn.Linear(status_dim, status_dim)
        self.write_proj = nn.Sequential(
            nn.Linear(local_channels + fact_channels + 1 + status_dim, 32), nn.SiLU())
        self.cell = nn.GRUCell(32, latent)

    def forward(self, M_old: Tensor, local_cells: Tensor, prev_facts: Tensor,
                age: Tensor, status16: Tensor, xy: Tensor) -> Tensor:
        """M_old [B,16,H,W]; local_cells [B,49,32]; prev_facts [B,49,42];
        age [B,49]; status16 [B,16]; xy [B,49,2] long. Returns M_new."""
        B, N = xy.shape[:2]
        st = self.status_proj(status16)[:, None, :].expand(B, N, -1)
        w = torch.cat([local_cells, prev_facts, age[..., None], st], dim=-1)
        w = self.write_proj(w)
        b = torch.arange(B, device=M_old.device)[:, None].expand(B, N)
        old = M_old[b, :, xy[..., 1], xy[..., 0]]                  # [B, N, 16]
        new = self.cell(w.reshape(B * N, -1), old.reshape(B * N, -1)).view(B, N, -1)
        # scatter out of place; duplicate coordinates would coalesce to the last
        # write, but a valid window on a >= 7x7 board never has any
        M_new = M_old.clone()
        M_new[b, :, xy[..., 1], xy[..., 0]] = new.to(M_new.dtype)
        return M_new


# -- 5.2 multiscale recurrent spatial computation --------------------------

class ConvGRUCell(nn.Module):
    """Separable-convolution gates, conditioned by a broadcast vector."""

    def __init__(self, c: int, cond_dim: int):
        super().__init__()
        self.gates = SeparableConv(2 * c, 2 * c)
        self.cand = SeparableConv(2 * c, c)
        self.cond_gates = nn.Linear(cond_dim, 2 * c)
        self.cond_cand = nn.Linear(cond_dim, c)

    def forward(self, x: Tensor, h: Tensor, cond: Tensor) -> Tensor:
        g = self.gates(torch.cat([x, h], 1)) + self.cond_gates(cond)[:, :, None, None]
        r, z = torch.sigmoid(g).chunk(2, dim=1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)) + self.cond_cand(cond)[:, :, None, None])
        return (1 - z) * n + z * h


class PortalExchange(nn.Module):
    """Neural message passing between the two tiles owning a confirmed portal
    pair. Tiles owning an unpaired portal receive an unknown-partner vector."""

    def __init__(self, c: int):
        super().__init__()
        self.msg = nn.Linear(c + 2, c)      # + axis one-hot
        self.unknown = nn.Parameter(torch.zeros(c))
        nn.init.normal_(self.unknown, std=0.02)

    def forward(self, x: Tensor, pairs: Tensor, pair_mask: Tensor, axis: Tensor,
                loose: Tensor, loose_mask: Tensor) -> Tensor:
        """pairs [B,P,2,2] (y,x); pair_mask [B,P]; axis [B,P]; loose [B,U,2]."""
        B, C, H, W = x.shape
        out = x
        if pairs.shape[1]:
            b = torch.arange(B, device=x.device)[:, None]
            a_y, a_x = pairs[:, :, 0, 0], pairs[:, :, 0, 1]
            b_y, b_x = pairs[:, :, 1, 0], pairs[:, :, 1, 1]
            fa = x[b, :, a_y, a_x]                                      # [B,P,C]
            fb = x[b, :, b_y, b_x]
            ax = Fn.one_hot(axis.clamp(0, 1), 2).to(x.dtype)
            m_to_a = self.msg(torch.cat([fb, ax], -1)) * pair_mask[..., None].to(x.dtype)
            m_to_b = self.msg(torch.cat([fa, ax], -1)) * pair_mask[..., None].to(x.dtype)
            # accumulate through a channel-last scratch so index_put can sum
            scratch = torch.zeros(B, H, W, C, device=x.device, dtype=x.dtype)
            scratch = scratch.index_put((b.expand(B, pairs.shape[1]), a_y, a_x), m_to_a, accumulate=True)
            scratch = scratch.index_put((b.expand(B, pairs.shape[1]), b_y, b_x), m_to_b, accumulate=True)
            out = out + scratch.permute(0, 3, 1, 2)
        if loose.shape[1]:
            b = torch.arange(B, device=x.device)[:, None].expand(B, loose.shape[1])
            add = self.unknown[None, None, :] * loose_mask[..., None].to(x.dtype)
            scratch = torch.zeros(B, H, W, C, device=x.device, dtype=x.dtype)
            scratch = scratch.index_put((b, loose[..., 0], loose[..., 1]), add.expand(B, loose.shape[1], C), accumulate=True)
            out = out + scratch.permute(0, 3, 1, 2)
        return out


class MultiscaleReader(nn.Module):
    def __init__(self, in_channels: int = N_FACT_CHANNELS + 16,
                 widths: tuple[int, int, int] = (16, 24, 32), iterations: int = 2,
                 cond_dim: int = 128 + 64):
        super().__init__()
        f, m, c = widths
        self.iterations = iterations
        self.fine_in = nn.Conv2d(in_channels, f, 1)
        self.fine_block = ResidualBlock(f)
        self.portal_in = PortalExchange(f)
        self.half_in = nn.Conv2d(f, m, 1)
        self.half_block = ResidualBlock(m)
        self.coarse_in = nn.Conv2d(m, c, 1)
        self.gru = ConvGRUCell(c, cond_dim)
        self.half_out = nn.Conv2d(m + c, m, 1)
        self.fine_out = nn.Conv2d(f + m, f, 1)
        self.portal_out = PortalExchange(f)
        # max/mean summaries of occupancy and edge evidence ride along so
        # narrow threats are not averaged away (5.3)
        self.summary = nn.Conv2d(2 * 8, c, 1)
        self.widths = widths

    @staticmethod
    def _evidence(facts: Tensor) -> Tensor:
        occ = facts[:, 6 + 1:6 + 7].amax(dim=1, keepdim=True)           # any occupant
        enemy = facts[:, 6 + 5:6 + 7].amax(dim=1, keepdim=True)
        kelp = facts[:, 21:21 + 4].amax(dim=1, keepdim=True)
        portal = facts[:, 25:25 + 4].amax(dim=1, keepdim=True)
        return torch.cat([occ, enemy, kelp, portal, facts[:, 3:3 + 1],
                          facts[:, 0:0 + 1], facts[:, 1:1 + 1],
                          facts[:, 38:38 + 1]], dim=1)          # [B,8,H,W]

    def forward(self, facts: Tensor, M: Tensor, valid: Tensor, cond: Tensor,
                pairs: Tensor, pair_mask: Tensor, axis: Tensor, loose: Tensor,
                loose_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (fine [B,f,H,W], coarse [B,c,Hc,Wc], coarse_valid [B,1,Hc,Wc])."""
        B, _, H, W = facts.shape
        v = valid[:, None].to(facts.dtype)
        x = torch.cat([facts, M], dim=1) * v
        fine = self.fine_block(Fn.silu(self.fine_in(x)))
        fine = self.portal_in(fine, pairs, pair_mask, axis, loose, loose_mask)

        half, v_half = masked_downsample(fine, v)
        half = self.half_block(Fn.silu(self.half_in(half)))
        coarse, v_coarse = masked_downsample(half, v_half)
        coarse = Fn.silu(self.coarse_in(coarse))

        ev = self._evidence(facts)
        ev_max = Fn.max_pool2d(Fn.pad(ev, (0, W % 2, 0, H % 2)), 2)
        ev_max = Fn.max_pool2d(Fn.pad(ev_max, (0, ev_max.shape[-1] % 2, 0, ev_max.shape[-2] % 2)), 2)
        ev_mean, _v1 = masked_downsample(ev, v)
        ev_mean, _v2 = masked_downsample(ev_mean, v_half)
        coarse = coarse + self.summary(torch.cat([ev_max, ev_mean], dim=1))

        state = coarse
        for _iter in range(self.iterations):
            state = self.gru(coarse, state, cond) * v_coarse
        coarse = state

        up = upsample_to(coarse, half.shape[-2], half.shape[-1])
        half = Fn.silu(self.half_out(torch.cat([half, up], dim=1))) * v_half
        up = upsample_to(half, H, W)
        fine = Fn.silu(self.fine_out(torch.cat([fine, up], dim=1))) * v
        fine = self.portal_out(fine, pairs, pair_mask, axis, loose, loose_mask)
        return fine, coarse, v_coarse


# -- 5.3 reads -----------------------------------------------------------------

class LocalMapRead(nn.Module):
    def __init__(self, c: int = 16, out: int = 128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(c * VISION * VISION, out), nn.SiLU())

    def forward(self, fine: Tensor, head_xy: Tensor) -> Tensor:
        return self.proj(crop_window(fine, head_xy).flatten(1))


class GlobalReads(nn.Module):
    """Four learned cross-attention reads over the coarse tokens. Queries are
    exchangeable at initialisation and never named or supervised."""

    def __init__(self, c: int = 32, n_reads: int = 4, width: int = 32,
                 query_in: int = 128 + 128 + 64):
        super().__init__()
        self.n, self.w = n_reads, width
        self.token = nn.Linear(c + 4 + 2, width)     # + head-relative sin/cos + observed, age
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.query = nn.Linear(query_in, n_reads * width)
        self.out_dim = n_reads * width

    def forward(self, coarse: Tensor, valid: Tensor, facts: Tensor, head_xy: Tensor,
                query_src: Tensor) -> Tensor:
        B, C, Hc, Wc = coarse.shape
        H, W = facts.shape[-2:]
        dev = coarse.device
        # wrapped head-relative coordinates of each coarse token's centre
        cy = (torch.arange(Hc, device=dev) * 2 + 0.5)[None, :, None].expand(B, Hc, Wc)
        cx = (torch.arange(Wc, device=dev) * 2 + 0.5)[None, None, :].expand(B, Hc, Wc)
        ry = 2 * math.pi * ((cy - head_xy[:, 1, None, None]) % H) / H
        rx = 2 * math.pi * ((cx - head_xy[:, 0, None, None]) % W) / W
        obs_frac, _o1 = masked_downsample(facts[:, 0:0 + 1], torch.ones_like(facts[:, :1]))
        obs_frac, _o2 = masked_downsample(obs_frac, torch.ones_like(obs_frac))
        age, _a1 = masked_downsample(facts[:, 2:2 + 1], torch.ones_like(facts[:, :1]))
        age, _a2 = masked_downsample(age, torch.ones_like(age))
        extra = torch.stack([torch.sin(ry), torch.cos(ry), torch.sin(rx), torch.cos(rx)], dim=1)
        tokens = torch.cat([coarse, extra, obs_frac[:, :, :Hc, :Wc], age[:, :, :Hc, :Wc]], dim=1)
        tokens = self.token(tokens.flatten(2).transpose(1, 2))          # [B, T, w]
        k, v = self.key(tokens), self.value(tokens)
        q = self.query(query_src).view(B, self.n, self.w)
        scores = torch.einsum("bnw,btw->bnt", q, k) / math.sqrt(self.w)
        mask = valid.flatten(1) > 0                                      # [B, T]
        scores = scores.masked_fill(~mask[:, None, :], -3.0e4)
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("bnt,btw->bnw", attn, v).flatten(1)
