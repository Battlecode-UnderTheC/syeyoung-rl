"""From dragons-env observation records to the model's inputs.

A record is the judge's round block in numbers (dragons-env/cpp/dragons/
vec_env.hpp); everything here is derived from it alone, so a bot parsing the
judge's text computes the same inputs. MODEL.md lists every channel.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import dragons_env
from dragons_env import SCALARS as S, STATE_CHANNELS, STATE_SCALARS as G, BLOCKS

TILES = 49                 # 7 x 7 window
WINDOW_CHANNELS = 21       # float channels per tile (portal ids are embedded separately)
PORTAL_IDS = 64            # portal id mod 64; index 64 = no portal
NUM_SCALARS = 16
NUM_ACTIONS = 24           # 4 steps, 16 two-step sprints, 4 splits
CELL = 2                   # a memory cell is 2 x 2 tiles
MAP_CELLS = 32             # cells a side: boards up to 64 x 64
CROP = 9                   # local read: 9 x 9 cells around the head
POOL = 8                   # global read: the map pooled to 8 x 8
SPLIT_FRACTIONS = (0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0)

N_, E_, S_, W_ = 0, 1, 2, 3
DR = (-1, 0, 1, 0)
DC = (0, 1, 0, -1)
CENTER = 3 * 7 + 3


def layout(inbox_cap: int = 16, state: bool = True) -> dict:
    return dragons_env._core.layout(inbox_cap, state)


class Inputs(NamedTuple):
    window: jax.Array       # [B, 49, 21] f32
    portal: jax.Array       # [B, 49, 4] i32, 0..64
    scalars: jax.Array      # [B, 16] f32
    mask: jax.Array         # [B, 24] bool, True = allowed
    write_cell: jax.Array   # [B, 49] i32, memory cell of each window tile
    crop_cell: jax.Array    # [B, 81] i32, the 9 x 9 cells around the head (repeats on boards under 18 tiles)
    crop_bin: jax.Array     # [B, 81] i32, pooled bin of each crop cell
    bin_cells: jax.Array    # [B, 64] f32, cells of the board in each bin (0 for bins a small board lacks)
    here: jax.Array         # [B, 64] f32, one-hot of the head's bin


class StateInputs(NamedTuple):
    grid: jax.Array         # [B, 8, 16, 16] f32
    scalars: jax.Array      # [B, 12] f32


def field32(rec: jax.Array, offset: int, n: int) -> jax.Array:
    b = rec.shape[0]
    return jax.lax.bitcast_convert_type(rec[:, offset:offset + 4 * n].reshape(b, n, 4), jnp.int32)


def field8(rec: jax.Array, offset: int, n: int, signed: bool = False) -> jax.Array:
    x = rec[:, offset:offset + n]
    return jax.lax.bitcast_convert_type(x, jnp.int8) if signed else x


def decode(rec: jax.Array, L: dict) -> Inputs:
    """`rec`: uint8 [B, R] records (R >= L['obs_bytes'])."""
    b = rec.shape[0]
    f = lambda x: x.astype(jnp.float32)
    sc = field32(rec, L["scalars"], len(S))
    col = lambda name: sc[:, S[name]]
    # padding records are all zero; clamping keeps them finite (they are masked out of every loss)
    length = col("LENGTH")
    width, height = jnp.maximum(col("WIDTH"), 1), jnp.maximum(col("HEIGHT"), 1)
    unit_limit = jnp.maximum(col("UNIT_LIMIT"), 1)

    pearl = f(field8(rec, L["pearl"], TILES))
    pin = field32(rec, L["pearl_in"], TILES)
    bed = pin >= 0
    countdown = f(jnp.clip(pin, 0, 50)) / 50.0 * f(bed)
    bid = field32(rec, L["body_id"], TILES)
    bteam = field8(rec, L["body_team"], TILES, signed=True).astype(jnp.int32)
    bhead = field8(rec, L["body_head"], TILES) == 1
    bdir = field8(rec, L["body_dir"], TILES, signed=True).astype(jnp.int32)
    occ = bid >= 0
    me = bid == col("ID")[:, None]
    same = bteam == col("TEAM")[:, None]
    direction = jax.nn.one_hot(jnp.maximum(bdir, 0), 4) * f(bdir >= 0)[..., None]

    he = field32(rec, L["hedge"], 56).reshape(b, 8, 7)
    ve = field32(rec, L["vedge"], 56).reshape(b, 7, 8)
    sides = jnp.stack([he[:, :7], ve[:, :, 1:], he[:, 1:], ve[:, :, :7]], -1).reshape(b, TILES, 4)   # N E S W
    kelp = sides == dragons_env.EDGE_KELP
    portal = sides >= 0

    ch = lambda x: f(x)[..., None]
    window = jnp.concatenate([
        pearl[..., None], ch(bed), countdown[..., None], ch(pin == 1),
        ch(me), ch(occ & ~me & same & bhead), ch(occ & ~me & same & ~bhead),
        ch(occ & ~same & bhead), ch(occ & ~same & ~bhead),
        direction, f(kelp), f(portal)], -1)

    hx, hy, rnd, facing = col("HEAD_X"), col("HEAD_Y"), col("ROUND"), col("DIR")
    tau = 2 * np.pi
    scalars = jnp.stack([
        jnp.log1p(f(length)) / 4.0, f(col("UNIT_COUNT")) / f(unit_limit),
        f(rnd) / 500.0, f(499 - rnd) / 500.0,
        f(facing == 0), f(facing == 1), f(facing == 2), f(facing == 3),
        f(col("TEAM")), f(width) / 64.0, f(height) / 64.0,
        jnp.sin(tau * f(hx) / f(width)), jnp.cos(tau * f(hx) / f(width)),
        jnp.sin(tau * f(hy) / f(height)), jnp.cos(tau * f(hy) / f(height)),
        jnp.log1p(f(col("TURN"))) / 6.0], 1)

    # action mask: remove only what is certain death from what the dragon sees
    tile = lambda r, c: r * 7 + c
    allowed, blocked1, via_portal = [], [], []
    for d in range(4):
        p = portal[:, CENTER, d]
        blk = kelp[:, CENTER, d] | (~p & occ[:, tile(3 + DR[d], 3 + DC[d])])
        via_portal.append(p)
        blocked1.append(blk)
        allowed.append(~blk)
    can_sprint = length >= 3
    for a in range(4):
        r1, c1 = 3 + DR[a], 3 + DC[a]
        t1 = tile(r1, c1)
        first = ~blocked1[a] & ~via_portal[a] & can_sprint   # past a portal the next step is out of view
        for d in range(4):
            p2 = portal[:, t1, d]
            blk2 = kelp[:, t1, d] | (~p2 & occ[:, tile(r1 + DR[d], c1 + DC[d])])
            allowed.append(first & ~blk2)
    can_split = (length >= 4) & (col("UNIT_COUNT") < col("UNIT_LIMIT"))
    allowed += [can_split] * 4
    mask = jnp.stack(allowed, 1)
    none = ~mask.any(1, keepdims=True)
    mask = mask | (none & (jnp.arange(NUM_ACTIONS) < 4)[None])

    # memory geometry
    wc, hc = (width + 1) // CELL, (height + 1) // CELL
    off7 = jnp.arange(7) - 3
    xs = (hx[:, None] + off7) % width[:, None]
    ys = (hy[:, None] + off7) % height[:, None]
    write_cell = ((ys // CELL)[:, :, None] * MAP_CELLS + (xs // CELL)[:, None, :]).reshape(b, TILES)
    cx, cy = hx // CELL, hy // CELL
    off9 = jnp.arange(CROP) - CROP // 2
    rows = (cy[:, None] + off9) % hc[:, None]
    cols = (cx[:, None] + off9) % wc[:, None]
    crop_cell = (rows[:, :, None] * MAP_CELLS + cols[:, None, :]).reshape(b, CROP * CROP)
    # pooled bins: cell row cy falls in bin row cy * POOL // hc
    bin_of = lambda c, n: c * POOL // n[:, None]
    crop_bin = (bin_of(rows, hc)[:, :, None] * POOL + bin_of(cols, wc)[:, None, :]).reshape(b, CROP * CROP)
    g = jnp.arange(MAP_CELLS)[None, :]
    per_row = jax.vmap(lambda r, n: jax.ops.segment_sum((g[0] < n).astype(jnp.float32), r, POOL))
    rows_in = per_row(jnp.minimum(bin_of(g, hc), POOL - 1), hc)          # [B, 8] cell rows per bin row
    cols_in = per_row(jnp.minimum(bin_of(g, wc), POOL - 1), wc)
    bin_cells = (rows_in[:, :, None] * cols_in[:, None, :]).reshape(b, POOL * POOL)
    here = jax.nn.one_hot((cy * POOL // hc) * POOL + cx * POOL // wc, POOL * POOL)
    return Inputs(window, jnp.where(portal, sides % PORTAL_IDS, PORTAL_IDS).astype(jnp.int32), scalars, mask,
                  write_cell.astype(jnp.int32), crop_cell.astype(jnp.int32), crop_bin.astype(jnp.int32),
                  bin_cells, here)


def decode_state(rec: jax.Array, L: dict) -> StateInputs:
    b = rec.shape[0]
    n = len(STATE_CHANNELS) * BLOCKS * BLOCKS
    grid = jnp.log1p(field8(rec, L["state_grid"], n).astype(jnp.float32)).reshape(b, len(STATE_CHANNELS), BLOCKS, BLOCKS) / 3.0
    sc = field32(rec, L["state_scalars"], len(G)).astype(jnp.float32)
    c = lambda name: sc[:, G[name]]
    limit = jnp.maximum(c("UNIT_LIMIT"), 1.0)
    scalars = jnp.stack([
        c("ROUND") / 500.0, c("WIDTH") / 64.0, c("HEIGHT") / 64.0, limit / 64.0,
        c("MY_COUNT") / limit, jnp.log1p(c("MY_LONGEST")) / 5.0, jnp.log1p(c("MY_TOTAL")) / 6.0,
        c("THEIR_COUNT") / limit, jnp.log1p(c("THEIR_LONGEST")) / 5.0, jnp.log1p(c("THEIR_TOTAL")) / 6.0,
        jnp.log1p(c("ACTING_LENGTH")) / 5.0, c("ACTING_TEAM")], 1)
    return StateInputs(grid, scalars)


def action_command(action: int, length: int) -> tuple[int, list[int], int]:
    """(kind, steps, split) for the engine; the C++ runtime mirrors this."""
    if action < 4:
        return 0, [action], 0
    if action < 20:
        return 0, [(action - 4) // 4, (action - 4) % 4], 0
    child = int(round(SPLIT_FRACTIONS[action - 20] * length))
    return 1, [], max(2, min(child, length - 2))
