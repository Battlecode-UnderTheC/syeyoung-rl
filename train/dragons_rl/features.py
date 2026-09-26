"""Spec sections 1, 4 and 8: the float inputs of the actor and the critic,
computed from the exact integer records of dragons_rl._native (the codec).

Every function here takes ONE decision or ONE event (no batch axis); callers
vmap. Batch padding is marked by explicit masks and never becomes an unknown
in-board tile. All arithmetic is float32.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from . import _native

F32 = jnp.float32
T = _native.TILE
E = _native.ENTITY
S = _native.STATUS
CD = _native.CRITIC_DRAGON
CC = {k: i for i, k in enumerate(("TILES", "PEARLS", "BEDS", "TIMER_SUM", "HEAD_A", "BODY_A", "HEAD_B", "BODY_B",
                                  "KELP", "PORTAL"))}      # the critic's cells (native CriticCellField)
CRITIC_CELL_FEATURES = 11
CG = _native.CRITIC_GLOBAL
WINDOW = 7
WIN = _native.FIELDS["WIN"]                  # simple1's window side (17): the head is its centre cell
CENTRE = (WIN // 2) * WIN + WIN // 2
G = {k: i for i, k in enumerate(("TILES", "SEEN", "VISIBLE", "PEARL", "BED", "VISITED", "SELF_BODY", "ALLY", "ENEMY",
                                 "EMPTY", "KELP", "PORTAL", "SEEN_ROUND_SUM", "VISIT_ROUND_SUM"))}
GRID_FEATURES = 15


def window_crop(side: int):
    """[side, side] window-cell indices of the square centred on the head"""
    off = jnp.arange(side, dtype=jnp.int32) - side // 2 + WIN // 2
    return off[:, None] * WIN + off[None, :]


def cell_rel(cell, W, H):
    """the relative-position features of a window cell (as rel() of its tile)"""
    return angles(cell % WIN - WIN // 2, cell // WIN - WIN // 2, W, H)
MAX_TOKENS = _native.FIELDS["MAX_MOVE_TOKENS"]
LOG501 = math.log(501.0)


# ---- section 1 helpers ------------------------------------------------------------

def f32(x):
    return jnp.asarray(x, F32)


def clip01(x):
    return jnp.clip(x, 0.0, 1.0)


def age(t, now, valid):
    """clip01((now - t) / 500), zero when the record is invalid."""
    return jnp.where(valid, clip01((f32(now) - f32(t)) / 500.0), 0.0).astype(F32)


def cnt(x):
    return jnp.minimum(jnp.log1p(jnp.maximum(f32(x), 0.0)) / LOG501, 1.0)


def loglen(x, area):
    return jnp.log1p(jnp.maximum(f32(x), 0.0)) / jnp.log1p(f32(area))


def timer(x):
    x = f32(x)
    return jnp.sign(x) * jnp.minimum(jnp.log1p(jnp.abs(x)) / LOG501, 1.0)


def angles(x, y, w, h):
    tx = 2.0 * jnp.pi * f32(x) / f32(w)
    ty = 2.0 * jnp.pi * f32(y) / f32(h)
    return jnp.stack([jnp.sin(tx), jnp.cos(tx), jnp.sin(ty), jnp.cos(ty)], -1).astype(F32)


def rel(x, y, hx, hy, w, h):
    return angles(f32(x) - f32(hx), f32(y) - f32(hy), w, h)


def one_hot(i, n, valid=True):
    """one-hot of i (all zero where i is out of range or not valid)"""
    return (jax.nn.one_hot(i, n, dtype=F32) * jnp.asarray(valid, F32)[..., None]).astype(F32)


def edge_types(packed):
    """[..., 4] edge type per direction from 2-bit packing"""
    d = jnp.arange(4, dtype=jnp.int32)
    return (packed[..., None] >> (2 * d)) & 3


# ---- section 4: the actor ------------------------------------------------------------

def actor_inputs(raw: dict) -> dict:
    """One decision's codec rows -> the section 4 tensors (plus what the
    reasoning and decoding need: J, masks, the crop, the head)."""
    # simple1: the 17x17 window of tile records around the head (J as window cells,
    # -1 unknown, -2 known but outside) and the 8x8 head-relative grid of the whole map
    tile, dest, status = raw["win"], raw["wdest"], raw["status"]
    n_pad = tile.shape[0]
    W, H = status[S["WIDTH"]], status[S["HEIGHT"]]
    A = W * H
    L = status[S["LENGTH"]]
    now = status[S["ROUND"]]
    hx, hy = status[S["HEAD_X"]], status[S["HEAD_Y"]]
    head = jnp.int32(CENTRE)
    t = jnp.arange(n_pad, dtype=jnp.int32)
    valid = tile[:, T["VALID"]] == 1
    dx, dy = t % WIN - WIN // 2, t // WIN - WIN // 2                    # offsets from the head

    seen_round = tile[:, T["SEEN_ROUND"]]
    seen = seen_round >= 0
    countdown = tile[:, T["COUNTDOWN"]]
    bed = countdown != -1
    visits = tile[:, T["VISITS"]]
    visit_round = tile[:, T["VISIT_ROUND"]]
    index = tile[:, T["BODY_INDEX"]]
    known_body = index >= 0
    prev_round = tile[:, T["PREV_ROUND"]]
    prev_valid = (prev_round >= 0) & seen
    prev_cd = tile[:, T["PREV_COUNTDOWN"]]
    occ = tile[:, T["OCC"]]
    order = tile[:, T["OCC_ORDER"]]
    facing = tile[:, T["OCC_FACING"]]
    etype = edge_types(tile[:, T["EDGES"]])                       # [N, 4]
    flags = tile[:, T["FLAGS"]]
    d4 = jnp.arange(4, dtype=jnp.int32)
    seam = (flags[:, None] >> d4) & 1
    paired = (flags[:, None] >> (4 + d4)) & 1
    known_dest = dest != -1                                          # -2: known, outside the window
    sm = f32(seen)

    cols = [
        # 0-5
        sm, f32(tile[:, T["VISIBLE"]]), age(seen_round, now, seen),
        f32(tile[:, T["PEARL"]]) * sm, f32(bed) * sm, timer(countdown) * sm,
        # 6-8
        f32(visits > 0), cnt(visits), age(visit_round, now, visit_round >= 0),
        # 9-12
        f32(known_body & (index == 0)), f32(known_body),
        jnp.where(known_body, f32(index) / f32(jnp.maximum(L - 1, 1)), 0.0),
        f32(known_body & (index == L - 1)),
        # 13-18
        f32(prev_valid), age(prev_round, now, prev_valid),
        f32(tile[:, T["PREV_PEARL"]]) * f32(prev_valid),
        f32(prev_cd != -1) * f32(prev_valid),
        timer(prev_cd) * f32(prev_valid),
        jnp.where(prev_valid, clip01((f32(seen_round) - f32(prev_round)) / 500.0), 0.0),
        # 19-20
        f32(order == 1) * sm, f32(order == 2) * sm,
        # 21-27: empty, self head, self body, ally head, ally body, enemy head, enemy body
        *[f32(seen & (occ > 0) & (occ - 1 == k)) for k in range(7)],
        # 28-31
        *[f32(seen & (facing >= 0) & (facing == k)) for k in range(4)],
        # 32-47: per direction unknown, empty, kelp, portal
        *[f32(etype[:, d] == k) for d in range(4) for k in range(4)],
        # 48-51
        *[f32(known_dest[:, d]) for d in range(4)],
        # 52-55
        *[angles(dx, dy, W, H)[:, i] for i in range(4)],
    ]
    vf = f32(valid)
    cols = [c * vf for c in cols]                                   # 56 columns of [N]
    X = jnp.stack(cols, axis=-1)

    dest_seen = jnp.where(dest >= 0, seen[jnp.clip(dest, 0, n_pad - 1)], False)
    D = jnp.concatenate([
        jnp.broadcast_to(jnp.eye(4, dtype=F32), (n_pad, 4, 4)),
        one_hot(etype, 4),
        f32(known_dest)[..., None],
        f32((seam == 1) & (etype == 1))[..., None],
        f32((paired == 1) & (etype == 3))[..., None],
        f32(dest_seen)[..., None],
    ], axis=-1) * f32(valid)[:, None, None]

    # 4.3 entities, sorted by id by the codec
    ent, seg = raw["entity"], raw["segment"]
    e_valid = ent[:, E["VALID"]] == 1
    seg_valid = seg[:, 3] == 1
    seg_rel = rel(seg[:, 1], seg[:, 2], hx, hy, W, H) * f32(seg_valid)[:, None]
    n_ent = ent.shape[0]
    seg_sum = jax.ops.segment_sum(seg_rel, jnp.where(seg_valid, seg[:, 0], n_ent), num_segments=n_ent + 1)[:n_ent]
    count = ent[:, E["COUNT"]]
    head_vis = ent[:, E["HEAD_VISIBLE"]] == 1
    U = jnp.concatenate([
        one_hot(ent[:, E["RELATION"]], 3),
        one_hot(ent[:, E["ORDER"]], 3),
        (f32(count) / 49.0)[:, None],
        f32(head_vis)[:, None],
        rel(ent[:, E["HEAD_X"]], ent[:, E["HEAD_Y"]], hx, hy, W, H) * f32(head_vis)[:, None],
        one_hot(ent[:, E["HEAD_FACING"]], 4, head_vis),
        seg_sum / f32(jnp.maximum(count, 1))[:, None],
    ], axis=-1) * f32(e_valid)[:, None]

    # 4.4 status
    prev_ok = status[S["PREV_VALID"]] == 1
    prev_move = prev_ok & (status[S["PREV_KIND"]] == 0)
    prev_split = prev_ok & (status[S["PREV_KIND"]] == 1)
    prev_len = status[S["PREV_LENGTH"]]
    unit_count, unit_limit = status[S["UNIT_COUNT"]], status[S["UNIT_LIMIT"]]
    turns = status[S["TURN_COUNT"]]
    pv = f32(prev_ok)
    Sv = jnp.concatenate([
        jnp.stack([f32(L) / f32(A), loglen(L, A), f32(now) / 499.0, (500.0 - f32(now)) / 500.0,
                   f32(W) / 64.0, f32(H) / 64.0,
                   f32(unit_count) / f32(unit_limit), f32(unit_limit) / 64.0,
                   f32(status[S["KNOWN_BODY"]]) / f32(L), f32(status[S["TAIL_KNOWN"]]),
                   f32(turns == 0), f32(turns) / 500.0]),
        one_hot(status[S["FACING"]], 4),
        angles(hx, hy, W, H),
        jnp.stack([pv, f32(prev_move), f32(prev_split),
                   pv * f32(status[S["PREV_STEPS"]]) / 8.0,
                   pv * f32(status[S["PREV_SPLIT"]]) / f32(jnp.maximum(prev_len, 1)),
                   pv * (f32(L) - f32(prev_len)) / f32(A),
                   f32(status[S["SEEN_TILES"]]) / f32(A), f32(status[S["KNOWN_EDGES"]]) / f32(2 * A),
                   cnt(status[S["INBOX_COUNT"]]), f32(status[S["ECHO_PRESENT"]]),
                   f32(status[S["HEAD_BED"]]), f32(status[S["HEAD_PEARL"]])]),
    ])

    # 4.5 previous command and sonar
    prev_dirs = raw["prev_dirs"]
    C_prev = physical_vector(jnp.where(prev_split, 1, 0), prev_dirs, status[S["PREV_SPLIT"]], prev_len, A) * pv
    mask = status[S["PREV_SONAR_MASK"]]
    prev_flags = f32(((mask >> d4) & 1) == 1) * pv

    # the local crop: north-up 7x7 around the head (the window's centre)
    crop = window_crop(WINDOW)                                          # [7, 7]

    # the whole map, 8x8 head-relative cells: shares of each cell's tiles, and mean ages
    g = raw["grid"]
    tiles = f32(g[:, G["TILES"]])
    per = lambda k: f32(g[:, G[k]]) / jnp.maximum(tiles, 1.0)
    n_seen, n_vis = f32(g[:, G["SEEN"]]), f32(g[:, G["VISITED"]])
    grid = jnp.stack([
        f32(tiles > 0), tiles / 64.0, per("SEEN"), per("VISIBLE"), cnt(g[:, G["PEARL"]]), cnt(g[:, G["BED"]]),
        per("VISITED"), per("SELF_BODY"), cnt(g[:, G["ALLY"]]), cnt(g[:, G["ENEMY"]]), per("EMPTY"),
        f32(g[:, G["KELP"]]) / jnp.maximum(2.0 * tiles, 1.0), cnt(g[:, G["PORTAL"]]),        # 2 canonical edges a tile
        jnp.where(n_seen > 0, clip01((f32(now) * n_seen - f32(g[:, G["SEEN_ROUND_SUM"]])) / jnp.maximum(n_seen, 1.0) / 500.0), 0.0),
        jnp.where(n_vis > 0, clip01((f32(now) * n_vis - f32(g[:, G["VISIT_ROUND_SUM"]])) / jnp.maximum(n_vis, 1.0) / 500.0), 0.0),
    ], -1) * f32(tiles > 0)[:, None]                                    # [64, 15]

    k_pad = raw["inbox"].shape[0]
    return {
        "X": X, "cols": cols, "J": jnp.where(valid[:, None], dest, -1), "D": D, "tile_mask": valid, "grid": grid,
        "U": U, "entity_mask": e_valid, "S": Sv,
        "inbox": raw["inbox"], "inbox_mask": jnp.arange(k_pad) < status[S["INBOX_COUNT"]],
        "inbox_count": status[S["INBOX_COUNT"]],
        "echo": raw["echo"], "echo_len": raw["echo_len"], "echo_present": status[S["ECHO_PRESENT"]] == 1,
        "C_prev": C_prev, "prev_flags": prev_flags, "prev_words": raw["prev_words"], "prev_valid": pv,
        "crop": crop, "head": head, "facing": status[S["FACING"]], "hx": hx, "hy": hy, "W": W, "H": H, "A": A,
        "L": L, "unit_count": unit_count, "unit_limit": unit_limit,
    }


def physical_vector(kind, dirs, split, length_before, area):
    """C[36]: kind one-hot, eight direction one-hots (unused zero), split_n/A
    (zero for MOVE), length_before/A. `dirs` holds -1 past the move."""
    is_split = kind == 1
    return jnp.concatenate([
        one_hot(kind, 2),
        one_hot(dirs, 4, (dirs >= 0) & ~is_split).reshape(-1),
        jnp.stack([jnp.where(is_split, f32(split) / f32(area), 0.0), f32(length_before) / f32(area)]),
    ]).astype(F32)


# ---- section 8: the critic ----------------------------------------------------------

def critic_inputs(raw: dict, team) -> dict:
    """One event from perspective `team` (0 A, 1 B): C[64, 11] (the board's 8x8 cells), R[D,207], Q[16]."""
    cells, q, rows, summary = raw["cells"], raw["global"], raw["dragons"], raw["summary"]
    W, H = q[CG["WIDTH"]], q[CG["HEIGHT"]]
    A = W * H
    c = lambda k: f32(cells[:, CC[k]])
    tiles = c("TILES")
    share = lambda x: x / jnp.maximum(tiles, 1.0)
    mine = lambda a_key, b_key: jnp.where(team == 0, c(a_key), c(b_key))
    theirs = lambda a_key, b_key: jnp.where(team == 0, c(b_key), c(a_key))
    beds = c("BEDS")
    C = jnp.stack([
        f32(tiles > 0), tiles / 64.0, cnt(c("PEARLS")), cnt(beds),
        jnp.where(beds > 0, timer(c("TIMER_SUM") / jnp.maximum(beds, 1.0)), 0.0),
        cnt(mine("HEAD_A", "HEAD_B")), share(mine("BODY_A", "BODY_B")),
        cnt(theirs("HEAD_A", "HEAD_B")), share(theirs("BODY_A", "BODY_B")),
        c("KELP") / jnp.maximum(2.0 * tiles, 1.0), cnt(c("PORTAL")),                           # 2 canonical edges a tile
    ], -1) * f32(tiles > 0)[:, None]

    d_valid = rows[:, CD["VALID"]] == 1
    alive = q[CG["ALIVE"]]
    length = rows[:, CD["LENGTH"]]
    ally = rows[:, CD["TEAM"]] == team
    R = jnp.concatenate([
        f32(ally)[:, None], f32(~ally)[:, None],
        (f32(length) / f32(A))[:, None], loglen(length, A)[:, None],
        angles(rows[:, CD["HEAD_X"]], rows[:, CD["HEAD_Y"]], W, H),
        one_hot(rows[:, CD["FACING"]], 4),
        (f32(rows[:, CD["RANK"]]) / f32(jnp.maximum(alive - 1, 1)))[:, None],
        f32(rows[:, CD["ACTED"]])[:, None],
        summary,
    ], axis=-1) * f32(d_valid)[:, None]

    pick = lambda a_key, b_key: jnp.where(team == 0, q[CG[a_key]], q[CG[b_key]])
    other = lambda a_key, b_key: jnp.where(team == 0, q[CG[b_key]], q[CG[a_key]])
    limit = q[CG["UNIT_LIMIT"]]
    next_ally = q[CG["NEXT_TEAM"]] == team
    Q = jnp.stack([
        f32(q[CG["ROUND"]]) / 499.0, (500.0 - f32(q[CG["ROUND"]])) / 500.0, f32(W) / 64.0, f32(H) / 64.0,
        f32(pick("LONGEST_A", "LONGEST_B")) / f32(A), f32(other("LONGEST_A", "LONGEST_B")) / f32(A),
        f32(pick("TOTAL_A", "TOTAL_B")) / f32(A), f32(other("TOTAL_A", "TOTAL_B")) / f32(A),
        f32(pick("COUNT_A", "COUNT_B")) / f32(limit), f32(other("COUNT_A", "COUNT_B")) / f32(limit),
        f32(next_ally), f32(~next_ally),
        f32(q[CG["NEXT_RANK"]]) / f32(jnp.maximum(alive - 1, 1)),
        f32(limit) / 64.0,
        f32(q[CG["DUE"]]) / f32(jnp.maximum(alive, 1)),
        f32(q[CG["NEXT_LENGTH"]]) / f32(A),
    ])
    return {"C": C, "R": R, "dragon_mask": d_valid, "Q": Q}
