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
CG = _native.CRITIC_GLOBAL
WINDOW = 7
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
    tile, dest, status = raw["tile"], raw["dest"], raw["status"]
    n_pad = tile.shape[0]
    W, H = status[S["WIDTH"]], status[S["HEIGHT"]]
    A = W * H
    L = status[S["LENGTH"]]
    now = status[S["ROUND"]]
    hx, hy = status[S["HEAD_X"]], status[S["HEAD_Y"]]
    head = status[S["HEAD_TILE"]]
    t = jnp.arange(n_pad, dtype=jnp.int32)
    valid = tile[:, T["VALID"]] == 1
    x, y = t % W, t // W

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
    known_dest = dest >= 0
    sm = f32(seen)[:, None]

    X = jnp.concatenate([
        # 0-5
        sm, f32(tile[:, T["VISIBLE"]])[:, None], age(seen_round, now, seen)[:, None],
        f32(tile[:, T["PEARL"]])[:, None] * sm, f32(bed)[:, None] * sm, timer(countdown)[:, None] * sm,
        # 6-8
        f32(visits > 0)[:, None], cnt(visits)[:, None], age(visit_round, now, visit_round >= 0)[:, None],
        # 9-12
        f32(known_body & (index == 0))[:, None], f32(known_body)[:, None],
        jnp.where(known_body, f32(index) / f32(jnp.maximum(L - 1, 1)), 0.0)[:, None],
        f32(known_body & (index == L - 1))[:, None],
        # 13-18
        f32(prev_valid)[:, None], age(prev_round, now, prev_valid)[:, None],
        f32(tile[:, T["PREV_PEARL"]])[:, None] * f32(prev_valid)[:, None],
        f32(prev_cd != -1)[:, None] * f32(prev_valid)[:, None],
        timer(prev_cd)[:, None] * f32(prev_valid)[:, None],
        jnp.where(prev_valid, clip01((f32(seen_round) - f32(prev_round)) / 500.0), 0.0)[:, None],
        # 19-20
        f32(order == 1)[:, None] * sm, f32(order == 2)[:, None] * sm,
        # 21-27: empty, self head, self body, ally head, ally body, enemy head, enemy body
        one_hot(occ - 1, 7, seen & (occ > 0)),
        # 28-31
        one_hot(facing, 4, seen & (facing >= 0)),
        # 32-47: per direction unknown, empty, kelp, portal
        one_hot(etype, 4).reshape(n_pad, 16),
        # 48-51
        f32(known_dest),
        # 52-55
        rel(x, y, hx, hy, W, H),
    ], axis=-1) * f32(valid)[:, None]

    dest_seen = jnp.where(known_dest, seen[jnp.clip(dest, 0, n_pad - 1)], False)
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

    # the local crop: north-up 7x7 around the head, toroidal
    off = jnp.arange(WINDOW, dtype=jnp.int32) - WINDOW // 2
    crop = ((hy + off)[:, None] % H) * W + (hx + off)[None, :] % W       # [7, 7]

    k_pad = raw["inbox"].shape[0]
    return {
        "X": X, "J": jnp.where(valid[:, None], dest, -1), "D": D, "tile_mask": valid,
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
    """One event from perspective `team` (0 A, 1 B): F[N,28], J, R[D,207], Q[16]."""
    grid, btile, bdest = raw["grid"], raw["board_tile"], raw["board_dest"]
    q, rows, summary = raw["global"], raw["dragons"], raw["summary"]
    n_pad = grid.shape[0]
    W, H = q[CG["WIDTH"]], q[CG["HEIGHT"]]
    A = W * H
    valid = btile[:, 1] == 1
    t = jnp.arange(n_pad, dtype=jnp.int32)
    pearl = grid & 1
    occ = (grid >> 1) & 7                     # 0 empty, 1 A head, 2 A body, 3 B head, 4 B body
    bed = (grid >> 4) & 1
    countdown = (grid >> 8) - _native.FIELDS["COUNTDOWN_BIAS"]
    occ_team = (occ - 1) // 2
    occ_body = (occ - 1) % 2
    rel_occ = jnp.where(occ == 0, 0, 1 + 2 * (occ_team != team) + occ_body)   # empty, ally h, ally b, enemy h, enemy b
    etype = edge_types(btile[:, 0])           # 1 empty 2 kelp 3 portal
    F = jnp.concatenate([
        f32(pearl)[:, None], f32(bed)[:, None], jnp.where(bed == 1, timer(countdown), 0.0)[:, None],
        one_hot(rel_occ, 5),
        one_hot(etype - 1, 3).reshape(n_pad, 12),
        f32(bdest >= 0),
        angles(t % W, t // W, W, H),
    ], axis=-1) * f32(valid)[:, None]

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
    return {"F": F, "J": jnp.where(valid[:, None], bdest, -1), "tile_mask": valid, "R": R, "dragon_mask": d_valid,
            "Q": Q}
