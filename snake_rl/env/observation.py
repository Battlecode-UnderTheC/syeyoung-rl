"""Turn a raw turn block plus private lifetime state into actor tensors.

This is the `prepare(raw_turn, private_facts, body_history)` interface of spec
section 13. It reads only the lifetime's own state and the turn block the
engine delivered; the result is what `encode_and_update` consumes.

`TurnBlock` from `dragons_env.protocol` is the raw contract of section 4.1: it
keeps every tile's absolute coordinates, the exact reported countdowns, every
segment with its id, facing and head flag, and every edge token including the
portal identifier. Nothing is collapsed before it reaches memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dragons_env.protocol import TurnBlock

from .memory import LifetimeState, NO_ROUND
from .schema import (DIR_INDEX, EF, F, MAX_ENTITIES, MAX_ROUNDS, LAST_ROUND,
                     MAX_BOARD, N_ENTITY_FEATURES, N_FACT_CHANNELS, N_STATUS, S,
                     TILES, VISION, VISION_RADIUS, MIN_SIZE)


class ObservationError(ValueError):
    pass


def validate_block(block: TurnBlock, width: int, height: int) -> None:
    """Check the supplied coordinates against the board and its wrapping."""
    if not (0 <= block.head_x < width and 0 <= block.head_y < height):
        raise ObservationError(f"head {block.head} outside {width}x{height}")
    xs, ys = np.asarray(block.xs), np.asarray(block.ys)
    if xs.shape != (TILES,) or ys.shape != (TILES,):
        raise ObservationError("window must have 49 tiles")
    if xs.min() < 0 or xs.max() >= width or ys.min() < 0 or ys.max() >= height:
        raise ObservationError("window coordinates outside the board")
    col = (np.arange(TILES) % VISION) - VISION_RADIUS
    row = (np.arange(TILES) // VISION) - VISION_RADIUS
    if not (np.all((block.head_x + col) % width == xs)
            and np.all((block.head_y + row) % height == ys)):
        raise ObservationError("window coordinates do not wrap around the head")
    for part in block.parts:
        if not (0 <= part.x < width and 0 <= part.y < height):
            raise ObservationError(f"segment {part} outside the board")
    if block.length < 1:
        raise ObservationError("length must be positive")


@dataclass
class PreparedObservation:
    """Tensors for one decision. Every array is numpy, batch-free."""
    local: np.ndarray                 # [42, 7, 7] factual crop, north up
    facts: np.ndarray                 # [42, H, W]
    board_valid: np.ndarray           # [H, W] bool
    visible_xy: np.ndarray            # [49, 2] int64 (x, y)
    visible_prev: np.ndarray          # [49, 42] the cells' features before this write
    visible_age_before: np.ndarray    # [49] age since previous sighting / 500 (0 if never)
    status: np.ndarray                # [N_STATUS]
    history: np.ndarray               # [32, N_HISTORY_FEATURES]
    history_mask: np.ndarray          # [32] bool
    entities: np.ndarray              # [E, N_ENTITY_FEATURES]
    entity_mask: np.ndarray           # [E] bool
    portal_pairs: np.ndarray          # [P, 2, 2] int64 (y, x) of each end's owning tile
    portal_axis: np.ndarray           # [P] int64, 0 north edge / 1 west edge
    portal_unpaired: np.ndarray       # [U, 2] int64 (y, x) owning tiles with an unknown partner
    inbox: np.ndarray                 # [M, 4] uint8, each delivered word as bytes
    # scalars the decoder and the protocol adapter need
    head: tuple[int, int]
    heading: int
    length: int
    unit_count: int
    unit_limit: int
    width: int
    height: int
    round: int
    dragon_id: int
    team: str
    split_legal: bool


def _status(block: TurnBlock, state: LifetimeState, max_steps: int) -> np.ndarray:
    out = np.zeros(N_STATUS, dtype=np.float32)
    w, h = state.width, state.height
    area = float(w * h)
    L = block.length
    out[S.LENGTH_LINEAR] = L / area
    out[S.LENGTH_LOG] = np.log1p(L) / np.log1p(area)
    out[S.ROUND] = block.round / LAST_ROUND
    out[S.ROUNDS_REMAINING] = max(LAST_ROUND - block.round, 0) / MAX_ROUNDS
    out[S.WIDTH] = w / MAX_BOARD
    out[S.HEIGHT] = h / MAX_BOARD
    out[S.UNIT_FRACTION] = block.unit_count / max(state.unit_limit, 1)
    out[S.HEADING_N + DIR_INDEX[block.facing]] = 1.0
    out[S.SPLIT_LEGAL] = float(split_legal(block, state.unit_limit))
    out[S.FIRST_OWN_TURN] = float(state.own_turns == 0)
    out[S.OWN_TURNS] = state.own_turns / MAX_ROUNDS
    out[S.KNOWN_BODY_FRACTION] = state.body.known_fraction()
    out[S.TAIL_KNOWN] = float(state.body.tail_known)
    ax, ay = 2 * np.pi * block.head_x / w, 2 * np.pi * block.head_y / h
    out[S.HEAD_SIN_X], out[S.HEAD_COS_X] = np.sin(ax), np.cos(ax)
    out[S.HEAD_SIN_Y], out[S.HEAD_COS_Y] = np.sin(ay), np.cos(ay)
    prev = state.history.last
    if prev is None:
        out[S.PREV_NONE] = 1.0
    elif prev.kind == "split":
        out[S.PREV_SPLIT] = 1.0
        out[S.PREV_SPLIT_FRACTION] = prev.split_n / max(prev.old_length, 1)
        out[S.PREV_RETAINED_FRACTION] = (prev.old_length - prev.split_n) / max(prev.old_length, 1)
        out[S.PREV_INTERMEDIATES_KNOWN] = 1.0
    else:
        out[S.PREV_MOVE] = 1.0
        out[S.PREV_STEPS] = len(prev.dirs) / max_steps
        for ch in prev.dirs:
            out[S.PREV_DIR_N + DIR_INDEX[ch]] += 1.0 / max_steps
        if prev.dirs:
            out[S.PREV_LAST_N + DIR_INDEX[prev.dirs[-1]]] = 1.0
        out[S.PREV_INTERMEDIATES_KNOWN] = float(prev.intermediates_known)
    out[S.INBOX_COUNT] = min(len(block.messages), 8) / 8.0
    return out


def split_legal(block: TurnBlock, unit_limit: int) -> bool:
    """SPLIT n needs n >= 2 and a remainder >= 2, and room under the unit limit."""
    return block.length - MIN_SIZE >= MIN_SIZE and block.unit_count < unit_limit


def _entities(block: TurnBlock, dragon_id: int, team: str
              ) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[int, list] = {}
    for part in block.parts:
        if part.dragon_id == dragon_id:
            continue
        groups.setdefault(part.dragon_id, []).append(part)
    out = np.zeros((MAX_ENTITIES, N_ENTITY_FEATURES), dtype=np.float32)
    mask = np.zeros(MAX_ENTITIES, dtype=bool)
    for slot, (other_id, parts) in enumerate(sorted(groups.items())):
        if slot >= MAX_ENTITIES:
            break            # cannot happen: 49 tiles bound the number of ids
        row = out[slot]
        mask[slot] = True
        n = len(parts)
        row[EF.COUNT] = n / TILES
        row[EF.COUNT_LOG] = np.log1p(n) / np.log(50.0)
        rel = []
        for p in parts:
            i = block.window_index(p.x, p.y)
            if i is None:
                continue
            rel.append(((i % VISION) - VISION_RADIUS, (i // VISION) - VISION_RADIUS))
            if p.is_head:
                row[EF.HEAD_PRESENT] = 1.0
                row[EF.HEAD_FACING_N + DIR_INDEX[p.facing]] = 1.0
                row[EF.HEAD_DX] = rel[-1][0] / VISION_RADIUS
                row[EF.HEAD_DY] = rel[-1][1] / VISION_RADIUS
        if rel:
            arr = np.asarray(rel, dtype=np.float32)
            row[EF.MEAN_DX], row[EF.MEAN_DY] = arr.mean(axis=0) / VISION_RADIUS
            row[EF.MIN_DIST] = np.abs(arr).max(axis=1).min() / VISION_RADIUS
        ally = parts[0].team == team
        row[EF.ALLY], row[EF.ENEMY] = float(ally), float(not ally)
        row[EF.ID_EARLIER], row[EF.ID_LATER] = float(other_id < dragon_id), float(other_id > dragon_id)
    return out, mask


def _portal_tables(state: LifetimeState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = state.facts.confirmed_portal_pairs()
    if pairs:
        ends = np.array([[[a[2], a[1]], [b[2], b[1]]] for a, b in pairs], dtype=np.int64)
        axis = np.array([a[0] for a, _ in pairs], dtype=np.int64)
    else:
        ends = np.zeros((0, 2, 2), dtype=np.int64)
        axis = np.zeros((0,), dtype=np.int64)
    unpaired = [e for ends_ in state.facts.portals.values() if len(ends_) != 2 for e in ends_]
    if unpaired:
        loose = np.array([[y, x] for _, x, y in sorted(unpaired)], dtype=np.int64)
    else:
        loose = np.zeros((0, 2), dtype=np.int64)
    return ends, axis, loose


def prepare(state: LifetimeState, block: TurnBlock, max_steps: int = 8
            ) -> PreparedObservation:
    """Advance the lifetime's factual state by this turn and build its tensors.

    Idempotent: a second call with the same round rebuilds the same tensors
    without shifting memory again.
    """
    validate_block(block, state.width, state.height)
    dragon_id, team = state.key.dragon_id, state.key.team
    xs = np.asarray(block.xs, dtype=np.int64)
    ys = np.asarray(block.ys, dtype=np.int64)

    redelivery = block.round == state.last_processed_round
    if redelivery and state._prewrite is not None:
        visible_prev, age_before = state._prewrite
    else:
        pre = state.facts.materialize(block.round, dragon_id, state.body)
        visible_prev = np.ascontiguousarray(pre[:, ys, xs].T)          # [49, 42]
        last = state.facts.last_seen_round[ys, xs].astype(np.int32)
        age_before = np.where(last >= 0, np.clip(block.round - last, 0, MAX_ROUNDS)
                              / MAX_ROUNDS, 0.0).astype(np.float32)
        state._prewrite = (visible_prev, age_before)
    state.begin_turn(block)

    facts = state.facts.materialize(block.round, dragon_id, state.body)
    local = np.ascontiguousarray(facts[:, ys, xs].reshape(N_FACT_CHANNELS, VISION, VISION))
    history, history_mask = state.history.features(block.round, state.width,
                                                   state.height, max_steps)
    entities, entity_mask = _entities(block, dragon_id, team)
    pairs, axis, loose = _portal_tables(state)
    words = np.zeros((len(block.messages), 4), dtype=np.uint8)
    for i, value in enumerate(block.messages):
        words[i] = np.frombuffer((int(value) & 0xFFFFFFFF).to_bytes(4, "little"), dtype=np.uint8)

    return PreparedObservation(
        local=local, facts=facts,
        board_valid=np.ones((state.height, state.width), dtype=bool),
        visible_xy=np.stack([xs, ys], axis=1),
        visible_prev=visible_prev, visible_age_before=age_before,
        status=_status(block, state, max_steps),
        history=history, history_mask=history_mask,
        entities=entities, entity_mask=entity_mask,
        portal_pairs=pairs, portal_axis=axis, portal_unpaired=loose,
        inbox=words,
        head=(block.head_x, block.head_y), heading=DIR_INDEX[block.facing],
        length=block.length, unit_count=block.unit_count, unit_limit=state.unit_limit,
        width=state.width, height=state.height, round=block.round,
        dragon_id=dragon_id, team=team,
        split_legal=split_legal(block, state.unit_limit),
    )


def portal_id_table(spec) -> list[int]:
    """[2*W*H] portal id per canonical edge (north edges, then west), -1 elsewhere.
    The native sim keeps only partner links, so the ids come from the map spec."""
    w, h = spec.width, spec.height
    ids = [-1] * (2 * w * h)
    stride = 2 * (w + 1)
    for idx, pid in spec.portals:
        y, off = divmod(idx, stride)
        vertical = off >= w + 1
        if vertical:
            off -= w + 1
        x, y = off % w, y % h
        ids[(w * h if vertical else 0) + y * w + x] = pid
    return ids
