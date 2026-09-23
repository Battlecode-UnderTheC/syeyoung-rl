"""Private factual memory for one dragon lifetime (spec sections 3, 4.2, 4.4).

Everything here is built from the dragon's own turn blocks and its own issued
commands. There is no argument through which the simulator's state could
arrive, and nothing is ever copied from a parent into a child.

Three stores live together in a `LifetimeState`:

* `FactualMemory` -- per-tile records of what was last observed, one preceding
  observation, canonical north/west edges and the portal endpoints seen so far.
  Ages are computed on read; updates overwrite only what was actually seen.
* `BodyHistory` -- the head-first order of the dragon's own segments, as far as
  it can be reconstructed from its own moves. Unknown entries stay unknown.
* `CommandRing` -- the last 32 completed commands with what they did.

Repeated delivery of the same turn is idempotent, and the whole state can be
snapshotted and restored losslessly.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from dragons_env.protocol import TurnBlock, KELP, EMPTY

from .schema import (DIR_INDEX, DIR_OFFSET, DIRS, EdgeType, F, HF, HISTORY_TURNS,
                     LAST_ROUND, MAX_ROUNDS, N_FACT_CHANNELS, N_HISTORY_FEATURES,
                     Occupant, OPPOSITE, TILES, VISION, VISION_RADIUS)

NO_ROUND = -1
NO_ID = -1
UNKNOWN = None          # a body cell whose coordinate is not privately known

_LOG_VISITS = float(np.log(501.0))


def _window_offsets() -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(TILES)
    return (i % VISION) - VISION_RADIUS, (i // VISION) - VISION_RADIUS


_WIN_DX, _WIN_DY = _window_offsets()


# -- factual map -------------------------------------------------------------

class FactualMemory:
    """Structured per-tile storage for one board, allocated at its real size."""

    def __init__(self, width: int, height: int):
        if not (1 <= width <= 64 and 1 <= height <= 64):
            raise ValueError(f"board {width}x{height} outside the 1..64 range")
        self.width, self.height = width, height
        shape = (height, width)
        self.seen = np.zeros(shape, dtype=bool)
        self.last_seen_round = np.full(shape, NO_ROUND, dtype=np.int16)
        self.pearl = np.zeros(shape, dtype=bool)
        self.countdown = np.zeros(shape, dtype=np.int32)
        self.prev_pearl = np.zeros(shape, dtype=bool)
        self.prev_countdown = np.zeros(shape, dtype=np.int32)
        self.prev_seen_round = np.full(shape, NO_ROUND, dtype=np.int16)
        self.occupant_kind = np.zeros(shape, dtype=np.uint8)
        self.occupant_id = np.full(shape, NO_ID, dtype=np.int32)
        self.occupant_facing = np.zeros(shape, dtype=np.uint8)
        self.visit_count = np.zeros(shape, dtype=np.uint16)
        self.last_visit_round = np.full(shape, NO_ROUND, dtype=np.int16)
        # canonical edges: north edge of (x, y) and west edge of (x, y)
        self.edge_known = np.zeros((2,) + shape, dtype=bool)
        self.edge_type = np.zeros((2,) + shape, dtype=np.uint8)
        self.edge_portal = np.full((2,) + shape, NO_ID, dtype=np.int32)
        # portal id -> set of canonical endpoints (axis, x, y)
        self.portals: dict[int, set[tuple[int, int, int]]] = {}
        self._visible_round = NO_ROUND
        self._visible = np.zeros(shape, dtype=bool)

    # -- edges ------------------------------------------------------------

    def canonical(self, x: int, y: int, d: int) -> tuple[int, int, int]:
        """(axis, x, y) of the stored edge you cross stepping `d` out of (x, y).
        Axis 0 is a north edge, axis 1 a west edge."""
        if d == 0:
            return 0, x, y
        if d == 2:
            return 0, x, (y + 1) % self.height
        if d == 3:
            return 1, x, y
        return 1, (x + 1) % self.width, y

    def edge(self, x: int, y: int, d: int) -> tuple[bool, int, int]:
        a, ex, ey = self.canonical(x, y, d)
        return (bool(self.edge_known[a, ey, ex]), int(self.edge_type[a, ey, ex]),
                int(self.edge_portal[a, ey, ex]))

    def _set_edge(self, x: int, y: int, d: int, kind: int, portal_id: int) -> None:
        a, ex, ey = self.canonical(x, y, d)
        self.edge_known[a, ey, ex] = True
        self.edge_type[a, ey, ex] = kind
        self.edge_portal[a, ey, ex] = portal_id
        if kind == EdgeType.PORTAL:
            self.portals.setdefault(portal_id, set()).add((a, ex, ey))

    def portal_partner(self, x: int, y: int, d: int) -> tuple[int, int, int] | None:
        """Canonical (axis, x, y) of the other end of the portal on that side,
        if both ends have been observed. Identity is by portal id, never by
        numeric order."""
        a, ex, ey = self.canonical(x, y, d)
        if not self.edge_known[a, ey, ex] or self.edge_type[a, ey, ex] != EdgeType.PORTAL:
            return None
        ends = self.portals.get(int(self.edge_portal[a, ey, ex]), set())
        others = [e for e in ends if e != (a, ex, ey)]
        if len(others) != 1:
            return None
        return others[0]

    def step(self, x: int, y: int, d: int) -> tuple[str, tuple[int, int] | None]:
        """Where a step lands, from private knowledge only.

        Returns ("empty", (x, y)), ("portal", (x, y)), ("portal", None) for a
        portal whose partner is unknown, ("kelp", None) or ("unknown", None).
        """
        known, kind, _ = self.edge(x, y, d)
        if not known:
            return "unknown", None
        if kind == EdgeType.KELP:
            return "kelp", None
        if kind == EdgeType.EMPTY:
            dx, dy = DIR_OFFSET[d]
            return "empty", ((x + dx) % self.width, (y + dy) % self.height)
        partner = self.portal_partner(x, y, d)
        if partner is None:
            return "portal", None
        axis, px, py = partner
        # The partner is the north (axis 0) or west (axis 1) edge of (px, py):
        # entering from the positive side lands on that tile, from the
        # negative side on the tile before it -- the engine's TileAfterStep.
        if axis == 1:
            lx, ly = (px if d == 1 else (px - 1) % self.width), py
        else:
            lx, ly = px, (py if d == 2 else (py - 1) % self.height)
        return "portal", (lx, ly)

    def confirmed_portal_pairs(self) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        out = []
        for pid, ends in sorted(self.portals.items()):
            if len(ends) == 2:
                a, b = sorted(ends)
                out.append((a, b))
        return out

    # -- observation ------------------------------------------------------

    def observe(self, block: TurnBlock, dragon_id: int, team: str) -> None:
        """Write one turn's window. Calling it twice for the same round is a
        no-op the second time: the preceding observation is not shifted."""
        rnd = block.round
        xs, ys = np.asarray(block.xs, dtype=np.int64), np.asarray(block.ys, dtype=np.int64)
        fresh = self.last_seen_round[ys, xs] != rnd
        shift = self.seen[ys, xs] & fresh
        sy, sx = ys[shift], xs[shift]
        self.prev_pearl[sy, sx] = self.pearl[sy, sx]
        self.prev_countdown[sy, sx] = self.countdown[sy, sx]
        self.prev_seen_round[sy, sx] = self.last_seen_round[sy, sx]

        self.seen[ys, xs] = True
        self.last_seen_round[ys, xs] = rnd
        self.pearl[ys, xs] = np.asarray(block.pearl) != 0
        self.countdown[ys, xs] = np.asarray(block.pearl_time, dtype=np.int32)
        self.occupant_kind[ys, xs] = Occupant.EMPTY
        self.occupant_id[ys, xs] = NO_ID
        self.occupant_facing[ys, xs] = 0
        for part in block.parts:
            if part.dragon_id == dragon_id:
                kind = Occupant.SELF_HEAD if part.is_head else Occupant.SELF_BODY
            elif part.team == team:
                kind = Occupant.ALLY_HEAD if part.is_head else Occupant.ALLY_BODY
            else:
                kind = Occupant.ENEMY_HEAD if part.is_head else Occupant.ENEMY_BODY
            self.occupant_kind[part.y, part.x] = kind
            self.occupant_id[part.y, part.x] = part.dragon_id
            self.occupant_facing[part.y, part.x] = DIR_INDEX[part.facing]

        for i in range(TILES):
            x, y = int(xs[i]), int(ys[i])
            for d in range(4):
                token = block.edge(i, DIRS[d])
                if token == EMPTY:
                    self._set_edge(x, y, d, EdgeType.EMPTY, NO_ID)
                elif token == KELP:
                    self._set_edge(x, y, d, EdgeType.KELP, NO_ID)
                else:
                    self._set_edge(x, y, d, EdgeType.PORTAL, int(token))

        self._visible_round = rnd
        self._visible.fill(False)
        self._visible[ys, xs] = True

    def mark_visited(self, cells, round_num: int) -> None:
        for cell in cells:
            if cell is UNKNOWN:
                continue
            x, y = cell
            if self.last_visit_round[y, x] == round_num:
                continue          # idempotent within a round
            self.visit_count[y, x] = min(int(self.visit_count[y, x]) + 1, 65535)
            self.last_visit_round[y, x] = round_num

    # -- reads ------------------------------------------------------------

    def materialize(self, round_num: int, own_id: int,
                    body: "BodyHistory | None" = None) -> np.ndarray:
        """The [42, H, W] float32 feature map at `round_num`."""
        H, W = self.height, self.width
        out = np.zeros((N_FACT_CHANNELS, H, W), dtype=np.float32)
        seen = self.seen
        out[F.SEEN] = seen
        if self._visible_round == round_num:
            out[F.VISIBLE] = self._visible
        age = np.clip(round_num - self.last_seen_round.astype(np.int32), 0, MAX_ROUNDS)
        out[F.AGE_SEEN] = np.where(seen, age / float(MAX_ROUNDS), 0.0)
        out[F.PEARL] = self.pearl & seen
        active = seen & (self.countdown >= 0)
        out[F.COUNTDOWN] = np.where(active, 1.0 / (1.0 + np.maximum(self.countdown, 0)), 0.0)
        out[F.NEVER] = seen & (self.countdown < 0)
        for k in range(len(Occupant)):
            out[F.OCC + k] = seen & (self.occupant_kind == k)
        occupied = seen & (self.occupant_kind != Occupant.EMPTY)
        for d in range(4):
            out[F.FACING + d] = occupied & (self.occupant_facing == d)
        # directional edge views derived from the canonical stores
        n_known, n_type = self.edge_known[0], self.edge_type[0]
        w_known, w_type = self.edge_known[1], self.edge_type[1]
        s_known, s_type = np.roll(n_known, -1, axis=0), np.roll(n_type, -1, axis=0)
        e_known, e_type = np.roll(w_known, -1, axis=1), np.roll(w_type, -1, axis=1)
        for d, (known, kind) in enumerate(((n_known, n_type), (e_known, e_type),
                                           (s_known, s_type), (w_known, w_type))):
            out[F.EDGE_KNOWN + d] = known
            out[F.KELP + d] = known & (kind == EdgeType.KELP)
            out[F.PORTAL + d] = known & (kind == EdgeType.PORTAL)
        visited = self.visit_count > 0
        out[F.VISIT_LOG] = np.clip(np.log1p(self.visit_count.astype(np.float32)) / _LOG_VISITS, 0, 1)
        vage = np.clip(round_num - self.last_visit_round.astype(np.int32), 0, MAX_ROUNDS)
        out[F.AGE_VISIT] = np.where(visited, vage / float(MAX_ROUNDS), 0.0)
        out[F.VISITED] = visited
        prev = self.prev_seen_round >= 0
        out[F.PREV_EXISTS] = prev
        out[F.PREV_PEARL] = prev & self.prev_pearl
        pactive = prev & (self.prev_countdown >= 0)
        out[F.PREV_COUNTDOWN] = np.where(pactive, 1.0 / (1.0 + np.maximum(self.prev_countdown, 0)), 0.0)
        interval = self.last_seen_round.astype(np.int32) - self.prev_seen_round.astype(np.int32)
        out[F.INTERVAL] = np.where(prev, np.clip(interval, 0, MAX_ROUNDS) / float(MAX_ROUNDS), 0.0)
        nonself = occupied & (self.occupant_id != NO_ID) & (self.occupant_id != own_id)
        out[F.ID_EARLIER] = nonself & (self.occupant_id < own_id)
        out[F.ID_LATER] = nonself & (self.occupant_id > own_id)
        if body is not None:
            body.paint(out)
        return out

    def snapshot(self) -> dict:
        return {
            "arrays": {k: v.copy() for k, v in self.__dict__.items()
                       if isinstance(v, np.ndarray)},
            "portals": {k: set(v) for k, v in self.portals.items()},
            "visible_round": self._visible_round,
            "width": self.width, "height": self.height,
        }

    def restore(self, snap: dict) -> None:
        for k, v in snap["arrays"].items():
            getattr(self, k)[...] = v
        self.portals = {k: set(v) for k, v in snap["portals"].items()}
        self._visible_round = snap["visible_round"]


# -- own body order ----------------------------------------------------------

class BodyHistory:
    """Head-first private record of where the dragon's own segments are.

    Initialised from the first observation as `[head] + [UNKNOWN] * (L - 1)`;
    the spawn body is never filled from the simulator. Every completed move
    prepends the head positions it produced, so after enough turns the whole
    body is known through the dragon's own trail.
    """

    def __init__(self, length: int, head: tuple[int, int]):
        self.cells: list = [tuple(head)] + [UNKNOWN] * (length - 1)

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def head(self) -> tuple[int, int]:
        return self.cells[0]

    @property
    def tail_known(self) -> bool:
        return self.cells[-1] is not UNKNOWN

    def known_fraction(self) -> float:
        return sum(c is not UNKNOWN for c in self.cells) / max(len(self.cells), 1)

    def index_of(self) -> dict[tuple[int, int], int]:
        return {c: i for i, c in enumerate(self.cells) if c is not UNKNOWN}

    def apply_move(self, path: list, new_length: int) -> None:
        """`path` is the temporal list of head positions p1..pk (UNKNOWN where
        not privately derivable); pk is the observed head and always known."""
        self.cells = list(reversed(path)) + self.cells
        del self.cells[new_length:]
        if len(self.cells) < new_length:
            self.cells += [UNKNOWN] * (new_length - len(self.cells))

    def apply_split(self, detached: int) -> None:
        del self.cells[len(self.cells) - detached:]

    def reconcile(self, block: TurnBlock, dragon_id: int) -> None:
        """Invalidate known cells the window contradicts; never repair from truth."""
        self.cells[0] = (block.head_x, block.head_y)
        own = {(p.x, p.y) for p in block.parts if p.dragon_id == dragon_id}
        visible = set(zip(map(int, block.xs), map(int, block.ys)))
        seen_cells: set = set()
        for i in range(1, len(self.cells)):
            c = self.cells[i]
            if c is UNKNOWN:
                continue
            if (c in visible and c not in own) or c in seen_cells or c == self.cells[0]:
                self.cells[i] = UNKNOWN
            else:
                seen_cells.add(c)
        if len(self.cells) != block.length:
            # a length change we did not predict (e.g. an unseen pearl during a
            # portal hop): keep what is known, pad or drop from the tail end
            del self.cells[block.length:]
            self.cells += [UNKNOWN] * (block.length - len(self.cells))

    def paint(self, out: np.ndarray) -> None:
        L = len(self.cells)
        for i, c in enumerate(self.cells):
            if c is UNKNOWN:
                continue
            x, y = c
            out[F.BODY_KNOWN, y, x] = 1.0
            out[F.BODY_INDEX, y, x] = i / max(L - 1, 1)
        if self.tail_known and L > 1:
            x, y = self.cells[-1]
            out[F.TAIL_KNOWN, y, x] = 1.0
        x, y = self.cells[0]
        out[F.HEAD, y, x] = 1.0


# -- command history ---------------------------------------------------------

@dataclass
class CommandRecord:
    kind: str                          # "move" or "split"
    dirs: str = ""                     # absolute tokens for a move
    split_n: int = 0
    round: int = 0
    old_head: tuple[int, int] = (0, 0)
    new_head: tuple[int, int] = (0, 0)
    old_heading: int = 0
    new_heading: int = 0
    old_length: int = 0
    new_length: int = 0
    intermediates_known: bool = True


class CommandRing:
    def __init__(self, size: int = HISTORY_TURNS):
        self.size = size
        self.records: deque[CommandRecord] = deque(maxlen=size)

    def push(self, record: CommandRecord) -> None:
        self.records.append(record)

    @property
    def last(self) -> CommandRecord | None:
        return self.records[-1] if self.records else None

    def features(self, round_num: int, width: int, height: int, max_steps: int
                 ) -> tuple[np.ndarray, np.ndarray]:
        """[32, F] features (oldest first, right-aligned) and a [32] mask."""
        out = np.zeros((self.size, N_HISTORY_FEATURES), dtype=np.float32)
        mask = np.zeros(self.size, dtype=bool)
        area = float(width * height)
        start = self.size - len(self.records)
        for j, rec in enumerate(self.records):
            row = out[start + j]
            mask[start + j] = True
            row[HF.IS_MOVE] = rec.kind == "move"
            row[HF.IS_SPLIT] = rec.kind == "split"
            for base, (x, y) in ((HF.OLD_SIN_X, rec.old_head), (HF.NEW_SIN_X, rec.new_head)):
                ax, ay = 2 * np.pi * x / width, 2 * np.pi * y / height
                row[base:base + 4] = (np.sin(ax), np.cos(ax), np.sin(ay), np.cos(ay))
            row[HF.OLD_HEADING_N + rec.old_heading] = 1.0
            row[HF.NEW_HEADING_N + rec.new_heading] = 1.0
            row[HF.ROUND] = rec.round / MAX_ROUNDS
            row[HF.ROUNDS_AGO] = max(round_num - rec.round, 0) / MAX_ROUNDS
            row[HF.OLD_LENGTH] = rec.old_length / area
            row[HF.NEW_LENGTH] = rec.new_length / area
            row[HF.LENGTH_DELTA] = (rec.new_length - rec.old_length) / max_steps
            row[HF.STEPS] = len(rec.dirs) / max_steps
            for ch in rec.dirs:
                row[HF.DIR_N + DIR_INDEX[ch]] += 1.0 / max_steps
            row[HF.SPLIT_FRACTION] = rec.split_n / max(rec.old_length, 1)
            row[HF.INTERMEDIATES_KNOWN] = rec.intermediates_known
        return out, mask


# -- one lifetime ------------------------------------------------------------

@dataclass(frozen=True)
class LifetimeKey:
    match_id: int
    team: str
    dragon_id: int
    generation: int = 0


@dataclass
class PendingCommand:
    kind: str
    dirs: str
    split_n: int
    round: int
    head: tuple[int, int]
    heading: int
    length: int


@dataclass
class LifetimeState:
    """Everything private to one lifetime. Created empty at spawn; a child
    never inherits any of it."""
    key: LifetimeKey
    width: int
    height: int
    unit_limit: int
    facts: FactualMemory = field(init=False)
    body: BodyHistory | None = None
    history: CommandRing = field(default_factory=CommandRing)
    pending: PendingCommand | None = None
    own_turns: int = 0
    last_processed_round: int = NO_ROUND
    inbox: list[int] = field(default_factory=list)
    _prewrite: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.facts = FactualMemory(self.width, self.height)

    # -- turn lifecycle ---------------------------------------------------

    def begin_turn(self, block: TurnBlock) -> bool:
        """Reconcile the previous command with this observation, then write the
        observation. Returns False when this round was already processed, in
        which case nothing changes (idempotent redelivery)."""
        if block.round == self.last_processed_round:
            return False
        head = (block.head_x, block.head_y)
        heading = DIR_INDEX[block.facing]
        if self.body is None:
            self.body = BodyHistory(block.length, head)
        else:
            self._complete_pending(block, head, heading)
            self.body.reconcile(block, self.key.dragon_id)
        self.facts.observe(block, self.key.dragon_id, self.key.team)
        self.facts.mark_visited([head], block.round)
        self.inbox = list(block.messages)
        self.last_processed_round = block.round
        return True

    def commit_command(self, block: TurnBlock, kind: str, dirs: str = "",
                       split_n: int = 0) -> None:
        """Record the command actually issued this turn. The next observation
        reconciles what it did; nothing is simulated speculatively."""
        self.pending = PendingCommand(kind, dirs, split_n, block.round,
                                      (block.head_x, block.head_y),
                                      DIR_INDEX[block.facing], block.length)
        self.own_turns += 1
        if kind == "split":
            # the parent's body order is truncated immediately: the detached
            # suffix belongs to a different lifetime now
            self.body.apply_split(split_n)

    def _complete_pending(self, block: TurnBlock, head: tuple[int, int],
                          heading: int) -> None:
        pending, self.pending = self.pending, None
        if pending is None:
            return
        if pending.kind == "split":
            self.history.push(CommandRecord(
                "split", split_n=pending.split_n, round=pending.round,
                old_head=pending.head, new_head=head, old_heading=pending.heading,
                new_heading=heading, old_length=pending.length,
                new_length=block.length, intermediates_known=True))
            return
        path, _ = self.trace_path(pending.head, pending.dirs)
        path[-1] = head            # the observed final head is always known
        known = all(c is not UNKNOWN for c in path)
        self.body.apply_move(path, block.length)
        self.facts.mark_visited(path, pending.round)
        self.history.push(CommandRecord(
            "move", dirs=pending.dirs, round=pending.round, old_head=pending.head,
            new_head=head, old_heading=pending.heading, new_heading=heading,
            old_length=pending.length, new_length=block.length,
            intermediates_known=known))

    def trace_path(self, start: tuple[int, int], dirs: str) -> tuple[list, bool]:
        """Head positions p1..pk of a MOVE, from private edge knowledge only.
        Once a position is unknown every later one is too."""
        path: list = []
        at: tuple[int, int] | None = start
        known = True
        for ch in dirs:
            if at is None:
                path.append(UNKNOWN)
                continue
            kind, landing = self.facts.step(at[0], at[1], DIR_INDEX[ch])
            if landing is None:
                at = None
                known = False
                path.append(UNKNOWN)
            else:
                at = landing
                path.append(landing)
        return path, known

    # -- snapshots --------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "key": self.key, "width": self.width, "height": self.height,
            "unit_limit": self.unit_limit,
            "facts": self.facts.snapshot(),
            "body": None if self.body is None else list(self.body.cells),
            "history": copy.deepcopy(list(self.history.records)),
            "pending": copy.deepcopy(self.pending),
            "own_turns": self.own_turns,
            "last_processed_round": self.last_processed_round,
            "inbox": list(self.inbox),
        }

    @classmethod
    def from_snapshot(cls, snap: dict) -> "LifetimeState":
        state = cls(snap["key"], snap["width"], snap["height"], snap["unit_limit"])
        state.restore(snap)
        return state

    def restore(self, snap: dict) -> None:
        self.facts.restore(snap["facts"])
        if snap["body"] is None:
            self.body = None
        else:
            self.body = BodyHistory(1, (0, 0))
            self.body.cells = list(snap["body"])
        self.history = CommandRing()
        for rec in snap["history"]:
            self.history.push(copy.deepcopy(rec))
        self.pending = copy.deepcopy(snap["pending"])
        self.own_turns = snap["own_turns"]
        self.last_processed_round = snap["last_processed_round"]
        self.inbox = list(snap["inbox"])
