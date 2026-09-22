"""The conditional command codec (spec section 6).

One command per turn is either MOVE d1..dk (absolute directions, k <= K) or
SPLIT n. The decoder emits move directions autoregressively; what this module
supplies is the *private* prefix simulation that conditions each token, the
per-step token masks, the split candidate table, the protocol text, and a
lossless trace so PPO can teacher-force the stored command.

The prefix simulator sees only the lifetime's own memory and the current
window. Anything it cannot know is flagged unknown, never guessed from truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import (DIR_OFFSET, F, MIN_SIZE as _MIN, N_PREFIX_FEATURES, N_TOKENS,
                     Occupant, PF, TOKEN_STOP, TOP_MOVE, TOP_SPLIT, SONAR_SILENT)

_DIR_CHARS = "NESW"


@dataclass
class MoveTrace:
    """Everything needed to reproduce the decoder's conditional structure."""
    tokens: np.ndarray        # [K] int64; STOP after the last real direction
    active: np.ndarray        # [K] bool; a token was actually sampled here
    masks: np.ndarray         # [K, 5] bool; allowed tokens before each step
    prefix: np.ndarray        # [K, N_PREFIX_FEATURES] float32


@dataclass
class Command:
    top: int                             # TOP_MOVE or TOP_SPLIT
    dirs: str = ""
    split_n: int = 0
    sonar: int = SONAR_SILENT            # 0 silent, j+1 emits code j
    move: MoveTrace | None = None
    split_candidates: np.ndarray | None = None   # [C] int64 detached sizes
    split_features: np.ndarray | None = None     # [C, 4] float32
    split_index: int = -1

    @property
    def text(self) -> str:
        line = f"MOVE {self.dirs}" if self.top == TOP_MOVE else f"SPLIT {self.split_n}"
        if self.sonar != SONAR_SILENT:
            line += f"\nSONAR {self.sonar - 1}"
        return line


class PrefixSimulator:
    """Tracks what the dragon can privately know along a partial MOVE."""

    def __init__(self, obs, state, max_steps: int):
        self.obs = obs
        self.facts = state.facts
        self.K = max_steps
        self.area = float(obs.width * obs.height)
        self.pos: tuple[int, int] | None = obs.head
        self.old_length = obs.length
        self.length = obs.length
        self.length_known = True
        self.fatal_known = False
        self.eaten = 0
        self.steps = 0
        self.last = -1
        self.trail: list[tuple[int, int]] = []
        self.body_index = state.body.index_of()
        facts = obs.facts
        self.visible = facts[F.VISIBLE] > 0
        kinds = facts[F.OCC:F.OCC + 7].argmax(axis=0)
        self.self_here = self.visible & ((kinds == Occupant.SELF_HEAD) | (kinds == Occupant.SELF_BODY))
        self.other_here = self.visible & (kinds >= Occupant.ALLY_HEAD)
        self.head_here = self.visible & ((kinds == Occupant.ALLY_HEAD) | (kinds == Occupant.ENEMY_HEAD))
        self.pearl_here = self.visible & (facts[F.PEARL] > 0)

    # -- what is on a tile, from private knowledge -------------------------

    def _old_present(self, index: int) -> bool:
        """Is old-body cell `index` still part of the body before the next step?"""
        return index < self.length - len(self.trail)

    def _occupancy(self, cell: tuple[int, int]) -> str:
        """'self', 'other', 'head', 'empty' or 'unknown'."""
        x, y = cell
        if cell in self.trail:
            return "self"
        if self.other_here[y, x]:
            return "head" if self.head_here[y, x] else "other"
        if self.self_here[y, x]:
            idx = self.body_index.get(cell)
            if idx is not None:
                return "self" if self._old_present(idx) else "empty"
            # the index is not known: at the first step every old cell is still
            # there, later on it may have been the tail that moved
            return "self" if self.steps == 0 else "unknown"
        if not self.visible[y, x]:
            return "unknown"
        return "empty"

    def _look(self, d: int) -> tuple[str, tuple[int, int] | None, str, bool]:
        """(edge kind, landing, occupancy, pearl known) for stepping `d`."""
        if self.pos is None:
            return "unknown", None, "unknown", False
        kind, landing = self.facts.step(self.pos[0], self.pos[1], d)
        if landing is None:
            return kind, None, "unknown", False
        occ = self._occupancy(landing)
        pearl = bool(self.pearl_here[landing[1], landing[0]])
        return kind, landing, occ, pearl

    # -- decoder interface ---------------------------------------------------

    def features(self) -> np.ndarray:
        out = np.zeros(N_PREFIX_FEATURES, dtype=np.float32)
        out[PF.STEP] = self.steps / self.K
        out[PF.POSITION_KNOWN] = self.pos is not None
        out[PF.ALIVE_KNOWN] = not self.fatal_known and self.pos is not None
        out[PF.FATAL_KNOWN] = self.fatal_known
        out[PF.LENGTH_KNOWN] = self.length_known
        out[PF.LENGTH] = self.length / self.area if self.length_known else 0.0
        out[PF.CAN_PAY] = self.length_known and self.length > _MIN
        out[PF.RESOURCE_UNCERTAIN] = not self.length_known
        if self.last >= 0:
            out[PF.PREV_N + self.last] = 1.0
        for d in range(4):
            kind, landing, occ, pearl = self._look(d)
            if kind == "empty":
                out[PF.AHEAD_EMPTY_N + d] = 1.0
            elif kind == "kelp":
                out[PF.AHEAD_KELP_N + d] = 1.0
            elif kind == "portal":
                out[PF.AHEAD_PORTAL_N + d] = 1.0
            else:
                out[PF.AHEAD_UNKNOWN_N + d] = 1.0
            if occ == "self":
                out[PF.OCC_SELF_N + d] = 1.0
            elif occ == "other":
                out[PF.OCC_OTHER_N + d] = 1.0
            elif occ == "head":
                out[PF.OCC_HEAD_N + d] = 1.0
            elif occ == "unknown":
                out[PF.OCC_UNKNOWN_N + d] = 1.0
            out[PF.PEARL_N + d] = pearl
        out[PF.PEARLS_EATEN] = self.eaten / self.K
        return out

    def mask(self) -> np.ndarray:
        """Allowed tokens before the next one. Fatal directions stay allowed;
        only impossible-to-pay steps and post-fatal continuation are removed."""
        m = np.ones(N_TOKENS, dtype=bool)
        if self.steps == 0:
            m[TOKEN_STOP] = False
            return m
        if self.fatal_known or (self.length_known and self.length <= _MIN):
            m[:TOKEN_STOP] = False
        return m

    def advance(self, token: int) -> None:
        if token == TOKEN_STOP:
            return
        pay = self.steps >= 1
        kind, landing, occ, pearl = self._look(token)
        if kind == "kelp" or occ in ("self", "other", "head"):
            self.fatal_known = True
        if landing is None:
            # unknown edge or unpaired portal: position and everything after it
            self.pos = None
            if pay:
                self.length -= 1
            self.length_known = False
        else:
            self.pos = landing
            self.trail.append(landing)
            if self.visible[landing[1], landing[0]]:
                if pearl:
                    self.eaten += 1
                    self.length += 1
            else:
                self.length_known = False
            if pay:
                self.length -= 1
        self.steps += 1
        self.last = token


class CommandCodec:
    """Builds, traces and serialises dragons-spatial-v1 commands."""

    def __init__(self, max_steps: int = 8):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = max_steps

    # -- split ----------------------------------------------------------------

    def split_table(self, obs) -> tuple[np.ndarray, np.ndarray]:
        """Every legal detached size n in [2, L-2] with its generic features."""
        L = obs.length
        if not obs.split_legal:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
        n = np.arange(_MIN, L - _MIN + 1, dtype=np.int64)
        area = float(obs.width * obs.height)
        feats = np.stack([n / L, (L - n) / L,
                          np.log1p(n) / np.log1p(area),
                          np.log1p(L - n) / np.log1p(area)], axis=1).astype(np.float32)
        return n, feats

    # -- move -----------------------------------------------------------------

    def begin_move(self, obs, state) -> PrefixSimulator:
        return PrefixSimulator(obs, state, self.max_steps)

    def empty_trace(self) -> MoveTrace:
        K = self.max_steps
        return MoveTrace(np.full(K, TOKEN_STOP, dtype=np.int64), np.zeros(K, dtype=bool),
                         np.zeros((K, N_TOKENS), dtype=bool),
                         np.zeros((K, N_PREFIX_FEATURES), dtype=np.float32))

    def trace_move(self, obs, state, dirs: str) -> MoveTrace:
        """Replay a given direction string through the private simulator,
        producing the same masks/features sampling would have produced."""
        sim = self.begin_move(obs, state)
        trace = self.empty_trace()
        tokens = [_DIR_CHARS.index(c) for c in dirs] + [TOKEN_STOP]
        for j in range(self.max_steps):
            m = sim.mask()
            trace.masks[j] = m
            trace.prefix[j] = sim.features()
            tok = tokens[j] if j < len(tokens) else TOKEN_STOP
            if not m[tok]:
                raise ValueError(f"token {tok} at step {j} is masked for MOVE {dirs}")
            trace.tokens[j] = tok
            trace.active[j] = m.sum() > 1
            if tok == TOKEN_STOP:
                break
            sim.advance(tok)
        return trace

    @staticmethod
    def dirs_of(trace: MoveTrace) -> str:
        out = []
        for tok in trace.tokens:
            if tok == TOKEN_STOP:
                break
            out.append(_DIR_CHARS[int(tok)])
        return "".join(out)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def retained(length: int, detached: int) -> int:
        return length - detached
