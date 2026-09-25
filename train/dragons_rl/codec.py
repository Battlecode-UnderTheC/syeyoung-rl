"""Buffers for the native codec, and the padding buckets that keep the number
of compiled shapes small. Padding is always masked (features.py), so a bucket
never changes an unpadded output."""

from __future__ import annotations

import hashlib

import numpy as np

from . import _native

FIELDS = _native.FIELDS
HIDDEN = FIELDS["HIDDEN"]


def pow2(n: int, floor: int) -> int:
    p = floor
    while p < n:
        p *= 2
    return p


def geometric(n: int, floor: int, ratio: int) -> int:
    """the smallest floor * ratio^k >= n: coarse buckets keep compiled shapes few"""
    p = floor
    while p < n:
        p *= ratio
    return p


# Coarse paddings keep the number of compiled programs (each ~0.1 GB of host
# memory, per device) small: nearly every decision fits the first size.
def inbox_pad(n: int) -> int:
    return geometric(n, 16, 8)


def echo_pad(n: int) -> int:
    return geometric(n, 4, 8)


def split_pad(length: int) -> int:
    """room for every split size 2..L-2 (L - 3 of them)"""
    return geometric(length - 3, 32, 8)


# Device calls are latency-bound and cost about the same per decision on any
# board, so few, big calls beat tight padding.
BOARD_BUCKETS = (128, 256, 512, 1024, 2048, 4096)


def board_bucket(tiles: int) -> int:
    for b in BOARD_BUCKETS:
        if tiles <= b:
            return b
    raise ValueError(f"{tiles} tiles is beyond a 64x64 board")


def actor_buffers(rows: int, n_pad: int, k_pad: int, e_pad: int) -> dict:
    return {
        "tile": np.zeros((rows, n_pad, FIELDS["TILE"]), np.int32),
        "dest": np.zeros((rows, n_pad, 4), np.int32),
        "entity": np.zeros((rows, FIELDS["MAX_ENTITIES"], FIELDS["ENTITY"]), np.int32),
        "segment": np.zeros((rows, FIELDS["WINDOW_TILES"], FIELDS["SEGMENT"]), np.int32),
        "status": np.zeros((rows, FIELDS["STATUS"]), np.int32),
        "prev_dirs": np.zeros((rows, FIELDS["PREV_DIRS"]), np.int32),
        "prev_words": np.zeros((rows, 32), np.uint8),
        "inbox": np.zeros((rows, k_pad, 8), np.uint8),
        "echo": np.zeros((rows, 5, e_pad), np.uint8),
        "echo_len": np.zeros((rows, 5), np.int32),
        "h": np.zeros((rows, HIDDEN), np.float32),
    }


ACTOR_KEYS = ("tile", "dest", "entity", "segment", "status", "prev_dirs", "prev_words", "inbox", "echo", "echo_len")


def critic_buffers(rows: int, n_pad: int, d_pad: int) -> dict:
    return {
        "board_tile": np.zeros((rows, n_pad, 2), np.int32),
        "board_dest": np.zeros((rows, n_pad, 4), np.int32),
        "grid": np.zeros((rows, n_pad), np.int32),
        "dragons": np.zeros((rows, d_pad, FIELDS["CRITIC_DRAGON"]), np.int32),
        "global": np.zeros((rows, FIELDS["CRITIC_GLOBAL"]), np.int32),
        "summary": np.zeros((rows, d_pad, HIDDEN + 1), np.float32),
    }


def command_arrays(n: int) -> dict:
    return {"kind": np.zeros(n, np.uint8), "nsteps": np.zeros(n, np.uint8), "dirs": np.zeros((n, 8), np.uint8),
            "split": np.zeros(n, np.int32), "sonar_mask": np.zeros(n, np.uint8), "words": np.zeros((n, 4), np.uint64)}


def to_native(command: dict) -> dict:
    """the policy's command (sonar indices 0 silent, 1 + j sends j) -> the codec's"""
    sonar = np.asarray(command["sonar"], np.int64)
    n = sonar.shape[0]
    out = command_arrays(n)
    out["kind"][:] = command["kind"]
    out["nsteps"][:] = command["nsteps"]
    out["dirs"][:] = command["dirs"]
    out["split"][:] = command["split"]
    out["sonar_mask"][:] = ((sonar > 0) * (1 << np.arange(4))).sum(1)
    out["words"][:] = np.maximum(sonar - 1, 0).astype(np.uint64)
    return out


def from_native(arrays: dict) -> dict:
    """the codec's command -> the policy's (for teacher forcing)"""
    mask = arrays["sonar_mask"].astype(np.int64)
    bits = (mask[:, None] >> np.arange(4)) & 1
    words = arrays["words"].astype(np.int64)
    if (words[bits == 1] > 255).any():
        raise ValueError("a stored word is outside the v1 transmit vocabulary")
    return {"kind": arrays["kind"].astype(np.int32), "nsteps": arrays["nsteps"].astype(np.int32),
            "dirs": arrays["dirs"].astype(np.int32), "split": arrays["split"].astype(np.int32),
            "sonar": np.where(bits == 1, words + 1, 0).astype(np.int32)}


# ---- section 10.2: private sampling streams ---------------------------------------------

UNIFORMS = 13          # kind, up to eight tokens (or a split size), four sonar heads


def lifetime_seed(training_seed: int, update: int, match: int, team: int, dragon: int) -> int:
    digest = hashlib.sha256(np.array([training_seed, update, match, team, dragon], "<u8").tobytes()).digest()
    return int.from_bytes(digest[:8], "little")


class Streams:
    """One PCG64 stream per lifetime; one float64 uniform per sampled categorical."""

    def __init__(self, training_seed: int, update: int):
        self.base = (training_seed, update)
        self.gens: dict[tuple, np.random.Generator] = {}

    def gen(self, match: int, team: int, dragon: int) -> np.random.Generator:
        key = (match, team, dragon)
        g = self.gens.get(key)
        if g is None:
            g = self.gens[key] = np.random.Generator(np.random.PCG64(lifetime_seed(*self.base, match, team, dragon)))
        return g

    def draw(self, keys) -> tuple[np.ndarray, list]:
        """UNIFORMS candidates per key, and the states to rewind to."""
        out = np.zeros((len(keys), UNIFORMS), np.float64)
        states = []
        for i, k in enumerate(keys):
            g = self.gen(*k)
            states.append(g.bit_generator.state)
            out[i] = g.random(UNIFORMS)
        return out, states

    def consume(self, keys, states, used) -> None:
        """leave each stream exactly `used` draws past where `draw` found it"""
        for k, s, n in zip(keys, states, used):
            g = self.gens[k]
            g.bit_generator.state = s
            g.random(int(n))
