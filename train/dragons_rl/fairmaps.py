"""Fair generated maps for training: every feature under one symmetry.

A map is drawn for one half and mirrored whole: pearl beds, kelp, portals and
team A's dragons (team B's are their mirror images, head first). Symmetries:
"x" flips y, "y" flips x, "xy" is the 180-degree rotation, as in the engine's
SYMMETRY line. Both teams therefore start identically up to the mirror, and
turn order alternates A, B, A, B with the first team drawn at random.
"""

from __future__ import annotations

import random

from dragons_env.mapgen import MapSpec

DX = (0, 1, 0, -1)
DY = (-1, 0, 1, 0)


class Mirror:
    def __init__(self, sym: str, w: int, h: int):
        self.w, self.h = w, h
        self.fx = sym in ("y", "xy")      # x is flipped
        self.fy = sym in ("x", "xy")      # y is flipped

    def tile(self, x, y):
        return (self.w - 1 - x if self.fx else x, self.h - 1 - y if self.fy else y)

    def h_edge(self, x, y):
        """north side of (x, y), mirrored: the edge between (x, y-1) and (x, y)"""
        return (self.w - 1 - x if self.fx else x, (self.h - y) % self.h if self.fy else y)

    def v_edge(self, x, y):
        """west side of (x, y), mirrored: the edge between (x-1, y) and (x, y)"""
        return ((self.w - x) % self.w if self.fx else x, self.h - 1 - y if self.fy else y)

    def dir(self, d):
        if d in (1, 3) and self.fx:
            return 4 - d
        if d in (0, 2) and self.fy:
            return 2 - d
        return d


def _walk(spec: MapSpec, rng: random.Random, length: int, blocked: set) -> list | None:
    for _ in range(200):
        at = (rng.randrange(spec.width), rng.randrange(spec.height))
        if at in blocked:
            continue
        body = [at]
        while len(body) < length:
            options = [spec.step(*body[-1], d) for d in range(4)]
            options = [p for p in options if p is not None and p not in blocked and p not in body]
            if not options:
                break
            body.append(rng.choice(options))
        if len(body) == length:
            head_first = body[::-1]
            if all(any(spec.step(*head_first[i], d) == head_first[i - 1] for d in range(4)) for i in range(1, length)):
                return head_first
    return None


def fair_map(rng: random.Random, min_side: int, max_side: int, name: str) -> str:
    while True:
        text = _attempt(rng, min_side, max_side, name)
        if text is not None:
            return text


def _attempt(rng, min_side, max_side, name):
    w, h = rng.randint(min_side, max_side), rng.randint(min_side, max_side)
    sym = rng.choice(["x", "y", "xy", "xy"])
    m = Mirror(sym, w, h)
    spec = MapSpec(w, h, symmetry=sym, name=name)
    # pearl beds, decided for a tile and its mirror together
    bed_fraction = rng.choice([0.05, 0.1, 0.2])
    for y in range(h):
        for x in range(w):
            mx, my = m.tile(x, y)
            if (my, mx) < (y, x):
                if (mx, my) in spec.gaps:
                    spec.gaps[(x, y)] = spec.gaps[(mx, my)]
            elif rng.random() < bed_fraction:
                lo = rng.choice([5, 10, 20, 30])
                spec.gaps[(x, y)] = (lo, lo + rng.choice([0, 5, 20]))
    # kelp, each edge with its mirror
    density = rng.choice([0.0, 0.02, 0.05, 0.1])
    for y in range(h):
        for x in range(w):
            if rng.random() < density:
                spec.kelp_h.update({(x, y), m.h_edge(x, y)})
            if rng.random() < density:
                spec.kelp_v.update({(x, y), m.v_edge(x, y)})
    # portals, each pair with its mirror pair
    used = {("h", e) for e in spec.kelp_h} | {("v", e) for e in spec.kelp_v}
    pid = rng.randrange(1, 500) * 2
    for _ in range(rng.choice([0, 1, 2])):
        o = rng.choice("hv")
        mir = m.h_edge if o == "h" else m.v_edge
        a = (rng.randrange(w), rng.randrange(h))
        b = (rng.randrange(w), rng.randrange(h))
        ends = [a, b, mir(*a), mir(*b)]
        keys = [(o, e) for e in ends]
        if len(set(keys)) != 4 or any(k in used for k in keys):
            continue                        # a self-mirrored or clashing end: no portal here
        used.update(keys)
        spec.portals.append((o, a, b, pid))
        spec.portals.append((o, mir(*a), mir(*b), pid + 1))
        pid += 2
    # team A's dragons, then their mirror images for team B
    occupied: set = set()
    pairs = []
    for _ in range(rng.randint(1, 3)):
        body = _walk(spec, rng, rng.randint(3, 6), occupied)
        if body is None:
            continue
        mirrored = [m.tile(*p) for p in body]
        if set(mirrored) & (occupied | set(body)):
            continue
        occupied.update(body)
        occupied.update(mirrored)
        pairs.append((body, mirrored))
    if not pairs:
        return None
    first = rng.randrange(2)
    for a_body, b_body in pairs:
        bodies = (a_body, b_body) if first == 0 else (b_body, a_body)
        teams = (0, 1) if first == 0 else (1, 0)
        for team, body in zip(teams, bodies):
            spec.dragons.append((team, body))
    text = spec.to_text()
    if not check(text, spec, m):
        return None
    return text


def check(text: str, spec: MapSpec, m: Mirror) -> bool:
    """the engine loads it, both teams start alike, and every step mirrors"""
    from dragons_env import Game
    from dragons_rl import _native
    try:
        g = Game(text)
    except Exception:
        return False
    step = _native.step_table(text)
    w = spec.width
    for y in range(spec.height):
        for x in range(w):
            t = y * w + x
            mx, my = m.tile(x, y)
            for d in range(4):
                to = step[t * 4 + d]
                mt = step[(my * w + mx) * 4 + m.dir(d)]
                if (to < 0) != (mt < 0):
                    return False
                if to >= 0 and m.tile(to % w, to // w) != (mt % w, mt // w):
                    return False
    dragons = g.dragons()
    lengths = [sorted(d["length"] for d in dragons if d["team"] == t) for t in (0, 1)]
    return lengths[0] == lengths[1]
