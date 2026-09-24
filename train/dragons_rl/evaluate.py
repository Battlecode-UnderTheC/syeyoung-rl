"""Spec section 11: validation (every 100 updates) and the final test.

Every (map, opponent) pair is played on both sides with four sampling seeds;
its score is the mean of win + 0.5 draw over those eight matches. Candidate
and incumbent are compared by the paired group differences, bootstrapped."""

from __future__ import annotations

import numpy as np

from . import _native, collect, league, manifest

VALIDATION_SEEDS = (100, 101, 102, 103)
TEST_SEEDS = (200, 201, 202, 203)
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 777


def pairs(maps: list[dict], opponents: list[dict]) -> list[tuple[dict, dict]]:
    """(map, opponent) pairs in lexicographic (map hash, opponent hash) order"""
    return sorted(((m, o) for m in maps for o in opponents), key=lambda p: (p[0]["sha256"], p[1]["sha256"]))


def scores(candidate, opponents_weights: dict, critic, maps: list[dict], opponents: list[dict], seeds, devices,
           threads: int, log=None) -> dict:
    """{"map_sha|opponent_sha": score} for `candidate`. opponents_weights maps a
    learned opponent's sha256 to its weights; scripted ones need none."""
    ps = pairs(maps, opponents)
    policies = [candidate]
    slot_of = {}
    for _, o in ps:
        if o["adapter_id"] in _native.SCRIPTED:
            slot_of[o["sha256"]] = _native.SCRIPTED[o["adapter_id"]]
        elif o["sha256"] not in slot_of:
            slot_of[o["sha256"]] = len(policies)
            policies.append(opponents_weights[o["sha256"]])
    total = {}
    for seed in seeds:
        setups, index, meta = [], [], []
        for k, (m, o) in enumerate(ps):
            text = open(m["path"]).read()
            for side in (0, 1):
                policy = [0, 0]
                policy[side] = 0
                policy[1 - side] = slot_of[o["sha256"]]
                setups.append({"map": text, "policy": policy, "record": [False, False],
                               "seed": manifest_seed(seed, k, side)})
                index.append(k)
                meta.append((k, side))
        cols, parts = collect.play_parallel(setups, policies, critic, seed, 0, index, devices, threads, log)
        for col, positions in zip(cols, parts):
            winners = col.results()["winner"]
            for w, i in zip(winners, positions):
                k, side = meta[i]
                s = 1.0 if w == side else 0.5 if w < 0 else 0.0
                key = f"{ps[k][0]['sha256']}|{ps[k][1]['sha256']}"
                total[key] = total.get(key, 0.0) + s / (2 * len(seeds))
    return total


def manifest_seed(seed: int, pair: int, side: int) -> int:
    return int.from_bytes(bytes.fromhex(manifest.sha256_text(f"scripted/{seed}/{pair}/{side}"))[:8], "little")


def compare(candidate: dict, incumbent: dict) -> dict:
    """paired bootstrap of the mean group difference (PCG64 seed 777)"""
    keys = sorted(candidate)
    diff = np.array([candidate[k] - incumbent[k] for k in keys])
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    means = diff[rng.integers(0, len(diff), (BOOTSTRAP, len(diff)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"groups": len(keys), "mean_difference": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "candidate_score": float(np.mean([candidate[k] for k in keys])),
            "incumbent_score": float(np.mean([incumbent[k] for k in keys])), "promote": bool(lo > 0)}
