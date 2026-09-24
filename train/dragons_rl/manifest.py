"""Spec section 10.1: the frozen run manifest (schema dragons-competitive-v1).

Maps are experiment inputs: the contest maps of unswbc 1.0.0 plus seeded
generated families, written once as files and referenced by sha256. Every
variant of a family shares its split. Nothing generates maps during training.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random

from dragons_env import mapgen, maps as contest

SCHEMA = "dragons-competitive-v1"
# sha256 of unswbc 1.0.0's unswbc_engine.wasm, the target release (dragons-env is held to it)
ENGINE_RELEASE_SHA256 = "48d249aa9cd98cba947e3cf51d731da5cb8da0a2bcddf89a02ca96880216aedd"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OPPONENTS = ROOT / "opponents"

# generated families: (side range, variants, split); contest-like parameters,
# the default unit limit, beds that refill on sensible gaps
FAMILIES = (
    [((7, 12), 4, "train"), ((12, 20), 4, "train"), ((20, 32), 4, "train"), ((32, 48), 4, "train"),
     ((48, 64), 4, "train")] * 4
    + [((7, 16), 2, "validation"), ((16, 32), 2, "validation"), ((32, 64), 2, "validation")]
    + [((7, 16), 2, "test"), ((16, 32), 2, "test"), ((32, 64), 2, "test")]
)


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _generated(family: int, variant: int, sides: tuple[int, int]) -> str:
    rng = random.Random(f"dragons-competitive-v1/family-{family}/variant-{variant}")
    spec = mapgen.random_map(rng, min_side=sides[0], max_side=sides[1], dragons_per_team=(1, 3), length=(3, 6),
                             kelp=rng.choice([0.0, 0.02, 0.05, 0.1]), portal_pairs=rng.choice([0, 1, 2, 4]),
                             bed_fraction=rng.choice([0.05, 0.1, 0.2]), unit_limit=None)
    spec.gaps = {k: (max(lo, 5), max(hi, lo, 5)) for k, (lo, hi) in spec.gaps.items()}
    spec.padding_edges = 0
    spec.name = f"family{family}-variant{variant}"
    return spec.to_text()


def build_pool(directory: pathlib.Path) -> list[dict]:
    """Writes the map files (idempotently) and returns the manifest's map entries."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in contest.names():
        path = directory / f"contest-{name}.map"
        path.write_text(contest.load(name))
        entries.append({"path": str(path), "family_id": f"contest-{name}", "split": "train", "weight": 1.0})
    for f, (sides, variants, split) in enumerate(FAMILIES):
        for v in range(variants):
            path = directory / f"family{f:02d}-v{v}.map"
            path.write_text(_generated(f, v, sides))
            entries.append({"path": str(path), "family_id": f"family{f:02d}", "split": split, "weight": 1.0})
    for e in entries:
        e["sha256"] = sha256_file(e["path"])
    return entries


def opponent(adapter_id: str, path=None) -> dict:
    path = pathlib.Path(path) if path else OPPONENTS / f"{adapter_id}.json"
    return {"artifact_path": str(path), "sha256": sha256_file(path), "adapter_id": adapter_id}


def write(run_dir: pathlib.Path, training_seed: int, entries: list[dict], validation: list[dict], test: list[dict],
          fixture_hash: str) -> dict:
    m = {"schema": SCHEMA, "engine_release_hash": ENGINE_RELEASE_SHA256, "maps": entries,
         "validation_opponents": validation, "final_test_opponents": test, "training_seed": training_seed,
         "protocol_fixture_results_hash": fixture_hash}
    check(m)
    path = run_dir / "run_manifest.json"
    if path.exists():
        raise FileExistsError(f"{path} is frozen")
    path.write_text(json.dumps(m, indent=1, sort_keys=True))
    return m


def check(m: dict) -> None:
    if m["schema"] != SCHEMA:
        raise ValueError(f"manifest schema {m['schema']} is not {SCHEMA}")
    splits = {"train": [], "validation": [], "test": []}
    family_split = {}
    for e in m["maps"]:
        if family_split.setdefault(e["family_id"], e["split"]) != e["split"]:
            raise ValueError(f"family {e['family_id']} spans two splits")
        splits[e["split"]].append(e)
        if e["split"] == "train" and not e["weight"] > 0:
            raise ValueError("training weights must be positive")
    if not all(splits.values()) or not m["validation_opponents"] or not m["final_test_opponents"]:
        raise ValueError("a complete reference run needs nonempty splits and opponent lists")


def load(run_dir: pathlib.Path) -> dict:
    """The frozen manifest, with every map and opponent hash verified."""
    m = json.loads((run_dir / "run_manifest.json").read_text())
    check(m)
    for e in m["maps"]:
        if sha256_file(e["path"]) != e["sha256"]:
            raise ValueError(f"map {e['path']} does not match its hash")
    for o in m["validation_opponents"] + m["final_test_opponents"]:
        if sha256_file(o["artifact_path"]) != o["sha256"]:
            raise ValueError(f"opponent {o['artifact_path']} does not match its hash")
    return m


def split(m: dict, name: str) -> list[dict]:
    return [e for e in m["maps"] if e["split"] == name]


def digest(m: dict) -> str:
    return sha256_text(json.dumps(m, sort_keys=True))
