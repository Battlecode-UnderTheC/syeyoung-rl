"""Spec section 10.3: opponent sampling, and weights on disk."""

from __future__ import annotations

import hashlib
import io
import os
import pathlib

import jax
import numpy as np

CURRENT = "current"        # the frozen copy of this batch's behaviour actor


def choose_opponent(rng: np.random.Generator, history: list) -> object:
    """20% the current behaviour actor, 40% one of the eight most recent
    snapshots, 40% an older one (its share falls to the recent set when there
    is none; everything is the current actor when there is no history)."""
    if not history:
        return CURRENT
    recent, older = history[-8:], history[:-8]
    u = rng.random()
    if u < 0.2:
        return CURRENT
    if u < 0.6 or not older:
        return recent[int(rng.integers(len(recent)))]
    return older[int(rng.integers(len(older)))]


def save_params(path: pathlib.Path, params) -> str:
    """writes a weight tree as .npz (keys are the tree paths); returns its sha256"""
    flat = {jax.tree_util.keystr(k): np.asarray(v) for k, v in jax.tree_util.tree_leaves_with_path(params)}
    buf = io.BytesIO()
    np.savez(buf, **flat)
    data = buf.getvalue()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def load_params(path: pathlib.Path, template, sha256: str | None = None):
    data = pathlib.Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path} does not match its hash")
    flat = np.load(io.BytesIO(data))
    leaves = []
    for k, v in jax.tree_util.tree_leaves_with_path(template):
        name = jax.tree_util.keystr(k)
        if name not in flat.files:          # a snapshot of an older model: the template's value
            leaves.append(np.asarray(v))
            continue
        a = flat[name]
        if a.shape != v.shape or a.dtype != v.dtype:
            raise ValueError(f"{path}: {jax.tree_util.keystr(k)} is {a.shape} {a.dtype}, expected {v.shape} {v.dtype}")
        leaves.append(a)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template), leaves)
