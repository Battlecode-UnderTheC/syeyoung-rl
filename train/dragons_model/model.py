"""The actor (what ships in the bot) and the critic (training only), in plain JAX.

Actor, one turn of one dragon (MODEL.md has the reasoning):
  window 49 tiles x (21 channels + 4 x 4-dim portal-id embeddings)
    -> per-tile linear 37->32 (shared by the tiles) -> flatten -> linear 256
  memory map M [32 x 32 cells of 2 x 2 tiles, 16], read before this turn's write:
    local  9 x 9 cells around the head -> linear 128
    global the map averaged over 8 x 8 bins, plus "you are here" -> linear 128
  scalars 16 -> linear 64
  concat 576 -> LayerNorm -> linear 512 -> GRU 512 -> linear 256 -> 24 logits
  write: per tile [tile embedding, 32-dim window summary] -> content u (tanh) and
    gate g (sigmoid), 16 channels, averaged per cell: M <- M (1 - mean g) + mean(g u)

A turn touches only the 9 x 9 crop: the window's cells lie inside it, and the
global read comes from per-bin sums S [64, 16] kept up to date by the changes
the write makes. The map is stored in float16 (rollout and replay alike, so a
replay reproduces the rollout exactly); S in float32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .features import (CROP, MAP_CELLS, NUM_ACTIONS, NUM_SCALARS, POOL, PORTAL_IDS, TILES, WINDOW_CHANNELS,
                       Inputs, StateInputs)
from dragons_env import BLOCKS, STATE_CHANNELS, STATE_SCALARS

TILE_EMBED = 32
MAP_CHANNELS = 16
HIDDEN = 512
PORTAL_EMBED = 4
CELLS = MAP_CELLS * MAP_CELLS
BINS = POOL * POOL
CONTEXT = 32


def _linear(key, n_in, n_out, scale=1.0):
    # PyTorch's default: U(-1/sqrt(n_in), 1/sqrt(n_in)) for weights and bias
    k1, k2 = jax.random.split(key)
    bound = 1.0 / np.sqrt(n_in)
    return {"w": jax.random.uniform(k1, (n_in, n_out), jnp.float32, -bound, bound) * scale,
            "b": jax.random.uniform(k2, (n_out,), jnp.float32, -bound, bound) * scale}


# Matmuls and convolutions take float16 inputs (tensor cores) and accumulate in
# float32; parameters and everything elementwise stay float32. Rollout and
# replay run the same ops, so a replay still reproduces the rollout closely.
COMPUTE = jnp.float16


def set_compute(dtype) -> None:
    """float16 (default) or float32 for the model's matmuls and convolutions."""
    global COMPUTE
    COMPUTE = dtype


def _apply(p, x):
    y = jnp.matmul(x.astype(COMPUTE), p["w"].astype(COMPUTE), preferred_element_type=jnp.float32)
    return y + p["b"]


def init_actor(key) -> dict:
    ks = iter(jax.random.split(key, 16))
    tile_in = WINDOW_CHANNELS + 4 * PORTAL_EMBED
    fused = 256 + 64 + 128 + 128
    p = {
        "portal_embed": jax.random.normal(next(ks), (PORTAL_IDS + 1, PORTAL_EMBED)).at[PORTAL_IDS].set(0.0),
        "tile": _linear(next(ks), tile_in, TILE_EMBED),
        "window": _linear(next(ks), TILES * TILE_EMBED, 256),
        "scalars": _linear(next(ks), NUM_SCALARS, 64),
        "local": _linear(next(ks), MAP_CHANNELS * CROP * CROP, 128),
        "global": _linear(next(ks), MAP_CHANNELS * BINS + BINS, 128),
        "norm": {"scale": jnp.ones(fused), "bias": jnp.zeros(fused)},
        "fuse": _linear(next(ks), fused, HIDDEN),
        "gru_x": _linear(next(ks), HIDDEN, 3 * HIDDEN),     # r, z, n
        "gru_h": _linear(next(ks), HIDDEN, 3 * HIDDEN),
        "policy1": _linear(next(ks), HIDDEN, 256),
        "policy2": _linear(next(ks), 256, NUM_ACTIONS, scale=0.01),   # near-uniform start
        "write_context": _linear(next(ks), 256, CONTEXT),
        "write_content": _linear(next(ks), TILE_EMBED + CONTEXT, MAP_CHANNELS),
        "write_gate": _linear(next(ks), TILE_EMBED + CONTEXT, MAP_CHANNELS),
    }
    p["write_gate"]["b"] = jnp.zeros(MAP_CHANNELS)
    return p


def _layer_norm(p, x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * p["scale"] + p["bias"]


def _gru(p, x, h):
    # the PyTorch GRUCell equations
    gx, gh = _apply(p["gru_x"], x), _apply(p["gru_h"], h)
    rx, zx, nx = jnp.split(gx, 3, -1)
    rh, zh, nh = jnp.split(gh, 3, -1)
    r = jax.nn.sigmoid(rx + rh)
    z = jax.nn.sigmoid(zx + zh)
    n = jnp.tanh(nx + r * nh)
    return (1 - z) * n + z * h


def actor_step(p: dict, x: Inputs, h: jax.Array, m: jax.Array, bins: jax.Array):
    """One turn for B dragons. h f32[B, 512]; m f16[B, 1024, 16] (cells x channels);
    bins f32[B, 64, 16] (per-bin sums of m). Returns (logits [B, 24], h', m', bins')."""
    b = x.window.shape[0]
    relu = jax.nn.relu
    portals = p["portal_embed"][x.portal].reshape(b, TILES, 4 * PORTAL_EMBED)
    e = relu(_apply(p["tile"], jnp.concatenate([x.window, portals], -1)))           # [B, 49, 32]
    w = relu(_apply(p["window"], e.reshape(b, -1)))                                  # [B, 256]
    # reads, from the map as it was before this turn
    crop16 = jnp.take_along_axis(m, x.crop_cell[:, :, None], axis=1)                # [B, 81, 16] f16
    crop = crop16.astype(jnp.float32)
    loc = relu(_apply(p["local"], crop.reshape(b, -1)))
    avg = bins / jnp.maximum(x.bin_cells, 1.0)[:, :, None]
    glo = relu(_apply(p["global"], jnp.concatenate([avg.reshape(b, -1), x.here], 1)))
    s = relu(_apply(p["scalars"], x.scalars))
    fused = relu(_apply(p["fuse"], _layer_norm(p["norm"], jnp.concatenate([w, s, loc, glo], 1))))
    h2 = _gru(p, fused, h)
    logits = _apply(p["policy2"], relu(_apply(p["policy1"], h2)))
    # write this turn's window: every crop position gets the mean write of its cell
    ctx = relu(_apply(p["write_context"], w))
    we = jnp.concatenate([e, jnp.broadcast_to(ctx[:, None, :], (b, TILES, CONTEXT))], -1)
    u = jnp.tanh(_apply(p["write_content"], we))                                     # [B, 49, 16]
    g = jax.nn.sigmoid(_apply(p["write_gate"], we))
    hit = (x.crop_cell[:, :, None] == x.write_cell[:, None, :]).astype(jnp.float32)  # [B, 81, 49]
    n = jnp.maximum(hit.sum(-1, keepdims=True), 1.0)
    g_mean = hit @ g / n
    gu_mean = hit @ (g * u) / n
    new16 = (crop * (1 - g_mean) + gu_mean).astype(m.dtype)
    m2 = jax.vmap(lambda mm, idx, v: mm.at[idx].set(v))(m, x.crop_cell, new16)
    # bin sums follow the change; a cell repeated in the crop counts once
    same = (x.crop_cell[:, :, None] == x.crop_cell[:, None, :]).sum(-1, keepdims=True).astype(jnp.float32)
    delta = (new16.astype(jnp.float32) - crop) / same
    bins2 = bins + jax.nn.one_hot(x.crop_bin, BINS, dtype=jnp.float32).transpose(0, 2, 1) @ delta
    return logits, h2, m2, bins2


def masked_log_probs(logits: jax.Array, mask: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(jnp.where(mask, logits, -jnp.inf), -1)


def entropy(log_probs: jax.Array, mask: jax.Array) -> jax.Array:
    return -(jnp.exp(log_probs) * jnp.where(mask, log_probs, 0.0)).sum(-1)


# ---- critic (training only) ------------------------------------------------------

def _conv(key, c_in, c_out, k=3):
    k1, k2 = jax.random.split(key)
    bound = 1.0 / np.sqrt(c_in * k * k)
    return {"w": jax.random.uniform(k1, (c_out, c_in, k, k), jnp.float32, -bound, bound),
            "b": jax.random.uniform(k2, (c_out,), jnp.float32, -bound, bound)}


def _conv_apply(p, x, stride=1):
    y = jax.lax.conv_general_dilated(x.astype(COMPUTE), p["w"].astype(COMPUTE), (stride, stride), ((1, 1), (1, 1)),
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"), preferred_element_type=jnp.float32)
    return y + p["b"][None, :, None, None]


def init_critic(key) -> dict:
    ks = iter(jax.random.split(key, 8))
    return {
        "conv1": _conv(next(ks), len(STATE_CHANNELS), 32),
        "conv2": _conv(next(ks), 32, 64),
        "grid": _linear(next(ks), 64 * (BLOCKS // 2) ** 2, 256),
        "scalars": _linear(next(ks), len(STATE_SCALARS), 64),
        "hidden": _linear(next(ks), HIDDEN, 128),
        "value1": _linear(next(ks), 256 + 64 + 128, 256),
        "value2": _linear(next(ks), 256, 1),
    }


def critic_value(p: dict, x: StateInputs, h: jax.Array) -> jax.Array:
    """The acting team's value; `h` (the actor's state after this turn) gets no gradient."""
    relu = jax.nn.relu
    b = x.grid.shape[0]
    g = relu(_conv_apply(p["conv2"], relu(_conv_apply(p["conv1"], x.grid)), stride=2))
    g = relu(_apply(p["grid"], g.reshape(b, -1)))
    s = relu(_apply(p["scalars"], x.scalars))
    k = relu(_apply(p["hidden"], jax.lax.stop_gradient(h)))
    return _apply(p["value2"], relu(_apply(p["value1"], jnp.concatenate([g, s, k], 1))))[:, 0]
