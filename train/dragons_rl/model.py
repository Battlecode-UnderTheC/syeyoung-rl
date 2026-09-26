"""Spec sections 5-8: the actor (530,115 parameters) and the privileged critic
(83,457), their initialisation, and the action distributions.

Functions take ONE decision / event; callers vmap (with per-example weights
during collection, where each match's teams play different snapshots).
All arithmetic is float32 (float64 is slow on these GPUs and x64 stays off).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax

from . import features as ft


F32 = jnp.float32
HIDDEN = 192
WIDTH = 32
ITERATIONS = 4
NEAR = 15                     # side of the memory crop around the head (simple model)
HEADS = 4
MAX_TOKENS = 8
BOS, STOP = 5, 4                      # token embedding rows: N E S W STOP BOS
SONAR_SYMBOLS = 257                   # 0 silent, 1 + j sends the uint64 j
NEG = -jnp.inf

ACTOR_PARAMETERS = 676_374       # simple1 (simple0 had 547,062; the spec actor 530,115)
CRITIC_PARAMETERS = 325_217      # simple1 (simple0 had 66,753; the spec critic 83,457)


# ---- operators (section 5.1) -------------------------------------------------------

def L(p, x):
    return x @ p["w"].T + p["b"]


def P(p, x):
    return jax.nn.silu(L(p, x))


def MLP(p, x):
    return jax.nn.silu(L(p["l2"], jax.nn.silu(L(p["l1"], x))))


def tile_MLP(p, x, half: bool = False):
    """MLP over every tile of a board. half: fp16 inputs and weights with fp32
    accumulation (tensor cores); ~3x faster at these skinny shapes. Collection
    uses it; training stays in float32 (PPO's ratio absorbs the small difference)."""
    if not half:
        return MLP(p, x)
    H16 = jnp.float16
    h = jnp.dot(x.astype(H16), p["l1"]["w"].T.astype(H16), preferred_element_type=F32) + p["l1"]["b"]
    h = jax.nn.silu(h).astype(H16)
    z = jnp.dot(h, p["l2"]["w"].T.astype(H16), preferred_element_type=F32) + p["l2"]["b"]
    return jax.nn.silu(z)


def tile_MLP_columns(p, cols):
    """MLP(p, stack(cols, -1)) without the stack: the first layer as a sum of
    column x weight products (one fused elementwise kernel with the feature
    arithmetic), the second in fp16 with fp32 accumulation. Collection only."""
    w1 = p["l1"]["w"]                                                   # [out, 56]
    h = p["l1"]["b"] + sum(c[:, None] * w1[:, f] for f, c in enumerate(cols))
    h = jax.nn.silu(h).astype(jnp.float16)
    z = jnp.dot(h, p["l2"]["w"].T.astype(jnp.float16), preferred_element_type=F32) + p["l2"]["b"]
    return jax.nn.silu(z)


def Score(p, x):
    return L(p["l2"], jax.nn.silu(L(p["l1"], x)))


def GRU(p, x, h):
    gi = jnp.einsum("ghi,...i->...gh", p["w_i"], x) + p["b_i"]
    gh = jnp.einsum("ghk,...k->...gh", p["w_h"], h) + p["b_h"]
    r = jax.nn.sigmoid(gi[..., 0, :] + gh[..., 0, :])
    z = jax.nn.sigmoid(gi[..., 1, :] + gh[..., 1, :])
    n = jnp.tanh(gi[..., 2, :] + r * gh[..., 2, :])
    return (1.0 - z) * n + z * h


def LN(p, x, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * p["scale"] + p["bias"]


def Attention(p, query, keys, key_mask):
    """multihead attention, width 32, 4 heads of 8, softmax over unmasked keys"""
    q = L(p["q"], query).reshape(query.shape[0], HEADS, WIDTH // HEADS)
    k = L(p["k"], keys).reshape(keys.shape[0], HEADS, WIDTH // HEADS)
    v = L(p["v"], keys).reshape(keys.shape[0], HEADS, WIDTH // HEADS)
    s = jnp.einsum("qhd,khd->hqk", q, k) / math.sqrt(WIDTH // HEADS)
    s = jnp.where(key_mask[None, None, :], s, NEG)
    w = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum("hqk,khd->qhd", w, v).reshape(query.shape[0], WIDTH)
    return L(p["o"], o)


def masked_mean_max(x, mask):
    m = mask[:, None].astype(F32)
    n = m.sum()
    mean = (x * m).sum(0) / jnp.maximum(n, 1.0)
    mx = jnp.where(mask[:, None], x, NEG).max(0)
    return jnp.where(n > 0, mean, 0.0), jnp.where(n > 0, mx, 0.0)


def Conv(p, x, stride: int = 1):
    """3x3 convolution over a [h, w, c] crop, zero padding outside it ("SAME").
    Written as nine shifted matmuls summed: under vmap with per-example weights
    (a stack of policies) each is a batched GEMM, not a grouped conv, and no 9x
    patch tensor is ever materialised (it dominated training memory)."""
    h, w, c = x.shape
    oh, ow = -(-h // stride), -(-w // stride)
    ph, pw = max((oh - 1) * stride + 3 - h, 0), max((ow - 1) * stride + 3 - w, 0)
    xp = jnp.pad(x, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    y = p["b"]
    for dy in range(3):
        for dx in range(3):
            tap = xp[dy:dy + stride * (oh - 1) + 1:stride, dx:dx + stride * (ow - 1) + 1:stride]   # [oh, ow, c]
            y = y + tap @ p["w"][:, :, dy, dx].T
    return y


# ---- initialisation (section 7) -----------------------------------------------------

class Init:
    def __init__(self, key):
        self.key = key

    def next(self):
        self.key, k = jax.random.split(self.key)
        return k

    def orth(self, shape, gain):
        return jax.nn.initializers.orthogonal(scale=gain)(self.next(), shape, F32)

    def L(self, a, b, gain=math.sqrt(2.0), bias=None):
        return {"w": self.orth((b, a), gain), "b": jnp.zeros((b,), F32) if bias is None else jnp.asarray(bias, F32)}

    def MLP(self, a, b, c):
        return {"l1": self.L(a, b), "l2": self.L(b, c)}

    def Score(self, a, b, c, out_gain=math.sqrt(2.0), out_bias=None):
        return {"l1": self.L(a, b), "l2": self.L(b, c, out_gain, out_bias)}

    def GRU(self, i, h):
        return {"w_i": jnp.stack([self.orth((h, i), 1.0) for _ in range(3)]),
                "w_h": jnp.stack([self.orth((h, h), 1.0) for _ in range(3)]),
                "b_i": jnp.zeros((3, h), F32), "b_h": jnp.zeros((3, h), F32)}

    def LN(self, d):
        return {"scale": jnp.ones((d,), F32), "bias": jnp.zeros((d,), F32)}

    def Conv(self, cin, cout):
        w = self.orth((cout, cin * 9), math.sqrt(2.0)).reshape(cout, cin, 3, 3)
        return {"w": w, "b": jnp.zeros((cout,), F32)}

    def embed(self, *shape):
        return 0.02 * jax.random.normal(self.next(), shape, F32)

    def Attention(self):
        return {k: self.L(WIDTH, WIDTH) for k in ("q", "k", "v", "o")}


def init_actor(key):
    i = Init(key)
    sonar_bias = jnp.zeros((SONAR_SYMBOLS,), F32).at[0].set(math.log(256.0))
    return {
        "local": {"conv1": i.Conv(56, 32), "conv2": i.Conv(32, 32), "out": i.L(96, 64)},
        "tile": i.MLP(56, 32, 32),
        "entity": i.MLP(20, 32, 32),
        "status": i.MLP(32, 32, 32),
        "word": {"byte": i.embed(256, 8), "position": i.embed(8, 8), "mlp": i.MLP(64, 32, 32)},
        "inbox": i.L(66, 32),
        "echo": {"gru": i.GRU(8, 16), "out": i.L(85, 32)},
        "physical": i.L(36, 32),
        "prev_sonar": i.L(37, 16),
        "previous": i.L(97, 32),
        "near": {"conv1": i.Conv(32, 32), "conv2": i.Conv(32, 32), "out": i.L(16 * 32, 64)},
        "grid": {"position": i.embed(64, 8), "cell": i.MLP(ft.GRID_FEATURES + 8, 32, 32), "out": i.L(64 * 32, 64)},
        "readout": {"direction": i.L(44, 32), "unknown": i.embed(WIDTH), "entities": i.L(64, 64)},
        "fusion": i.L(64 + 64 + 64 + 64 + 32 + 32 + 32 + 32 + 128, HIDDEN),
        "gru": i.GRU(HIDDEN, HIDDEN),
        "kind": i.L(HIDDEN, 2, 0.01, [math.log(19.0), 0.0]),
        "move": {"init": i.L(HIDDEN, 64), "token": i.embed(6, 8), "gru": i.GRU(18, 64),
                 "direction": i.Score(314, 64, 1, 0.01, [0.0]),
                 "stop": i.Score(266, 64, 1, 0.01, [math.log(8.0)])},
        "split": i.Score(198, 64, 1, 0.01, [0.0]),
        "sonar": i.Score(228, 64, SONAR_SYMBOLS, 0.01, sonar_bias),
    }


def init_critic(key):
    i = Init(key)
    return {
        "cells": {"position": i.embed(64, 8), "cell": i.MLP(ft.CRITIC_CELL_FEATURES + 8, 32, 32), "out": i.L(64 * 32, 128)},
        "dragon": i.MLP(207, 64, 64),
        "head": {"l1": i.L(272, 128), "l2": i.L(128, 64), "l3": i.L(64, 1, 1.0)},
    }


def count(params):
    return sum(int(x.size) for x in jax.tree.leaves(params))


# ---- the actor's encoder and recurrence (sections 5.2-5.4) -------------------------------

def encode_word(p, byte_rows):
    """[..., 8] uint8 little-endian bytes -> [..., 32]"""
    e = p["byte"][byte_rows.astype(jnp.int32)] + p["position"]
    return MLP(p["mlp"], e.reshape(*byte_rows.shape[:-1], 64))


def encode_echo(params, echo, lengths, present):
    p = params["echo"]
    emb = params["word"]["byte"][echo.astype(jnp.int32)]              # [5, e_pad, 8]

    def step(h, t):
        h_new = GRU(p["gru"], emb[:, t], h)
        return jnp.where((t < lengths)[:, None], h_new, h), None

    h, _ = lax.scan(step, jnp.zeros((5, 16), F32), jnp.arange(echo.shape[1]))
    flags = jnp.broadcast_to(present.astype(F32), (5,))
    return P(p["out"], jnp.concatenate([h.reshape(-1), flags]))


def encode(params, inp, h_old, half: bool = False):
    """Everything up to h_new: returns (h_new, the tile states)."""
    fused, z_tile = encode_fused(params, inp, half)
    return GRU(params["gru"], fused, h_old), z_tile


def encode_fused(params, inp, half: bool = False):
    """Everything before the GRU (none of it reads h): returns (fused, the tile states)."""
    X, J, D, tmask = inp["X"], inp["J"], inp["D"], inp["tile_mask"]
    n_pad = X.shape[0]
    # local
    pl = params["local"]
    if half:                                                          # X is never built for the whole board
        crop = jnp.stack([c[inp["crop"]] for c in inp["cols"]], -1)  # [7, 7, 56]
    else:
        crop = X[inp["crop"]]                                         # [7, 7, 56]
    y = jax.nn.silu(Conv(pl["conv2"], jax.nn.silu(Conv(pl["conv1"], crop))))
    flat = y.reshape(49, WIDTH)
    local = P(pl["out"], jnp.concatenate([y[3, 3], flat.mean(0), flat.max(0)]))
    tile = tile_MLP_columns(params["tile"], inp["cols"]) if half else MLP(params["tile"], X)
    entity = MLP(params["entity"], inp["U"])
    status = MLP(params["status"], inp["S"])
    # inbox: streaming mean/max over every word
    words = encode_word(params["word"], inp["inbox"])
    w_mean, w_max = masked_mean_max(words, inp["inbox_mask"])
    empty = (inp["inbox_count"] == 0).astype(F32)
    inbox = P(params["inbox"], jnp.concatenate([w_mean, w_max, ft.cnt(inp["inbox_count"])[None], empty[None]]))
    echo = encode_echo(params, inp["echo"], inp["echo_len"], inp["echo_present"])
    # previous command
    phys_prev = P(params["physical"], inp["C_prev"])
    prev_words = encode_word(params["word"], inp["prev_words"].reshape(4, 8))
    flags = inp["prev_flags"]
    sonar_prev = P(params["prev_sonar"], jnp.concatenate([flags[:, None] * prev_words, flags[:, None],
                                                          jnp.eye(4, dtype=F32)], -1))
    previous = P(params["previous"], jnp.concatenate([phys_prev, sonar_prev.reshape(-1), inp["prev_valid"][None]]))
    previous = previous * inp["prev_valid"]

    # the simple model: per-tile features without context, pooled and cropped
    z_tile = tile                                                     # [N, 32], also read by the move decoder
    emask = inp["entity_mask"]
    # memory near the head: the window's central 15x15, two stride-2 convs
    near_idx = ft.window_crop(NEAR)
    pn = params["near"]
    y = jax.nn.silu(Conv(pn["conv1"], z_tile[near_idx], stride=2))              # [8, 8, 32]
    y = jax.nn.silu(Conv(pn["conv2"], y, stride=2))                             # [4, 4, 32]
    near = P(pn["out"], y.reshape(-1))
    # the whole remembered map: 8x8 head-relative cells, each encoded with its position
    pg = params["grid"]
    cells = MLP(pg["cell"], jnp.concatenate([inp["grid"], pg["position"]], -1))      # [64, 32]
    glob = P(pg["out"], cells.reshape(-1))
    # readout
    ro = params["readout"]
    head = inp["head"]
    jh = J[head]
    dvec = jnp.where((jh >= 0)[:, None], z_tile[jnp.clip(jh, 0, n_pad - 1)], ro["unknown"])
    directional = P(ro["direction"], jnp.concatenate([dvec, D(head)], -1))          # [4, 32]
    e_mean, e_max = masked_mean_max(entity, emask)
    ents = P(ro["entities"], jnp.concatenate([e_mean, e_max]))
    fused = P(params["fusion"], jnp.concatenate([local, near, glob, ents, status, inbox, echo, previous,
                                                 directional.reshape(-1)]))
    return fused, z_tile


# ---- categorical distributions -------------------------------------------------------------

def log_softmax(logits, mask):
    """log-probabilities over the unmasked categories (all -inf if none).
    Masked entries never enter an arithmetic path, so gradients stay finite."""
    x = jnp.where(mask, logits, 0.0)
    m = lax.stop_gradient(jnp.max(jnp.where(mask, x, NEG)))
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    s = jnp.sum(jnp.where(mask, jnp.exp(x - m), 0.0))
    lse = jnp.log(jnp.where(mask.any(), s, 1.0)) + m
    return jnp.where(mask, x - lse, NEG)


def entropy(logp, mask):
    safe = jnp.where(mask, logp, 0.0)
    return -jnp.sum(jnp.where(mask, jnp.exp(safe) * safe, 0.0))


def sample(logits, mask, u):
    """Inverse-CDF categorical sample from one uniform u in [0, 1): a parallel
    float32 cumulative sum (the spec's float64 index-order loop was the
    collection bottleneck at small batch); masked categories never chosen."""
    p = jnp.where(mask, jax.nn.softmax(jnp.where(mask, logits, NEG)), 0.0).astype(F32)
    cum = jnp.cumsum(p)
    above = (cum > u.astype(F32) * cum[-1]) & mask
    idx = jnp.arange(p.shape[0])
    last_valid = jnp.max(jnp.where(mask, idx, -1))
    return jnp.where(above.any(), jnp.argmax(above), last_valid).astype(jnp.int32)


# ---- the action heads (section 6) -----------------------------------------------------------

def split_candidates(L_len, A, s_pad):
    """per split size n = 2.. its 6 features, and which sizes exist (n <= L - 2)"""
    n = jnp.arange(s_pad, dtype=jnp.int32) + 2
    Lf, Af, nf = ft.f32(L_len), ft.f32(A), ft.f32(n)
    x = jnp.stack([nf / Af, (Lf - nf) / Af, nf / Lf, (Lf - nf) / Lf, ft.loglen(n, A), ft.loglen(L_len - n, A)], -1)
    return x, n <= L_len - 2


def split_logits_of(p, h_new, L_len, A, s_pad):
    """Score(p, [h_new, candidate]) for every candidate, with the h_new part of the
    first layer computed once (the same function and weights, without an
    [s_pad, 198] input): wide s_pad is cheap"""
    x, mask = split_candidates(L_len, A, s_pad)
    w1 = p["l1"]["w"]                                                # [64, HIDDEN + 6]
    hidden = jax.nn.silu(h_new @ w1[:, :HIDDEN].T + p["l1"]["b"] + x @ w1[:, HIDDEN:].T)   # [s_pad, 64]
    return L(p["l2"], hidden)[:, 0], mask


def decode(params, h_new, z_tile, inp, s_pad, forced=None, uniforms=None):
    """Kind, physical parameters and the four sonar indices, either sampled
    (uniforms [13] float64) or teacher-forced (forced: the stored command).
    Returns the command, its joint log-probability and the entropy terms."""
    J, D = inp["J"], inp["D"]
    n_pad = J.shape[0]
    A, L_len = inp["A"], inp["L"]
    split_ok = (L_len >= 4) & (inp["unit_count"] < inp["unit_limit"])
    kind_mask = jnp.array([True, True]) & jnp.stack([jnp.array(True), split_ok])
    kind_logits = L(params["kind"], h_new)
    kind_lp = log_softmax(kind_logits, kind_mask)
    kind = forced["kind"] if forced is not None else sample(kind_logits, kind_mask, uniforms[0])
    kind = kind.astype(jnp.int32)

    # SPLIT n = 2..L-2
    split_logits, smask = split_logits_of(params["split"], h_new, L_len, A, s_pad)
    split_lp = log_softmax(split_logits, smask)
    split_idx = (forced["split"] - 2) if forced is not None else sample(split_logits, smask, uniforms[1])
    split_idx = jnp.clip(split_idx, 0, s_pad - 1)
    split_logp = jnp.where(smask[split_idx], split_lp[split_idx], 0.0)
    split_H = entropy(split_lp, smask) / jnp.log(jnp.maximum(2.0, ft.f32(L_len) - 3.0))

    # MOVE: autoregressive direction tokens with a geometric prefix tracker
    pm = params["move"]
    unknown = params["readout"]["unknown"]
    r0 = jnp.tanh(L(pm["init"], h_new))
    eye4 = jnp.eye(4, dtype=F32)
    synthetic = jnp.concatenate([eye4, jnp.broadcast_to(jnp.array([1.0, 0, 0, 0], F32), (4, 4)),
                                 jnp.zeros((4, 4), F32)], -1)                  # [4, 12]

    def token_step(carry, t):
        r, pos, known, heading, prev_tok, alive, n_dirs, lp, H = carry
        q = jnp.concatenate([
            jnp.stack([ft.f32(n_dirs) / 8.0, ft.f32(known)]),
            jnp.where(known, ft.cell_rel(pos, inp["W"], inp["H"]), 0.0),
            eye4[heading]])
        r = GRU(pm["gru"], jnp.concatenate([pm["token"][prev_tok], q]), r)
        jp = jnp.where(known, J[jnp.clip(pos, 0, n_pad - 1)], -1)                # [4]
        vfeat = jnp.where((jp >= 0)[:, None], z_tile[jnp.clip(jp, 0, n_pad - 1)], unknown)
        edge = jnp.where(known, D(jnp.clip(pos, 0, n_pad - 1)), synthetic)
        dx = jnp.concatenate([jnp.broadcast_to(r, (4, 64)), jnp.broadcast_to(h_new, (4, HIDDEN)), vfeat, edge,
                              jnp.broadcast_to(q, (4, 10)), eye4], -1)
        logits = jnp.concatenate([Score(pm["direction"], dx)[:, 0],
                                  Score(pm["stop"], jnp.concatenate([r, h_new, q]))])
        mask = jnp.array([True, True, True, True, False]) | (t > 0) & (jnp.arange(5) == STOP)
        lps = log_softmax(logits, mask)
        if forced is not None:
            tok = jnp.where(t < forced["nsteps"], forced["dirs"][jnp.minimum(t, MAX_TOKENS - 1)], STOP)
        else:
            tok = sample(logits, mask, uniforms[1 + t])
        tok = tok.astype(jnp.int32)
        lp = lp + jnp.where(alive & mask[tok], jnp.where(mask[tok], lps[tok], 0.0), 0.0)
        H = H + jnp.where(alive, entropy(lps, mask), 0.0)
        is_dir = alive & (tok != STOP)
        nxt = jp[jnp.minimum(tok, 3)]
        pos = jnp.where(is_dir, jnp.where(known & (nxt >= 0), nxt, 0), pos)
        known = jnp.where(is_dir, known & (nxt >= 0), known)
        heading = jnp.where(is_dir, jnp.minimum(tok, 3), heading)
        prev_tok = jnp.where(is_dir, tok, prev_tok)
        consumed = alive                                 # one uniform per sampled token
        alive = is_dir
        return (r, pos, known, heading, prev_tok, alive, n_dirs + is_dir.astype(jnp.int32), lp, H), \
               (jnp.where(is_dir, tok, 0).astype(jnp.int32), consumed)

    carry0 = (r0, inp["head"].astype(jnp.int32), jnp.array(True), inp["facing"].astype(jnp.int32),
              jnp.int32(BOS), jnp.array(True), jnp.int32(0), jnp.zeros((), F32), jnp.zeros((), F32))
    (_, _, _, _, _, _, n_dirs, move_logp, move_H), (toks, consumed) = lax.scan(
        token_step, carry0, jnp.arange(MAX_TOKENS), unroll=True)
    move_draws = consumed.sum().astype(jnp.int32)

    is_move = kind == 0
    phys_logp = jnp.where(is_move, move_logp, split_logp)
    phys_draws = jnp.where(is_move, move_draws, 1)
    nsteps = jnp.where(is_move, n_dirs, 0)
    dirs = jnp.where(jnp.arange(MAX_TOKENS) < nsteps, toks, -1)
    split_n = jnp.where(is_move, 0, split_idx + 2)

    # sonar, conditioned on the physical command
    C = ft.physical_vector(kind, dirs, split_n, L_len, A)
    phys = P(params["physical"], C)
    sx = jnp.concatenate([jnp.broadcast_to(h_new, (4, HIDDEN)), jnp.broadcast_to(phys, (4, WIDTH)),
                          jnp.eye(4, dtype=F32)], -1)
    sonar_logits = Score(params["sonar"], sx)                                    # [4, 257]
    all_ok = jnp.ones((SONAR_SYMBOLS,), bool)
    sonar_lp = jax.vmap(lambda lg: log_softmax(lg, all_ok))(sonar_logits)
    if forced is not None:
        sonar = forced["sonar"].astype(jnp.int32)
    else:
        sonar = jnp.stack([sample(sonar_logits[d], all_ok, uniforms[1 + phys_draws + d]) for d in range(4)])
    sonar_logp = jnp.take_along_axis(sonar_lp, sonar[:, None], 1)[:, 0].sum()
    sonar_H = jax.vmap(lambda lp: entropy(lp, all_ok))(sonar_lp).sum() / (4.0 * math.log(SONAR_SYMBOLS))

    logp = jnp.where(kind_mask[kind], kind_lp[kind], 0.0) + phys_logp + sonar_logp
    H_phys = 0.5 * (entropy(kind_lp, kind_mask) / math.log(2.0)
                    + jnp.where(is_move, move_H / (8.0 * math.log(5.0)), split_H))
    command = {"kind": kind, "nsteps": nsteps, "dirs": jnp.maximum(dirs, 0), "split": split_n, "sonar": sonar}
    return command, logp, H_phys, sonar_H, 1 + phys_draws + 4


def act(params, raw, h_old, uniforms, s_pad, half: bool = False):
    """Collection: one decision, sampled. Returns (command, logp, h_new, draws)."""
    inp = ft.actor_inputs(raw)
    h_new, z_tile = encode(params, inp, h_old, half)
    command, logp, _, _, draws = decode(params, h_new, z_tile, inp, s_pad, uniforms=uniforms)
    return command, logp, h_new, draws


def score(params, raw, h_old, command, s_pad):
    """Training: one stored decision, teacher-forced. Returns (logp, H_phys, H_sonar, h_new)."""
    inp = ft.actor_inputs(raw)
    h_new, z_tile = encode(params, inp, h_old)
    _, logp, H_phys, H_sonar, _ = decode(params, h_new, z_tile, inp, s_pad, forced=command)
    return logp, H_phys, H_sonar, h_new


# ---- the critic (section 8) -------------------------------------------------------------------

def value(params, raw, team):
    x = ft.critic_inputs(raw, team)
    pc = params["cells"]
    board = P(pc["out"], MLP(pc["cell"], jnp.concatenate([x["C"], pc["position"]], -1)).reshape(-1))   # [128]
    r = MLP(params["dragon"], x["R"])
    r_mean, r_max = masked_mean_max(r, x["dragon_mask"])
    ph = params["head"]
    h = jax.nn.silu(L(ph["l1"], jnp.concatenate([board, r_mean, r_max, x["Q"]])))
    return jnp.tanh(L(ph["l3"], jax.nn.silu(L(ph["l2"], h))))[0]
