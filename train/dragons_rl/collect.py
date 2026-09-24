"""Collection (spec section 9.1): complete matches, one dragon-turn per match
per step, each team played by its policy slot, with the critic's two values
recorded at every event.

Policies are a stack of actor weights on the device; slot k is row k. Every
decision of a step runs in one call per board-size bucket, each example with
its own slot's weights, so a batch mixes the learner and any number of frozen
opponents.
"""

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import _native, codec, model

ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


def stack_policies(params_list, slots: int):
    """slots rows of actor weights (the list, then copies of its first entry)"""
    padded = list(params_list) + [params_list[0]] * (slots - len(params_list))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *padded)


@partial(jax.jit, static_argnames=("s_pad",))
def policy_step(stack, slot, raw, h, uniforms, s_pad):
    def one(slot, raw, h, u):
        params = jax.tree.map(lambda a: a[slot], stack)
        return model.act(params, raw, h, u, s_pad)
    return jax.vmap(one)(slot, raw, h, uniforms)


@jax.jit
def critic_step(params, raw):
    def one(raw):
        return jnp.stack([model.value(params, raw, 0), model.value(params, raw, 1)])
    return jax.vmap(one)(raw)


def _rows(n: int) -> int:
    for b in ROW_BUCKETS:
        if n <= b:
            return b
    return codec.pow2(n, 128)


def _pad_rows(buf: dict, used: int) -> dict:
    """fill unused rows with row 0 (a real decision), so padding computes finite values"""
    for a in buf.values():
        a[used:] = a[0]
    return buf


def unit_limit(map_text: str) -> int:
    for line in map_text.splitlines():
        if line.startswith("UNIT_LIMIT"):
            return int(line.split()[1])
    return 64


def play(setups: list[dict], stack, critic_params, training_seed: int, update: int, match_index, *,
         device=None, threads: int = 1, log=None) -> _native.Collector:
    """Plays every setup to its end. Lifetime (match m, team, dragon) samples
    from the stream of (training_seed, update, match_index[m], team, dragon)."""
    col = _native.Collector(setups, threads)
    d_pad = codec.pow2(2 * max(unit_limit(s["map"]) for s in setups), 8)
    streams = codec.Streams(training_seed, update)
    steps, decisions, t0 = 0, 0, time.time()
    put = (lambda x: jax.device_put(x, device)) if device is not None else jnp.asarray
    rows = _rows(len(setups))           # one shape per collection: every step pads to it
    while not col.done():
        pend = col.prepare()
        B = len(pend["match"])
        if B == 0:
            continue
        # one call for every waiting decision, padded to the largest board in play
        n_pad = codec.tile_bucket(int(pend["tiles"].max()))
        matches = pend["match"]
        k_pad = codec.inbox_pad(int(pend["inbox"].max()))
        e_pad = codec.echo_pad(int(pend["echo_bytes"].max()))
        s_pad = codec.split_pad(int(pend["length"].max()))
        abuf = codec.actor_buffers(rows, n_pad, k_pad, e_pad)
        col.encode_actors(matches, abuf)
        cbuf = codec.critic_buffers(rows, n_pad, d_pad)
        col.encode_critics(matches, cbuf)
        _pad_rows(abuf, B)
        _pad_rows(cbuf, B)
        keys = [(int(match_index[m]), int(t), int(d)) for m, t, d in zip(matches, pend["team"], pend["dragon"])]
        uniforms = np.zeros((rows, codec.UNIFORMS), np.float64)
        uniforms[:B], states = streams.draw(keys)
        slot = np.zeros(rows, np.int32)
        slot[:B] = pend["policy"]
        raw = {k: put(abuf[k]) for k in codec.ACTOR_KEYS}
        command, logp, h_new, draws = policy_step(stack, put(slot), raw, put(abuf["h"]), put(uniforms), s_pad)
        values = critic_step(critic_params, {k: put(v) for k, v in cbuf.items()})
        command = {k: np.asarray(v)[:B] for k, v in command.items()}
        streams.consume(keys, states, np.asarray(draws)[:B])
        col.apply(matches, codec.to_native(command), np.asarray(logp, np.float32)[:B],
                  np.asarray(h_new, np.float32)[:B], np.asarray(values, np.float32)[:B])
        decisions += B
        steps += 1
        if log is not None and steps % 200 == 0:
            log(f"collect: {steps} steps, {decisions} decisions, {decisions / (time.time() - t0):.0f}/s")
    return col


SLOTS = 34      # the stack's fixed size: slot 0 the behaviour actor, then up to 33 opponents


def play_parallel(setups: list[dict], policies: list, critic_params, training_seed: int, update: int,
                  match_index: list[int], devices: list, threads: int, log=None):
    """play() over devices, the matches dealt round-robin. `policies[k]` is the
    weights of slot k. Returns one finished Collector per device, and for
    each device the positions (into setups) of its matches."""
    from concurrent.futures import ThreadPoolExecutor
    if len(policies) > SLOTS:
        raise ValueError(f"at most {SLOTS} policies in one collection")
    parts = [list(range(d, len(setups), len(devices))) for d in range(len(devices))]
    parts = [(dev, p) for dev, p in zip(devices, parts) if p]

    def run(item):
        dev, positions = item
        stack = jax.device_put(stack_policies(policies, SLOTS), dev)
        critic = jax.device_put(critic_params, dev)
        return play([setups[i] for i in positions], stack, critic, training_seed, update,
                    [match_index[i] for i in positions], device=dev, threads=max(1, threads // len(parts)), log=log)

    with ThreadPoolExecutor(len(parts)) as pool:
        collectors = list(pool.map(run, parts))
    return collectors, [p for _, p in parts]
