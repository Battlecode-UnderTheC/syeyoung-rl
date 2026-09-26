"""Build step: the model as compiled programs for the C++ runtime (no Python at run time).

    python -m dragons_rl.export --out programs --slots 224 [--from runs/simple1/ckpt/N.pkl]

Each program is a jitted JAX function exported as StableHLO (<name>.mlirbc), its
XLA compile options (<name>.options) and a manifest (<name>.txt): one line per
flat input or output, `in|out <name> <dtype> <rank> <dims...>`, in call order.
Names are pytree paths joined with '/', e.g. learner/tile/l1/w or lraw/win.

Weights travel as named-array files (see write_arrays): actor.bin, critic.bin.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import struct

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import compiler

from . import codec, collect, model

DEVICES = 8                  # compile options are written for devices 0..DEVICES-1
DTYPES = {"float32": "f32", "int32": "s32", "uint8": "u8", "float16": "f16", "bool": "pred"}


def _name(path) -> str:
    parts = []
    for k in path:
        parts.append(str(getattr(k, "key", getattr(k, "idx", getattr(k, "name", k)))))
    return "/".join(parts)


def flat_named(tree) -> list[tuple[str, np.ndarray]]:
    return [(_name(p), np.asarray(v)) for p, v in jax.tree_util.tree_leaves_with_path(tree)]


def write_program(out: pathlib.Path, name: str, fn, args: dict, platform: str = "cuda"):
    """args: {argument name: pytree}; the function takes them in that order"""
    names = list(args)
    exp = jax.export.export(jax.jit(fn), platforms=(platform,))(*[args[n] for n in names])
    (out / f"{name}.mlirbc").write_bytes(exp.mlir_module_serialized)
    for d in range(DEVICES):                       # one compile per device: its assignment
        opts = compiler.get_compile_options(num_replicas=1, num_partitions=1, device_assignment=np.array([[d]]))
        (out / f"{name}.options.{d}").write_bytes(opts.SerializeAsString())
    lines = []
    for n in names:
        for leaf_name, a in flat_named(args[n]):
            full = f"{n}/{leaf_name}" if leaf_name else n
            lines.append(f"in {full} {DTYPES[str(a.dtype)]} {a.ndim} {' '.join(map(str, a.shape))}")
    outs = jax.eval_shape(fn, *[args[n] for n in names])
    for leaf_name, a in [(_name(p), v) for p, v in jax.tree_util.tree_leaves_with_path(outs)]:
        lines.append(f"out {leaf_name or 'out'} {DTYPES[str(a.dtype)]} {len(a.shape)} {' '.join(map(str, a.shape))}")
    (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
    return exp


def write_arrays(path: pathlib.Path, tree, prefix: str = ""):
    """named arrays: u32 count, then per array: u32 name length, name, u8 dtype code
    (0 f32, 1 s32, 2 u8, 3 f16), u32 rank, i64 dims, raw little-endian data"""
    codes = {"float32": 0, "int32": 1, "uint8": 2, "float16": 3}
    items = flat_named(tree)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(items)))
        for name, a in items:
            full = (prefix + name).encode()
            f.write(struct.pack("<I", len(full)))
            f.write(full)
            f.write(struct.pack("<BI", codes[str(a.dtype)], a.ndim))
            f.write(struct.pack(f"<{a.ndim}q", *a.shape))
            f.write(np.ascontiguousarray(a).tobytes())


def act_program(out: pathlib.Path, actor, critic, slots: int):
    """the workers' step: learner rows [slots], opponent rows [slots // 4 + 16], critic rows [max(8, slots // 4)]"""
    M, Mo, Mc = slots, slots // 4 + 16, max(8, slots // 4)
    k, e = codec.FIELDS["MAX_INBOX"], codec.FIELDS["MAX_ECHO"]
    lb, ob, cb = codec.actor_buffers(M, k, e), codec.actor_buffers(Mo, k, e), codec.critic_buffers(Mc, 128)

    def step(learner, opponent, critic, lraw, lh, oraw, oh, craw, cteam, seed):
        command, logp, h, value = collect.native_step.__wrapped__(learner, opponent, critic, lraw, lh, oraw, oh, craw,
                                                                  cteam, seed[0], codec.MAX_SPLIT)
        return {"command": command, "logp": logp, "h": h, "value": value}
    args = {"learner": actor, "opponent": actor, "critic": critic,
            "lraw": {k_: lb[k_] for k_ in codec.ACTOR_KEYS}, "lh": lb["h"],
            "oraw": {k_: ob[k_] for k_ in codec.ACTOR_KEYS}, "oh": ob["h"],
            "craw": cb, "cteam": np.zeros(Mc, np.int32), "seed": np.zeros(1, np.int32)}
    write_program(out, "act", step, args)
    (out / "act.shape").write_text(f"{M} {Mo} {Mc}\n")


def _fair(seed: int) -> str:
    import random
    from . import fairmaps
    return fairmaps.fair_map(random.Random(seed), 7, 64, f"pool-{seed}")


def write_maps(path: pathlib.Path, n: int, contest_share: float = 0.1):
    """n fresh fair maps and the contest maps (their share of the pool), as `<bytes>\n<text>` records"""
    import multiprocessing as mp
    import random
    from dragons_env import maps as contest
    with mp.get_context("spawn").Pool() as pool:
        fresh = pool.map(_fair, [random.randrange(2**62) for _ in range(n)], chunksize=16)
    extra = list(contest.contest_maps().values())
    k = int(n * contest_share / (1 - contest_share))
    pool_maps = fresh + [extra[i % len(extra)] for i in range(k)]
    random.shuffle(pool_maps)
    with open(path, "wb") as f:
        for text in pool_maps:
            b = text.encode()
            f.write(f"{len(b)}\n".encode() + b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--slots", type=int, default=224)
    p.add_argument("--from", dest="source", type=pathlib.Path, default=None, help="a trainer checkpoint (.pkl)")
    p.add_argument("--maps", type=int, default=0, help="also write a pool of this many fresh maps (maps.txt)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.source:
        with open(args.source, "rb") as f:
            s = pickle.load(f)
        actor, critic = s["actor"], s["critic"]
    else:
        actor = model.init_actor(jax.random.PRNGKey(0))
        critic = model.init_critic(jax.random.PRNGKey(1))
    actor = jax.tree.map(lambda a: np.asarray(a, np.float32), actor)
    critic = jax.tree.map(lambda a: np.asarray(a, np.float32), critic)
    write_arrays(args.out / "actor.bin", actor)
    write_arrays(args.out / "critic.bin", critic)
    act_program(args.out, actor, critic, args.slots)
    if args.maps:
        write_maps(args.out / "maps.txt", args.maps)
    print(f"exported to {args.out}")


if __name__ == "__main__":
    main()
