"""Compile the training programs ahead of time for the C++ trainer.

    python -m dragons_model.export --out artifacts/v1 [--num-envs 8192 ...]

writes, into --out:
    <program>.mlirbc         StableHLO bytecode (act, value, gather, grad_step, apply)
    <program>.sig            one line per flat input / output: "in|out <i> <dtype> <d0,d1,...> [donated|dropped]"
                             (a dropped input is not an argument of the compiled module)
    compile_options.pb       XLA compile options: a portable executable, whose device
                             is chosen per call (PJRT execute_device)
    state.txt, state.bin     the initial parameters and Adam state: one line per leaf
                             "<i> <group> <path> <dtype> <shape> <offset> <bytes>", raw bytes in state.bin
    config.txt               Config and derived sizes, "key value" per line

The flat order of every program's inputs is jax.tree_util's: the parameter
leaves first (as listed in state.txt), then the other arguments in order.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import export
from jax._src.lib import xla_client

from . import program as P

DTYPES = {np.dtype("float32"): "f32", np.dtype("float16"): "f16", np.dtype("int32"): "s32",
          np.dtype("uint32"): "u32", np.dtype("uint8"): "u8", np.dtype("bool"): "pred"}


def _sig_line(kind: str, i: int, aval, donated: bool = False) -> str:
    shape = ",".join(str(d) for d in aval.shape) or "-"
    return f"{kind} {i} {DTYPES[np.dtype(aval.dtype)]} {shape}" + (" donated" if donated else "")


def export_program(name: str, fn, args, donate: tuple, out: pathlib.Path, platform: str) -> None:
    exp = export.export(jax.jit(fn, donate_argnums=donate), platforms=[platform])(*args)
    n_in = len(exp.in_avals)
    # inputs a program never reads are left out of the compiled module; the
    # signature still lists them (marked "dropped") so callers keep one flat order
    kept = set(exp.module_kept_var_idx)
    donated_flat = []
    for a, arg in enumerate(args):
        donated_flat += [a in donate] * len(jax.tree_util.tree_leaves(arg))
    (out / f"{name}.mlirbc").write_bytes(exp.mlir_module_serialized)
    lines = [_sig_line("in", i, av, donated_flat[i]) if i in kept else _sig_line("in", i, av) + " dropped"
             for i, av in enumerate(exp.in_avals)]
    lines += [_sig_line("out", i, av) for i, av in enumerate(exp.out_avals)]
    (out / f"{name}.sig").write_text("\n".join(lines) + "\n")
    print(f"{name}: {n_in} inputs, {len(exp.out_avals)} outputs, {len(exp.mlir_module_serialized) / 1e6:.1f} MB")


def write_state(params, opt_state, out: pathlib.Path) -> None:
    lines, blobs, offset = [], [], 0
    groups = [("params", params), ("opt", opt_state)]
    i = 0
    for group, tree in groups:
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            a = np.asarray(leaf)
            name = jax.tree_util.keystr(path).replace(" ", "")
            shape = ",".join(str(d) for d in a.shape) or "-"
            lines.append(f"{i} {group} {name} {DTYPES[a.dtype]} {shape} {offset} {a.nbytes}")
            blobs.append(a.tobytes())
            offset += a.nbytes
            i += 1
    (out / "state.txt").write_text("\n".join(lines) + "\n")
    (out / "state.bin").write_bytes(b"".join(blobs))
    print(f"state: {i} leaves, {offset / 1e6:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--platform", default="cuda")
    for f in dataclasses.fields(P.Config):
        p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = p.parse_args()
    cfg = P.Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(P.Config)})
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shapes = P.shapes(cfg)
    train_step, adam = P.make_train_step(cfg)
    export_program("act", P.make_act(cfg), shapes["act"], (4, 5, 6), out, args.platform)
    export_program("value", P.make_value(cfg), shapes["value"], (), out, args.platform)
    export_program("gather", P.make_gather(cfg), shapes["gather"], (), out, args.platform)
    export_program("grad_step", train_step.grad_step, shapes["grad_step"], (), out, args.platform)
    export_program("apply", train_step.apply, shapes["apply"], (0, 1), out, args.platform)

    params = P.init_params(jax.random.PRNGKey(cfg.seed))
    write_state(params, adam.init(params), out)
    opts = xla_client.CompileOptions()
    opts.compile_portable_executable = True
    (out / "compile_options.pb").write_bytes(opts.SerializeAsString())
    (out / "config.txt").write_text("".join(f"{k} {v}\n" for k, v in cfg.items().items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
