"""Ahead-of-time export of `programs.build(...)` for the C++ trainer.

    python -m dragons_model.export --out artifacts/<name> [ProgramConfig flags]

writes into --out:
    <program>.mlirbc     StableHLO bytecode
    <program>.sig        one line per input and output: "<in|out> <name> <dtype> <shape> [donated]";
                         pytree arguments expand to "<arg>/<path>" names (e.g. "params/actor/gru/w")
    compile_options.pb   a portable executable (device chosen per call)
    state.bin, state.txt the initial parameters and Adam state, one named leaf per line
    manifest.txt         ProgramConfig, the code revision, and both layout descriptions,
                         which train::Artifacts::Open checks against the C++ build
"""

from __future__ import annotations


def export(config, out_dir: str, platform: str = "cuda") -> None:
    ...


def main() -> None:
    ...
