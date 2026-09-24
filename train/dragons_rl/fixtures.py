"""Spec section 2 / 12.1: the environment's black-box conformance against the
target release, run before a reference run starts.

    python -m dragons_rl.fixtures OUT.json

runs dragons-env's sonar fixtures and its lockstep suite against the unswbc
1.0.0 wasm and writes the outcome (hashed into the run manifest)."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ENV = pathlib.Path(__file__).resolve().parents[2] / "dragons-env"


def main():
    out = pathlib.Path(sys.argv[1])
    commit = subprocess.run(["git", "-C", str(ENV), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ENV), "status", "--porcelain", "cpp", "dragons_env"], capture_output=True,
                           text=True).stdout.strip()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", "--color=no",
                        "tests/test_sonar_fixtures.py", "tests/test_official.py"], cwd=ENV, capture_output=True, text=True)
    tests = sorted(line for line in r.stdout.splitlines() if line.startswith(("PASSED", "FAILED", "ERROR")))
    result = {"passed": r.returncode == 0 and not dirty, "dragons_env_commit": commit, "dirty": bool(dirty),
              "tests": tests, "summary": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]}
    out.write_text(json.dumps(result, indent=1))
    print(result["summary"], "->", out)
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
