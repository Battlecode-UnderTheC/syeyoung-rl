"""The agent's side of the environment.

The game itself -- the rules, the official engine driver, the wire protocol and
map generation -- lives in the `dragons-env` submodule and is re-exported here.
Nothing in this package reimplements it: there is one copy of the rules and it
is the one the differential tests pin to the real engine.
"""

import pathlib
import sys

_SUBMODULE = pathlib.Path(__file__).resolve().parents[2] / "dragons-env"
if _SUBMODULE.is_dir() and str(_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE))

from dragons_env import mapgen                                    # noqa: E402
from dragons_env.engine import MatchResult, WasmEngine            # noqa: E402
from dragons_env.protocol import (SpawnInfo, TurnBlock,           # noqa: E402
                                  VISION, VISION_RADIUS,
                                  parse_spawn, parse_turn)

from .actions import ActionSpace                                  # noqa: E402
from .match import Action, MatchOutcome, MatchRunner, Obs         # noqa: E402
from .observation import ObservationEncoder                       # noqa: E402

__all__ = ["WasmEngine", "MatchResult", "TurnBlock", "SpawnInfo", "parse_turn",
           "parse_spawn", "VISION", "VISION_RADIUS", "MatchRunner", "Obs",
           "Action", "MatchOutcome", "ActionSpace", "ObservationEncoder",
           "mapgen"]
