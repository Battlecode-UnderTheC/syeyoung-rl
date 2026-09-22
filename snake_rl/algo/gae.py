"""Team GAE over arena time.

The advantage is computed once per team per *round*, never per dragon. A dragon
dying does not end its team's episode and must not reset the trace -- only the
match ending does. Every dragon of that team that acted in round t is then
handed the same `A[t]`, so an action is credited with the whole team's future,
including what happens after that particular dragon is dead.
"""

from __future__ import annotations

import numpy as np


def team_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float,
             terminal: bool, bootstrap_value: float = 0.0
             ) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and value targets for one team over one match.

    `rewards[t]` and `values[t]` are per round. `terminal` says whether the
    match really ended (so the value after the last round is 0) as opposed to
    being cut short, where `bootstrap_value` carries the rest.
    """
    steps = len(rewards)
    advantages = np.zeros(steps, dtype=np.float32)
    next_value = 0.0 if terminal else float(bootstrap_value)
    running = 0.0
    for t in range(steps - 1, -1, -1):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values
