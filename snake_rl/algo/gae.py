"""Event-indexed GAE (spec section 8)."""

from __future__ import annotations

import numpy as np

def event_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
              delta_rounds: np.ndarray, terminal: np.ndarray, gamma_round: float = 1.0,
              lambda_round: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """Advantages over one match's *action-event* timeline, for one team.

    Event e is the instant before a living dragon acts. `rewards[e]` is zero
    except at the official terminal, `values[e]` is V_old(S_e), `next_values[e]`
    is V_old(S_{e+1}) (ignored when `terminal[e]`), and `delta_rounds[e]` is the
    number of round boundaries between S_e and S_{e+1} -- usually 0 or 1, so
    within a round extra children cause no extra trace decay.

        g_e = gamma ** dr_e;  l_e = lambda ** dr_e
        d_e = r_e + (1-done_e) g_e V(S_{e+1}) - V(S_e)
        A_e = d_e + (1-done_e) g_e l_e A_{e+1}

    With gamma = lambda = 1 over a complete match this gives exactly
    A_e = G - V(S_e) for every event, including the actions of dragons that
    died before the end; tests pin that identity.
    """
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    running = 0.0
    for e in range(n - 1, -1, -1):
        g = gamma_round ** float(delta_rounds[e])
        lam = lambda_round ** float(delta_rounds[e])
        cont = 0.0 if terminal[e] else 1.0
        delta = rewards[e] + cont * g * next_values[e] - values[e]
        running = delta + cont * g * lam * running
        adv[e] = running
    adv = adv.astype(np.float32)
    return adv, adv + values.astype(np.float32)
