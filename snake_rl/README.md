# snake_rl

RL environment built on the **official engine** (`unswbc_engine.wasm`), driven
in-process through wasmtime. The rules are ground truth by construction — there
is no reimplementation to drift out of sync.

## Shape of the API

The engine runs a whole match inside one call and calls back per dragon-turn, so
this is **not** a gym `step()` loop. You hand it a policy:

```python
from snake_rl.env import WasmEngine, MatchRunner, mapgen
from snake_rl.policies import GreedyPearlPolicy

engine = WasmEngine()                     # compile once per process, reuse
runner = MatchRunner(engine)
policy = GreedyPearlPolicy(runner.space)

outcome = runner.run(mapgen.open_board(21, 21).to_bytes(), policy)
print(outcome.result.winner, outcome.death_reasons())
```

A policy is `(Obs) -> Action | int`. Per-dragon recurrent state belongs in the
policy's closure, keyed by `obs.dragon_id` — ids are assigned in spawn order,
are never reused, and a split child gets a fresh one, so they are lifetime ids.

## Observation

`Obs.grid` is `[21, 7, 7]` float32 — pearls (present / how soon / never), own
head and body, ally and enemy heads and bodies, part facings, and kelp and
portal edges on each of a tile's four sides. By default the window is rotated
into the dragon's frame so forward is always up, with the directional channel
groups rolled to match; with relative actions this makes the policy heading
invariant. `Obs.scalars` is `[8]`, `Obs.messages` is `[K, 4]` bytes plus a mask.

Nothing global ever reaches the policy. Everything comes from the 7×7 window and
the dragon's own status line.

## Actions

One action per turn: a move path or a split.

* **Moves** — paths of 1..`max_sprint` steps over {forward, left, right}; 39
  actions at the default `max_sprint=3`. A k-step sprint costs k−1 length. A
  reverse step is never included because it is always fatal (step 0 is your
  neck, and after any step the tile behind you is your own body).
* **Splits** — `SPLIT n` for n in 2..9. A split replaces the move, so the parent
  stays put; it is also the only way to survive a turn with no legal move.

`Obs.action_mask` is **exact**: an unmasked action cannot kill the dragon by its
own doing. It rules out kelp, occupied tiles (checked per sprint step), sprints
you cannot pay for, and illegal splits. Occupancy is truth at decision time
because dragons move in id order and the window shows the post-move state of
everyone earlier in the order.

Two documented holes, both unavoidable:

1. An enemy can still move onto your head (`H` deaths) — nothing observable
   rules that out.
2. A step through a **portal** lands where no window can see. Construct the
   space with `ActionSpace(allow_portal_steps=False)` to refuse those and make
   the mask airtight, at the cost of never using portals.

`tests/test_env.py` fuzzes this: random masked play over generated maps, and
every death must be `H` or a genuinely boxed-in dragon.

## Maps

The engine seeds its RNG from a **fixed constant**, so a given map always plays
out with identical pearl timers. Variety comes from generating maps:
`mapgen.random_map(rng, ...)` samples size, kelp, portals, pearl rates and
symmetric spawns; `mapgen.open_board(...)` is a plain torus for early training.

## Cost per dragon-turn

Measured on an M-series laptop, single core:

| stage | µs/turn |
|---|---|
| engine + FFI | 27–55 |
| parse | 26–40 |
| grid encode | 29–46 |
| action mask | 18–25 |
| **total** | **100–170** → 6–10k turns/s/core |

The spread is body length: a long dragon puts more parts in the window. The
engine scales linearly across processes and **anti-scales across threads** (the
callback holds the GIL), so parallelism must be process-based.

## Tests

```
.venv/bin/python -m pytest tests -q
```

`tests/test_rules.py` pins every rule in `../GAME_ANALYSIS.md` against the real
engine — if the toolkit ships a new engine and these fail, the document is stale.

## Model (`snake_rl.model`)

Implements the actor and critic from `SNAKE_MARL_IMPLEMENTATION_SPEC.md` §3–§4.

**Actor** (`Actor`, 659k params) — conv(C,32) → conv(32,64,s2) → conv(64,64,s2) →
Linear(·,256), a scalar MLP to 64 and a message encoder to 64, fused to 256 and
run through a single-layer GRU(256,256). Heads: the 47-way masked action head,
a transmit gate, and 4 × 16-way symbol heads packed one per byte of the uint32
sonar payload.

Its inputs are only its own window, scalars, messages, mask and hidden state —
there is no argument a global state could arrive through, and a test asserts it.

The heads are **conditional**: the joint log-prob and entropy sum over the
*active* heads, so a dragon that stays silent contributes nothing from its
payload heads to either. `sample_with_log_prob` (rollout) and `log_prob`
(update) are tested to agree, which is what makes the PPO ratio meaningful.

One module serves both paths — `step()` for a single timestep and `unroll()` for
a contiguous sequence — so the weights a rollout acted with are exactly the
weights the update differentiates.

**Critic** (`TeamCritic`, 652k params) — training only. A CNN over the whole
board (adaptive-pooled, so boards from 7×7 to 64×64 work) plus a permutation-
invariant masked mean/max pool over a variable dragon set with own/enemy flags.
Both team values come from the same weights evaluated from each team's point of
view, so `V(s)[A] == V(swap(s))[B]` holds by construction; `zero_sum=True`
additionally forces the two to sum to zero.

**`TorchPolicy`** wraps an actor for `MatchRunner`, holding one private GRU state
per dragon id — split children start from zeros because they get new ids, and a
dead dragon's state is dropped.

### Inference cost (CPU, 1 thread)

| batch | µs per dragon-turn |
|---|---|
| 1 | 302 |
| 8 | 93 |
| 64–256 | ~108 |

A full rollout loop (engine + encode + mask + batch-1 actor) runs at **534
µs/dragon-turn, ~1.9k turns/s/core**. Over half of the batch-1 cost is torch
dispatch overhead on tiny tensors, not arithmetic — an exported numpy forward
(which the judge submission needs anyway, since numpy is pre-imported in the
sandbox and torch is not) should be substantially faster. That is the next
optimisation, not a rewrite.
