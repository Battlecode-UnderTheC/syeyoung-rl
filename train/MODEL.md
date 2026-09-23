# Model and trainer

Long-horizon self-play PPO on the dragons game, with the model in JAX and
everything else in C++. The simulator is `dragons-env`, a port of the
official unswbc 1.0.0 engine.

## Layout

```
dragons_model/   JAX: inputs (features.py), actor and critic (model.py), the programs the
                 trainer runs (program.py), ahead-of-time export (export.py)
src/             C++: PJRT runtime (pjrt.*), exported artifacts (artifacts.*), rollout,
                 data-parallel PPO update, the trainer (main.cpp)
tests/           the model against the real environment; the programs' replay parity
```

```sh
python -m dragons_model.export --out artifacts/v1          # on the GPU machine
cmake -S . -B build && cmake --build build
build/trainer --artifacts artifacts/v1 --plugin <.venv>/.../jax_plugins/xla_cuda13/xla_cuda_plugin.so \
              --out runs/NAME [--resume runs/NAME/ckpt/<it>.bin]
```

## Inputs: only what the judge sends

A dragon-env observation record is the judge's round block in numbers.

- **Window: 21 channels per tile over the 7×7 window.**
  - pearl;
  - pearl bed, its countdown clipped to 50, and "spawns next round";
  - body segments, split into self, ally head, ally body, enemy head, enemy body;
  - each segment's facing, one-hot;
  - kelp on each of the four sides, and portal on each of the four sides.
- **Portal ids:** each side also carries a learned 4-dim embedding of the portal id mod 64.
- **16 scalars:**
  - log length, unit count ÷ limit, round and rounds left;
  - facing and team;
  - board size;
  - sin/cos of the head's position;
  - log(turns taken).

## Actor

This is the network that ships in the bot.

```
window 49 × 37 → per-tile linear 37→32 (shared) → flatten 1568 → linear 256
map crop 9×9 cells × 16 around the head → linear 128
map bins 8×8 × 16 (averages) + "you are here" 64 → linear 128
scalars 16 → linear 64
concat 576 → LayerNorm → linear 512 → GRU 512 → linear 256 → 24 logits (masked)
```

**Memory.** Each dragon has a learned map of 16 channels over cells of 2×2 tiles (32×32 cells on a 64×64 board).

- Each turn the map is read before it is written, so the reads only hold what the dragon remembers.
- The write works per tile: a content vector u = tanh(·) and a gate g = sigmoid(·), both computed from the tile embedding plus a 32-dim summary of the window.
- Writes are averaged per cell: M ← M·(1 − mean g) + mean(g·u).
- A turn only touches the 9×9 crop. The global read uses per-bin sums that are updated by each write's changes.
- The global read is average-only. A max-pooled read was considered, but it can't be updated incrementally.
- The map is stored in float16, in both rollout and replay, which keeps replay exact.
- A dragon's memory starts at zero at birth; split children get nothing from their parent, as in the engine.

**Actions.** There are 24:

| index | action |
|---|---|
| 0–3 | a single step: N, E, S, W |
| 4–19 | a two-step sprint, 16 direction pairs |
| 20–23 | split off a child of clamp(round(f·L), 2, L−2), for f ∈ {¼, ⅓, ½, ⅔} |

**Mask.** It removes moves that are certain death from what the dragon can see: kelp, or a body on the target tile (unless the side is a portal). Two further cases are masked conservatively; neither is certain death:

- the second step of a sprint whose first step goes through a portal, because that step can't be seen;
- the second step of a sprint onto the dragon's own body, because that segment might be its tail, which moves away.

If every action is masked, the four single steps are allowed. There is no sonar yet: the 1.0.0 sonar source is unpublished.

## Critic

The critic is used in training only. It sees the full state from dragons-env's state section:

- the board in 16×16 blocks, 8 channels: pearls, beds, kelp, my body, my heads, their body, their heads, and the actor;
- 12 scalars: the round, the board, and both teams' dragon count, longest and total length;
- the actor's GRU state, with gradients stopped.

Network: conv 8→32 → conv s2 32→64 → linear 256, concatenated with MLPs of the scalars and the GRU state, then 256 → V. V is the acting team's value.

## Returns

- **Reward:** the game's outcome for the dragon's team, +1 for a win, −1 for a loss, 0 for a draw, given at the game's end. There is no shaping.
- **Trajectories:** a dragon's turns form one trajectory, with a discount of γ = 0.999 per round between turns and GAE λ = 0.95 per turn.
- **Deaths:** a dragon that dies bootstraps from its team's result.
- **Rollout end:** trajectories still open bootstrap from the critic's value of the position, negated for the team not acting there, since the game is zero-sum.
- **Opponents:** self-play, with both teams played by the learner.

## Training loop

`act`, `value`, `gather`, `grad_step` and `apply` are fixed-shape XLA programs exported from JAX and run from C++ through the CUDA PJRT plugin.

**Rollout (GPU 0).**

- 8192 envs × 256 steps per iteration, which is 2.1M dragon-turns.
- Each dragon's state is kept in a slot of a device pool.
- A dragon's turns are cut into chunks of up to 32. Each chunk's exact starting state is copied out (`gather`) before `act` overwrites it.

**Update (all four GPUs).**

- Each round, every GPU replays one minibatch of 512 chunks from their start states (`lax.scan` with rematerialisation) and computes its gradient.
- `apply` averages the gradients, clips the actor and the critic separately (norm 1.0), and takes an Adam step. The new parameters are broadcast to the other GPUs.
- 2 epochs, learning rate 2.5e-4, clip 0.2, entropy coefficient 0.01, value coefficient 0.5.

**Precision.** Matmuls and convolutions take float16 inputs and accumulate in float32; parameters, optimiser state and all elementwise work stay float32. `grad_step` scales the loss by 1024 before differentiating, and `apply` skips any step whose gradients are not finite (logged as grad norm −1). Measured on a 2080 Ti against float32: `act` is 30% faster; the update only 5–8%, because it is bound by elementwise work (decoding records, the action mask, map gathers and scatters), not matmuls.

**Replay parity.** The first round of each update runs before any parameter change, so its approximate KL and clip fraction must be about 0. The trainer logs them every iteration.

## Throughput

Measured on 4× RTX 2080 Ti: about 41k dragon-turns/s end to end, of which the update takes about 55% and rollout about 45% (`act` 7 s, `gather` 8 s per iteration).

The headroom, in rough order of payoff:

- fp16 matmuls: the model runs in fp32 today;
- a lighter critic;
- overlapping rollout with the update;
- pinned-memory uploads.
