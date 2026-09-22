# Snake PvP MARL — Implementation Specification for Claude Code

**Status:** implementation-ready design, with explicit configuration points for unconfirmed game mechanics.  
**Audience:** coding agent implementing a Python/PyTorch training system against an existing or newly written Snake simulator.  
**Priority:** correctness and strict decentralized execution before throughput optimizations.

## 0. Mission and non-negotiable rules

Implement recurrent, parameter-shared multi-agent PPO for a two-player Snake PvP game. Each player can own multiple independently controlled snakes created by splitting. Every snake has private local vision and private recurrent memory. **The only in-game information channel between snakes is a forward-traveling 4-byte message** that can also be intercepted by opponents. There is **no team coordinator, shared map, shared recurrent state, or automatic transfer of parent memory to a child**. All snakes may execute identical policy weights; this is not information sharing. A centralized critic may inspect full simulator state **only during training**, never during actor inference.

The winning criterion is lexicographic: maximize the length of the player's longest living snake; if both players' longest lengths tie, compare total living snake length (interpretation of user's “total max len”; expose as a configurable tiebreak function until confirmed). If both tie, draw. Score and win checks must use the authoritative simulator rules, not guessed alternatives.

**Implementation instruction:** inspect the existing repository before writing code. Reuse its game rules, simulator, observations, action semantics, and conventions. Do not silently implement unspecified mechanics. Where no existing implementation answers a question, isolate it behind an interface/configuration and document the assumption. First deliver a working end-to-end baseline, then optional features.

## 1. Required game-adapter contract

Do not assume discrete movement, food behavior, sprint costs, split mechanics, portal topology, death/respawn, message propagation, message visibility, match duration, or max snake count until checked in the simulator. Expose all as adapter-provided behavior.

```python
@dataclass(frozen=True)
class SnakeKey:
    arena_id: int
    lifetime_id: int  # never reused within an arena; owner tracked separately

@dataclass
class SnakeObservation:
    grid: np.ndarray          # [C,H,W] float32, local visible cells only
    scalars: np.ndarray       # [D] float32, ONLY legally observable fields
    received: np.ndarray      # [K,4] uint8, delivered messages, padded
    received_mask: np.ndarray # [K] bool
    # optional: arrival direction/age ONLY if actually observable
    legal_actions: dict       # masks for steer/sprint/split/transmit etc.

@dataclass
class SnakeAction:
    steer: int | float
    sprint: bool
    split: bool
    transmit: bool
    payload: bytes            # exactly 4 bytes if transmit, else b''

@dataclass
class StepResult:
    observations: dict[SnakeKey, SnakeObservation]
    owners: dict[SnakeKey, int]     # player 0 or 1
    full_state: object              # training critic only
    team_scores: tuple             # authoritative scores from simulator
    match_done: bool
    winner: int | None              # 0/1/draw if done
    events: list                   # spawn, split, death, message delivery, etc.
```

`reset(seed) -> StepResult`, `step(dict[SnakeKey,SnakeAction]) -> StepResult`. Snapshot **pre-action** observations and global state, then step simultaneously; align each action with its pre-action observation and resulting reward. When a snake splits, the parent retains its identity and memory if the engine says so; the child receives a new lifetime ID and zero memory. If a snake dies, close its actor trajectory; do **not** terminate its player's match or erase the team's future return. Respawned snakes receive new lifetime IDs and zero memory. Distinguish match termination from time-limit truncation; bootstrap at truncation when appropriate.

### Information-flow invariants

1. Actor inputs: its own observation, delivered messages, own previous GRU state, its own legally visible action mask, and optionally its own previous action if the environment makes that information available.
2. Actor MUST NOT read full_state, teammate observations, team-wide stats unavailable in game, unseen portal exits, hidden enemy positions, other snakes' hidden states, or message delivery events before arrival.
3. Shared weights are allowed; **no shared runtime hidden states, pooled team features, central planner, or implicit parent-child knowledge transfer**.
4. Critic may read global state only in the learner/training worker. Do not serialize critic features into actor observations or deploy the critic as a policy input.
5. Messages are generated and delivered by simulator physics. Do not directly write message content into a teammate's observation or memory.

## 2. Scoring and rewards

For player p at state s, let `L_i` be length of each *living* owned snake:

```
M_p = max(L_i for owned living snakes, default=0)
B_p = sum(L_i for owned living snakes)
score_p = (M_p, B_p)   # lexicographic; replace B_p if simulator defines another tiebreak
```

Terminal reward: `R_0 = +1` if `score_0 > score_1`, `-1` if less, else `0`; `R_1 = -R_0`. Only pay at true match termination, not individual death or time-limit truncation unless the game resolves a winner there. This is the **baseline reward** and should be implemented first.

Optional dense potential shaping, enabled via config and tested against terminal-only baseline:

```
# Only if B_max is a proven upper bound on total living length:
phi_p(s) = M_p(s) + B_p(s)/(B_max + 1)
phi_rel(s,p) = phi_p(s) - phi_opponent(s)
r_p(t) = terminal_reward_p(t) + lambda_g * (
    gamma * phi_rel(s_next,p) - phi_rel(s,p)
)
```

For true terminal transitions use `phi_rel(s_next,p)=0` (absorbing terminal potential); use correct bootstrapping at time-limit truncation. `B_max` must be an actual upper bound, and integer primary scores are assumed for lexicographic preservation. If not provable, use terminal-only or a clearly labeled approximate potential `M + beta*B` and keep terminal win as the authoritative objective. The shaping expression above is potential-based; **do not replace it with raw delta score and claim policy invariance**. At the start of each episode, potential offsets may depend on initial state, which is fine. Do not award a separate split bonus, sprint bonus, message bonus, food bonus, or arbitrary death penalty by default. Growth, resource costs, deaths and splitting already change authoritative score; a child's creation must not multiply rewards. Keep a single reward stream per player per environment timestep. If a teammate dies, earlier actions still receive the subsequent team return; do not truncate team GAE at snake death.

**Caveat:** the actual game may end on elimination or have no fixed horizon. Implement its authoritative terminal conditions. Discounted terminal-only optimization is not identical to undiscounted win probability when episode lengths vary; report this limitation and tune gamma based on actual match lengths.

## 3. Actor architecture: strictly decentralized

Each active snake has `h_i in R^256`, unique to its lifetime. Its actor sees only `o_i(t)` and its own `h_i(t-1)`.

- Local grid: use actual visibility radius; optional ego-heading rotation if consistent with simulator and orientation semantics. Channels for visible walls, portal entrances, visible food, own body, visible allied/enemy bodies and heads, and a **visibility/unknown mask**. Never encode unseen as known-empty. Only include segment ages and portal destinations if observable.
- Scalar vector: own length, own heading/speed/sprint resource/split cooldown, own legal status and game time **only if observable**. No uncommunicated global/team score or allied positions.
- Message encoder: delivered `[K,4]` bytes + mask; embed bytes (e.g. 16 or 256 symbols per byte), aggregate with masked mean or small attention, include an explicit no-message embedding. Include sender/arrival direction only if exposed by simulator. Process only arrived messages; multiple messages in a step must be handled deterministically.
- Spatial encoder: Conv2d(C,32,3,pad=1) -> SiLU -> Conv2d(32,64,3,stride=2,pad=1) -> SiLU -> Conv2d(64,64,3,stride=2,pad=1) -> SiLU -> flatten -> Linear(…,256) -> SiLU. Scalar MLP to 64, message embedding to 64, concatenate, Linear(384,256).
- Single-layer GRUCell(256,256) for batched inference or GRU(256,256) for sequence training; keep weights equivalent. `h_new = GRU(encoder(o), h_old)`; actor heads use `h_new`.
- Action heads: steering (discrete 3-way **only if** simulator uses relative left/straight/right; otherwise continuous or simulator-specific), sprint Bernoulli, split Bernoulli plus any required parameters, transmit Bernoulli, and 4 categorical byte heads. Apply legal masks to action distributions; don't backprop through invalid actions. For a non-transmit action, do not include payload log-prob/entropy in joint action log-prob. Likewise mask parameter heads when their parent action is inactive.
- First communication baseline: 4 independent 16-way categorical heads encoded into four actual bytes in `[0,15]`, plus `transmit` gate; make vocabulary size configurable. The environment can use an external codec/authentication layer **only if it exists in actual gameplay**. Do not assume 4 bytes offer strong encryption/authentication, and do not pass the secret or decoded metadata to the actor unless the runtime itself exposes it. If hostile messages fail authentication, the actor receives only the same invalid/no-message indicator available at deployment.

Per-snake runtime:

```python
hidden: dict[SnakeKey, Tensor] = {}
for key, obs in active_observations.items():
    h_prev = hidden.get(key, zeros(256))
    action, h_new, logp = policy.sample(obs, h_prev)
    hidden[key] = h_new.detach()
# after stepping: create zero hidden for newborn keys; delete dead keys
```

In production batch all active snakes and scatter/gather by stable `SnakeKey`. Keep inference states detached from autograd; store sequence-start states and observations for training. **No inheritance of parent's h or map knowledge** on split.

## 4. Critic and multi-agent PPO

Use a training-only team critic `V_phi(full_state, player_id)` with global map CNN and permutation-invariant pooling/attention over a variable number of snakes, including owner/team embeddings and lengths. It outputs one scalar per player per timestep; zero-sum symmetry may be exploited but is not mandatory. Use separate actor and critic encoders initially to avoid privileged-feature leakage. Critic input is not a policy input.

Recurrent PPO: per active snake store observation, pre-step hidden state, action and active-head masks, old joint log-prob, owner, `SnakeKey`, policy version, plus team reward, critic value, match termination/truncation and next critic state per arena timestep. Joint log-prob is the **sum of log-probs of active action heads**; entropy similarly handles conditional heads. Recompute policy log-probs using the same conditional action-head structure at update time.

Compute team GAE once per player/arena across **arena time**, not independently per snake lifetime:

```
delta_t = r_team[t] + gamma*(1-match_terminal[t])*V[t+1] - V[t]
A_t = delta_t + gamma*gae_lambda*(1-match_terminal[t])*A[t+1]
```

For true terminal use V[next]=0. For time-limit truncation bootstrap from critic state before reset. Assign the team advantage `A_{arena,player,t}` to every allied snake that acted at that time. A snake's actor sequence ends on its death, but its earlier actions retain advantages based on the *full team future*. A newborn's sequence begins at zero h. A parent continues its own sequence across splitting. Normalize advantages over sampled active snake-steps for the actor. Normalize actor loss by number of active snake-steps, not raw sum across snake count. This avoids a trivial gradient-scale dependence on splitting, **but does not guarantee perfect credit assignment or prevent strategic incentives from changing with agent count**. Value loss is computed **once per team-timestep**, not duplicated once for every active snake.

PPO objective for sampled snake-steps:

```
ratio = exp(new_joint_logp - old_joint_logp)
actor_loss = -mean(min(ratio*A, clamp(ratio,1-eps,1+eps)*A))
value_loss = mean((V_pred - return_target)**2)  # team-timestep samples
loss = actor_loss + c_v*value_loss - c_e*mean(active_head_entropy)
```

Use clipped value loss optionally; gradient clipping (e.g. norm 0.5). For recurrent training, sample **contiguous per-snake sequences**, never shuffled individual timesteps; use stored sequence-start hidden state and optionally 16–32 burn-in steps with no loss. Prevent sequences crossing lifetime boundaries. If the sequence starts mid-lifetime, replay preceding observations for burn-in where available, otherwise use stored rollout hidden state and document that it was produced by the behavior policy. Keep PPO on-policy; discard old rollout after update, and tag policy versions. Avoid using an opponent's frozen policy actions in learner actor loss (they may still appear in critic state and environment trajectories).

### Parameter sharing and team assignment

A single learner policy may control both player sides in symmetric self-play, but collect learner actor gradients only from sides explicitly assigned to the current learner version. Frozen historical opponents are inference-only. If both sides are current learner, include both sides' transitions with their own signed team rewards. Opponents may also split and have their own private states.

## 5. Message delivery and adversarial channel

Implement message propagation entirely in the simulator: a transmitted payload is exactly four bytes and travels forward according to the engine's collision, range, portal, speed, lifetime, and interception rules. Never invent a direct ally broadcast. Both allied and enemy snakes may encounter/intercept messages according to actual rules. Preserve exact arrival timing. Unit tests must verify that a message is invisible before delivery, including across split and death events.

Keep cryptography/protocol framing **outside the policy**. If the real runtime provides a shared secret and authenticated decoding, implement an interchangeable `MessageCodec.encode(symbols)->bytes[4]` and `decode(bytes[4])->symbols|INVALID` with tests. Do not claim confidentiality/authentication without a specified threat model, nonce/counter and replay design; a 32-bit total frame is severely constrained. The training simulator must match actual deployed decoding behavior. If no verified codec exists, train raw bytes and treat hostile messages as untrusted observations. Avoid hard-coded assumptions that all received messages are friendly.

## 6. Rollout and training orchestration

1. Create `N_arenas` independent seeded arenas; each arena has two players and dynamically many snakes.
2. Assign one current learner side and one sampled opponent (current, historical, or scripted). Track policy ID per side. For same-policy self-play both sides can be learner.
3. At each step, gather all active snake observations across arenas; batch actor inference by policy ID, using each snake's private hidden state. Build action dicts; step arenas simultaneously.
4. Record team reward **once per player per arena timestep** and snake action transitions only for learner-controlled snakes. Keep team critic state/value at each timestep and bootstrap value at rollout boundary. Maintain opponent recurrent states separately and never train from frozen-opponent action log-probs.
5. Process events: parent retains hidden state; newborn gets zero state; dead snake's actor sequence closes; match reset clears every lifetime state for that arena. Never reuse keys.
6. After `T` arena steps, compute per-team GAE, join to learner snake-steps by `(arena_id,player_id,t)`, assemble contiguous recurrent minibatches and independent team critic minibatches. Run PPO for `K` epochs; discard rollout.
7. Periodically freeze checkpoint into opponent pool, run fixed-seed evaluation against frozen policies and scripted baselines, save best-by-held-out-evaluation and latest checkpoints separately. Do not use training opponents alone as the evaluation set.

### Suggested starting configuration (tune, do not treat as guaranteed optimal)

```yaml
seed: 42
arenas: 1024
rollout_steps: 128
actor:
  hidden_size: 256
  grid_embedding: 256
  message_vocab_per_byte: 16
  max_received_per_step: 8  # handle overflow deterministically; match simulator
ppo:
  epochs: 3
  sequence_length: 64
  burn_in: 16
  clip_epsilon: 0.2
  gamma: 0.999
  gae_lambda: 0.95
  learning_rate: 0.0003
  value_coef: 0.5
  entropy_coef: 0.01
  grad_clip_norm: 0.5
reward:
  terminal_win: 1.0
  terminal_loss: -1.0
  shaping_enabled: false
  shaping_lambda: 0.01
opponents:
  current: 0.5
  historical: 0.3
  scripted: 0.2
```

`gamma` must be reconsidered after measuring match lengths: `0.999^1000≈0.368`; long matches may require a value nearer 1 or a different temporal abstraction. Measure throughput and memory before scaling to thousands of arenas. Do not assume the maximum received-message count is 8 if actual engine behavior differs; expose truncation/overflow as explicit policy and log it.

## 7. Curriculum and ablations

Milestones, in order:

1. **Environment + observation legality:** deterministic tests, strict visibility and message-delivery tests, authoritative scoring and tiebreak tests.
2. **Single-snake navigation:** no split or communication; train steering, food, walls, portals and sprint if implemented. Use a curriculum-specific growth objective if needed, then switch to actual PvP objective.
3. **Two-player competition:** one snake per side, target vision, recurrent PPO, terminal outcome, scripted and self-play opponents.
4. **Splitting:** enable dynamic agents; team GAE; zero-initialized child; evaluate whether split improves final longest length/win rate.
5. **Communication:** enable physical messages and codec if real; train sender/receiver jointly, compare with messages disabled, shuffled and delayed; no direct communication bonus initially.
6. **Population self-play and generalization:** unseen maps, portal layouts, spawn positions, lengths and opponent snapshots.

Ablations: feedforward vs GRU; recurrent IPPO local critic vs CTDE team critic; terminal-only vs potential shaping; split on/off; messages on/off; historical pool on/off. Evaluate against frozen opponents with equal seeds and report win/draw/loss, primary length margin, total-length tiebreak outcomes, snake counts, split frequency, sprint costs, collision causes, portal use, delivered/authenticated message counts and wall-clock samples/sec. Do not infer communication usefulness from number of transmissions alone.

## 8. Repository layout (adapt to existing code)

```
snake_rl/
  config.py
  env/adapter.py
  env/observations.py
  env/scoring.py
  env/message_codec.py
  model/encoder.py
  model/actor.py
  model/critic.py
  model/distributions.py
  rollout/active_agents.py
  rollout/buffer.py
  algo/gae.py
  algo/recurrent_ppo.py
  training/self_play.py
  training/train.py
  evaluation/evaluate.py
  tests/test_scoring.py
  tests/test_visibility.py
  tests/test_split_memory.py
  tests/test_messages.py
  tests/test_rollout_alignment.py
  tests/test_gae.py
  tests/test_ppo.py
```

Use PyTorch, NumPy, dataclasses and pytest unless the repository already establishes another stack. Type annotate interfaces; keep simulator and policy decoupled. Add deterministic seeding for simulator, action sampling and opponent selection. Checkpoint model/optimizer/config/RNG state/opponent-pool metadata; record policy version and environment ruleset hash.

## 9. Acceptance tests (must pass before claiming completion)

- **Lexicographic score:** `[100,20]` beats `[99,80]`; `[100,20]` beats `[100,19]`; identical `(max,total)` is a draw. Splitting `[100]` into `[50,50]` reduces primary score from 100 to 50 even though total is unchanged.
- **Visibility:** modifying an unseen enemy, food item or portal exit cannot change the actor input at that instant, except through prior private memory or a legitimately delivered message. Critic input may change.
- **Split memory:** child's hidden state is exactly zero and child has new lifetime ID; parent's hidden state is unchanged by the split event itself; child receives no parent map/observations. Other snakes' states are unchanged.
- **Message physics:** no receive before actual delivery; enemies can intercept if engine permits; payload exactly four bytes; invalid codec frames do not magically become friendly information.
- **Independent inference:** permuting other snakes' private observations/hidden states cannot affect a target snake's action distribution while its own input and state remain fixed. Shared weights do not imply shared runtime state.
- **Trajectory alignment:** each stored action log-prob corresponds to the same pre-action observation, legal masks, recurrent state, and policy version. No stale hidden state survives death or match reset.
- **GAE:** a snake's death does not reset the player's GAE; true match terminal does. Truncation bootstraps correctly. Team critic value loss is counted once per team-timestep, not once per active snake.
- **Conditional actions:** when transmit=False, payload log-prob contributes zero; when split is illegal, it cannot be sampled. PPO old/new log-probs use identical masks and head conditions.
- **PPO smoke test:** short deterministic simulation produces finite loss/gradients and changes policy parameters; evaluation actor runs without full-state critic inputs; save/load reproduces deterministic inference with the same inputs and hidden states.
- **Reproducibility:** fixed seed, same config, same single-process execution produces identical short rollouts, modulo documented nondeterministic GPU operations.

## 10. Deliverables and working method for Claude

1. Inspect repository and write a short mapping from actual simulator mechanics to this spec, including unresolved rule questions. Do not change game semantics to fit this document.
2. Implement the adapter, score/reward, visibility and message interfaces, plus tests.
3. Implement batched actor and private-state lifecycle, then centralized critic, team GAE, recurrent PPO, and opponent pool in that order.
4. Provide a minimal executable training command, evaluation command, checkpoint format, example config, and README.
5. Run tests and a short end-to-end training smoke test; report actual results, any unsupported mechanics, and remaining TODOs. Do not claim successful training or benchmark results without running them.

### Open mechanics to resolve from existing code or user

- Exact movement/action space and simultaneous collision ordering.
- Whether split snakes can merge, transfer food/length, or eat one another; whether a player can control an unlimited number.
- Sprint resource/cost and speed; whether splitting consumes length or has a cooldown.
- Exact food/growth mechanics, portal teleport/visibility, map dimensions and vision radius.
- Match termination, time limit, death/respawn, and whether the secondary score really is sum of living allied lengths.
- Message emission range/speed/lifetime, collision/interception, receive metadata, per-step capacity, and real codec/authentication availability.
- Whether snakes can observe their own player ID, allied identity, score, remaining time, and initial parent-child relationship. **Do not add these fields to actor inputs without engine support.**

**Bottom line:** shared neural weights, private snake memories, zero-memory children, strictly physical 4-byte communication, training-only privileged critic, authoritative lexicographic team reward, and correct team-level credit assignment across snake deaths and splits.
