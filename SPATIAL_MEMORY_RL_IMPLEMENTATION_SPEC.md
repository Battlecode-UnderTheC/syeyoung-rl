# Spatial-memory recurrent PPO for Dragons

Status: implementation specification, not an implementation or a trained result.
Scope: replace the current local-observation actor and its training contracts.
The simulator's rules remain authoritative. Strategies must emerge from team-outcome RL; no tactical teaching curriculum.
This document supersedes conflicting
model, observation, action, memory and rollout prescriptions in
`SNAKE_MARL_IMPLEMENTATION_SPEC.md`; retain unaffected simulator validation work.
`COMPETITIVE_RL_DESIGN.md` supplies the strategic analysis and literature review.

## 1. Decision and objective

Implement a decentralized actor with:

1. An exact, private, coordinate-indexed record of observations.
2. A small learned spatial memory with sparse observation-addressed writes.
3. A multiscale spatial reader with recurrent internal computation.
4. A tactical GRU whose state survives the parent's split.
5. Conditional move/sprint/split heads, trained from team outcomes with PPO.
6. A separate privileged critic and a historical-opponent league.

All strategic decisions are learned. There is no A*, frontier selector, scripted
hunter, prescribed leader role, hand-selected target, or privileged actor map.
Indexing observations by their supplied coordinates is input processing, not a
navigation policy. Model capacity is an initial design choice, not a claim that
this architecture is empirically the best. Promotion requires the evaluations
in section 18, including the actual judge budget.

Optimize expected final team reward: win +1, loss -1, draw 0. Read the engine's
result: elimination precedes longest living dragon, which precedes total living
length. Never substitute total food, total biomass or individual survival.

## 2. Emergent strategy, not prescribed tactics

The policy must have enough information and action expressiveness to discover
strategies through game outcomes. Exploration, harvesting, blocking, sacrifice,
splitting and communication are possible behaviors, not labeled subskills.

Do not provide:

* Demonstrations, scripts or scenario rewards teaching split-then-sprint.
* Leader, attacker, scout or replacement-contender role assignments.
* Parent-to-child intent or memory transfer absent from the real protocol.
* A special attack target, head/tail tactical query, or retained sprint-budget
  category chosen because a human anticipates a particular combination.
* A promotion requirement that the trained policy use a named human strategy.

Use natural game observations, generic spatial/history representations, the
actual action parameters, and terminal team reward. Maps and initial conditions
may vary in difficulty, but action combinations must be discovered by RL. A
mechanical engine test may execute split then movement to verify semantics;
that test must not enter the learner's training data or supply its rewards.

Splitting and sprinting expose a potentially strong combination. This motivates
not accidentally excluding large splits, suicidal moves or memory persistence.
It does not justify teaching that combination. A model that wins more without
using it should be preferred over one that imitates our tactical expectations.

## 3. Non-negotiable information and lifecycle boundaries

Actor inputs may depend only on spawn metadata, its actual observations, its
own issued actions, and messages delivered by the engine. A process owns one
lifetime's state. Shared weights do not imply shared dynamic state.

Use a lifetime key `(match_id, team, dragon_id, generation)`; generation can be
omitted only if simulator IDs provably never repeat within a match. Do not index
fixed-size storage by an unbounded lifetime ID. The unit limit bounds living
dragons, not necessarily all IDs ever created.

| Event | Factual map | Learned map | Tactical GRU | Private body history |
|---|---|---|---|---|
| Initial spawn | Empty | Zero | Zero | Length-sized unknown body with observed head |
| Normal own turn | Update observed facts | Write observed cells | Advance once | Reconcile from own history/observation |
| Parent split | Retain; stale records stay labeled stale | Retain | Retain, including intent | Truncate parent's body order |
| Child spawn | Empty | Zero | Zero | Initialize independently |
| Other dragon dies out of sight | No oracle update | No oracle update | No oracle update | No change |
| Own death | Delete after trajectory closure | Delete | Delete | Delete |
| New match | Clear all lifetimes | Clear | Clear | Clear |

Padded training steps do not advance any state. Never reset a parent's GRU
because length decreases. Do not feed rewards, privileged termination reasons
of unseen dragons, full-map spawn files, or critic features into the actor.

## 4. Observation and state schema

Version every schema. New checkpoints use `schema_version: dragons-spatial-v1`.
Old checkpoints are incompatible and must fail with an explicit error rather
than silently reshaping weights or treating missing coordinates as zero.

### 4.1 Raw observation contract

Preserve these fields before neural encoding:

* Spawn: own ID/team, width, height, unit limit.
* Turn: round, own heading/length, own team unit count, delivered sonar words.
* All 49 tile coordinates, pearl presence and exact reported countdown.
* All visible segments: team, ID, coordinate, head flag, reported facing.
* All observed edges with empty/kelp/portal type and actual portal identifier.

Use supplied absolute coordinates to address memory. Validate their consistency
with dimensions and wrapping. Do not discard portal identities or collapse heads
into generic occupancy. Body facing points toward the head; it is not velocity.

### 4.2 Factual memory: structured storage

Allocate per actual board size, at most 64x64, with no time-growing dense cache.
Store integers/booleans in their natural types; do not store IDs as normalized
image brightness or quantize timestamps through uint8 observation compression.

Per tile:

```
seen: bool
last_seen_round: int16             # -1 unknown; rounds 0..499
pearl_at_last_seen: bool
countdown_at_last_seen: int32      # protocol sentinel preserved
previous_pearl: bool
previous_countdown: int32
previous_seen_round: int16         # -1 if only one observation
occupant_kind: uint8               # empty/self head/self body/ally head/body/enemy head/body
occupant_id: int32                 # -1 for none/unknown, disambiguated by kind/seen
occupant_facing: uint8
visit_count: uint16
last_visit_round: int16            # -1 unknown
```

Store canonical north/west edges with `(known, type, portal_id)` and derive
south/east by indexing. This avoids inconsistent duplicate edge copies at the
toroidal seam. Resolve portal endpoint equality by identifiers, not a learned
numeric ordering of IDs. A partner outside accumulated observations is unknown.

Updates overwrite only observed facts. Compute ages on read. Do not mark an old
enemy position as currently occupied, erase unseen enemies on death callbacks,
or treat a predicted pearl respawn as a confirmed pearl. Store one preceding
observation to expose changes/intervals without pretending to know an unseen
respawn-gap distribution. Repeated delivery of the same turn must be idempotent.

A confirmed visited location is an actual observed head position or a privately
reconstructable intermediate head position. Do not mark speculative portal
landings visited. Missing visit history must not masquerade as a confirmed visit.

### 4.3 Materialized factual feature map: [B, 42, H, W]

This is derived from structured storage at the current round, not saved as the
canonical database. Unknown scalar values use zero with their explicit flags.

| Indices | Count | Meaning |
|---|---:|---|
| 0 | 1 | Ever observed |
| 1 | 1 | Visible in this turn |
| 2 | 1 | Clipped age since last observation / 500 |
| 3 | 1 | Last observed pearl presence |
| 4 | 1 | `1/(1+countdown)` for known active countdown; otherwise 0 |
| 5 | 1 | Reported never-spawns flag |
| 6..12 | 7 | Last occupancy kind one-hot, including observed empty |
| 13..16 | 4 | Last occupant facing N/E/S/W; zero if no occupant |
| 17..20 | 4 | Directional edge-known flags |
| 21..24 | 4 | Directional kelp flags |
| 25..28 | 4 | Directional portal flags |
| 29 | 1 | `log1p(visit_count)/log(501)` clipped to 1 |
| 30 | 1 | Clipped age since confirmed visit / 500 |
| 31 | 1 | Ever visited |
| 32 | 1 | Previous observation exists |
| 33 | 1 | Pearl presence at preceding observation |
| 34 | 1 | Transformed preceding countdown |
| 35 | 1 | Interval between two last observations / 500 |
| 36 | 1 | Occupant ID earlier than self, for recorded nonself occupant |
| 37 | 1 | Occupant ID later than self, for recorded nonself occupant |
| 38 | 1 | Privately known current own-body position |
| 39 | 1 | Known own-body index / max(length-1,1) |
| 40 | 1 | Privately known current own tail |
| 41 | 1 | Current own head |

A separate board-valid mask distinguishes padded storage from unknown in-board
cells. Same-ID grouping and portal links are side tables, not extra numeric-ID
channels. Keep a generated schema registry shared by Python/C++ tests.

### 4.4 Own-body order and trail

Keep a private head-first deque of known coordinates or UNKNOWN, initialized
from length and the first observation. Never fill it from the global simulator.
Also keep a ring of the last 32 completed commands with old/new head, headings,
rounds, lengths and whether intermediate coordinates are known.

On the next surviving observation, a prior MOVE command completed. Prepend its
k intermediate head coordinates in reverse temporal order, then truncate to the
new observed length. Derive intermediate coordinates only through known edges
and known portal pairs; otherwise insert UNKNOWN. The observed final head is
always known. This records growth/sprint tail removal without inventing which
unseen intermediate tile contained food. Check consistency with visible own
segments; invalidate contradicted hypotheses rather than repairing from truth.

On parent SPLIT n, retain the first L-n deque entries. A newborn does not receive
the suffix. Map observations can later resolve body order when linkage is
unambiguous, but that inference is optional in v1. Expose how much order is known.
If the tail is unknown, the model receives tail_unknown=true and must reason
under uncertainty; do not query the real tail to make the attack easier.

### 4.5 Status vector and history embedding

Define a named schema rather than anonymous magic offsets. Include:

* Linear and log own length, normalized by board area and log(1+area).
* Round/499, remaining rounds/500, width/64, height/64, unit count/limit.
* Four heading bits; legal-to-split flag based on length/count/limit.
* First-own-turn flag, number of own completed turns / 500.
* Known-body fraction and known-tail flag.
* Wrapped head sin/cos coordinates; actual dimensions retained separately.
* Previous command type, split fraction/retained length, movement direction
  tokens and count; explicit no-previous-command flag.

Encode the 32-command history with a small masked temporal convolution to 32
features. No observation of successful children or kills is inferred from an
issued action alone. The reference configuration disables sonar; the enabled experiment adds the
32-dimensional inbox embedding specified in section 6.5. Raw inboxes stay in
the protocol adapter in both cases.

## 5. Actor architecture and tensor interfaces

Use explicit state objects. Do not hide a 64x64 map inside a flat `hidden=256`
argument or serialize a full map at every transition.

```
ActorState:
    facts: FactualMemory                # runtime-side, deterministic
    spatial: float32[16,H,W]            # persistent learned state
    tactical: float32[128]              # persistent GRU state
    body_order: private deque
    command_history: private ring
    last_processed_turn: turn key

PreparedObservation:
    local: float32[42,7,7]              # current local factual crop
    facts: float32[42,H,W]
    board_valid: bool[H,W]
    visible_xy: int64[49,2]
    visible_previous_features: float32[49,42]
    visible_age_before_write: float32[49]
    status/history tensors
    observed_portal_links: int64[P,2,2]
    observed_entity grouping tables
```

Batch by dimensions where practical. Mixed-size batching must implement per-row
wrapping and masked pooling; periodic padding around a padded 64x64 tensor is
incorrect for a 15x19 board. Local crop padding is zero/unknown, not circular.

### 5.1 Local encoder and sparse learned writes

Encode the north-up 7x7 crop with two 3x3 convolutions, 32 channels, SiLU.
Retain [49,32] per-cell features. Flatten/project to a local vector of 128.

For each visible coordinate u:

```
w_u = concat(local_feature_u, previous_factual_features_u, prewrite_age_u,
             status_projection_16)
M_new[u] = GRUCell(write_projection(w_u), M_old[u])  # hidden width 16
```

Unobserved learned cells are unchanged. Factual observations are preserved on a
separate path regardless of learned gates. Pass the pre-overwrite age/features
explicitly, so the writer can distinguish a revisit from continuous visibility.
No backward pass through deterministic raw storage is needed; gradients flow
through learned writes inside the training unroll and through all neural reads.

Writer input, old memory and output use absolute north-up orientation. Duplicate
coordinates must be coalesced before scatter, although valid 7x7 windows on
boards of at least 7x7 have unique tile coordinates.

### 5.2 Multiscale spatial computation

Input is factual features concatenated with learned memory: 58 channels.

Initial implementation:

1. Fine: 1x1 projection to 16, then depthwise 3x3 + pointwise 1x1 residual block.
2. Half: masked area downsample by 2, project to 24, separable residual block.
3. Coarse: downsample by 2 again, project to 32.
4. Coarse computation: two shared ConvGRU iterations, width 32, using separable
   convolution gates, conditioned on projected previous tactical state and status.
5. Upsample coarse to half and fine; concatenate with skip features; project
   each back to its target width. Preserve fine directional edge information.

Coarse ConvGRU state is **working memory within a decision**, initialized from
the current coarse features. It is not a third persistent map. Persistent memory
is only M and h. Increasing iterations from 2 to 4/8 is an ablation and changes
runtime even though parameter count stays similar.

Handle odd dimensions without adding false adjacency. All resampling has
explicit actual-size/mask metadata. Kelp features influence learned route
reasoning; they do not prevent the neural network from exchanging information.
For confirmed portal pairs, exchange learned endpoint messages before/after the
coarse block, retaining orientation and fine endpoint indices. This is a neural
relation operation, not a pathfinder. Unpaired portals get an unknown-partner flag.

A few local iterations do not guarantee long-range planning. The multiscale
branch and global reads provide broad context; long-detour evaluation decides
whether extra depth or the value-propagation alternative is justified.

### 5.3 Reads and tactical controller

Read two kinds of context:

* Local: a 7x7 crop around the current head from fine features, encoded to 128.
* Global: four 32-dimensional cross-attention reads over coarse spatial tokens.
  Each token includes wrapped head-relative sin/cos coordinates, observed
  fraction and age summary. Query = linear(local128, old_h128, status64).

All global queries are learned and exchangeable at initialization. Do not name
or supervise them as tail, target, food or exploration heads. They may specialize
if that improves returns. No explicit tail-centered crop is supplied.

Unknown but in-board tokens remain available for exploration reasoning. Only
padding is excluded. Keep max/mean summaries of occupancy and edge evidence in
the multiscale path so narrow threats are not erased solely by average pooling.

A shared entity MLP pools currently observed segments by ID, with counts, head
presence/facing, relative locations and earlier/later-ID flags. Attention-pool
entity features to 64. The observed segment count is a lower bound, not the true
length. No arbitrary truncation of an input line may hide a visible head; the
7x7 occupancy bound already limits this set to at most 49 occupied tiles.

Fuse local128 + map_local128 + global128 + entity64 + status64 + history32;
When sonar is enabled, append inbox32. Apply Linear -> SiLU -> 128, then
GRU(128,128). The resulting h_new drives all action
heads. A single shared actor learns its behavior from these inputs and outcomes;
no semantic role is assigned to any lifetime.

## 6. Action distribution

### 6.1 Top-level choice

A two-way categorical chooses MOVE or SPLIT. SPLIT is masked only when its
length/unit-count constraints fail. MOVE always permits at least one direction;
a syntactically valid fatal move remains an action.

### 6.2 MOVE decoder

Use an autoregressive GRUCell(64) initialized from h_new. Each token is one of
N/E/S/W/STOP in absolute orientation. Absolute directions avoid assuming that
geometric reversing is always equivalent after a portal. The local encoder and
scalar heading provide the orientation information.

* First token cannot be STOP.
* Maximum emitted directions: K=8 in the initial candidate. K is part of the
  checkpoint, not an invisible runtime setting. K=3/16 are separate ablations.
* After K directions, terminate deterministically; there is no sampled STOP term.
* Decoder conditions on h_new, previous token and known/unknown prefix features.
* Submit the whole sequence in one command. Never obtain an intermediate
  observation or choose the next token after the simulator has moved.

The prefix feature builder may simulate only the actor's private information to
track known position, cost/growth, edge type and encountered recorded occupancy.
Unknown after an unpaired portal is explicit. It cannot inspect the hidden board.
Never use stale occupancy as an exact fact for masking. Do not turn a heuristic
safety score into a hard mask.

When a prefix is known to be alive and length<=2, mask another paid step. If
resource state is uncertain, leave the uncertain continuation available and flag
it rather than consulting truth. After a known fatal collision, force STOP to
canonicalize commands that the engine would truncate anyway. Fatal direction
tokens remain selectable, including enemy heads, self/body collisions and kelp.
Do not introduce curriculum safety overrides that decide which sacrifices
are strategically acceptable.

K=8 is an explicit restriction: the engine permits longer commands. Test K=3,
8 and 16 and retain a longer cap only if its win-rate benefit justifies cost.

### 6.3 SPLIT decoder

Score detached size n in [2,L-2] with a shared MLP for every legal candidate:

```
score(n) = MLP(h_new, n/L, (L-n)/L, log1p(n), log1p(L-n))
```

Normalize the log inputs by log(1+board_area). Emit SPLIT n directly. These are
generic action-parameter features; there are no hand-selected sprint-budget
classes or privileged newborn survival features. All legal split sizes remain
available. Enumerate using actual length and profile large states. A restricted
candidate catalogue is not v1: if later necessary, it is a separately versioned
and evaluated approximation.

### 6.4 Probability, entropy and teacher forcing

For MOVE, joint log probability is log p(MOVE) plus all sampled direction and
STOP token log probabilities. For SPLIT it is log p(SPLIT)+log p(n|SPLIT).
Inactive branches contribute zero. Store complete tokens, token validity and
all conditional masks or enough lossless information to reproduce them.

PPO recomputes log probabilities by teacher forcing the stored command. Use one
ratio for the whole command, not an independently clipped ratio per token.
Sampling, export and NumPy inference use exactly the same decoder order/masks.

For entropy, v1 uses top-level entropy plus the sum of conditional entropies
along stored active prefixes, scaled by fixed 1/K for movement and fixed 1 for
split. This is an explicitly documented exploration surrogate, not exact joint
entropy of the current autoregressive policy. Monitor mean command length and
branch frequencies for regularizer-induced bias. Do not sum inactive payloads
or unexecuted deterministic termination tokens.

### 6.5 Sonar: learned discrete communication experiment

Implement sonar behind a model/config flag and compare training from the same
initialization distribution with sonar enabled versus disabled. Do not assume
communication helps; it is a directional, intercepted, delayed channel with no
sender identity in the reported payloads, not a friendly broadcast.

Initial output: a 17-way categorical conditioned on h_new and an embedding of
the already selected physical command. Choices are SILENT and 16 message codes.
Code j emits the uint32 value j for j in [0,15]. SILENT emits no SONAR command;
it is distinct from emitting zero. Codes have no human-assigned meanings. This
is an explicit small-vocabulary restriction, not use of all 32 payload bits.
If learning/evaluation justify it, compare 256 codes or four conditional byte
heads; do not predefine messages as coordinates, roles or attack instructions.

Input: encode every actually received uint32 as four raw bytes, with a shared
256-entry byte embedding plus byte-position embeddings, then an MLP per word.
Pool messages using a streaming mean and max, plus count/log-count and an
explicit empty flag. The actor receives no sender/team metadata unless the
protocol actually reports it. Streaming avoids a hard eight-message truncation;
implementation must check the real protocol's valid input limits. Enemy words
outside our transmit vocabulary still receive a defined representation.

Fuse the pooled 32-dimensional inbox representation into actor context before
the tactical GRU. The outgoing distribution conditions on both that state and
the physical command because delivery geometry is post-action. A command that
kills its sender is resolved by the real engine; do not invent delivery.

Joint log probability adds the chosen 17-way log probability. No inactive
payload head exists in this version. Include its categorical entropy with a
separate tunable coefficient; do not give a transmit, delivery, agreement or
communication-success reward. PPO credit comes only from eventual team outcome,
including after sender death. Gradients do not pass directly through delivery
or sampled messages to another process.

Train in the actual simulator, including kelp blocking, portals, toroidal rays,
first-segment interception, self-reception and next-turn delivery. Do not inject
an ally inbox or inherit messages at child birth. Test a sender/receiver fixture
only for physics, not to teach a codebook.

Evaluate complete games for enabled/disabled learned policies. On a trained
communication policy also test suppressed, shuffled, and hostile messages to
diagnose whether it uses useful information; these interventions induce a shift
and do not replace matched-training comparisons. The deployment default remains
sonar off unless enabled training produces a reproducible win improvement within
the resource budget. No hard-coded emergency or coordination protocol.

## 7. Centralized critic

Use a separately parameterized training-only critic. Consume exact native
simulator state at the event boundary, round/turn phase and next acting ID.
Do not describe the current Python union-of-windows tracker as exact full state;
it is an approximate fallback requiring a distinct experiment label.

Inputs:

* Full board pearls/countdowns, directional edges and portal relations.
* Per-dragon head, heading, length, team, ID-order information and alive flag.
* Per-team max length, total length, unit count and remaining rounds.
* Detached summaries of recurrent actor state/known-map coverage for active
  dragons, plus opponent policy ID. History matters to future recurrent actions.

A map encoder (32/64 channels), per-entity MLP, separate team mean/max/count
pooling and MLP produce two team values. Evaluate from each team's perspective
with shared weights; optional antisymmetric output is valid for the win-only
zero-sum objective. No critic outputs feed the actor at execution.

State-value conditioning may still approximate unrepresented private histories;
do not claim the physical state alone makes recurrent-policy returns Markov.
Use complete-match returns as a diagnostic reference. Do not attempt COMA or
learned reward redistribution until simpler credit assignment is evaluated.

## 8. Sequential event timeline and advantages

Use one **global action-event timeline per match**, not independent lifetime
returns and not a simultaneous batch of stale observations.

Event e is the instant before the next living dragon acts, after any automatic
pearl tick. Store state S_e and acting lifetime. Execute that action, then advance
through dead-ID skips and any complete-round adjudication/next-round tick until
another living dragon can act, or the match is terminal. This produces S_(e+1).
Newborns are scheduled by the actual engine. Results are never decided mid-round.

For each team p, record r_e^p=0 except for terminal official +1/-1/0. Critic
values and rewards exist once per team-event. Actor samples attach to their own
team/event. Frozen opponents contribute events/critic state, but never actor loss.

Default gamma_round=1. Define Δr_e as the number of round boundaries between
pre-action states (usually 0 or 1), and:

```
g_e = gamma_round ** Δr_e
l_e = lambda_round ** Δr_e
δ_e^p = r_e^p + (1-d_e) * g_e * V_old^p(S_(e+1)) - V_old^p(S_e)
A_e^p = δ_e^p + (1-d_e) * g_e * l_e * A_(e+1)^p
Rhat_e^p = A_e^p + V_old^p(S_e)
```

Terminal d_e=1 zeroes bootstrap regardless of Δr. No next physical state is
required for terminal values. Within a round l_e=1, so extra children do not
cause extra trace decay. This is an explicit adaptation of GAE to the sequential
game, to be tested against complete-return advantages; it is not an unmodified
MAPPO recipe. A collection cutoff bootstraps and stops the stored recursion but
is not an environment terminal. The final round 499 is a true terminal.

For gamma=1 and lambda_round=1 over a complete match, require exactly
A_e^p=G^p-V_old^p(S_e), including actions of dragons that died before the end.
This identity is the reference credit-assignment test. Dead actor trajectories
end for BPTT, but their actions retain event-indexed advantages from team future.

Default collection uses a frozen behavior version for complete matches, which
avoids rollout-bootstrap ambiguity and online state refresh during initial
implementation. Batch 32 concurrent arenas and finish in-flight matches before
updating; do not silently truncate long survivors to satisfy a step target.
Later streaming collection must explicitly bootstrap and refresh/version
surviving recurrent states after weight updates.

## 9. PPO and recurrent training

Initial configuration (tunable, not claimed optimal):

```yaml
schema: dragons-spatial-v1
actor:
  latent_channels: 16
  tactical_hidden: 128
  decoder_hidden: 64
  spatial_widths: [16, 24, 32]
  spatial_iterations: 2
  global_reads: 4
  read_width: 32
  history_turns: 32
  max_move_steps: 8
  sonar: false
rollout:
  arenas: 32
  complete_matches: true
  behavior_frozen_until_update: true
ppo:
  gamma_round: 1.0
  lambda_round: 0.99
  clip: 0.1
  epochs: 3
  learning_rate: 0.0003
  sequence_length: 64
  burn_in: 16
  sequences_per_microbatch: 8
  accumulation_microbatches: 8
  value_coef: 0.5
  entropy_surrogate_coef: 0.005
  grad_norm: 0.5
  target_kl: 0.02
auxiliary:
  coefficient: 0.05
  enabled_after_parity: true
```

Use contiguous per-lifetime sequences. Store state before the first burn-in
step; replay burn-in with current weights without loss/gradient, detach, then
unroll scored steps with gradients. Include full newborn prefixes without
artificially dropping the first 16 turns. Prefix observations overlap chunks,
but each valid actor decision contributes to scored loss exactly once per epoch.
Parent sequences cross splits; child sequences start from zero. Padding advances
neither factual nor neural state and contributes no loss/entropy/normalization.

PPO policy loss is the clipped joint-command surrogate. Sum valid actor terms
and normalize by a fixed batch count of learner team-rounds (with gradient
accumulation preserving the denominator). Extra genuine decisions contribute
extra terms, as in the shared-policy trajectory gradient; do not create extra
reward streams for extra dragons. Do not independently average each dragon's
loss and then add those averages. Report effective decision count and gradient
norm. An active-decision mean is a separately labeled weighting ablation.

Compute value loss over unique team-events; do not duplicate it for each token
or decoder candidate. Advantage normalization uses valid learner samples only
and fixed statistics for the entire update, not each microbatch. Clip gradients
after accumulation. Track joint KL, clip fraction, decoder lengths, split
fractions, explained variance and memory gradients. Early-stop remaining PPO
epochs when the update's monitored KL exceeds the configured threshold.

Learned state snapshots were generated by old weights. Burn-in mitigates but
does not eliminate this staleness. Factual memory is weight-independent and can
be rebuilt exactly. Ablate 128 scored steps and 32 burn-in if memory tests fail;
longer storage duration does not imply longer backpropagation credit.

## 10. Auxiliary objectives and reward discipline

Final competitive rewards are only team outcomes. No permanent personal death
penalty, food bonus, survival bonus, kill bounty or split bonus.

Initial auxiliary heads, trained without deployment-only extra inputs:

1. On a revisit, predict pearl presence from the **pre-update** memory and elapsed
   time; score against the arriving observation. Stratify positive/negative
   samples and ages so predicting empty everywhere does not dominate.
2. Predict the next observed enemy-head location relative to its last sighting,
   including a not-observed category. This predicts observation outcomes, not
   certain enemy death or true absence outside vision.
3. Predict next-own-turn length change/alive outcome from history and command,
   labeling censoring when the match ends before another own turn. These are
   behavior-policy outcomes, not proof of what all alternative actions do.

Auxiliary heads train the representation, not a new reward. Prevent label
leakage: forward predictions before consuming target observations, detach labels,
and test that targets cannot alter actor inference. Disable each loss separately
in ablations. Normalize loss magnitudes before combining; the stated coefficient
is only a starting point. Do not use prediction error from moving enemies as an
intrinsic reward for standing in familiar territory.

The reference training run uses terminal team reward throughout. Do not include
tactic-specific pretraining or coverage/food bonuses in the default specification.
Any future intrinsic reward or potential shaping is a separately proposed
experiment requiring an explicit objective and terminal-only comparison. Do not
hide raw score-delta shaping in native code.

## 11. Environment curriculum and self-play

Train on complete games generated from the actual rules. Vary board dimensions,
kelp density, food ranges, portal count, starting lengths and initial unit counts.
Start with cheaper small maps if needed and gradually broaden map sizes and
geometry, while keeping some games from the target distribution throughout.
Splitting, movement and sprinting remain real policy choices; do not prescribe
an order in which to use them. No tactical demonstrations or special attack
fixtures enter training. Reward remains terminal team outcome.

All genuine initial dragons and children start without inherited memories.
Random initial lengths expose the same model to different resources without
naming roles. Map generation must preserve legal non-overlapping bodies and
actual game constraints; no oracle-filled actor memories are allowed.

Initial opponent mixture: 50% recent snapshots and 50% diverse historical
snapshots. Before a history exists, use current-policy self-play. Keep a
nonzero sampling floor for older opponents. Later allocate a portion to RL
exploiters trained solely to win against frozen main checkpoints, not to carry
out a designated attack or exploit. Opponent assignment is fixed per episode.
Scripted agents can be evaluation baselines, not teacher policies.

Track matchup results and preserve opponents that reveal weaknesses; select
checkpoints on held-out win rate rather than training self-play wins. Do not
reward or reject a learner for failing to produce a strategy anticipated by a
human. Sonar-enabled and sonar-disabled models follow the same game distribution
and win objective, with their own legitimate private states.

## 12. Buffer and checkpoint contracts

### 12.1 Actor transition

```
match/lifetime/event/team/round IDs
behavior_policy_version, schema_hash, action_codec_version
lossless local observation delta (including coordinates/edges/IDs/timers)
status and previous command/history references
full sampled command, active token flags, conditional mask data
old joint log_probability
actor-sequence boundary flags and snapshot reference
reference to team-event advantage/return
```

Do not store privileged labels inside an actor observation object. Auxiliary
labels and critic states have separate buffers with separate interfaces.

### 12.2 Team-event record

Store pre-action critic state, old two-team values, two-team reward, next-state
reference, Δround, actual terminal/truncation flags and acting lifetime. A child
birth or individual death does not set match-terminal. Record automatic phase
transitions sufficient to reproduce next-event selection and pearl ticks.

### 12.3 Memory storage

Checkpoint each lifetime's raw facts, learned M, h, body deque and command ring
at chunk boundaries. Between checkpoints store lossless observation/action
updates. Maintain a snapshot at or before every burn-in prefix. Raw-map snapshots
may be compressed losslessly. Start learned states in float32; lower precision
requires replay/log-prob parity measurements and an explicit version.

Do not save dense maps at every turn: 16 float32 learned channels at 64x64 cost
256 KiB per lifetime checkpoint before facts. Training activations and spatial
maps are much larger than a 128-element hidden vector. Estimate peak host/device
memory using actual lifetimes/chunks, then scale arenas/microbatches.

Checkpoint weights, optimizer, config, schema/codec hashes, normalization stats,
RNG states, curriculum stage, opponent pool and evaluation manifest. Resume
mid-match only with complete actor/critic/simulator state and policy version;
otherwise resume at an explicit episode boundary. Distinguish latest and
best-held-out checkpoints.

## 13. Pure model and runtime APIs

Define conceptual interfaces before coding:

```
prepare(raw_turn, private_facts, body_history) -> prepared, fact_delta
encode_and_update(prepared, M_old, h_old) -> context, M_new, h_new
decode_sample(context, prepared, rng) -> command, joint_logp, trace
decode_evaluate(context, prepared, stored_command) -> joint_logp, entropy_proxy
critic(full_training_state, histories) -> values[2]
```

`encode_and_update` is pure with respect to caller state. A second call to score
an action/value must not advance memory twice. `decode_evaluate` never resamples.
`act` and teacher-forced `evaluate_sequence` share modules and mask definitions.
The deployed runtime commits the new private state once and logs the issued
command. Next observation reconciles execution, not a speculative simulator step.

TorchScript export exposes separate neural encoding and decoder evaluation/
sampling entry points as supported by its type system. Keep factual bookkeeping
and command serialization deterministic in the runtime. Export dimensions and
schema metadata with the module; do not reuse config.hidden as total state size.

## 14. Repository integration plan

This table describes future work; creating this document does not perform it.

| Location | Required change |
|---|---|
| `snake_rl/env/observation.py` | Preserve raw coordinates, IDs, countdowns and portal identities; materialize versioned channels |
| New `snake_rl/env/memory.py` | Factual map, canonical edges, private body/trail updates, snapshot/delta format |
| `snake_rl/env/actions.py` | Conditional codec, split retained/detached mapping, private prefix features/masks |
| New `snake_rl/model/spatial_memory.py` | Sparse learned writer, multiscale recurrence and local/global reads |
| `snake_rl/model/encoder.py` | Local/entity/status/history encoders |
| `snake_rl/model/actor.py` | Explicit M/h state and shared conditional decoder; remove flat-action assumption |
| `snake_rl/model/policy.py` | Per-lifetime lifecycle, exactly-once state commits, raw observation preparation |
| `snake_rl/model/critic.py` | Team-event critic with phase, IDs, counts and detached history summaries |
| `snake_rl/model/export.py` | Export same learned modules and codec metadata; separate encoding/evaluation APIs |
| `snake_rl/rollout/collector.py` | Lossless deltas, event joins, chunk snapshots, no per-turn dense map copies |
| `snake_rl/algo/gae.py` | Event GAE with per-round trace/discount factors, no death cutoff |
| `snake_rl/algo/ppo.py` | Joint-command ratios, valid burn-in, unique critic records and fixed loss denominator |
| `cpp/agent/encoder.*` and bindings | Match Python factual encoding/update/codec from visible information only |
| `cpp/train/train.cpp` | Exact event critic, sparse lifetime state, snapshots, new exported interfaces and full commands |
| `tools/export_for_native.py` | Versioned model/config metadata; explicit state tensors rather than one hidden size |
| `tools/play_replay.py` | New schema/runtime, proper per-match/per-lifetime reset, real engine evaluation |
| New deployment runtime | NumPy encoder/forward/decoder plus packaged weights and bot entry point |

Retain existing local-GRU model behind an explicit baseline configuration for
controlled comparisons. Do not silently mix its encoders/actions with new actors.
The documented native actor-value/per-dragon trainer cannot support splitting
correctly by merely swapping the neural network file.

Implementation order:

1. Schema and factual-memory reference, lifecycle tests, protocol fixtures.
2. Learned actor with sampling/evaluation parity and gradient tests.
3. Small NumPy/export parity and judge budget probe before scaling training.
4. Exact native event timeline/critic, team advantages, recurrent buffers/PPO.
5. Short full-loop optimization smoke test and official-engine evaluation.
6. Curriculum/league training and ablations, then deployment packaging.

Do not change unrelated user edits while performing this future work.

## 15. Deployment and resource budgets

Judge constraints from game.md: 100M CPU points per turn, 48 MB memory, 10-second
wall limit, 10 KB output and 4 MB zipped submission. NumPy is available; PyTorch
is training-only. Profile the full process, including parsing, raw/learned maps,
body history, temporary arrays, model weights, attention and variable decoding.

Aim initially for <=750,000 deployed parameters (<=3 MB float32 raw weights),
leaving archive room for runtime code. This is a design budget, not a counted
parameter result. Use separable spatial operations and bounded reads; allocate
per real board dimensions and reuse inference scratch arrays. Read weights with
frombuffer. Any reduced-precision export requires long-sequence parity checks.

Gate: no candidate is deployable until worst-case fixtures pass actual judge
CPU/memory/output limits. Measure 64x64 full history, high visible occupancy,
maximum tested sprint cap, large split candidate count and repeated turns.
Reported host milliseconds or parameter bytes alone cannot establish compliance.

If over budget, first reduce spatial iterations/width and profile, then consider
distilling the same history-aware actor to a smaller student. Never respond by
dropping map memory silently. Reevaluate full-match performance after changes.
The communication experiment must also pass this gate. Large attention/world-model
additions are deferred.

## 16. Correctness acceptance tests

All are required before long training; deterministic fixtures unless stated.

### Observation, privacy and memory

* Same raw observation/action history produces identical factual maps in Python
  and C++, including toroidal seams, odd rectangular sizes and portal IDs.
* Modifying unseen simulator terrain/enemies/food cannot alter a target actor's
  prepared tensors or command masks. Privileged critic tensors may change.
* Known empty differs from unknown. Last-seen enemies/pearls retain age and do
  not become current truth. Empty observation clears only the observed cell.
* All four headings map writes to identical absolute cells. Local and full-map
  padding behave differently as specified. Reprocessing a turn is idempotent.
* Known own-body deque updates correctly across growth, paid steps, known and
  unknown portals, and parent splits; unknown spawn segments stay unknown.
* Parent M/h/facts persist across split. Child M/h/facts are empty before its
  first observation. No parent target or tail map is copied to the child.
* Permuting other batch rows/lifetimes cannot change one actor's result.
* Local/global reads use only private memory and supplied coordinates; no
  dedicated target/tail query is populated from simulator truth.

### Actions and timing

* A length-40 parent selecting detached n=36 emits SPLIT 36 and retains four.
* A parent cannot split and move in the same turn. Newborn acts in birth round;
  parent may next sprint only on its next scheduled turn.
* Head collisions stay selectable and kill both regardless of length. Body
  collisions have a different outcome. Synthetic masks do not erase sacrifice.
* Three-step no-food sprint is affordable at length four; length three cannot
  pay its third step unless earlier food changes the budget.
* Food on an unaffordable destination cannot pay for entry. Prefix simulation
  consumes food once and never reads an unseen portal exit.
* Sampled-command log probability equals teacher-forced evaluation; inactive
  branches, padding and forced STOP contribute zero probability terms.
* Full decoder output is valid uppercase protocol text and respects output caps.

### Credit, recurrence and optimization

* Engine still executes later turns before complete-round adjudication.
* Team reward occurs once at actual terminal; child birth does not duplicate it.
* With gamma=lambda=1, event advantages equal G-V for every action, including
  the sacrificed parent. With lambda<1, decay counts rounds, not child count.
* Bootstrap cutoffs differ from terminal round 499. No GRU/death flag is reused
  as a match-terminal mask.
* Stepping vs contiguous unroll agrees for M, h, logits and commands, including
  a split and a cold-start child. Chunk reconstruction matches uninterrupted
  factual/learned memory with unchanged weights.
* Burn-in ends with detach, includes available prefixes, and does not discard
  newborn scored turns. Padded states remain unchanged.
* PPO updates reach spatial writer, reader, tactical GRU and active decoder
  parameters with finite gradients. Actor has no critic-input dependency.
* Microbatch accumulation matches a full-batch loss/gradient within tolerance.
* Save/load, TorchScript and NumPy agree across long observation sequences.
  Start with float32 logit/state tolerance 1e-5; diagnose rather than loosen
  tolerances blindly, and compare near-tied argmax cases separately.

### Compound-action mechanics fixture, not training data

Construct a legal prefix and scripted command sequence exercising a split,
newborn turn, later parent sprint and mutual head death. Verify memory isolation,
retained parent state, real action scheduling and team-return continuation. Also
check a target that moves away and a child that dies. These are deterministic
correctness tests only; never add them to the training buffer, teach the model
these commands, or require an untrained policy to select them.

## 17. Statistical training acceptance

Do not equate passing unit tests with learning the desired strategy.

* Report at least three training seeds for comparisons where feasible; otherwise
  label results exploratory. Compare equal environment-step and wall-clock budgets.
* Hold out map families and opponent policies, use both team assignments and
  paired maps. Fixed official pearl RNG means changing an invented seed is not
  independent timing variation; vary legitimate map parameters.
* Measure W/D/L and draw-as-half score with intervals clustered by map/opponent
  pairing. Keep a final untouched test suite separate from checkpoint selection.
* Inspect emergent splitting, newborn survival and risk-taking as diagnostic
  behavior, without using tactical labels to train or select the policy.
* Track head trades, trap deaths, retained split sizes, sprint spending, food
  revisitation, map coverage and cold-start survival as diagnostics, not rewards.
* Compare historical opponents and trained exploiters; no promotion based solely
  on defeating latest-self or a single script.

## 18. Ablations and promotion gates

Required comparisons:

1. Current local CNN+GRU baseline.
2. Baseline plus absolute coordinates/dimensions.
3. Factual map + learned reader, with no learned persistent map channels.
4. Full factual + learned spatial memory model.
5. Full model without generic body-order/history features.
6. Split disabled vs full-range split; K=3 vs 8 (16 if promising).
7. Complete-return advantages vs the proposed event GAE.
8. Historical league vs latest-self; auxiliary losses on/off.

Recall tests must include delay beyond the BPTT horizon and paired histories
with identical current local observations but different relevant past facts.
Memory erasure interventions are useful diagnostics but introduce distribution
shift; confirm findings with models trained without memory as well.

Promotion sequence:

* Gate A: privacy, semantics, exact replay and PPO smoke test pass.
* Gate B: export/NumPy parity and judge resource constraints pass.
* Gate C: spatial-memory model improves held-out recall/navigation and competitive
  outcomes over the local baseline; an extra module must earn its cost.
* Gate D: full action policy improves team outcomes against a diverse pool.
  Neither high split rate nor use of any named tactic is a success criterion.
* Gate E: official-engine held-out matches confirm improvements and no systematic
  exploit invalidates them. Select by win strength subject to deployment limits.

Keep sonar as the controlled experiment in section 6.5. Defer explicit role
hierarchies, large transformers, model-based search, COMA and return redistribution
until a named failed evaluation motivates them.

## 19. Literature-to-design traceability

These are methodological influences, not evidence that the combined design is
optimal for Dragons:

* [Neural Map](https://arxiv.org/abs/1702.08360): spatially addressed external
  memory and distinct reads/writes. Here exact coordinates remove localization.
* [MERLIN](https://arxiv.org/abs/1803.10760): predictive representation/memory
  training; motivates pre-update auxiliary forecasts rather than reward-only writes.
* [An Investigation of Model-Free Planning](https://arxiv.org/abs/1901.03559):
  recurrent spatial computation as a learned alternative to scripted planning.
* [Value Propagation Networks](https://arxiv.org/abs/1805.11199) and
  [Value Iteration Networks](https://arxiv.org/abs/1602.02867): alternatives if
  long-route reasoning fails; not a claim that tile values solve body dynamics.
* [Never Give Up](https://arxiv.org/abs/2002.06038): separate exploratory behavior
  from exploitation and avoid rewarding uncontrollable changes. We do not import
  its off-policy algorithm into PPO without correction.
* [PPO](https://arxiv.org/abs/1707.06347) and
  [MAPPO](https://arxiv.org/abs/2103.01955): conservative policy updates and
  centralized training with decentralized actors. Our event GAE is game-specific.
* [COMA](https://arxiv.org/abs/1705.08926): possible action-contribution baseline
  if sacrifice credit fails, adapted to sequential decisions rather than copied.
* [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z): historical league
  and exploitation pressure to prevent narrow self-play strategies.
* [RUDDER](https://arxiv.org/abs/1806.07857): targeted delayed-credit alternative,
  not an excuse to add arbitrary reward deltas.
