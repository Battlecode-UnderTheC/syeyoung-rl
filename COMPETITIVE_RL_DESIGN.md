# Dragons: literature-grounded RL design

Implementation authority: `SPATIAL_MEMORY_RL_IMPLEMENTATION_SPEC.md`. Its later
user-directed requirement supersedes the tactical curricula suggested below:
strategies must emerge from game-outcome RL, with no tactic-specific teaching.

Design only. No implementation or training results are claimed. This replaces
prior sketches in this document. The game rules in `dragons-env/game.md` are the
source of the strategic analysis; the cited papers motivate particular methods,
not claims of competitive performance in Dragons.

## 1. Start from the strategy we want to learn

The objective is expected terminal team outcome: elimination first, then longest
living dragon, then total living length. A good policy must balance information,
food, maneuverability, and threats under that objective.

| Desired behavior | Why it matters | Necessary representation or learning signal |
|---|---|---|
| Purposeful exploration | Find useful food regions, passages, opponents and portal connections | Persistent known/unknown map and visitation history; navigation curriculum |
| Harvesting circuits | Revisiting replenished territory can outperform exploring forever | Countdown observations, elapsed time, resource history and body history |
| Preserve escape space | Eating can delay tail clearance and turn a safe-looking pocket into a trap | Ordered own-motion history and spatial recurrence; growth/trap scenarios |
| Spend length to buy position | Sprint to escape, intercept, or fund later steps with food | Conditional path actions and intermediate-step reasoning |
| Body blocking and interception | A body is an obstacle, while a head is a mutual-kill target | Head/body separation, identities, heading, turn order and predicted movement |
| Concentrate winning length | One 40 beats two 30s at the horizon | Team outcome reward rather than total-food reward |
| Cheap head trades | A short attacker can eliminate the opponent's contender | Keep suicidal attacks available; credit team returns beyond individual death |
| Selective splitting | Preserve length in a child while the informed parent prepares a head trade; newborn survival matters | Parent-head and tail geometry, retained parent intent, empty child memory, delayed team credit |
| Recover corpse resources | Roughly half a dead dragon's length becomes pearls | Predicted death locations and access to dropped food |
| Endgame risk adjustment | A likely leader should defend; a likely loser may need a risky trade | Remaining time and uncertainty about opponents, not an oracle score |
| Distributed roles | A large contender and a small attacker should behave differently | Shared policy conditioned on own length, observations and private history |

These are hypotheses to test, not rules to hard-code into the actor.

Three subtleties matter:

* Sight through kelp does not imply reachability. A visible pearl across a wall
  is not necessarily a good target. Toroidal topology and portal connections
  invalidate ordinary Euclidean route intuition.
* Splitting is a spatial action: the child appears at the tail, not next to the
  parent's head, and starts with no inherited memory. A child may explore or
  attack, but extra scouts cannot automatically share their discoveries.
* Deliberate death is not universally bad. For example, partial transfer of a
  secondary dragon's length through corpse pearls could increase the team's
  maximum length. Access, opportunity cost, and enemy interception can make it
  a terrible choice instead. Do not remove this possibility by definition.

### Split to preserve length, then attack with the informed parent

The main planned attack is **parent splits, parent later sprints**. The parent
keeps its map, target history and recurrent intent; the child starts without
inherited memory. Detach most of the length as a replacement contender and
retain enough parent length to execute an expendable head attack.

Example against enemy lengths [60, 30]: a length-40 parent splits off 36,
leaving a length-4 parent. If that parent subsequently completes a three-step
head trade against the length-60 enemy and its child survives, the resulting
[36] versus [30] reverses the primary-score lead. It does not guarantee a win.

The exact sequence matters:

1. On round t the parent executes SPLIT 36. It stays at its head position,
   keeps its private memory, and cannot also sprint during this turn.
2. The length-36 child starts at the old tail, facing back along the trail,
   with empty inherited memory. It acts later in round t and must choose a
   viable move from its own observation.
3. On the parent's next turn it observes again, updates its target belief,
   and chooses whether/how to sprint. The opponent may have moved or killed
   either dragon in the meantime; this is a setup attack, not an atomic combo.
4. A successful head collision kills the small parent and target. The child
   remains the team's contender if it survived the intervening turns.

Without intermediate food, a k-step sprint requires at least k+1 parent
segments (minimum two for one step). A three-step attack needs four retained
segments. Food already consumed along the prefix may fund later steps; a pearl
on an unaffordable next destination cannot pay for entering it. All intermediate
collisions matter, and hitting a body kills only the attacker.

The parent must learn **two-location planning**: attack feasibility near its
head and newborn survival near its old tail. It should position its tail in a
safe region before committing, accounting for imperfect knowledge of that
region. Giving the child most of the length also gives a memoryless policy a
large body to manage: preserving biomass is not equivalent to preserving the
old dragon's learned map knowledge or immediate safety.

Parent and child share policy weights, but receive no hidden role labels or
memory transfer. The parent can retain attack intent in its recurrent state;
the child needs strong cold-start behavior conditioned on its length and actual
view. The parent's GRU must not reset when it splits, even if its length changes
drastically. Conversely, both factual and learned maps must reset for the child.
The child can later build its own new memory normally.

A newly born child may independently discover a local sprint attack, but that
is a separate opportunistic behavior. It cannot execute an unseen parent plan
using memory it never received, and is not the principal planned attack here.

Train the chain explicitly: position the tail, preserve length in a viable
child, retain sufficient sprint budget, survive the setup round, reobserve,
and conditionally attack or abort. Include losing variants: the child is boxed
in, target escapes, ID order changes, head path is blocked, or splitting removes
an existing winning lead. Team returns must credit the parent for child survival
and eventual victory even after the parent sacrifices itself.

## 2. What the literature contributes

### Spatial memory: Neural Map and MERLIN

[Neural Map, Parisotto & Salakhutdinov](https://arxiv.org/abs/1702.08360)
separates spatial storage from a recurrent controller, with position-addressed
writes and global/content reads. Its navigation results motivate a spatial
memory rather than asking one vector to retain an entire board.

[MERLIN, Wayne et al.](https://arxiv.org/abs/1803.10760) trains memory
representations using predictive objectives as well as return-related signals.
The transferable lesson is that sparse policy rewards need not be the only
supervision teaching a memory what information to retain.

Application: use exact absolute addresses supplied by Dragons. Preserve direct
observations explicitly, and learn latent spatial features plus reads. We do
not need to learn localization or reconstruct raw visual inputs. Train forecasts
on later observations before writing those observations, so a decoder cannot
solve its task by copying the answer.

### Spatial computation: DRC and learned value propagation

[An Investigation of Model-Free Planning, Guez et al.](https://arxiv.org/abs/1901.03559)
shows planning-like behavior from deep recurrent convolutional networks, with
additional internal computation helping on its evaluated tasks. This supports
spatial recurrence as an architectural candidate; it does not prove arbitrary
lookahead or transfer to adversarial snakes.

[Value Propagation Networks, Nardelli et al.](https://arxiv.org/abs/1805.11199)
provides parameter-efficient differentiable planning modules trained with RL.
[Value Iteration Networks, Tamar et al.](https://arxiv.org/abs/1602.02867)
also makes explicit why local propagation needs enough iterations or a hierarchy
to communicate over long routes.

Application: retain full-resolution local geometry and propagate information
at multiple spatial scales. Compare a small recurrent convolutional reader with
an edge-conditioned value-propagation reader. The latter must handle kelp on
edges and observed portal links. Neither is an exact game solver: position alone
is not a Markov state when body layout, growth, heading and opponents matter.
Do not substitute a shortest-path objective for winning.

### Exploration: Never Give Up

[NGU, Badia et al.](https://arxiv.org/abs/2002.06038) distinguishes episodic
novelty from longer-term novelty, and trains policies with different intrinsic
reward strengths. It also avoids rewarding every uncontrollable visual change.

Application: a changing pearl timer or moving enemy must not create endless
novelty credit while the snake circles familiar ground. Exact coordinates make
simple episodic geographic coverage a useful warm-up signal. Final competitive
training must value profitable revisits and strategic restraint. We borrow the
exploration/exploitation distinction, not NGU's off-policy replay machinery.

### Team learning: PPO, MAPPO and COMA

[PPO, Schulman et al.](https://arxiv.org/abs/1707.06347) supplies the clipped
policy update, not memory or strategy by itself.
[MAPPO, Yu et al.](https://arxiv.org/abs/2103.01955) supports centralized-critic
PPO as a serious cooperative baseline and studies sensitivity to critic inputs,
clipping and training-data reuse. Its evaluated tasks do not establish the right
hyperparameters for this sequential competitive game.

[COMA, Foerster et al.](https://arxiv.org/abs/1705.08926) addresses individual
contributions to team outcomes through a centralized action-value critic and a
counterfactual baseline. This is relevant if a team value cannot distinguish
useful sacrifices from incidental deaths. Dragons acts sequentially, so directly
copying COMA's simultaneous joint-action construction would be inappropriate.

### Competition: AlphaStar

[AlphaStar, Vinyals et al.](https://www.nature.com/articles/s41586-019-1724-z)
uses a league and exploitation pressure to address diverse strategies and
self-play cycles. Borrow that principle at small scale: maintain historical
opponents and deliberately train opponents that find the current policy's
weaknesses. A latest-self win rate alone is not evidence of strength.

### Delayed credit: RUDDER

[RUDDER, Arjona-Medina et al.](https://arxiv.org/abs/1806.07857) studies return
redistribution for delayed rewards. Keep it as a targeted follow-up if successful
exploration or sacrifice occurs but PPO cannot assign credit. Learned outcome
changes are not automatically causal action contributions; do not casually turn
critic deltas into a new reward and claim equivalent objectives.

## 3. Recommended actor

Use a spatial-memory recurrent actor, trained with PPO. Shared weights; private
state per dragon. Start with one actor architecture, without fixed roles or a
handcrafted planner.

```
                     absolute-coordinate observations
                                  |
             +--------------------+-------------------+
             |                                        |
      factual history map                    current 7x7 local CNN
             |                                        |
      learned spatial features                        |
             |                                        |
     multiscale recurrent reader <--- previous GRU + own status
             |
      local read + multiple attention reads
             |                                        |
             +--------------------+-------------------+
                                  |
                          tactical GRU(128)
                                  |
                  movement / sprint / split distribution
```

### Factual and learned memory serve different purposes

The factual map stores only information available from this dragon's history:

* Known/unknown edges, kelp, observed portal IDs and established endpoint pairs.
* Last-seen pearl presence/countdown, with timestamps and a bounded history of
  revisits if needed to estimate renewal behavior.
* Last-seen head/body occupancy with team, identity and observation age.
* Visit count, most recent visit, and observed own-body/trail information.

Static facts persist. Dynamic records retain their age and are not asserted to
be current. Extrapolated food availability and enemy location are predictions,
not observations. Distinguish never observed, observed empty, and stale records.

Absolute coordinates determine where to write the 49 observed tiles, wrapping
at actual board dimensions. A coordinate cache is observation preprocessing,
not action selection. It is compatible with a fully learned RL controller.

Add 16 learned features per tile as a proposed starting capacity. A small shared
gated writer summarizes local observation changes/history at the observed tiles;
unobserved latent cells retain their content. The factual channels bypass this
writer so failure to learn a long-term write gate cannot erase known terrain.
Learned features can summarize recurring food activity and movement patterns.

Use a short ordered own-action/head-position history in addition to last-seen
occupancy, encoded by the recurrent network. A last-seen map alone loses trail
ordering. Do not infer exact unobserved spawn-body geometry or enemy body length.
A spawn child's own first observation may reveal only part of its body.

### Spatial computation and tactical recurrence

Starting candidate, subject to resource profiling:

* Local branch: two 3x3 convolutions, 32 channels, retaining 7x7 detail.
* Memory branch: 32-channel multiscale encoder with fine, half and quarter
  resolution, followed by 2-4 shared recurrent computation steps.
* Reads: features near the current head plus four learned attention queries over
  coarse spatial tokens, each retaining wrapped relative position information.
* Fusion: local/spatial/status embeddings into a 128-unit GRU and action heads.

The coarse branch gives a wide receptive field; fine features retain edge and
body geometry. Two to four recurrent steps are a budgeted starting experiment,
not a claim that arbitrary 64x64 detours are solved. Stress-test long detours
and compare more iterations or the learned value-propagation alternative.

Map storage uses absolute orientation. Policy reads may be centered/rotated to
heading, with directional edges transformed consistently. Full-board convolutions
wrap on the true board; local observation crops do not. Padded cells are masked.
Kelp blocks motion, not information propagation through the network. Confirmed
portal pairs need explicit relational features/nonlocal connections; unknown
exits must remain unknown.

Status includes length, time remaining, heading, board dimensions, team unit
count/limit, ID-order relations for observed dragons, and previous action.
No hidden team score, true unseen enemy length, or global teammate locations.
Risk adjustment therefore depends on a learned belief about competitive position.

### Action expressiveness

Train a categorical move/split gate. For movement, use a small autoregressive
command decoder emitting directions and a STOP token. No intermediate observation
is received: the entire sprint is one action. Begin with a three-step cap and
extend to eight in a controlled curriculum; both caps restrict the game's action
space and must be evaluated for lost tactics. Sum token log-probabilities once
for the PPO action ratio, including STOP when applicable.

A split head scores sizes n in [2, L-2], using a shared function of the state,
absolute size and size fraction, rather than a fixed small-size-only catalogue.
Variable size enumeration/runtime may require a bounded candidate version; that
would be an explicit ablation, not a claim of full coverage.

Hard masks enforce command constraints from legitimate information. They must
not automatically remove all collisions or head trades. A temporary conservative
navigation curriculum can exclude certain deaths, but the competitive policy
must regain strategically meaningful sacrifice actions. Affordability must
account for food consumed earlier in a sprint; an unseen portal exit is not a
known-clear tile. Split is available while boxed in because it leaves the parent
stationary. Invalid output and numerical all-masked states need a defined fallback.

Sonar is a later extension, with heads conditioned on the chosen physical action
and a receiver trained under actual ray/delivery physics. Exploration by a child
is not shared map coverage unless a real received message communicates it.

## 4. Training objectives and credit

Terminal reward is +1/-1/0 from the official team result. Personal death is not
an RL terminal for team returns. Do not reward number of dragons, raw total
food, or personal survival as permanent objectives.

Train a privileged critic from global state, turn phase, acting identity and
relevant history/memory summaries. Full physical state alone does not describe
all future behavior of recurrent policies. These critic features remain absent
from the actor and its masks.

A transparent reference estimator is complete-match return G minus a pre-action
baseline V: every learner decision, including those of dragons that died earlier,
gets credit for the eventual team result. Its variance can be high. Compare it
with round-based team GAE, broadcasting round advantages to actions in that round.
The latter has coarser tactical credit, but gives a consistent time unit despite
splits. Never accidentally discount once per extra child and change the horizon.
Only complete-round adjudication ends the match, including round 499.

Start gamma=1 for the finite-horizon outcome objective. Test lambda=.97 and .99
for GAE against the complete-return reference. Start PPO clipping at .1, 3-5
epochs and sequence length 64, then test 128. These are proposed settings, not
paper-derived optimal values. Larger lambda reduces trace decay but increases
variance; it does not repair insufficient memory gradients by itself.

Train contiguous per-dragon histories. Preserve factual map snapshots and
reconstruct updates exactly at sequence boundaries. Learned map/GRU states
need stored initial state and burn-in (initial trial: 16-32 steps); truncated
backpropagation still limits credit for learned writes. Auxiliary prediction
reduces this burden, while direct factual memory avoids it for raw observations.
Store map checkpoints/write deltas rather than a full dense snapshot every turn.

Useful small auxiliary heads, evaluated individually:

* Predict pearl availability on revisiting remembered cells, scored before the
  new observation enters memory.
* Predict visible enemy head displacement/occupancy distributions at a future
  observation, with an explicit unseen/not-observed outcome or evaluation mask.
* Predict short-horizon action consequences such as length change or collision
  under observed actions. Multi-step outcome labels depend on later policies.

These are representation losses, not reward bonuses. Do not reward prediction
error on moving enemies as if it were useful geographic exploration. Avoid a
large reconstruction loss that makes irrelevant details dominate winning.

## 5. Curriculum that produces the intended strategies

Keep a stream of full games throughout later stages so specialized training does
not replace the actual objective. Scenario resets must be legal; when history
matters, provide a real observation/action prefix, not a privileged filled map.

1. **Memory and navigation:** maps with food regions separated by detours,
   periodic harvesting opportunities, toroidal seams and portal shortcuts.
   Increase lengths so compact routes become dangerous. Use temporary coverage
   and food rewards only here, explicitly separate from final competitive reward.
2. **Body-aware growth:** pockets, corridors, self-crossing temptations and
   food choices that delay tail clearance. Test whether the policy passes up
   food when it would remove escape options.
3. **Tactical encounters:** randomized lengths and ID order, sprint-funded
   escapes, interception and head trades. Include both profitable and losing
   sacrifices so “always attack” cannot solve the distribution.
4. **Team allocation:** vary which dragon is a plausible contender, split near
   threats at the parent's tail, and create corpse-recovery opportunities for
   either side. Train the parent's split-then-sprint attack chain, including safe tail
   placement for the memoryless child and re-evaluation on the next turn. No hand-assigned leader/hunter labels are required.
5. **Endgames:** vary remaining rounds and true score configurations while
   preserving partial observations. Distinct hidden states may warrant the
   same action when the actor has identical information; evaluate expected
   outcomes over the compatible uncertainty, not oracle-optimal choices.
6. **League self-play:** historical policies, current learners and exploiters
   targeting passivity, predictable food circuits, excessive splitting or
   reckless aggression. Preserve opponents that expose strategic weaknesses.

If explicit curiosity is needed beyond warm-up, use separate exploration-trained
opponents/auxiliary policies while maintaining a win-only main policy. A
beta-conditioned NGU-style family is another experiment, but PPO trajectories
must remain associated with the beta/policy that generated them. Do not train
an exploitative head off-policy on exploratory samples without a valid correction.
Per-child novelty resets must not pay repeatedly for splitting in known territory.

## 6. Evaluation that can reject the design

Use both sides on held-out maps and opponents, multiple independent training
seeds, equal training budgets, and measured inference costs. Official-engine
matches remain necessary given documented simulator coverage/RNG limitations.

| Experiment | Failure it diagnoses |
|---|---|
| Local GRU vs coordinate-only vs factual map vs factual+learned map | Whether spatial memory earns its complexity |
| Erase old map records while keeping the current observation | Whether the policy actually uses history; interpret distribution shift carefully |
| History-dependent paired decision tasks | Whether past information changes action appropriately |
| Long-detour and harvesting-circuit suites | Whether it can use memory for navigation and revisitation |
| Same food geometry at different lengths | Whether it reasons about maneuverability rather than greed |
| Same encounter with ID order swapped | Whether it models sequential tactical timing |
| Profitable/unprofitable sacrifice pairs | Whether team credit supports conditional sacrifice |
| Short and long action-cap comparison | Whether sprint restrictions hide important strategies |
| Spatial recurrence vs simple attention vs value propagation | Which learned computation helps at the available CPU budget |
| Historical league and trained exploiters | Whether apparently strong play has a consistent counterstrategy |
| Coverage-pretrained vs win-only | Whether exploration training improves competitive outcomes |

Primary metric is held-out win/draw/loss, with confidence intervals grouped by
map/opponent pairing. Also measure longest-length margins, forced traps, head
trades, split outcomes, food revisitation, and p95/p99 runtime/memory. Coverage
alone is not success: a strong endgame policy may intentionally explore less.

The final deployment has 48 MB, 100M CPU points per turn and a 4 MB zipped
submission limit. Exact map storage is modest (32 float channels on 64x64 is
512 KiB), but repeated spatial computation and temporary arrays need profiling.
A small teacher/student distillation experiment is possible if the selected
architecture is too expensive; it must preserve history inputs and be evaluated
on full matches, not only one-step imitation accuracy.

Recommendation: factual spatial memory with learned latent augmentation,
multiscale recurrent spatial reasoning, a tactical GRU, expressive conditional
actions, team-outcome PPO and a competitive league. Treat learned propagation,
COMA-style credit and RUDDER as targeted alternatives when the corresponding
navigation or credit tests fail, rather than combining every paper at once.
