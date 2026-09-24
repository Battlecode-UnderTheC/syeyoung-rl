# Dragons competitive policy — model and training specification v1

**Status:** a concrete baseline specification, not an implementation or a trained result. This document replaces the earlier proposal in this file. No existing agent, model, trainer or simulator implementation source was inspected.

**Objective:** maximize expected official match result against other bots: win +1, draw 0, loss −1. Geometry use, resource collection, combat, splitting and sonar are means to that objective. There is no maze objective, assigned tactical role or prescribed winning strategy.

**Responsibility boundary:** the program parses legal observations and records facts. The model learns decisions, including path planning. No BFS, A*, computed path distances, selected destinations, scripted movement, teacher routes or action overrides appear in this baseline.

## 1. Frozen baseline and notation

| Item | v1 choice |
|---|---|
| Execution | Shared weights, strictly private state per dragon lifetime |
| Board support | 7–64 tiles per side, actual-size storage; reject larger inputs explicitly |
| Spatial representation | Uncompressed tile graph, built from observed edge geometry |
| Reasoning | Four shared-weight graph/global-attention iterations, width 32 |
| Persistent neural state | One GRU state of width 192 |
| Physical action | MOVE with 1–8 absolute direction tokens, or any legal SPLIT size |
| Sonar action | Four independent 257-way heads: SILENT or a symbol 0–255 sent as uint64 |
| Received sonar | All uint64 values supported, plus the five raw ECHOES fields |
| Learning | Recurrent PPO, complete-match undiscounted returns, separate privileged critic |
| Training distribution | Full game and all baseline action heads from update 0 |
| Auxiliary losses | None |
| Reference run | 2,000 updates, 32 complete matches/update, three PPO epochs/update |
| Deployment policy | Sampling at temperature 1, same as training |
| Actor parameters | 530,115, including biases, normalization and embeddings |

The eight-step cap and 256-symbol transmit vocabulary are deliberate policy restrictions. v1 does not represent every engine-valid movement length or every outgoing uint64. It does represent all four sonar directions simultaneously, every legal split size, and arbitrary received uint64 words. Changing these restrictions creates a new model version, not a runtime override.

Notation: W,H are actual board dimensions; A=W*H; N=A; B is a batch of independent actor decisions; E is the number of currently visible dragon IDs. Directions are N,E,S,W, indexed 0,1,2,3. Tile index is y*W+x. Coordinates increase east and south. Relative coordinate encoding is:

```
rel(x,y) = [sin(2*pi*(x-head_x)/W), cos(2*pi*(x-head_x)/W),
            sin(2*pi*(y-head_y)/H), cos(2*pi*(y-head_y)/H)]
abs(x,y) = [sin(2*pi*x/W), cos(2*pi*x/W),
            sin(2*pi*y/H), cos(2*pi*y/H)]
```

All model arithmetic is float32. Raw coordinates, IDs, timers and messages remain exact integers outside the network. Define:

```
clip01(x) = min(1,max(0,x))
age(t)    = clip01((current_round-t)/500)             # zero if record invalid
cnt(x)    = min(log1p(max(x,0))/log(501), 1)
length(x) = x/A
loglen(x) = log1p(max(x,0))/log1p(A)
timer(x)  = sign(x)*min(log1p(abs(x))/log(501), 1)
```

Unknown numeric features are zero with the specified validity flag. Factual records are never quantized to these features. Batch padding has a separate mask and does not become an unknown in-board tile.

## 2. Required game contract

The target is modified unswbc 1.0.0. The following defines the model's rule assumptions:

1. The world is toroidal. Kelp occupies edges, blocks movement/sonar, and does not occlude the geometric 7×7 view. Paired portal edges preserve movement direction and change the destination tile. The view does not look through portals.
2. Spawn metadata gives ID, team, dimensions and unit limit. A turn gives round, facing, own length, own-team living count, local tile coordinates/pearls/countdowns, visible segments with team/ID/facing/head flags, visible edges and portal IDs, inbox and ECHOES. The model receives no hidden scores, other dragons' complete lengths or unseen map files.
3. Dragons act sequentially in increasing ID order. A split child acts later in its birth round. Observations are generated at the actual actor's turn, not from a simultaneous snapshot. Pearl ticks and result checks occur at round boundaries.
4. MOVE submits a complete direction sequence. The first step is free; later steps require length >2 before entry and cost one segment. Growth can affect subsequent affordability. No intermediate observation is provided.
5. Kelp or any body collision kills the mover, including contact with its own tail. Contact with another head kills both regardless of length. Fatal movement is allowed as a policy choice. No WAIT action is assumed.
6. SPLIT n requires 2<=n<=L−2 and room under the team unit limit. The reversed last n segments become a new dragon. The parent remains in place, retains identity/memory, and loses that suffix. A child receives no parent state. Death leaves pearls on alternating segments starting at the head.
7. Results are checked after completing the round. Elimination wins first. At the end of round 499, compare longest living dragon, then total living length, then draw. That final round is a true terminal, not a training truncation.
8. Four directional commands use `SONAR <dir> <uint64>`. Opposite-facing sonar originates at the tail. Delivered words and the sender's next-turn `ECHOES kelp ally ally_head enemy enemy_head` report are separate inputs.

The supplied documentation does not fully define the newer echo field encoding or all directional-ray edge cases. **This specification does not invent them.** The encoder below handles the five raw field tokens without interpreting them as directions, counts or distances. The environment adapter must pass black-box conformance fixtures against the target release before training:

* Four different directional commands together, zero versus silence, duplicate-direction handling, and uint64 boundary values.
* Head/tail origins after turns, multistep movement and splits; emission when the physical action kills the sender.
* Range, kelp, portals, wrapping, self-interception, ally/enemy interception and loops.
* Exact echo/inbox contents and timing, including earlier/later IDs, first turns and newborns.

Old forward-only ray rules are not a fallback. A simulator without the modified sonar behavior is incompatible. These are external game-contract checks, not choices left open in the neural architecture.

## 3. Private factual state and update order

A lifetime is `(match_id, dragon_id)`, with IDs never reused inside a match. Allocate one object per lifetime:

```
Facts:
  tile[N]: seen, seen_round, pearl, countdown,
           previous_valid, previous_round, previous_pearl, previous_countdown,
           occupant_class, occupant_id, occupant_facing,
           visits, visited_valid, visit_round
  north_edge[N], west_edge[N]: known, type, portal_id
  portal_endpoints: map from exact portal_id to observed canonical edges
  body[L]: coordinate or UNKNOWN, head first
  previous_command: kind, directions[0:8], split_n, length_before,
                    send[4], words_uint64[4], head_before, facing_before
  previous_valid: bool
  own_turn_count: integer
  last_observation_key
Neural:
  h: float32[192]
```

The raw current observation, inbox and echo tokens are transient inputs, not privileged additions. Static geometry persists once observed. Old occupancy and pearls remain *last-seen* records with ages. Receiving a message never writes it into the factual map as verified terrain or occupancy.

On a new lifetime, all records are unknown, h=0, previous_valid=false and the body is L UNKNOWN entries with the observed head at index zero. A visible segment's facing points toward the next segment nearer the head. To fill body entries, repeatedly find a visible own segment whose known transition in its reported facing reaches body[i]; assign it to body[i+1] only if i+1<L and both the segment and index match uniquely. Use known portal transitions in the same lookup. Conflicting assignments stay UNKNOWN. Stop when no entries change. Do not guess indices from filtered body-line order when unseen segments can lie between them.

Process each own observation once, in this order:

1. Reconcile the previous command if one exists. A new turn means this lifetime survived the entire prior command. For MOVE, derive intermediate head coordinates only along privately known transitions, prepend them in reverse temporal order, and truncate to the newly reported length. Unknown intermediate locations stay UNKNOWN; the final reported head is always known. For SPLIT, retain the first L_before−n entries. Contradictions invalidate affected inferred entries; they are never repaired from simulator truth.
2. Promote each visible tile's previous record, then overwrite current observed tile/occupancy data. Empty observed tiles clear old occupants. Update observed canonical edges and portal endpoint identities. Seeing an edge also establishes its two geometric side coordinates, not the visibility of those tiles.
3. Reconcile provable visible own-body links by the procedure above. On spawn mark the observed head visited once. After a completed MOVE mark each reconstructable intermediate head, including the final observed head, once per passage; repeated visits to a coordinate in that command count separately. Do not count that final head a second time when reading the observation. SPLIT creates no new passage for the parent. Sonar echoes do not mark tiles visited or determine an exact enemy position.
4. Materialize the input tensors below, encode/sample, and commit h_new plus the issued command exactly once. Increment own_turn_count. A repeated observation key returns the cached response without a second state update.

Parent facts/h survive splitting. A child starts independently. A dragon's death removes its runtime state after its actor record is closed. Match reset clears all states. Nothing in one actor's preparation reads another actor's state. Padded training steps advance nothing.

## 4. Exact actor input tensors

Group batches by W,H. Use masks for variable E and inbox lengths. Shapes below describe one actor; prepend B when batching.

### 4.1 Tile features X: float32[N,56]

| Indices | Features, in order |
|---|---|
| 0–5 | seen; visible now; age(seen_round); last pearl; last countdown!=-1 (bed flag); timer(last countdown) |
| 6–8 | ever visited; cnt(visits); age(visit_round) |
| 9–12 | current own head; privately known current own-body cell; index/max(L−1,1); privately known current tail |
| 13–18 | previous observation valid; age(previous_round); previous pearl; previous bed flag; timer(previous countdown); clip01((seen_round−previous_round)/500) |
| 19–20 | recorded nonself occupant ID earlier than self; later than self |
| 21–27 | last occupant one-hot: empty, self head, self body, ally head, ally body, enemy head, enemy body |
| 28–31 | last occupant facing N,E,S,W; all zero for empty/unknown |
| 32–47 | four edge one-hots in direction order; each is unknown, empty, kelp, portal |
| 48–51 | four known-destination flags, defined below |
| 52–55 | rel(x,y), including for unseen in-board tiles |

For unseen tiles, features 0–5 and 13–31 are zero; edge records, private-body/visit facts and coordinates retain whatever is independently known. For no previous record, features 13–18 are all zero. Unknown occupancy is all zeros, distinct from observed empty. Known-body index is zero unless feature 10 is one. An old self-occupancy record in 21–27 is not current own-body truth; features 9–12 are the separate current private reconstruction.

### 4.2 Geometry J and edge features D

`J: int32[N,4]` contains the destination tile for an observed empty edge or a portal with a privately known pair. Otherwise J=−1. Walls, unknown edges and unpaired portals are distinguished by features. For an empty edge, use the wrapped geometric neighbor. For a portal, crossing its source edge in direction d lands on the side beyond its paired edge in the same direction d. Portal IDs are matched by equality, never by a numeric embedding. Opposite traversal uses the corresponding opposite side.

`D: float32[N,4,12]`, in order:

```
direction one-hot[4]
edge-type one-hot[4]                 # unknown, empty, kelp, portal
known_destination[1]                 # J>=0
ordinary_wrapped_seam_crossing[1]    # zero for nonempty/unknown/portal
confirmed_portal_pair[1]
destination_tile_seen[1]             # zero when J<0
```

Graph arcs are exactly J>=0. Bodies never remove arcs; they are dynamic features the model reasons about. Construct no multihop reachability, distance, corridor decomposition or frontier targets.

### 4.3 Visible entities U: float32[E,20]

Group all current visible segments by exact dragon ID, sorted by ID for reproducibility. Include the actor itself; E>=1 on a valid live observation. Do not include stale entities as fresh tokens; their old cell records remain in X.

| Indices | Features |
|---|---|
| 0–2 | self/ally/enemy one-hot |
| 3–5 | ID earlier/equal/later than self one-hot |
| 6 | visible segment count/49 |
| 7 | head is visible |
| 8–11 | rel(visible head), or zero if not visible |
| 12–15 | visible head facing one-hot, or zero |
| 16–19 | mean of rel(segment coordinate) over this entity's visible segments |

Count/49 is visible extent, never total length. No numeric ID value enters the network.

### 4.4 Status S: float32[32]

| Indices | Features |
|---|---|
| 0–5 | length(L); loglen(L); round/499; (500−round)/500; W/64; H/64 |
| 6–11 | unit_count/unit_limit; unit_limit/64; known body fraction; tail known; first own turn; own_turn_count/500 |
| 12–15 | current facing one-hot |
| 16–19 | abs(current head) |
| 20–25 | previous_valid; previous MOVE; previous SPLIT; previous move count/8; previous split_n/max(previous length,1); (L−previous length)/A |
| 26–31 | fraction of tiles seen; fraction of canonical edges known; cnt(inbox word count); echo line present; current head tile is bed; current head tile pearl |

Features 21–25 are zero without a previous command. Round and turn counts here are the reported round and count of earlier completed own turns. Ratios not explicitly clipped remain unclipped; in particular length change is signed.

### 4.5 Messages, echoes and previous command

* Inbox: the complete list of received uint64 words. Encode each as eight little-endian bytes; zero is a real word. No sender, team, direction or delivery acknowledgement is invented. Preserve all messages; process with streaming pooling rather than a silent length cap.
* Echo: exactly five raw field strings after ECHOES, in the documented field order. UTF-8 encode each string losslessly. The neural encoder reads their bytes; it assumes neither bitmask nor count semantics. If the line is absent where the qualified protocol permits absence, use five empty strings and zero presence flags. Present `0` is not absence. No numeric token is cast through float32 first.
* Physical command vector C: float32[36]. Kind MOVE/SPLIT one-hot[2], eight absolute direction one-hots[8,4] flattened (unused positions zero), split_n/A[1] (zero for MOVE), length_before/A[1]. No previous command means all zeros.
* Previous sonar: four send flags and four uint64 words. A silent direction has flag zero and a zero-filled word slot; its word encoding is multiplied by the flag.

No additional command-history encoder or learned spatial memory exists in v1. The recurrent state learns temporal summaries; factual geometry and raw last-observed records persist independently of it.

## 5. Exact neural modules

### 5.1 Operator conventions

`L(a,b)` is an affine layer with weight[b,a] and bias[b]. `MLP(a,b,c)` is L(a,b), SiLU, L(b,c), SiLU. `Score(a,b,c)` is L(a,b), SiLU, L(b,c) without final activation. `P(a,b)` is L(a,b), SiLU. All modules have separate parameters unless sharing is explicitly stated.

GRU(i,h) uses three gates and two biases per gate:

```
r = sigmoid(W_ir*x+b_ir + W_hr*h+b_hr)
z = sigmoid(W_iz*x+b_iz + W_hz*h+b_hz)
n = tanh(W_in*x+b_in + r*(W_hn*h+b_hn))
h_new = (1-z)*n + z*h
```

Its parameter count is 3hi+3h²+6h. LayerNorm(32) has learned scale/bias and epsilon 1e−5. No dropout or batch normalization is used. Masked mean/max exclude padding. Empty pools produce zero vectors and their specified empty flag. A graph max over no neighbors is zero; otherwise it is the componentwise maximum of valid messages.

### 5.2 Encoders

| Module | Exact operations | Output |
|---|---|---|
| Local | Gather the 7×7 north-up X crop using toroidal coordinates; Conv2d(56,32,3,pad=1), SiLU, Conv2d(32,32,3,pad=1), SiLU; concatenate center/mean/max; P(96,64) | local[64] |
| Tile | MLP(56,32,32), independently per tile | tile[N,32] |
| Entity | MLP(20,32,32), independently per entity | entity[E,32] |
| Status | MLP(32,32,32) | status[32] |
| Word | Byte embedding[256,8] + byte-position embedding[8,8]; flatten eight sums to 64; MLP(64,32,32) | word[32] |
| Inbox | Mean/max of word vectors plus cnt(count), empty flag; P(66,32) | inbox[32] |
| Echo | Shared GRU(8,16) over each field's UTF-8 byte embeddings, zero initial state; concatenate five final states and five presence flags; P(85,32) | echo[32] |
| Physical command | P(36,32), shared for previous and newly selected command | physical[32] |
| Previous sonar direction | Concatenate flag*word[32], flag[1], direction one-hot[4]; P(37,16), shared across directions | four vectors[16] |
| Previous command | Concatenate previous physical[32], four previous-sonar vectors[64], previous_valid[1]; P(97,32); multiply result by previous_valid | previous[32] |

The echo encoder shares the Word byte embedding, but does not use its byte-position table. The local convolutions use **zero padding outside the gathered crop**, not extra wrapping around the crop. Batch padding is never included in local/map/entity pooling.

### 5.3 Four iterations of spatial/entity reasoning

Decision context and initialization:

```
c = P(384,32)(concat(local64, status32, inbox32, echo32, previous32, h_old192))
b = L(32,32)(c)
z_tile   = tile   + b + type_embedding[TILE]
z_entity = entity + b + type_embedding[ENTITY]
slots[k] = learned_slot[k] + c, k=0..3
```

There are two learned type vectors[32] and four learned slot vectors[32]. The following parameters are shared across all four iterations:

* Message MLP(108,32,32) and gate L(108,1).
* Two distinct multihead attention modules: slot-to-token and token-to-slot, each model width 32, four heads of width 8. Each module has separate affine Q,K,V,O projections L(32,32), standard score scaling 1/sqrt(8), softmax over unmasked keys, and no attention dropout.
* Slot GRU(32,32), token GRU(32,32), two separate LayerNorm(32) modules, and token-input P(96,32).

One iteration is computed simultaneously from the iteration's old token states:

```
for each tile u and valid outgoing direction d:
    v = J[u,d]
    a = concat(z_tile[u], z_tile[v], D[u,d], c)       # 108
    message[u,d] = sigmoid(L_gate(a)) * MLP_message(a)
mean_u = sum_valid(message[u,d]) / 4                 # fixed denominator
max_u  = max_valid(message[u,d])                    # zero if none

T = concat_rows(z_tile, z_entity)
slot_read = Attention_slot(query=slots, keys=T, values=T)
slots_new = LN_slot(GRU_slot(slot_read, slots))
broadcast = Attention_token(query=T, keys=slots_new, values=slots_new)

for each tile token u:
    inp = P_token(concat(mean_u, max_u, broadcast[u])) # 96 -> 32
for each entity token e:
    inp = P_token(concat(zeros32, zeros32, broadcast[e]))
T_new = LN_token(GRU_token(inp, T))
```

Unseen in-board tile tokens participate in attention with their explicit unknown features; padded tokens do not. Global attention is an internal information operation, not a physical edge. No inferred traversability is written back to Facts. Slots and token latents are temporary and reinitialized on every own turn.

### 5.4 Readout and persistent recurrence

After iteration four:

* For each outgoing head direction d, gather z_tile[J[head,d]], or one shared learned unknown-destination vector[32] when J<0. Concatenate D[head,d,12] and apply shared P(44,32). This gives four directional vectors[32]. Store the final tile states for the movement decoder as well.
* Flatten slots[4,32] in their learned slot order and apply P(128,64), giving global[64].
* Concatenate mean/max over final visible entity states and apply P(64,64), giving entities[64].
* Concatenate local64, entities64, status32, inbox32, echo32, previous32, four directional vectors128, global64: **448 dimensions**.
* Apply P(448,192), then GRU(192,192) with h_old. The result h_new[192] is the actor context and next persistent state.

A single forward pass advances the persistent GRU once. Scoring a stored action uses that same context, not a second GRU update.

## 6. Exact action distributions

### 6.1 Kind and SPLIT

Kind logits are L(192,2)(h_new), ordered MOVE,SPLIT. Set SPLIT logit to −infinity exactly when L<4 or unit_count>=unit_limit. There is always a MOVE choice, even if all movement is fatal.

For SPLIT, enumerate n=2..L−2 and score with Score(198,64,1):

```
concat(h_new,
       n/A, (L-n)/A, n/L, (L-n)/L, loglen(n), loglen(L-n))
```

Softmax over all candidates and sample one. There is no top-k pruning or human-selected split catalogue.

### 6.2 MOVE

Modules: initialization L(192,64) followed by tanh; token embedding[6,8] ordered N,E,S,W,STOP,BOS; decoder GRU(18,64); shared directional Score(314,64,1); STOP Score(266,64,1).

A private geometric prefix tracker holds a hypothetical head coordinate and heading. It begins at the observed head. It follows J for each chosen direction; a missing destination makes the hypothetical position unknown for the remaining prefix. It does not predict enemy motion, body clearance, food, survival or paid-step affordability. Following a known edge is bookkeeping, not an action recommendation.

Before each token, form q[10]: prefix direction count/8; position-known flag; rel(hypothetical position)[4] or zeros; hypothetical heading one-hot[4]. Decoder steps are:

```
r_0 = tanh(L_init(h_new)); previous_token=BOS
r_t = GRU_move(concat(token_embedding[previous_token], q), r_(t-1))
for d in N,E,S,W:
    vfeat = final z at J[prefix_position,d], or learned unknown vector32
    edge  = D[prefix_position,d], or synthetic unknown-edge D[d,12]
    logit_d = Score_direction(concat(r_t64,h_new192,vfeat32,edge12,q10,onehot(d)4))
logit_STOP = Score_stop(concat(r_t64,h_new192,q10))
```

A synthetic unknown edge retains direction one-hot, edge-type UNKNOWN and zeros elsewhere. At token one, STOP is masked. At subsequent positions, all five logits are available. Sampling STOP ends the command. After eight direction tokens, terminate without sampling a ninth token or adding a STOP probability.

Fatal directions and unaffordable continuations are **not masked**. They remain possible sacrifices. v1 does not canonicalize suffixes after a predicted death; the commanded token sequence is the policy action, and the engine decides which steps execute. This avoids hidden-state-dependent safety masks. Learning useful movement is the policy's responsibility.

### 6.3 Four-direction sonar

Build C[36] for the sampled physical command, with current L as length_before, and encode it with the shared physical-command P(36,32). For each absolute direction d independently:

```
x_d = concat(h_new192, physical32, onehot(d)4)         # 228
logits_d = Score_sonar(228,64,257)(x_d)
s_d ~ Categorical(softmax(logits_d))
```

Index 0 means SILENT. Index 1+j sends the exact uint64 integer j for j in 0..255. Symbols have no assigned meanings. Directions are not rotated with the head. All four can be active. The conditional factorization is:

```
pi(action|history) = pi(kind) * pi(physical parameters|kind)
                    * product_d pi(s_d|physical command, history)
```

Serialize the physical command first, then non-silent SONAR lines in N,E,S,W order. The qualified engine determines post-action head/tail origins, delivery, interception and echoes. No ray casting or echo interpretation by the program supplies a tactical answer to the policy.

### 6.4 Probability and entropy accounting

Store the entire command, including sampled STOP if any, split size, four sonar indices, and the old joint log probability:

```
logp = logp(kind)
     + [sum sampled MOVE token logp OR logp(split size)]
     + sum_d logp(sonar index d)
```

Inactive branches and cap-forced termination contribute zero. Recompute by teacher forcing the stored command with the same kind mask and decoder order. Use one PPO ratio for the entire command.

Entropy regularization uses this explicit stored-prefix surrogate:

```
Hphysical = 0.5 * (H(kind)/log(2)
          + I_MOVE * sum_over_sampled_token_prefixes H(token)/(8*log(5))
          + I_SPLIT * H(split_size)/log(max(2,L-3)))
Hsonar = sum_d H(sonar_d)/(4*log(257))
```

A single legal split size has entropy zero. Token entropy includes only unmasked logits. Hphysical is not claimed to be exact joint autoregressive entropy. All head temperatures are one; there is no heuristic action postprocessing at deployment.

## 7. Initialization and parameter accounting

Initialize affine and convolution weights with orthogonal initialization, gain sqrt(2), and zero biases; flatten convolution input-channel/kernel dimensions into one matrix axis for this operation. Initialize each GRU input/recurrent gate matrix separately with gain 1 and zero biases. LayerNorm scale=1, bias=0. All embeddings, learned slots and the unknown-destination vector use independent Normal(0,0.02).

Override action-output weights with orthogonal gain 0.01. Kind biases are [log(19),0], favoring MOVE initially when both kinds are legal. Direction final bias is zero; STOP final bias is log(8), making initial long sprints uncommon without forbidding them. Split final bias is zero. Sonar final biases are [log(256),0,...,0], giving approximately half silence and half emission at initialization. Features can slightly perturb these starting probabilities.

| Actor component | Parameters |
|---|---:|
| Local encoder | 31,616 |
| Tile encoder | 2,880 |
| Entity encoder | 1,728 |
| Status encoder | 2,112 |
| Shared word encoder and byte/position tables | 5,248 |
| Inbox pool | 2,144 |
| Echo encoder | 4,000 |
| Physical/previous command encoders | 4,928 |
| Reasoning context, types, slots, messages, attention, GRUs, normalization | 42,573 |
| Direction/global/entity readouts and unknown vector | 13,888 |
| Fusion and persistent GRU | 308,544 |
| Kind head | 386 |
| Movement decoder | 65,906 |
| Split scorer | 12,801 |
| Sonar head | 31,361 |
| **Total** | **530,115** |

Counting rules: L(a,b)=(a+1)b; convolution includes one bias per output channel; GRU as defined above; LayerNorm(d)=2d. Shared parameters are counted once. Raw float32 actor weights occupy 2,120,460 bytes, approximately 2.02 MiB. This does not include executable code, parsing buffers, scratch arrays or private facts.

## 8. Exact centralized critic

The critic is training-only, shares no parameters with the actor, and produces V(S,p) in [−1,1] for each team p using the same critic weights from that team's perspective. Evaluate it before the event's actor consumes its next observation. Privileged inputs never enter actor tensors or masks.

### Inputs

Full-map F[N,28], in order: current pearl[1], bed flag[1], transformed actual countdown[1]; occupancy one-hot empty/ally-head/ally-body/enemy-head/enemy-body[5]; directional edge types empty/kelp/portal[4,3]; destination-known flags[4], all one for traversable true destinations and zero for kelp; abs(tile coordinates)[4]. Use full actual portal transitions for critic J. All team labels are relative to p. The critic knows current state, not future RNG draws.

Per living dragon R[207]:

```
ally/enemy one-hot[2], length/A[1], loglen(length)[1],
abs(head)[4], facing one-hot[4],
rank among living IDs/max(alive_count-1,1)[1],
already acted this round[1],
stored actor h before this event[192], private seen fraction[1]
```

Training opponents are frozen copies of this architecture, so their detached h is available to the training-only critic. A newborn not yet acting has zero h and zero coverage. These summaries need not make the critic fully Markov; value estimation remains an approximation.

Global Q[16], in order: round/499, remaining rounds/500, W/64,H/64; each team's max length/A[2], total length/A[2], living count/unit_limit[2]; next actor ally/enemy one-hot[2]; next actor normalized living-ID rank[1]; unit_limit/64[1]; fraction of living dragons still due this round[1]; next actor length/A[1]. Within each team pair, p precedes its opponent.

### Network

```
z = MLP(28,64,64)(F)
repeat 4 times with shared weights:
    a[u] = sum_{valid d} L_neighbor(64,64)(z[J[u,d]]) / 4
    z_new[u] = LN64(z[u] + Score_update(128,64,64)(concat(z[u],a[u])))
map_summary = concat(mean(z),max(z))                     # 128
r = MLP(207,64,64)(R)
entity_summary = concat(mean(r),max(r))                  # 128
v = tanh(L(64,1)(SiLU(L(128,64)(SiLU(L(272,128)(
        concat(map_summary,entity_summary,Q)))))))
```

LayerNorm64 uses epsilon 1e−5. Empty entity pools are zero; terminal states need no critic call. Critic parameter count is **83,457**. Initialize by the non-action rules in section 7; final value layer gain=1. No target critic, value clipping, auxiliary head or reward shaping is used.

## 9. Event records, returns and PPO

### 9.1 Collection timeline

An event is the boundary immediately before the next living dragon acts, after any pearl tick. Record its pre-action critic state and both team values. Generate that dragon's actual observation, act, execute the command, then process death skips, complete-round adjudication and any next-round tick until the next event or true terminal.

Each match has exactly one learner-controlled team. The other team uses a fixed opponent snapshot for the entire match. In same-version mirror play, the opponent is still frozen and excluded from actor loss. Collect 32 complete matches under one frozen learner behavior version. There is no artificial episode cutoff, mid-match weight update or reset on an individual death.

Actor records contain: match/lifetime/event IDs, raw legal observation delta, previous command, full issued command, old logp, behavior version, state-schema hash and action-codec hash. Event records contain: pre-action full-state delta/snapshot reference, both old values, next actor/round phase, detached private h/coverage summaries, and final official result reference. No dense full actor map is saved at every step; factual state can be reconstructed from birth and observation deltas. Full critic state uses lossless event deltas and a snapshot every 64 events.

### 9.2 Return and advantage

Let G_m,p be +1,0,−1 from the official result of match m. Gamma=1. For every event e in that match:

```
critic_target[e,p] = G_m,p
A[e,p] = G_m,p - V_old(S_e,p)
```

For each learner actor decision, use its team's A at the attached event. This includes actions of parents or children that died long before the result. There is no individual reward stream, birth reward, survival penalty, event discount, GAE or bootstrap. Complete-return variance is an explicit cost of this baseline. Do not label a collection failure or interrupted match as a loss: discard its unfinished records and replace the match under the same behavior version.

### 9.3 Exact losses

There are M=32 learner team-episodes per batch. Let I be all learner actor decisions, and T all unique team-event pairs, including both teams. Advantages and old values/logp are frozen across the three PPO epochs. Do not normalize advantages; their scale is bounded by terminal rewards and tanh values.

```
ratio_i = exp(logp_new_i - logp_old_i)
clip_i  = clip(ratio_i, 0.9, 1.1)
Lpolicy = -sum_i min(ratio_i*A_i, clip_i*A_i) / (512*M)
Lentropy = -sum_i (0.005*Hphysical_i + 0.001*Hsonar_i) / (512*M)
Lactor = Lpolicy + Lentropy
Lcritic = sum_(e,p in T) (V_new(S_e,p)-G_m,p)^2 / |T|
```

The factor 512 is a fixed numerical scale independent of actions, match length or number of children. Summing actor decisions per team-episode follows the trajectory-policy-gradient form; extra decisions do not create extra rewards. Do not independently average each dragon's lifetime loss. Value samples occur once per team-event, not per action token or actor lifetime.

### 9.4 Recurrent replay and optimizer steps

Every PPO epoch holds current weights fixed until all its gradients have accumulated. Replay **every lifetime from its birth with h=0 and empty facts**, consuming the recorded observations and teacher-forcing its recorded actions. Rebuild private facts exactly. Split replay into 64-decision gradient chunks. At a chunk boundary detach h, but carry its computed value and continue factual history. Newborns lose no prefix. No burn-in or old hidden-state snapshot is used for actor replay.

This gives current-weight forward states for the complete recorded history; gradients are truncated at 64 own decisions. It removes hidden-state staleness from the forward pass, not the limitation of truncated recurrent credit. Received messages remain the recorded behavior observations; PPO does not regenerate another actor's counterfactual messages.

Stream up to four chunks from independent lifetimes per microbatch, preserving order within each lifetime. Backpropagate contributions with the full-batch denominators above. After the entire batch, clip actor gradient norm to 0.5 and take **one** actor Adam step. Independently accumulate full critic loss with its own parameters, clip norm to 0.5 and take **one** critic Adam step. There are at most three of each per collected batch. Do not step either optimizer between microbatches.

Before each candidate actor update, compute over I:

```
KL_est = mean_i(exp(x_i)-1-x_i), x_i=logp_new_i-logp_old_i
```

If nonfinite, abort the update and diagnose. If KL_est>0.02, discard that candidate actor gradient and stop remaining actor epochs for this batch; critic epochs may finish. This is an empirical joint-command KL estimator, not a tokenwise threshold. Log entropy, clip fraction, joint KL, value MSE, action counts and official match results.

Optimizers: Adam, betas=(0.9,0.999), epsilon=1e−5, weight_decay=0. No mixed precision or gradient scaling in the baseline. Use identical actor/critic learning rate at update u:

```
lr(u) = 3e-4 + (3e-5 - 3e-4)*u/1999, u=0..1999
```

## 10. Reproducible training sequence

### 10.1 Run inputs and qualification

A run requires a frozen `run_manifest.json` with schema `dragons-competitive-v1`:

```
engine_release_hash
map entries: {path, sha256, family_id, split: train|validation|test, weight}
validation opponents: {artifact_path, sha256, adapter_id}
final-test opponents: {artifact_path, sha256, adapter_id}
training_seed
protocol_fixture_results_hash
```

Map files and opponent artifacts are experiment inputs, not facts loaded into an actor. Validate hashes before collection. All variants in a map family belong to the same split. Positive training weights are normalized to a categorical distribution; default weight is one. Validation/test weights are equal. Require nonempty splits and opponent lists before claiming a complete reference run. If external benchmarks are unavailable, report internal self-play results as such; do not call them evidence of beating other bots.

Do not alter rules, shorten matches, randomize the official fixed pearl RNG, or assign navigation rewards. Any map randomness is embodied in legitimate, hashed map files. Geometry variety is part of the match distribution, with no privileged maze quota. Policy sampling can vary even when the engine's RNG is fixed.

Before update 0, pass section 12's correctness/runtime gates. Initialize actor and critic, save initial actor snapshot S0, and freeze the training manifest. No existing model checkpoint is required.

### 10.2 Randomness and matching

Reference training seeds are 0,1,2 in separate runs. Use a PCG64 root RNG per run for map/opponent selection and shuffling, and seeded model initialization. Derive a private sampling seed for each lifetime from `(training_seed, update_index, match_index, team, dragon_id)` using SHA-256, interpreted as a little-endian uint64 from the first eight digest bytes. The encoded seed input is five uint64 little-endian integers, not language-dependent tuple serialization. Each lifetime owns a PCG64 stream; newborns do not inherit a parent's stream. The learner and opponent sample categorical actions at temperature one.

For reproducible categorical sampling, consume one uniform float64 value per sampled categorical, form probabilities by float32 softmax, then accumulate them in index order in float64 and normalize by their final sum. Select the first cumulative probability strictly exceeding the uniform value, with the last valid category as the rounding fallback. Masked categories have probability zero. For deployment outside a training run, replace the first three seed fields with a configured run seed, zero and match identifier supplied to the process; absent a match identifier, use zero. These values seed private randomness only and never enter model features.

For each batch, assign 16 learner matches to team A and 16 to team B, then shuffle these assignments. Sample each map independently from the fixed training weights. Batch across independent matches; never batch several stale turns of the same match as simultaneous actions.

### 10.3 Exact league sampling

Save a frozen historical snapshot after every 20 completed updates; retain all such snapshots for the run. Opponent selection for each match is:

* 20%: a frozen copy of the current batch's behavior actor.
* 40%: uniform among the eight most recent historical snapshots, or all if fewer than eight.
* 40%: uniform among older historical snapshots excluding those recent eight.

If the older set is empty, assign its probability to the recent set. If history is empty, assign all probability to the frozen behavior actor. S0 is included in history before update 0. Snapshots carry their own private runtime states per opponent lifetime. Never average weights or share hidden states between snapshots.

This is the entire baseline league algorithm. There is no underspecified exploiter process, adaptive difficulty curriculum or staged sonar vocabulary. Exploiters are an experiment after the reference run.

### 10.4 Reference loop

```
initialize actor, critic, optimizers, RNGs, history=[S0]
for u in 0..1999:
    behavior = frozen copy of actor
    select 32 maps, sides and frozen opponents
    collect all 32 complete matches; freeze logp, V_old and results
    derive complete-return advantages and targets
    for epoch in 0..2:
        replay actor lifetimes from birth, with 64-decision gradient chunks
        accumulate full-batch actor loss; apply KL rule and optimizer step
        replay privileged events; accumulate full-batch critic loss and step
    save resume checkpoint
    if (u+1)%20 == 0: append frozen actor to history
    if (u+1)%100 == 0: run fixed validation procedure
run final selection and untouched test procedure
```

All movement, splitting and four sonar heads train jointly from the first update. There is no competence gate for a named behavior. The reference loop collects 64,000 complete matches and takes at most 6,000 actor updates per seed. Report actual dragon decisions, round-work, matches and wall time; none can be inferred from another when splitting changes the number of actors.

A resume checkpoint is saved at completed-update boundaries only and contains actor/critic/optimizer states, update number, all RNG states, history snapshot hashes, training manifest, schema/codec hashes and validation records. Mid-match resume is outside v1. Checkpoints with mismatched schema, dimensions, move cap or sonar vocabulary fail explicitly.

This document specifies a run; writing it does not launch one or establish that 2,000 updates suffice for competitive strength.

## 11. Validation and checkpoint selection

The primary metric is `score = win + 0.5*draw`, averaged over the fixed validation opponent/map distribution. The training reward remains +1/0/−1; these metrics induce the same ordering in expectation.

Every 100 updates, evaluate the candidate against every validation map/opponent pair, both team assignments, with four policy-sampling seeds 100,101,102,103. For sampling-seed derivation use each of these as training_seed, use update_index=0, and use the pair's index in lexicographic `(map hash,opponent hash)` order as match_index; include actual team and dragon_id as usual. Candidate and incumbent use the same seed construction, independent of checkpoint update. The engine's pearl RNG remains unchanged. Run the current selected incumbent under the same manifest if its results are not cached. Average sides and sampling seeds within each `(map hash, opponent hash)` group.

Compare candidate and incumbent using the paired group score differences. Bootstrap groups with replacement 10,000 times using PCG64 seed 777. Promote only when the 2.5th percentile of the mean difference is >0 and all correctness/runtime gates pass. Start the incumbent at S0. Report the actual number of groups and interval width; small pools can make promotion inconclusive. Do not move validation failures into training while still claiming this validation set is held out.

After update 2,000, select the promoted incumbent, not automatically the last checkpoint. Evaluate once on every untouched test-map/test-opponent pair, both sides, four seeds 200..203. Report W/D/L, score, paired confidence intervals, runtime and the selected update number. Summarize each training seed separately before aggregating; a single run is exploratory evidence.

Diagnostic measurements explain losses but do not choose the champion independently of competitive score:

* Geometry/memory: long detours, identical local views after different histories, dead-end revisits, unseen portal exits, and recall beyond 64 turns.
* Interaction: head trades, body traps, ID-order disadvantage, sprint costs, split outcomes, newborn survival and late-game score changes.
* Sonar: per-direction emission, tail-origin cases, echo dependence, payload dependence and hostile messages.

An evaluation-only path solver may report a lower bound for a diagnostic; no output enters training or actor inference. A policy need not maximize coverage, minimize travel, split frequently, or ping every direction every turn. It should win.

## 12. Acceptance tests and runtime gates

These are requirements for a later implementation, not tests claimed to have been run while writing this document.

1. **Protocol:** pass the modified-sonar fixtures in section 2 and official movement, growth, split, newborn scheduling, collision, pearl and result fixtures. Parse/serialize 0 and 2^64−1 without precision loss.
2. **Privacy:** changing unseen state while holding a lifetime's observation/action history fixed cannot change any actor tensor, mask, state or action distribution. Another dragon's private state cannot affect it except through genuine game observations/messages.
3. **Geometry:** canonical seams and portal side/direction mapping are consistent. Unknown is distinct from empty. Kelp does not hide visible tiles. No received word creates a factual edge. No multihop route computation occurs in actor preparation.
4. **Lifecycle:** cold-start children; retained parent h/facts; no stale state after death/reset; body UNKNOWN propagation; repeated input idempotence.
5. **Network contract:** tile width 56, entity width 20, status width 32, context input 384, fusion input 448, movement score inputs 314/266; exactly 530,115 actor and 83,457 critic parameters. Batching/padding does not alter unpadded outputs.
6. **Actions:** every split size available when legal; STOP cannot be first; no ninth move token; fatal moves remain selectable; all four sonar outputs may be active; emitting zero differs from silence. Sampled and teacher-forced joint logp agree within 1e−5 float32 tolerance.
7. **Returns:** every prior learner action, including a dead parent's, gets G−V_old from the actual final team result. Extra births do not create rewards. No individual death causes a team bootstrap or terminal cutoff.
8. **Replay:** forward replay from birth reproduces rollout h/logits under unchanged weights. Chunking with detach changes gradients but not forward values. Every scored decision appears once per epoch. Full-batch and microbatch accumulation agree within numerical tolerance.
9. **Optimization:** finite losses/gradients reach every active encoder, reasoning module and action branch over a fixture batch exercising all branches. A tiny end-to-end learning run changes parameters and preserves all invariants. This is not a strength benchmark.
10. **Export:** long-sequence reference/deployment parity, deterministic seeded categorical sampling and exact unsigned protocol formatting. Diagnose near-tied logits separately; do not silently change inference to greedy.

Existing documentary specs report 48 MB memory, 100M CPU points/turn, 10 seconds wall time, 10 KB output and a 4 MB zipped submission. Confirm those limits against the target judge and measure the complete deployed process. The approximately 2.02 MiB weights do not prove a budget fit. At 64×64, a float32[N,56] feature map alone is 917,504 bytes; token buffers, observations, weights, facts and Python/runtime overhead also count.

Required stress cases: 64×64 maximum memory history; high visible occupancy; full inbox; long echo tokens within the protocol's valid limits; length near board capacity for split enumeration; eight-step decoder; all four emissions; many successive turns. Stream per-edge messages and inbox pooling, and reuse inference scratch. Do not create an N×N attention matrix: the specified global attention is 4×(N+E).

If v1 fails a resource gate, report it as a failed candidate. Changing width, reasoning iterations or decoding limits requires a new version and fresh evaluation; the implementation must not quietly choose a smaller model while claiming this specification.

## 13. Separate experiments, not baseline ambiguities

Only after the fixed reference is implemented and evaluated: compare reasoning iterations 4/8/16, a fresh local-only baseline, longer recurrent gradient windows, full-width learned transmit payloads, learned spatial memory, auxiliary prediction losses, GAE, opponent exploiters or smaller deployment models. Each changes one declared factor, uses matched compute where possible, and is judged by held-out competitive outcomes. None is an unresolved instruction in v1.

Documentary provenance: `dragons-env/game.md` for base rules; `dragons-env/README.md` for its modified-1.0.0 sonar note; `SNAKE_MARL_IMPLEMENTATION_SPEC.md` and `SPATIAL_MEMORY_RL_IMPLEMENTATION_SPEC.md` for information/lifecycle/training concerns; user clarification that the shipped game supports four directional sonars. Older forward-only and uint32 prescriptions are superseded. The exact new-ray mechanics remain an external adapter-conformance requirement; the model interface and baseline learning algorithm above are fixed.
