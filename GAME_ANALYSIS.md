# UNSW Battlecode "dragons" — verified game analysis

Everything below was established by **running the real engine**
(`unswbc_engine.wasm`, toolkit `unswbc` 0.3.6) with scripted probe bots, not by
reading documentation. Probe scripts: `/private/tmp/claude-501/-Users-syeyoung-Documents-postech-unswbattlecode-rl/dad3b4c0-152d-4358-9fcb-15e389945264/scratchpad/probe/`.

There is **no Python simulator in this repo** — the rules live only in the wasm
binary. `unswbc/engine.py` drives it through wasmtime with a `bot_reply(dragon_id,
block) -> bytes` callback, so the engine can be run **in-process** as an RL
environment (~20k dragon-turns/s single-threaded; ~5–10 full matches/s).

## Board

* Torus: x and y wrap. Maps are 11×11 … 64×64.
* Tiles hold at most one **pearl** and carry a respawn timer.
* **Edges** between tiles are `EMPTY`, `KELP` (impassable) or `PORTAL`.
* Map file format (fully decoded, custom maps verified):
  * `TILE x y minGap maxGap` — pearl respawn delay drawn from `[minGap,maxGap]`;
    `0 0` = never spawns (timer reported as `-1`).
  * `EDGE index kind portalId` with
    `index = y*2*(w+1) + x` (north edge of `(x,y)`) and
    `index = y*2*(w+1) + (w+1) + x` (west edge of `(x,y)`).
    Row `y=h` and column `x=w` mirror row/column 0; the engine validates the mirror.
  * Portals pair two edges of the **same orientation** (mixing them is a map error).
  * `DRAGON team n x y x y …` — head first; several per team is normal
    (`default.map` starts 4 a side, `big_empty.map` 3).
  * `UNIT_LIMIT` defaults to **64**.
* Runs are **deterministic**: the engine seeds a Mersenne twister from a **fixed
  constant**, so a given map always produces identical pearl timers. There is no
  seed directive, and the map name does not enter the seed — training variety has
  to come from generating different maps.
* The smallest legal board is **7x7**, where a single dragon's window covers every
  tile. That makes a board fully observable through one dragon, which is what
  makes exact differential testing against a reimplementation cheap.

## Round structure — *sequential*, not simultaneous

500 rounds, numbered 0–499. Within a round dragons act **one at a time in dragon-id
order**, and each one observes the board *after* the earlier dragons of that round
have already moved. A dragon created by a split takes the next free id and therefore
**acts later in the same round it was born**.

## What a dragon observes

At spawn: own id, team, map size, unit limit. Each turn: `ROUND`, own `DIR`, own
`LENGTH`, own team's `UNIT_COUNT`, sonar messages received since its last turn, a
**7×7 window centred on its head** (per tile: x, y, hasPearl, pearlTimer), every
dragon part in that window (team, dragon id, x, y, facing, isHead), and that
window's edges.

* Vision is a plain rectangle: **kelp does not block sight**, and you **cannot see
  through a portal** (the window shows the geometric neighbour, not the exit).
* No score, no ally positions outside vision, no enemy count, no time limit.

## Actions — one per turn, last one wins

`LOG`, `DOT`, `LINE`, `INDICATOR` and `SONAR` are *not* actions; `MOVE` and `SPLIT`
are, and only the **last** of them in a turn is applied.

| Command | Rules (verified) |
|---|---|
| `MOVE <dir>` | one step, absolute N/E/S/W |
| `MOVE <dirs…>` (sprint) | k steps in one turn, costs **k−1** length, needs `k ≤ L−1` (min length 2). Steps resolve one at a time — a collision on **any** intermediate tile kills. Eats every pearl along the path. |
| `SPLIT n` | detaches the last `n` segments as a new dragon; child's head is the old tail tip, **facing reversed**, new id, acts later the same round. Needs `n ≥ 2`, parent remainder `≥ 2`, team units `<` unit limit. **The parent does not move that turn.** |
| `SONAR v` | uint32 (taken mod 2³²); sent after the action resolves; one per turn |

**A turn with no legal action is death** (reason `A`): nothing sent, malformed or
mis-cased command, unaffordable sprint, illegal split, or split past the unit limit.

## Death

| Reason | Cause |
|---|---|
| `W` | crossed a kelp edge |
| `S` | moved into own body — including a **reverse** and including your own tail tile (**no tail-following**) |
| `O` | moved into any other dragon's body (ally or enemy) |
| `H` | moved onto an enemy **head** — **both dragons die, regardless of length** |
| `A` | no valid action |

No respawn. A dead dragon **drops pearls on alternating tiles of its corpse** — so
kills feed the killer and dying feeds the opponent. From the decompiled `Kill`:

```
for (c = 0; c < length; c += 2)  tiles[segment[c]].pearl = 1
alive = 0
```

i.e. segment indices 0, 2, 4, … from the head, written unconditionally (an
earlier guess that it skips tiles already holding a pearl was wrong — overlapping
corpses just write over each other).

## Pearls

Eating one gives **+1 length**. Each tile counts down to its next spawn attempt;
the timer is re-drawn from `[minGap,maxGap]` after a spawn.

## Sonar — the only communication channel

A uint32 travels in a straight line from the head along its facing, **instantly**:

* unlimited range — it **wraps around the torus**,
* **passes through portals**,
* is **blocked by kelp**,
* is delivered to the **first dragon part it meets** — ally, enemy, or the sender's
  own body (a ray that circles an empty row comes back to you).

The receiver reads it on its **next turn**: same round if it acts later in the id
order, otherwise the following round. One message per sender per turn.

## Portals

Crossing a portal edge teleports the head to the tile on the far side of the partner
edge, **keeping direction**; the body trails through over the following turns (the
dragon is split across two places meanwhile).

## Win condition — verified by controlled matches

At the end of round 500:

1. **Longest single living dragon** wins. (Verified: one 10 beats two 7s, i.e.
   total 10 beats total 14; one 9 beats two 8s.)
2. Tie → **greater total living length** wins. (Verified: equal max 10, totals 12 vs 15 → B.)
3. Still tied → **draw**. Dragon *count* is never a tiebreak.

Earlier: a team with no dragons left loses by elimination; both eliminated in the
same round is a draw. Note the CLI's `a_length`/`b_length` are **totals**, which is
*not* the deciding quantity.

## Execution model and deployment budget

* **One OS process per dragon**, forked from a pre-warmed zygote. Private per-dragon
  memory is enforced by the runtime, and a split child is a brand-new process with
  **no parent state**.
* Judge sandbox (`--sandbox` reproduces it): **100M wasm CPU points per turn**,
  **768 pages = 48 MB** memory, 10 s wall clock, 10 KB output per turn, submission
  zip **≤ 4 MB**, only files matching `bot.toml` `include` (`*.py`).
* **numpy 2.5.3 is pre-imported in the sandbox image** — importing it costs ~0.
* Measured costs: plain `helper.py` turn **≈ 3.9M points** (16M on turn 0);
  a forward pass of 12ch 7×7 → conv32 → conv64/s2 → Linear(1024,256) → GRU(256)
  **≈ 34M points, 27 MB RSS** — fits with ~2.5× headroom.
* Generating weights at runtime (`standard_normal`) **exceeds the budget**. Embed
  them as a `bytes` literal and `np.frombuffer` (400k float32 / 4.4 MB source works;
  700k / 8 MB fails to compile in the sandbox). The 4 MB zip cap means quantizing.

## Deltas against `SNAKE_MARL_IMPLEMENTATION_SPEC.md`

**Confirmed by the engine**

* Lexicographic score `(max living length, total living length)` — exactly §2.
* Split reduces the primary score (100 → 50+50) and costs no total length.
* The 4-byte message is real (uint32), travels forward physically, and is
  interceptable by enemies — §5's model is right, and richer than assumed.
* Zero-memory children and private per-snake memory are enforced by the runtime.

**Wrong or missing in the spec**

1. **Turn order is sequential, not simultaneous.** §1's "snapshot pre-action
   observations, then step simultaneously" does not match the engine. An arena
   "timestep" is a *round* containing N ordered decisions, and later movers see
   strictly more information. The adapter and the alignment tests must model this.
2. **The action space is different.** Not 3-way relative steering + sprint boolean:
   it is absolute N/E/S/W (reversing is legal to *send* and simply kills you), with
   sprint as a *variable-length direction sequence*, plus `SPLIT n` (which replaces
   the move), plus sonar. A workable discretization: `4 directions × sprint length
   1..k_max`, plus `split n`, exactly one per turn.
3. **Total length is not conserved** — deaths remove it and return ≈ half as pearls,
   so a `B_max` bound for potential shaping must be derived from map pearl supply,
   not from a conservation argument.
4. **Most maps start several dragons per side**, so the §7 "one snake per side"
   curriculum stage needs a custom map (the format is decoded; a generator exists in
   the probe directory).
5. **Illegal actions are fatal, not ignored.** Action masking is not an optimization
   here — an unaffordable sprint or an over-limit split kills the dragon.
6. **Head-to-head always kills both, regardless of length.** Combined with the
   longest-dragon score, the dominant tactic to model is trading a cheap 2-segment
   child for the opponent's longest dragon.


## Appendix: rules read out of the engine binary

`wasm-decompile --enable-all` (wabt) lifts `unswbc_engine.wasm` to pseudocode, and
the binary keeps its C++ name section, so the rule functions are directly
readable: `Step`, `Move`, `Split`, `CastSonar`, `Kill`, `PearlTick`,
`ResultAfterRound`, `TileAfterStep`, `ReadReply`. What follows is from the
binary, not inferred from experiments.

**`ResultAfterRound`** — per living dragon, `team = byte@+4` (`'A'`=65),
`length = int@+28`, `alive = byte@+48`; accumulate count, total and max per team.
If either side has no dragons: 0/0 is a draw, otherwise the side with dragons
wins. Otherwise, once `forced || round >= 499`: compare `max`, then `total`,
then draw. The lexicographic `(max, total)` rule, confirmed.

**`Step(state, dragon, dir, pay)`** in order:

1. `facing = dir`
2. `TileAfterStep` invalid (kelp) → `Kill('W')`
3. destination is one of the dragon's own segments → `Kill('S')`
4. destination occupied by another dragon → head of it? `Kill(other,'H')` **and**
   `Kill(self,'H')`; otherwise `Kill(self,'O')`
5. push the new head; if the tile holds a pearl, clear it and **keep the tail**
   (that is the growth), otherwise pop the tail

**`Move`** — `Step(dirs[0], pay=0)`, then for every later direction: **`if
(length > 2)`** `Step(dir, pay=1)` else `Kill('A')`. The affordability test is
re-checked *before each step*, so eating a pearl part-way through a sprint pays
for extra steps. A mask derived from the starting length alone is therefore
conservative, not exact.

**`Split(state, dragon, n)`** — rejects `n < 2` or `n > length - 2`, then rejects
`AliveUnitCount(team) >= unit_limit`; either rejection throws and the dragon dies
of `'A'`. The child takes the next id from a global counter.

**`CastSonar`** — walks `TileAfterStep` from the head along the dragon's *current*
facing (i.e. after its action), stopping at kelp or at the first tile an alive
dragon occupies, for at most **`width + height`** steps; the message is appended
to that dragon's inbox. Portals are traversed like any other step.

**`PearlTick`** — decrements each tile's countdown (and writes it through to the
mirrored tile, so symmetric maps stay in sync); at zero it plants a pearl **only
if the tile has no pearl and no living dragon is standing on it**.

**`TileAfterStep`** — edge byte `'#'` (35) is kelp and blocks; `'P'` (80) is a
portal carrying its partner tile and orientation; anything else steps one tile
and wraps modulo the board.

Death reason codes: `'W'`=87 wall, `'S'`=83 self, `'H'`=72 head-to-head, `'O'`=79
other dragon, `'A'`=65 no valid action.
