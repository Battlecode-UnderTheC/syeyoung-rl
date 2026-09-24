# dragons-env API changes this design assumes

The env stays the game: rules, the judge's observation, parity with the
official engine. What changes is the shape of its interface.

## Actions

```cpp
namespace dragons {
struct Action {
  struct Move { std::vector<uint8_t> steps; };   // directions 0..3, at least one
  struct Split { int32_t child_segments; };
  struct None {};                                // the engine's "no valid action"
  std::variant<Move, Split, None> what;
  std::optional<uint32_t> sonar;
};
void Game::Act(const Action& a);                 // replaces Act(kind, steps, nsteps, split, sonar)
}
```

## Stepping many games

```cpp
struct ActionsView {                             // one action per env, in env order
  std::span<const Action> actions;
};

struct GameEnd {
  int32_t env;
  Result result;                                 // winner, end reason, standings
  int32_t final_round;
};

struct StepResult {                              // valid until the next Step or Reset
  std::span<const Death> deaths;                 // env order, then engine order
  std::span<const GameEnd> ended;                // games that ended (and were restarted)
};

class VecEnv {
 public:
  VecEnv(const VecEnvConfig& config, NextMap next_map);   // next_map: the board for each new game
  StepResult Step(ActionsView actions);
  std::span<const uint8_t> Observation(int32_t env) const; // the acting dragon's record
  const Game& game(int32_t env) const;                    // full state, read-only (critic features)
};
```

- No public data members; results are returned as views.
- The board for each new game comes from a caller-supplied function, so the trainer's
  `MapSource` (files, generated, mixtures) plugs in without the env knowing about it.

## Removed from the env

- The critic's state summary (`WriteState`, the per-block counters in `Game`): moves to
  `train/critic.hpp`, built from `game(env)`.
- `Game::Prefetch`, `cell()`, `occ_pos()` leave the public interface; the fast observer
  becomes an internal part of VecEnv.
