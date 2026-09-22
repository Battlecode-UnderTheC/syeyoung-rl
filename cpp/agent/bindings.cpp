#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "agent/encoder.hpp"
#include "dragons/game.hpp"

namespace py = pybind11;
using namespace dragons;

namespace {

py::dict DragonToDict(const Dragon& dragon) {
  py::list segments;
  for (const Point& p : dragon.segments) segments.append(py::make_tuple(p.x, p.y));
  py::dict out;
  out["id"] = dragon.id;
  out["team"] = dragon.team;
  out["alive"] = dragon.alive;
  out["facing"] = std::string(1, "NESW"[static_cast<int>(dragon.facing)]);
  out["segments"] = segments;
  out["length"] = dragon.length();
  out["inbox"] = dragon.inbox;
  return out;
}

}  // namespace

void BindSpatial(py::module_& m);

PYBIND11_MODULE(_dragons, m) {
  m.doc() = "Native reimplementation of the UNSW Battlecode dragons engine";
  BindSpatial(m);

  py::class_<Action>(m, "Action")
      .def(py::init<>())
      .def_static("move", [](const std::string& dirs, bool sonar, uint32_t payload) {
        Action action;
        action.kind = Action::Kind::Move;
        for (char c : dirs) {
          const char* found = std::strchr("NESW", c);
          if (!found) throw std::invalid_argument("bad direction");
          action.dirs.push_back(static_cast<Dir>(found - "NESW"));
        }
        action.sonar = sonar;
        action.payload = payload;
        return action;
      }, py::arg("dirs"), py::arg("sonar") = false, py::arg("payload") = 0)
      .def_static("split", [](int size, bool sonar, uint32_t payload) {
        Action action;
        action.kind = Action::Kind::Split;
        action.split_size = size;
        action.sonar = sonar;
        action.payload = payload;
        return action;
      }, py::arg("size"), py::arg("sonar") = false, py::arg("payload") = 0)
      .def_static("none", [] { return Action{}; });

  py::class_<Game>(m, "Game")
      .def_static("from_map_text", &Game::FromMapText,
                  py::arg("text"), py::arg("seed") = 0)
      .def_property_readonly("width", &Game::width)
      .def_property_readonly("height", &Game::height)
      .def_property_readonly("round", &Game::round)
      .def("turn_order", &Game::TurnOrder)
      .def("begin_round", &Game::BeginRound)
      .def("take_turn", &Game::TakeTurn)
      .def("end_round", &Game::EndRound)
      .def("finished", &Game::Finished)
      .def("result", [](const Game& game) {
        switch (game.Result()) {
          case Outcome::TeamA: return std::string("A");
          case Outcome::TeamB: return std::string("B");
          case Outcome::Draw: return std::string("draw");
          default: return std::string("running");
        }
      })
      .def("team_score", [](const Game& game, int team) {
        const Score score = game.TeamScore(static_cast<uint8_t>(team));
        return py::make_tuple(score.max_length, score.total_length, score.dragons);
      })
      .def("dragons", [](const Game& game) {
        py::list out;
        for (const Dragon& dragon : game.dragons()) out.append(DragonToDict(dragon));
        return out;
      })
      .def("pearls", [](const Game& game) {
        py::list out;
        for (int y = 0; y < game.height(); ++y) {
          for (int x = 0; x < game.width(); ++x) {
            if (game.TileAt(Point{static_cast<int16_t>(x), static_cast<int16_t>(y)}).pearl) {
              out.append(py::make_tuple(x, y));
            }
          }
        }
        return out;
      })
      .def("deaths", [](const Game& game) {
        py::list out;
        for (const DeathEvent& death : game.deaths()) {
          out.append(py::make_tuple(death.dragon_id, death.round,
                                    std::string(1, static_cast<char>(death.reason))));
        }
        return out;
      })
      // dragons-spatial-v1: exact protocol text and full state
      .def_property_readonly("unit_limit", &Game::unit_limit)
      .def("alive_unit_count", [](const Game& game, int team) {
        return game.AliveUnitCount(static_cast<uint8_t>(team));
      })
      .def("spawn_block", &SpawnBlock)
      .def("turn_block", &TurnBlock)
      .def("tiles", [](const Game& game) {
        py::array_t<uint8_t> pearl({game.height(), game.width()});
        py::array_t<int32_t> countdown({game.height(), game.width()});
        DumpTiles(game, pearl.mutable_data(), countdown.mutable_data());
        return py::make_tuple(pearl, countdown);
      })
      .def("edges", [](const Game& game) {
        py::array_t<uint8_t> north({game.height(), game.width()});
        py::array_t<uint8_t> west({game.height(), game.width()});
        DumpEdges(game, north.mutable_data(), west.mutable_data());
        return py::make_tuple(north, west);
      });
}


// ---- dragons-spatial-v1: the native lifetime state, for parity tests --------
#include "agent/spatial/codec.hpp"
#include "agent/spatial/memory.hpp"

namespace {

using dragons::spatial::LifetimeState;
using dragons::spatial::PrefixSimulator;
using dragons::spatial::Window;

py::dict PrepareDict(LifetimeState& state, const Game& game, int index,
                     const std::vector<int>& portal_ids) {
  const Window w = dragons::spatial::ReadWindow(game, index, portal_ids);
  const int W = state.width(), H = state.height();
  namespace sp = dragons::spatial;
  py::array_t<float> local({sp::kFactChannels, sp::kVision, sp::kVision});
  py::array_t<float> facts({sp::kFactChannels, H, W});
  py::array_t<int64_t> visible_xy({sp::kTiles, 2});
  py::array_t<float> visible_prev({sp::kTiles, sp::kFactChannels});
  py::array_t<float> visible_age(sp::kTiles);
  py::array_t<float> status(sp::kStatus);
  py::array_t<float> history({sp::kHistoryTurns, sp::kHistoryFeatures});
  py::array_t<bool> history_mask(sp::kHistoryTurns);
  py::array_t<float> entities({sp::kMaxEntities, sp::kEntityFeatures});
  py::array_t<bool> entity_mask(sp::kMaxEntities);
  LifetimeState::Buffers buf{
      local.mutable_data(), facts.mutable_data(), visible_xy.mutable_data(),
      visible_prev.mutable_data(), visible_age.mutable_data(), status.mutable_data(),
      history.mutable_data(), reinterpret_cast<uint8_t*>(history_mask.mutable_data()),
      entities.mutable_data(), reinterpret_cast<uint8_t*>(entity_mask.mutable_data())};
  state.Prepare(w, buf);
  py::dict out;
  out["local"] = local;
  out["facts"] = facts;
  out["visible_xy"] = visible_xy;
  out["visible_prev"] = visible_prev;
  out["visible_age_before"] = visible_age;
  out["status"] = status;
  out["history"] = history;
  out["history_mask"] = history_mask;
  out["entities"] = entities;
  out["entity_mask"] = entity_mask;
  py::array_t<int64_t> pairs({static_cast<int>(state.portal_pairs.size()), 2, 2});
  py::array_t<int64_t> axis(static_cast<int>(state.portal_axis.size()));
  for (size_t i = 0; i < state.portal_pairs.size(); ++i) {
    for (int j = 0; j < 4; ++j) pairs.mutable_data()[4 * i + j] = state.portal_pairs[i][j];
    axis.mutable_data()[i] = state.portal_axis[i];
  }
  py::array_t<int64_t> loose({static_cast<int>(state.portal_unpaired.size()), 2});
  for (size_t i = 0; i < state.portal_unpaired.size(); ++i) {
    loose.mutable_data()[2 * i] = state.portal_unpaired[i][0];
    loose.mutable_data()[2 * i + 1] = state.portal_unpaired[i][1];
  }
  out["portal_pairs"] = pairs;
  out["portal_axis"] = axis;
  out["portal_unpaired"] = loose;
  out["inbox"] = state.inbox;
  out["head"] = py::make_tuple(state.head.x, state.head.y);
  out["heading"] = state.heading;
  out["length"] = state.length;
  out["unit_count"] = state.unit_count;
  out["round"] = state.round;
  out["split_legal"] = state.split_legal;
  return out;
}

py::tuple TraceMove(const LifetimeState& state, const std::string& dirs) {
  namespace sp = dragons::spatial;
  const int K = state.max_steps;
  py::array_t<int64_t> tokens(K);
  py::array_t<bool> active(K);
  py::array_t<bool> masks({K, sp::kTokens});
  py::array_t<float> prefix({K, sp::kPrefixFeatures});
  std::fill(tokens.mutable_data(), tokens.mutable_data() + K, sp::kStop);
  std::fill(active.mutable_data(), active.mutable_data() + K, false);
  std::fill(masks.mutable_data(), masks.mutable_data() + K * sp::kTokens, false);
  std::fill(prefix.mutable_data(), prefix.mutable_data() + K * sp::kPrefixFeatures, 0.0f);
  PrefixSimulator sim(state);
  std::vector<int> toks;
  for (char c : dirs) toks.push_back(static_cast<int>(std::string("NESW").find(c)));
  toks.push_back(sp::kStop);
  for (int j = 0; j < K; ++j) {
    std::array<bool, sp::kTokens> m;
    sim.Mask(m);
    int allowed = 0;
    for (int t = 0; t < sp::kTokens; ++t) {
      masks.mutable_data()[j * sp::kTokens + t] = m[t];
      allowed += m[t];
    }
    sim.Features(prefix.mutable_data() + j * sp::kPrefixFeatures);
    const int tok = j < static_cast<int>(toks.size()) ? toks[j] : sp::kStop;
    if (!m[tok]) throw std::invalid_argument("token masked at step " + std::to_string(j));
    tokens.mutable_data()[j] = tok;
    active.mutable_data()[j] = allowed > 1;
    if (tok == sp::kStop) break;
    sim.Advance(tok);
  }
  return py::make_tuple(tokens, active, masks, prefix);
}

}  // namespace

void BindSpatial(py::module_& m) {
  py::class_<LifetimeState>(m, "Lifetime")
      .def(py::init<int, int, int, int, int, int>(), py::arg("dragon_id"), py::arg("team"),
           py::arg("width"), py::arg("height"), py::arg("unit_limit"), py::arg("max_steps") = 8)
      .def("prepare", &PrepareDict, py::arg("game"), py::arg("index"), py::arg("portal_ids"))
      .def("commit_move", &LifetimeState::CommitMove)
      .def("commit_split", &LifetimeState::CommitSplit)
      .def("trace_move", &TraceMove)
      .def("split_table", [](const LifetimeState& state) {
        const auto table = dragons::spatial::MakeSplitTable(state);
        py::array_t<int64_t> sizes(static_cast<int>(table.sizes.size()));
        py::array_t<float> feats({static_cast<int>(table.sizes.size()), 4});
        for (size_t i = 0; i < table.sizes.size(); ++i) {
          sizes.mutable_data()[i] = table.sizes[i];
          for (int j = 0; j < 4; ++j) feats.mutable_data()[4 * i + j] = table.features[i][j];
        }
        return py::make_tuple(sizes, feats);
      })
      .def_property_readonly("own_turns", [](const LifetimeState& s) { return s.own_turns; })
      .def_property_readonly("body", [](const LifetimeState& s) {
        py::list out;
        if (!s.body) return out;
        for (const auto& c : s.body->cells) {
          if (c) out.append(py::make_tuple(c->x, c->y)); else out.append(py::none());
        }
        return out;
      });
}
