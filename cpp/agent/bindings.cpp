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

PYBIND11_MODULE(_dragons, m) {
  m.doc() = "Native reimplementation of the UNSW Battlecode dragons engine";

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

