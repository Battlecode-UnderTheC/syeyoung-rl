#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

void BindVec(py::module_& m);

PYBIND11_MODULE(_dragons, m) {
  m.doc() = "Native reimplementation of the UNSW Battlecode dragons engine";
  BindVec(m);

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
      });
}

// ---- observation encoding, masking and batched stepping --------------------
#include <pybind11/numpy.h>

#include "agent/encoder.hpp"

void BindVec(py::module_& m) {
  py::class_<ActionSpace>(m, "ActionSpace")
      .def(py::init<int, std::vector<int>, bool>(), py::arg("max_sprint") = 3,
           py::arg("split_sizes") = std::vector<int>{2, 3, 4, 5, 6, 7, 8, 9},
           py::arg("allow_portal_steps") = true)
      .def_property_readonly("size", &ActionSpace::size)
      .def_property_readonly("n_moves", &ActionSpace::n_moves)
      .def("is_split", &ActionSpace::is_split);

  m.def("encode", [](Game& game, int dragon_index, bool ego_rotate) {
    py::array_t<float> grid({kChannels, kVision, kVision});
    py::array_t<float> scalars(kScalars);
    EncodeObservation(game, game.dragons()[dragon_index], ego_rotate,
                      grid.mutable_data(), scalars.mutable_data());
    return py::make_tuple(grid, scalars);
  }, py::arg("game"), py::arg("dragon_index"), py::arg("ego_rotate") = true);

  m.def("legal_mask", [](Game& game, int dragon_index, const ActionSpace& space) {
    py::array_t<bool> mask(space.size());
    space.Mask(game, game.dragons()[dragon_index], mask.mutable_data());
    return mask;
  });

  py::class_<VecGame>(m, "VecGame")
      .def(py::init<const std::vector<std::string>&, uint64_t, ActionSpace, bool>(),
           py::arg("maps"), py::arg("seed") = 0,
           py::arg("space") = ActionSpace(), py::arg("ego_rotate") = true)
      .def_property_readonly("arenas", &VecGame::arenas)
      .def("begin_round", &VecGame::BeginRound)
      .def("end_round", &VecGame::EndRound)
      .def("game", &VecGame::game, py::return_value_policy::reference_internal)
      .def("prepare_slot", [](VecGame& vec, int slot) {
        const int cap = vec.arenas();
        const int actions = vec.space().size();
        py::array_t<float> grid({cap, kChannels, kVision, kVision});
        py::array_t<float> scalars({cap, kScalars});
        py::array_t<bool> mask({cap, actions});
        py::array_t<int32_t> arena(cap), dragon(cap);
        const int rows = vec.PrepareSlot(slot, grid.mutable_data(), scalars.mutable_data(),
                                         mask.mutable_data(), arena.mutable_data(),
                                         dragon.mutable_data());
        auto trim2 = [rows](py::array& a) { return a[py::slice(0, rows, 1)]; };
        return py::make_tuple(rows, trim2(grid), trim2(scalars), trim2(mask),
                              trim2(arena), trim2(dragon));
      })
      .def("submit_slot", [](VecGame& vec, py::array_t<int32_t> actions,
                             py::object sonar, py::object payloads) {
        const uint8_t* sonar_ptr = nullptr;
        const uint32_t* payload_ptr = nullptr;
        py::array_t<uint8_t> sonar_arr;
        py::array_t<uint32_t> payload_arr;
        if (!sonar.is_none()) {
          sonar_arr = py::cast<py::array_t<uint8_t>>(sonar);
          sonar_ptr = sonar_arr.data();
        }
        if (!payloads.is_none()) {
          payload_arr = py::cast<py::array_t<uint32_t>>(payloads);
          payload_ptr = payload_arr.data();
        }
        vec.SubmitSlot(actions.data(), sonar_ptr, payload_ptr);
      }, py::arg("actions"), py::arg("sonar") = py::none(),
         py::arg("payloads") = py::none())
      .def("benchmark_rounds", [](VecGame& vec, int rounds, bool encode) {
        const BenchResult r = BenchmarkRounds(vec, rounds, encode);
        return py::make_tuple(r.dragon_turns, r.rounds);
      }, py::arg("rounds"), py::arg("encode") = true)
      .def("results", [](VecGame& vec) {
        py::list out;
        for (int i = 0; i < vec.arenas(); ++i) {
          switch (vec.game(i).Result()) {
            case Outcome::TeamA: out.append("A"); break;
            case Outcome::TeamB: out.append("B"); break;
            case Outcome::Draw: out.append("draw"); break;
            default: out.append("running"); break;
          }
        }
        return out;
      });
}
