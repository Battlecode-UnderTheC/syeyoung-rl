// dragons_rl._native: facts, collection and replay for the Python trainer.
// Batches are dicts of C-contiguous numpy arrays that Python allocates
// (dragons_rl.codec.actor_buffers / critic_buffers) and C++ fills in place.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

#include "collector.hpp"

namespace py = pybind11;
using namespace rl;

namespace {

template <typename T>
T* Ptr(const py::dict& d, const char* key) {
  if (!d.contains(key)) throw std::invalid_argument(std::string("buffer '") + key + "' missing");
  auto a = d[key].cast<py::array>();
  if (!a.dtype().is(py::dtype::of<T>())) throw std::invalid_argument(std::string("buffer '") + key + "' has the wrong dtype");
  if (!(a.flags() & py::array::c_style)) throw std::invalid_argument(std::string("buffer '") + key + "' is not C-contiguous");
  if (!a.writeable()) throw std::invalid_argument(std::string("buffer '") + key + "' is read-only");
  return static_cast<T*>(a.mutable_data());
}

ActorBatch Actor(const py::dict& d) {
  ActorBatch b;
  b.k_pad = static_cast<int32_t>(d["inbox"].cast<py::array>().shape(d["inbox"].cast<py::array>().ndim() - 2));
  b.e_pad = static_cast<int32_t>(d["echo"].cast<py::array>().shape(d["echo"].cast<py::array>().ndim() - 1));
  b.win = Ptr<int32_t>(d, "win");
  b.wdest = Ptr<int32_t>(d, "wdest");
  b.grid = Ptr<int32_t>(d, "grid");
  b.entity = Ptr<int32_t>(d, "entity");
  b.segment = Ptr<int32_t>(d, "segment");
  b.status = Ptr<int32_t>(d, "status");
  b.prev_dirs = Ptr<int32_t>(d, "prev_dirs");
  b.prev_words = Ptr<uint8_t>(d, "prev_words");
  b.inbox = Ptr<uint8_t>(d, "inbox");
  b.echo = Ptr<uint8_t>(d, "echo");
  b.echo_len = Ptr<int32_t>(d, "echo_len");
  b.h = Ptr<float>(d, "h");
  return b;
}

CriticBatch Critic(const py::dict& d) {
  CriticBatch b;
  auto grid = d["grid"].cast<py::array>();
  b.n_pad = static_cast<int32_t>(grid.shape(grid.ndim() - 1));
  auto rows = d["dragons"].cast<py::array>();
  b.d_pad = static_cast<int32_t>(rows.shape(rows.ndim() - 2));
  b.board_tile = Ptr<int32_t>(d, "board_tile");
  b.board_dest = Ptr<int32_t>(d, "board_dest");
  b.grid = Ptr<int32_t>(d, "grid");
  b.dragons = Ptr<int32_t>(d, "dragons");
  b.global = Ptr<int32_t>(d, "global");
  b.summary = Ptr<float>(d, "summary");
  return b;
}

std::vector<Command> Commands(const py::dict& d, size_t n) {
  auto kind = d["kind"].cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
  auto nsteps = d["nsteps"].cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
  auto dirs = d["dirs"].cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
  auto split = d["split"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
  auto mask = d["sonar_mask"].cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
  auto words = d["words"].cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
  if (static_cast<size_t>(kind.size()) != n || static_cast<size_t>(dirs.size()) != n * kMaxMoveTokens ||
      static_cast<size_t>(words.size()) != n * 4)
    throw std::invalid_argument("command arrays do not match the batch");
  std::vector<Command> out(n);
  for (size_t b = 0; b < n; ++b) {
    Command& c = out[b];
    c.kind = kind.data()[b];
    c.nsteps = nsteps.data()[b];
    for (int i = 0; i < kMaxMoveTokens; ++i) c.dirs[i] = dirs.data()[b * kMaxMoveTokens + i];
    c.split = split.data()[b];
    c.sonar_mask = mask.data()[b];
    for (int d4 = 0; d4 < 4; ++d4) c.words[d4] = words.data()[b * 4 + d4];
    if (c.kind > 1 || (c.kind == 0 && (c.nsteps < 1 || c.nsteps > kMaxMoveTokens)) || c.sonar_mask > 15)
      throw std::invalid_argument("malformed command");
    for (int i = 0; i < c.nsteps; ++i)
      if (c.dirs[i] > 3) throw std::invalid_argument("move directions are 0..3");
  }
  return out;
}

py::dict CommandArrays(const std::vector<const Command*>& cs) {
  const py::ssize_t n = static_cast<py::ssize_t>(cs.size());
  py::array_t<uint8_t> kind(n), nsteps(n), mask(n);
  py::array_t<uint8_t> dirs({n, static_cast<py::ssize_t>(kMaxMoveTokens)});
  py::array_t<int32_t> split(n);
  py::array_t<uint64_t> words({n, static_cast<py::ssize_t>(4)});
  for (py::ssize_t b = 0; b < n; ++b) {
    const Command& c = *cs[b];
    kind.mutable_data()[b] = c.kind;
    nsteps.mutable_data()[b] = c.nsteps;
    mask.mutable_data()[b] = c.sonar_mask;
    split.mutable_data()[b] = c.split;
    for (int i = 0; i < kMaxMoveTokens; ++i) dirs.mutable_data()[b * kMaxMoveTokens + i] = c.dirs[i];
    for (int d = 0; d < 4; ++d) words.mutable_data()[b * 4 + d] = c.words[d];
  }
  py::dict d;
  d["kind"] = kind; d["nsteps"] = nsteps; d["dirs"] = dirs; d["split"] = split; d["sonar_mask"] = mask; d["words"] = words;
  return d;
}

std::vector<int32_t> Ints(const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& a) {
  return std::vector<int32_t>(a.data(), a.data() + a.size());
}

template <typename T>
const T* Rows(const py::array_t<T, py::array::c_style | py::array::forcecast>& a, size_t rows, size_t cols, const char* what) {
  if (static_cast<size_t>(a.size()) != rows * cols) throw std::invalid_argument(std::string(what) + " has the wrong size");
  return a.data();
}

MatchSetup Setup(const py::dict& d) {
  MatchSetup ms;
  ms.map_text = d["map"].cast<std::string>();
  auto policy = d["policy"].cast<std::vector<int32_t>>();
  auto record = d["record"].cast<std::vector<bool>>();
  if (policy.size() != 2 || record.size() != 2) throw std::invalid_argument("policy and record are per team");
  for (int t = 0; t < 2; ++t) {
    ms.policy[t] = policy[t];
    ms.record[t] = record[t];
  }
  if (d.contains("seed")) ms.seed = d["seed"].cast<uint64_t>();
  return ms;
}

py::dict MatchResults(const std::vector<MatchRecord>& ms) {
  const py::ssize_t n = static_cast<py::ssize_t>(ms.size());
  py::array_t<int8_t> winner(n);
  py::array_t<int32_t> rounds(n), events(n), tiles(n), policy({n, static_cast<py::ssize_t>(2)});
  py::array_t<bool> finished(n), record({n, static_cast<py::ssize_t>(2)});
  py::array_t<int32_t> deaths({n, static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(5)});
  for (py::ssize_t i = 0; i < n; ++i) {
    winner.mutable_data()[i] = ms[i].winner;
    rounds.mutable_data()[i] = ms[i].rounds;
    events.mutable_data()[i] = static_cast<int32_t>(ms[i].events.size());
    finished.mutable_data()[i] = ms[i].finished;
    tiles.mutable_data()[i] = ms[i].board->width * ms[i].board->height;
    for (int t = 0; t < 2; ++t) {
      policy.mutable_data()[2 * i + t] = ms[i].setup.policy[t];
      record.mutable_data()[2 * i + t] = ms[i].setup.record[t];
    }
    for (int k = 0; k < 10; ++k) deaths.mutable_data()[i * 10 + k] = ms[i].deaths[k];
  }
  py::dict d;
  d["winner"] = winner; d["rounds"] = rounds; d["events"] = events; d["finished"] = finished; d["tiles"] = tiles;
  d["policy"] = policy; d["record"] = record;
  d["deaths"] = deaths;   // [match, team, cause]: wall, self, other body, head-to-head, no valid action
  return d;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "Private facts, match collection and replay for the dragons trainer";
  py::dict fields;
  auto put = [&](const char* k, int v) { fields[k] = v; };
  put("TILE", kTileFields); put("ENTITY", kEntityFields); put("SEGMENT", kSegmentFields);
  put("STATUS", kStatusFields); put("PREV_DIRS", kPrevDirsFields); put("MAX_ENTITIES", kMaxEntities);
  put("WINDOW_TILES", kWindowTiles); put("HIDDEN", kHidden); put("MAX_MOVE_TOKENS", kMaxMoveTokens);
  put("CRITIC_DRAGON", kCriticDragonFields); put("CRITIC_GLOBAL", kCriticGlobalFields);
  put("COUNTDOWN_BIAS", kCountdownBias); put("SNAPSHOT_EVERY", kSnapshotEvery); put("PROTOCOL", kProtocol);
  put("WIN", kWin); put("WIN_TILES", kWinTiles); put("GRID_CELLS", kGridCells); put("GRID", kGridFields);
  m.attr("FIELDS") = fields;
  py::dict scripted;
  scripted["greedy-v1"] = static_cast<int>(kGreedy);
  scripted["random-v1"] = static_cast<int>(kRandom);
  m.attr("SCRIPTED") = scripted;
  auto names = [](std::initializer_list<const char*> l) {
    py::dict d;
    int i = 0;
    for (const char* s : l) d[s] = i++;
    return d;
  };
  m.attr("TILE") = names({"SEEN_ROUND", "PEARL", "COUNTDOWN", "VISIBLE", "VISITS", "VISIT_ROUND", "BODY_INDEX",
                          "PREV_ROUND", "PREV_PEARL", "PREV_COUNTDOWN", "OCC", "OCC_ORDER", "OCC_FACING", "EDGES",
                          "FLAGS", "VALID"});
  m.attr("ENTITY") = names({"VALID", "RELATION", "ORDER", "COUNT", "HEAD_VISIBLE", "HEAD_X", "HEAD_Y", "HEAD_FACING"});
  m.attr("STATUS") = names({"LENGTH", "ROUND", "WIDTH", "HEIGHT", "UNIT_COUNT", "UNIT_LIMIT", "KNOWN_BODY", "TAIL_KNOWN",
                            "TURN_COUNT", "FACING", "HEAD_X", "HEAD_Y", "PREV_VALID", "PREV_KIND", "PREV_STEPS",
                            "PREV_SPLIT", "PREV_LENGTH", "SEEN_TILES", "KNOWN_EDGES", "INBOX_COUNT", "ECHO_PRESENT",
                            "HEAD_BED", "HEAD_PEARL", "HEAD_TILE", "PREV_SONAR_MASK"});
  m.attr("CRITIC_DRAGON") = names({"TEAM", "LENGTH", "HEAD_X", "HEAD_Y", "FACING", "RANK", "ACTED", "ID", "VALID"});
  m.attr("CRITIC_GLOBAL") = names({"ROUND", "WIDTH", "HEIGHT", "UNIT_LIMIT", "LONGEST_A", "LONGEST_B", "TOTAL_A",
                                   "TOTAL_B", "COUNT_A", "COUNT_B", "NEXT_TEAM", "NEXT_RANK", "ALIVE", "DUE",
                                   "NEXT_LENGTH"});

  py::class_<Pack, std::shared_ptr<Pack>>(m, "Pack")
      .def_property_readonly("num_matches", [](const Pack& p) { return p.matches.size(); })
      .def("decisions", [](const Pack& p) {
        int64_t n = 0;
        for (const auto& mr : p.matches)
          for (const auto& l : mr.lives) n += static_cast<int64_t>(l.logp.size());   // recorded lifetimes only
        return n;
      }, "recorded (learner) decisions")
      .def("results", [](const Pack& p) { return MatchResults(p.matches); })
      .def("serialize", [](const Pack& p) {
        std::string b;
        {
          py::gil_scoped_release nogil;
          b = SerializePack(p);
        }
        return py::bytes(b);
      }, "the pack as bytes (rebuilt by Pack.deserialize, which replays the matches)")
      .def_static("deserialize", [](const py::bytes& b) {
        std::string s = b;
        py::gil_scoped_release nogil;
        return DeserializePack(s);
      });

  py::class_<Collector>(m, "Collector")
      .def(py::init([](const py::list& setups, int threads, bool keep_observations) {
             std::vector<MatchSetup> s;
             for (auto item : setups) s.push_back(Setup(item.cast<py::dict>()));
             py::gil_scoped_release nogil;
             return std::make_unique<Collector>(std::move(s), threads, keep_observations);
           }),
           py::arg("setups"), py::arg("threads") = 1, py::arg("keep_observations") = true,
           "setups: [{'map': text, 'policy': [slot A, slot B], 'record': [bool A, bool B], 'seed': int}]")
      .def_property_readonly("num_matches", &Collector::num_matches)
      .def("done", &Collector::done)
      .def("prepare", [](Collector& c) {
        std::vector<Pending> p;
        {
          py::gil_scoped_release nogil;
          p = c.Prepare();
        }
        const py::ssize_t n = static_cast<py::ssize_t>(p.size());
        py::dict d;
        const char* keys[] = {"match", "dragon", "team", "policy", "record", "tiles", "length", "inbox", "echo_bytes",
                              "alive"};
        std::vector<py::array_t<int32_t>> cols;
        for (const char* k : keys) {
          cols.emplace_back(n);
          d[k] = cols.back();
        }
        for (py::ssize_t i = 0; i < n; ++i) {
          const Pending& q = p[i];
          const int32_t v[] = {q.match, q.dragon, q.team, q.policy, q.record, q.tiles, q.length, q.inbox, q.echo_bytes,
                               q.alive};
          for (size_t k = 0; k < cols.size(); ++k) cols[k].mutable_data()[i] = v[k];
        }
        return d;
      })
      .def("encode_actors", [](Collector& c, py::array_t<int32_t, py::array::c_style | py::array::forcecast> matches,
                               const py::dict& buf) {
        auto ms = Ints(matches);
        ActorBatch b = Actor(buf);
        py::gil_scoped_release nogil;
        c.EncodeActors(ms, b);
      })
      .def("encode_critics", [](Collector& c, py::array_t<int32_t, py::array::c_style | py::array::forcecast> matches,
                                const py::dict& buf) {
        auto ms = Ints(matches);
        CriticBatch b = Critic(buf);
        py::gil_scoped_release nogil;
        c.EncodeCritics(ms, b);
      })
      .def("apply", [](Collector& c, py::array_t<int32_t, py::array::c_style | py::array::forcecast> matches,
                       const py::dict& commands, py::array_t<float, py::array::c_style | py::array::forcecast> logp,
                       py::array_t<float, py::array::c_style | py::array::forcecast> h,
                       py::array_t<float, py::array::c_style | py::array::forcecast> value) {
        auto ms = Ints(matches);
        auto cs = Commands(commands, ms.size());
        const float* lp = Rows(logp, ms.size(), 1, "logp");
        const float* hp = Rows(h, ms.size(), kHidden, "h");
        const float* vp = Rows(value, ms.size(), 1, "value");
        py::gil_scoped_release nogil;
        c.Apply(ms, cs, lp, hp, vp);
      }, py::arg("matches"), py::arg("commands"), py::arg("logp"), py::arg("h"), py::arg("value"),
         "value: V_old from each actor's team's perspective (NaN if not evaluated)")
      .def("reset", [](Collector& c, int32_t m, const py::dict& setup) { c.Reset(m, Setup(setup)); })
      .def("finished", &Collector::finished)
      .def("harvest", [](Collector& c) {
        py::gil_scoped_release nogil;
        return c.Harvest();
      })
      .def("results", [](const Collector& c) { return MatchResults(c.match_records()); });

  py::class_<Replay>(m, "Replay")
      .def(py::init([](const std::vector<std::shared_ptr<Pack>>& packs, int threads,
                       const std::optional<std::vector<int32_t>>& keep) {
             std::vector<std::shared_ptr<const Pack>> ps(packs.begin(), packs.end());
             return std::make_unique<Replay>(std::move(ps), threads, keep);
           }),
           py::arg("packs"), py::arg("threads") = 1, py::arg("keep") = std::nullopt)
      .def_property_readonly("num_lifetimes", &Replay::num_lifetimes)
      .def_property_readonly("num_matches", &Replay::num_matches)
      .def("reset", &Replay::Reset)
      .def("lifetimes", [](const Replay& r) {
        const py::ssize_t n = r.num_lifetimes();
        py::array_t<int32_t> match(n), dragon(n), team(n), length(n), tiles(n), death(n);
        for (py::ssize_t i = 0; i < n; ++i) {
          const LifetimeRecord& l = r.lifetime(static_cast<int32_t>(i));
          match.mutable_data()[i] = r.lifetime_match(static_cast<int32_t>(i));
          dragon.mutable_data()[i] = l.dragon;
          team.mutable_data()[i] = l.team;
          length.mutable_data()[i] = static_cast<int32_t>(l.observations.size());
          tiles.mutable_data()[i] = l.observations[0].spawn.width * l.observations[0].spawn.height;
          death.mutable_data()[i] = l.death;
        }
        py::dict d;
        d["match"] = match; d["dragon"] = dragon; d["team"] = team; d["decisions"] = length; d["tiles"] = tiles;
        d["death"] = death;   // engine cause (wall, self, other body, head-to-head, no valid action), -1 alive
        return d;
      }, "per learner lifetime: match, dragon id, team, number of decisions, board tiles")
      .def("decisions", [](const Replay& r) {
        // every learner decision, lifetime by lifetime in order
        std::vector<const Command*> cs;
        std::vector<int32_t> life, step, event, length, board, inbox, echo;
        std::vector<float> logp, h;
        for (int32_t i = 0; i < r.num_lifetimes(); ++i) {
          const LifetimeRecord& l = r.lifetime(i);
          for (size_t t = 0; t < l.observations.size(); ++t) {
            cs.push_back(&l.commands[t]);
            life.push_back(i);
            step.push_back(static_cast<int32_t>(t));
            event.push_back(l.events[t]);
            length.push_back(l.observations[t].length);
            board.push_back(l.observations[t].spawn.width * l.observations[t].spawn.height);
            inbox.push_back(static_cast<int32_t>(l.observations[t].inbox.size()));
            size_t e = 0;
            for (const auto& f : l.observations[t].echo) e = std::max(e, f.size());
            echo.push_back(static_cast<int32_t>(e));
            logp.push_back(l.logp[t]);
            h.insert(h.end(), l.h[t].begin(), l.h[t].end());
          }
        }
        py::dict d = CommandArrays(cs);
        d["lifetime"] = py::array_t<int32_t>(life.size(), life.data());
        d["step"] = py::array_t<int32_t>(step.size(), step.data());
        d["event"] = py::array_t<int32_t>(event.size(), event.data());
        d["length"] = py::array_t<int32_t>(length.size(), length.data());
        d["tiles"] = py::array_t<int32_t>(board.size(), board.data());
        d["inbox"] = py::array_t<int32_t>(inbox.size(), inbox.data());
        d["echo_bytes"] = py::array_t<int32_t>(echo.size(), echo.data());
        d["logp"] = py::array_t<float>(logp.size(), logp.data());
        d["h"] = py::array_t<float>({static_cast<py::ssize_t>(logp.size()), static_cast<py::ssize_t>(kHidden)}, h.data());
        return d;
      })
      .def("encode_chunk", [](Replay& r, py::array_t<int32_t, py::array::c_style | py::array::forcecast> which,
                              py::array_t<int32_t, py::array::c_style | py::array::forcecast> start, int steps,
                              const py::dict& buf) {
        auto w = Ints(which), s = Ints(start);
        ActorBatch b = Actor(buf);
        py::gil_scoped_release nogil;
        r.EncodeChunk(w, s, steps, b);
      }, py::arg("which"), py::arg("start"), py::arg("steps"), py::arg("buffers"))
      .def("encode_chunk_reference", [](Replay& r, py::array_t<int32_t, py::array::c_style | py::array::forcecast> which,
                                        py::array_t<int32_t, py::array::c_style | py::array::forcecast> start, int steps,
                                        const py::dict& buf) {
        auto w = Ints(which), s = Ints(start);
        ActorBatch b = Actor(buf);
        r.EncodeChunk(w, s, steps, b, true);
      }, "encode_chunk through the from-scratch reference encoder (tests)")
      .def("encode_list", [](Replay& r, py::array_t<int32_t, py::array::c_style | py::array::forcecast> which,
                             py::array_t<int32_t, py::array::c_style | py::array::forcecast> step, const py::dict& buf) {
        auto w = Ints(which), s = Ints(step);
        ActorBatch b = Actor(buf);
        py::gil_scoped_release nogil;
        r.EncodeList(w, s, b);
      }, py::arg("which"), py::arg("step"), py::arg("buffers"),
         "decision step[e] of lifetime which[e] into row e (which < 0: left as is)")
      .def("events", [](const Replay& r, int32_t match) {
        const MatchRecord& mr = r.match(match);
        const py::ssize_t n = static_cast<py::ssize_t>(mr.events.size());
        py::array_t<float> values({n, static_cast<py::ssize_t>(2)});
        py::array_t<int32_t> dragon(n);
        for (py::ssize_t e = 0; e < n; ++e) {
          values.mutable_data()[2 * e] = mr.events[e].value[0];
          values.mutable_data()[2 * e + 1] = mr.events[e].value[1];
          dragon.mutable_data()[e] = mr.events[e].dragon;
        }
        py::dict d;
        d["values"] = values; d["dragon"] = dragon; d["winner"] = mr.winner; d["tiles"] = mr.board->width * mr.board->height;
        d["finished"] = mr.finished;
        d["unit_limit"] = mr.board->unit_limit;
        return d;
      })
      .def("encode_events", [](const Replay& r, int32_t match,
                               py::array_t<int32_t, py::array::c_style | py::array::forcecast> events, const py::dict& buf) {
        auto ev = Ints(events);
        CriticBatch b = Critic(buf);
        py::gil_scoped_release nogil;
        r.EncodeEvents(match, ev, b);
      }, py::arg("match"), py::arg("events"), py::arg("buffers"));

  // One lifetime's facts, stepped by hand: for tests and the deployed bot.
  py::class_<Lifetime>(m, "Lifetime")
      .def(py::init([](const std::string& init_block) { return Lifetime(ParseSpawn(init_block)); }))
      .def("observe", [](Lifetime& l, const std::string& init_block, const std::string& round_block) {
        return l.Observe(ParseRound(ParseSpawn(init_block), round_block));
      })
      .def("commit", [](Lifetime& l, const py::dict& command) { l.Commit(Commands(command, 1)[0]); })
      .def("encode", [](const Lifetime& l, const std::string& init_block, const std::string& round_block,
                        const py::dict& buf) {
        ActorBatch b = Actor(buf);
        EncodeObservation(l, ParseRound(ParseSpawn(init_block), round_block), b, 0);
      })
      .def("tables", [](const Lifetime& l, const std::string& init_block, const std::string& round_block,
                        py::array_t<int32_t, py::array::c_style> tile, py::array_t<int32_t, py::array::c_style> dest,
                        bool reference) {
        // the whole-board records [n_pad, TILE] and J [n_pad, 4] (tests: incremental == from scratch)
        const Observation o = ParseRound(ParseSpawn(init_block), round_block);
        const int32_t n_pad = static_cast<int32_t>(tile.shape(0));
        std::vector<int32_t> entity(kMaxEntities * kEntityFields), segment(kWindowTiles * kSegmentFields),
            status(kStatusFields), prev(kPrevDirsFields);
        std::vector<uint8_t> words(32);
        DecisionRows rows{tile.mutable_data(), dest.mutable_data(), entity.data(), segment.data(), status.data(),
                          prev.data(), words.data(), n_pad};
        if (reference) l.EncodeReference(o, rows);
        else l.Encode(o, rows);
      }, py::arg("init_block"), py::arg("round_block"), py::arg("tile"), py::arg("dest"), py::arg("reference") = false)
      .def("dest", &Lifetime::Dest)
      .def("body", &Lifetime::body)
      .def_property_readonly("turns", &Lifetime::turns)
      .def_property_readonly("seen_fraction", &Lifetime::seen_fraction);

  m.def("step_table", [](const std::string& map_text) { return dragons::LoadBoard(map_text)->step_to; },
        "the engine's [tile * 4 + dir] -> tile table (-1 kelp), for tests");
  m.def("parse_round", [](const std::string& init_block, const std::string& round_block) {
    const Observation o = ParseRound(ParseSpawn(init_block), round_block);
    py::dict d;
    d["round"] = o.round; d["length"] = o.length; d["inbox"] = o.inbox; d["echo_present"] = o.echo_present;
    d["echo"] = std::vector<std::string>(o.echo.begin(), o.echo.end()); d["head"] = py::make_tuple(o.head_x(), o.head_y());
    return d;
  });
}
