// Checks the trainer's GPU-side derivations against the exact CPU state:
// sparse raw updates + Materialize() must equal LifetimeState::Materialize(),
// and CropWindows() must equal the local crop, on random play with sprints,
// splits, kelp and portals. Run by tools/build_trainer.sh.
#include <cstdio>
#include <random>
#include <sstream>
#include <string>

#include <torch/torch.h>

#include "agent/spatial/codec.hpp"
#include "agent/spatial/memory.hpp"
#include "dragons/game.hpp"
#include "train/gpu_state.hpp"

using namespace dragons;
using namespace dragons::train;
using dragons::spatial::Cell;
using dragons::spatial::LifetimeState;

namespace {

std::string RandomMap(std::mt19937& rng, int w, int h, int portals, std::vector<int>& portal_ids) {
  std::ostringstream out;
  out << "MAP " << w << " " << h << "\nMAP_NAME t\nTILE_COUNT " << w * h << "\n";
  std::uniform_int_distribution<int> gap(2, 6);
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) out << "TILE " << x << " " << y << " " << gap(rng) << " " << gap(rng) + 3 << "\n";
  const int stride = 2 * (w + 1);
  std::vector<std::string> edges;
  std::set<int> used;
  portal_ids.assign(2 * w * h, -1);
  auto spawn_free = [&](int x, int y) { return !((y == 2 && x >= 0 && x <= 6) || (y == h - 3 && x >= w - 7)); };
  std::uniform_real_distribution<double> u(0, 1);
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
    if (!spawn_free(x, y) || !spawn_free(x, (y + h - 1) % h) || !spawn_free((x + w - 1) % w, y)) continue;
    if (u(rng) < 0.05) { const int i = y * stride + x; used.insert(i); edges.push_back("EDGE " + std::to_string(i) + " 1 -1"); }
    if (u(rng) < 0.05) { const int i = y * stride + (w + 1) + x; used.insert(i); edges.push_back("EDGE " + std::to_string(i) + " 1 -1"); }
  }
  for (int pid = 0; pid < portals; ++pid) {
    const bool vertical = pid % 2 == 1;
    int placed = 0;
    for (int tries = 0; tries < 500 && placed < 2; ++tries) {
      const int x = rng() % w, y = rng() % h;
      if (!spawn_free(x, y) || !spawn_free(x, (y + h - 1) % h) || !spawn_free((x + w - 1) % w, y)) continue;
      const int i = y * stride + (vertical ? w + 1 : 0) + x;
      if (used.count(i)) continue;
      used.insert(i);
      edges.push_back("EDGE " + std::to_string(i) + " 2 " + std::to_string(pid));
      portal_ids[(vertical ? w * h : 0) + y * w + x] = pid;
      ++placed;
    }
  }
  out << "EDGE_COUNT " << edges.size() << "\n";
  for (const auto& e : edges) out << e << "\n";
  out << "DRAGON_COUNT 2\nDRAGON 0 6 6 2 5 2 4 2 3 2 2 2 1 2\nDRAGON 1 6 " << w - 7 << " " << h - 3;
  for (int i = 1; i < 6; ++i) out << " " << w - 7 + i << " " << h - 3;
  out << "\n";
  return out.str();
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  double worst = 0.0;
  int compared = 0;
  for (int seed = 0; seed < 4; ++seed) {
    const int w = 9 + seed * 2, h = 11 + seed;
    std::vector<int> ids;
    const std::string text = RandomMap(rng, w, h, 2, ids);
    Game game = Game::FromMapText(text, seed);
    std::vector<std::unique_ptr<LifetimeState>> states;
    std::vector<std::vector<Cell>> old_bodies;
    std::vector<Cell> old_heads;
    std::vector<bool> had_head;
    torch::Tensor raw = torch::zeros({64, kRaw, h, w});
    ResetSlots(raw, torch::arange(64, torch::kLong));
    auto spawn = [&] {
      while (states.size() < game.dragons().size()) {
        const Dragon& d = game.dragons()[states.size()];
        states.push_back(std::make_unique<LifetimeState>(d.id, d.team, w, h, 64, 8));
        old_bodies.emplace_back();
        old_heads.push_back(Cell{0, 0});
        had_head.push_back(false);
      }
    };
    spawn();
    std::vector<float> facts(spatial::kFactChannels * w * h), local(spatial::kFactChannels * 49);
    std::vector<int64_t> xy(98);
    std::vector<float> prev(49 * spatial::kFactChannels), age(49), status(spatial::kStatus);
    std::vector<float> hist(32 * spatial::kHistoryFeatures), ent(49 * spatial::kEntityFeatures);
    std::vector<uint8_t> hmask(32), emask(49);
    for (int round = 0; round < 40 && !game.Finished(); ++round) {
      game.BeginRound();
      for (size_t i = 0; i < game.dragons().size(); ++i) {
        spawn();
        if (!game.dragons()[i].alive) continue;
        LifetimeState& st = *states[i];
        const spatial::Window win = spatial::ReadWindow(game, static_cast<int>(i), ids);
        LifetimeState::Buffers buf{local.data(), facts.data(), xy.data(), prev.data(), age.data(),
                                   status.data(), hist.data(), hmask.data(), ent.data(), emask.data()};
        st.Prepare(win, buf);
        // sparse update of this slot's raw store, then compare the derivation
        std::vector<Cell> cells;
        ChangedCells(st, old_heads[i], had_head[i], old_bodies[i], cells);
        const auto body_index = st.body->IndexOf();
        std::vector<RawUpdate> updates;
        for (const Cell& c : cells) {
          RawUpdate u{static_cast<int32_t>(i), c.y, c.x, {}};
          RawCell(st, body_index, c.x, c.y, u.values);
          updates.push_back(u);
        }
        ApplyUpdates(raw, updates);
        old_heads[i] = st.head;
        had_head[i] = true;
        old_bodies[i].clear();
        for (const auto& c : st.body->cells) if (c) old_bodies[i].push_back(*c);
        torch::Tensor rounds = torch::full({1}, static_cast<int64_t>(win.round), torch::kLong);
        torch::Tensor got = Materialize(raw.slice(0, i, i + 1), rounds);
        torch::Tensor ref = torch::from_blob(facts.data(), {1, spatial::kFactChannels, h, w}).clone();
        const double diff = (got - ref).abs().max().item<double>();
        worst = std::max(worst, diff);
        torch::Tensor head_xy = torch::tensor({{static_cast<int64_t>(st.head.x), static_cast<int64_t>(st.head.y)}}, torch::kLong);
        torch::Tensor crop = CropWindows(got, head_xy);
        torch::Tensor ref_local = torch::from_blob(local.data(), {1, spatial::kFactChannels, 7, 7}).clone();
        worst = std::max(worst, (crop - ref_local).abs().max().item<double>());
        ++compared;
        if (diff > 1e-5) {
          std::printf("MISMATCH seed %d round %d dragon %zu diff %g\n", seed, round, i, diff);
          torch::Tensor per = (got - ref).abs().amax({0, 2, 3});
          for (int c = 0; c < spatial::kFactChannels; ++c) if (per[c].item<double>() > 1e-5) std::printf("  channel %d diff %g\n", c, per[c].item<double>());
          return 1;
        }
        // act: a move the private masks allow, or a split
        spatial::PrefixSimulator sim(st);
        std::string dirs;
        std::array<bool, spatial::kTokens> m;
        sim.Mask(m);
        std::vector<int> allowed;
        for (int t = 0; t < 4; ++t) if (m[t] && t != (st.heading + 2) % 4) allowed.push_back(t);
        Action action;
        if (st.split_legal && rng() % 8 == 0) {
          const int n = 2 + static_cast<int>(rng() % (st.length - 3));
          st.CommitSplit(n);
          action = Action{Action::Kind::Split, {}, n, false, 0};
        } else {
          const int t = allowed.empty() ? st.heading : allowed[rng() % allowed.size()];
          dirs = std::string(1, "NESW"[t]);
          if (st.length >= 5 && rng() % 3 == 0) dirs += "NESW"[t];
          st.CommitMove(dirs);
          action.kind = Action::Kind::Move;
          for (char c : dirs) action.dirs.push_back(static_cast<Dir>(std::string("NESW").find(c)));
        }
        game.TakeTurn(static_cast<int>(i), action);
      }
      game.EndRound();
    }
  }
  std::printf("gpu_state selftest: %d turns compared, worst |diff| %.3g\n", compared, worst);
  return worst > 1e-5 ? 1 : 0;
}
