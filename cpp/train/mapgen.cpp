#include "mapgen.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dragons::train {

namespace {

// edge indexing, verified against the engine (see dragons_env/mapgen.py)
int North(int x, int y, int w) { return y * 2 * (w + 1) + x; }
int West(int x, int y, int w) { return y * 2 * (w + 1) + (w + 1) + x; }

int Mod(int a, int m) { return ((a % m) + m) % m; }

// segments of a dragon whose head is at (x, y) facing `dir` (0 N, 1 E, 2 S, 3 W)
std::vector<std::pair<int, int>> Snake(int x, int y, int length, int dir, int w, int h) {
  static const int kDx[4] = {0, 1, 0, -1}, kDy[4] = {-1, 0, 1, 0};
  const int back = (dir + 2) % 4;   // the body extends behind the head
  std::vector<std::pair<int, int>> out;
  for (int i = 0; i < length; ++i) out.emplace_back(Mod(x + kDx[back] * i, w), Mod(y + kDy[back] * i, h));
  return out;
}

}  // namespace

std::string RandomMapText(std::mt19937_64& rng, int w, int h, int unit_limit) {
  auto pick = [&](auto&& choices) { return choices[rng() % choices.size()]; };
  auto uniform = [&] { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); };
  const int dragons_per_team = pick(std::vector<int>{1, 1, 2, 3});
  const int length = pick(std::vector<int>{3, 4, 6, 8});
  const double kelp_density = uniform() * 0.08;
  const int portal_pairs = pick(std::vector<int>{0, 0, 1, 2});
  const std::vector<std::pair<int, int>> gaps{{4, 12}, {8, 20}, {15, 40}, {20, 60}};
  const std::pair<int, int> pearl_gap = pick(gaps);

  // spawns: point-symmetric pairs, kept apart, one per team
  std::set<std::pair<int, int>> occupied;
  std::vector<std::pair<int, std::vector<std::pair<int, int>>>> dragons;
  for (int i = 0; i < dragons_per_team; ++i) {
    bool placed = false;
    for (int attempt = 0; attempt < 200 && !placed; ++attempt) {
      const int x = static_cast<int>(rng() % w), y = static_cast<int>(rng() % h);
      const int dir = static_cast<int>(rng() % 4);
      const auto a = Snake(x, y, length, dir, w, h);
      const auto b = Snake(Mod(x + w / 2, w), Mod(y + h / 2, h), length, (dir + 2) % 4, w, h);
      std::set<std::pair<int, int>> sa(a.begin(), a.end());
      bool ok = true;
      for (const auto& c : a) if (occupied.count(c)) ok = false;
      for (const auto& c : b) if (occupied.count(c) || sa.count(c)) ok = false;
      if (!ok) continue;
      occupied.insert(a.begin(), a.end());
      occupied.insert(b.begin(), b.end());
      dragons.emplace_back(0, a);
      dragons.emplace_back(1, b);
      placed = true;
    }
    if (!placed) throw std::runtime_error("could not place dragons; board too small or too crowded");
  }

  // kelp: never on a spawning body's tiles, so nobody dies on round 0
  std::set<int> kelp;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (occupied.count({x, y})) continue;
      if (uniform() < kelp_density) kelp.insert(North(x, y, w));
      if (uniform() < kelp_density) kelp.insert(West(x, y, w));
    }
  }

  // portals: two edges of the same orientation per pair, each edge used once
  std::vector<std::pair<int, int>> portals;
  std::set<int> used;
  for (int pid = 0; pid < portal_pairs; ++pid) {
    const bool horizontal = uniform() < 0.5;
    std::vector<int> picked;
    for (int attempt = 0; attempt < 500 && picked.size() < 2; ++attempt) {
      const int x = static_cast<int>(rng() % w), y = static_cast<int>(rng() % h);
      const int idx = horizontal ? North(x, y, w) : West(x, y, w);
      if (kelp.count(idx) || used.count(idx) || std::find(picked.begin(), picked.end(), idx) != picked.end()) continue;
      picked.push_back(idx);
    }
    if (picked.size() < 2) break;
    used.insert(picked.begin(), picked.end());
    portals.emplace_back(picked[0], pid);
    portals.emplace_back(picked[1], pid);
  }

  std::ostringstream out;
  out << "MAP " << w << ' ' << h << "\nMAP_NAME R" << w << 'x' << h << "\nUNIT_LIMIT " << unit_limit << '\n';
  out << "TILE_COUNT " << (w * h) << '\n';
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) out << "TILE " << x << ' ' << y << ' ' << pearl_gap.first << ' ' << pearl_gap.second << '\n';
  out << "EDGE_COUNT " << (kelp.size() + portals.size()) << '\n';
  for (int i : kelp) out << "EDGE " << i << " 1 -1\n";
  for (const auto& [i, pid] : portals) out << "EDGE " << i << " 2 " << pid << '\n';
  out << "DRAGON_COUNT " << dragons.size() << '\n';
  for (const auto& [team, segs] : dragons) {
    out << "DRAGON " << team << ' ' << segs.size();
    for (const auto& [x, y] : segs) out << ' ' << x << ' ' << y;
    out << '\n';
  }
  return out.str();
}

}  // namespace dragons::train
