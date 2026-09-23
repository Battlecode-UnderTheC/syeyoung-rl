#include "encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace dragons {
namespace {

int WrapTo(int value, int modulus) {
  const int r = value % modulus;
  return r < 0 ? r + modulus : r;
}

// Where a step would land if there were no portal: used to tell a portal jump
// apart from an ordinary step.
Point PlainStep(const Game& game, Point from, Dir dir) {
  const int d = static_cast<int>(dir);
  return Point{static_cast<int16_t>(WrapTo(from.x + kDx[d], game.width())),
               static_cast<int16_t>(WrapTo(from.y + kDy[d], game.height()))};
}

}  // namespace
}  // namespace dragons

// ---- dragons-spatial-v1 ----------------------------------------------------
namespace dragons {
namespace {

// The edge crossed stepping `d` out of p, as (is_west, owner tile).
std::pair<bool, Point> CanonicalEdge(const Game& game, Point p, Dir d) {
  const int w = game.width(), h = game.height();
  switch (d) {
    case Dir::N: return {false, p};
    case Dir::S: return {false, Point{p.x, static_cast<int16_t>((p.y + 1) % h)}};
    case Dir::W: return {true, p};
    default: return {true, Point{static_cast<int16_t>((p.x + 1) % w), p.y}};
  }
}

std::string EdgeToken(const Game& game, Point p, Dir d, const std::vector<int>& portal_ids) {
  const std::optional<Point> step = game.TileAfterStep(p, d);
  if (!step) return "w";
  if (*step == PlainStep(game, p, d)) return ".";
  const auto [west, owner] = CanonicalEdge(game, p, d);
  const size_t index = static_cast<size_t>(owner.y) * game.width() + owner.x;
  const int id = portal_ids[(west ? game.width() * game.height() : 0) + index];
  return std::to_string(id);
}

}  // namespace

std::string SpawnBlock(const Game& game, int dragon_index) {
  const Dragon& dragon = game.dragons()[dragon_index];
  std::ostringstream out;
  out << "ID " << dragon.id << "\nTEAM " << (dragon.team == 0 ? "A" : "B")
      << "\nMAP " << game.width() << " " << game.height()
      << "\nUNIT_LIMIT " << game.unit_limit() << "\n";
  return out.str();
}

std::string TurnBlock(const Game& game, int dragon_index, const std::vector<int>& portal_ids) {
  const Dragon& dragon = game.dragons()[dragon_index];
  const int w = game.width(), h = game.height();
  if (portal_ids.size() != static_cast<size_t>(2 * w * h)) {
    throw std::invalid_argument("portal_ids must have 2*W*H entries");
  }
  const Point head = dragon.head();
  std::ostringstream out;
  out << "ROUND " << game.round() << "\nDIR " << "NESW"[static_cast<int>(dragon.facing)]
      << "\nLENGTH " << dragon.length() << "\nUNIT_COUNT " << game.AliveUnitCount(dragon.team)
      << "\nNUM_MSGS " << dragon.inbox.size() << "\n";
  for (uint32_t word : dragon.inbox) out << word << "\n";
  auto tile_at = [&](int row, int col) {
    return Point{static_cast<int16_t>(WrapTo(head.x - kVisionRadius + col, w)),
                 static_cast<int16_t>(WrapTo(head.y - kVisionRadius + row, h))};
  };
  for (int row = 0; row < kVision; ++row) {
    for (int col = 0; col < kVision; ++col) {
      const Point p = tile_at(row, col);
      const Tile& tile = game.TileAt(p);
      out << p.x << " " << p.y << " " << (tile.pearl ? 1 : 0) << " "
          << (tile.spawns ? tile.countdown : -1) << "\n";
    }
  }
  // segments inside the window, dragons in index order, head first
  std::vector<std::string> parts;
  for (const Dragon& other : game.dragons()) {
    if (!other.alive) continue;
    for (size_t i = 0; i < other.segments.size(); ++i) {
      const Point p = other.segments[i];
      const int col = WrapTo(p.x - head.x + kVisionRadius, w);
      const int row = WrapTo(p.y - head.y + kVisionRadius, h);
      if (col >= kVision || row >= kVision) continue;
      std::ostringstream line;
      line << (other.team == 0 ? "A" : "B") << " " << other.id << " " << p.x << " " << p.y
           << " " << "NESW"[static_cast<int>(game.TileFacing(p))] << " " << (i == 0 ? 1 : 0);
      parts.push_back(line.str());
    }
  }
  out << "DRAGON_BODIES " << parts.size() << "\n";
  for (const std::string& line : parts) out << line << "\n";
  // 8 rows of 7 horizontal edges: north of each row, then south of the last
  for (int row = 0; row <= kVision; ++row) {
    for (int col = 0; col < kVision; ++col) {
      const std::string token = row < kVision
          ? EdgeToken(game, tile_at(row, col), Dir::N, portal_ids)
          : EdgeToken(game, tile_at(row - 1, col), Dir::S, portal_ids);
      out << (col ? " " : "") << token;
    }
    out << "\n";
  }
  // 7 rows of 8 vertical edges: west of each column, then east of the last
  for (int row = 0; row < kVision; ++row) {
    for (int col = 0; col <= kVision; ++col) {
      const std::string token = col < kVision
          ? EdgeToken(game, tile_at(row, col), Dir::W, portal_ids)
          : EdgeToken(game, tile_at(row, col - 1), Dir::E, portal_ids);
      out << (col ? " " : "") << token;
    }
    out << "\n";
  }
  return out.str();
}

void DumpTiles(const Game& game, uint8_t* pearl, int32_t* countdown) {
  const int w = game.width(), h = game.height();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Tile& tile = game.TileAt(Point{static_cast<int16_t>(x), static_cast<int16_t>(y)});
      pearl[y * w + x] = tile.pearl ? 1 : 0;
      countdown[y * w + x] = tile.spawns ? tile.countdown : -1;
    }
  }
}

void DumpEdges(const Game& game, uint8_t* north_kind, uint8_t* west_kind) {
  const int w = game.width(), h = game.height();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const Point p{static_cast<int16_t>(x), static_cast<int16_t>(y)};
      for (int side = 0; side < 2; ++side) {
        const Dir d = side == 0 ? Dir::N : Dir::W;
        const std::optional<Point> step = game.TileAfterStep(p, d);
        uint8_t kind = 0;
        if (!step) kind = 1;
        else if (!(*step == PlainStep(game, p, d))) kind = 2;
        (side == 0 ? north_kind : west_kind)[y * w + x] = kind;
      }
    }
  }
}

}  // namespace dragons
