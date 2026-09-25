#include "observation.hpp"

#include <charconv>
#include <stdexcept>
#include <string_view>

namespace rl {
namespace {

[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error("observation: " + what); }

class Lines {
 public:
  explicit Lines(const std::string& text) : text_(text) {}
  bool done() const { return at_ >= text_.size(); }
  std::string_view Next() {
    if (done()) Fail("unexpected end of block");
    const size_t end = text_.find('\n', at_);
    const size_t stop = end == std::string::npos ? text_.size() : end;
    std::string_view line(text_.data() + at_, stop - at_);
    at_ = stop + 1;
    return line;
  }
  std::string_view Peek() const {
    const size_t end = text_.find('\n', at_);
    return std::string_view(text_.data() + at_, (end == std::string::npos ? text_.size() : end) - at_);
  }

 private:
  const std::string& text_;
  size_t at_ = 0;
};

std::vector<std::string_view> Split(std::string_view line) {
  std::vector<std::string_view> out;
  size_t at = 0;
  while (at <= line.size()) {
    const size_t sp = line.find(' ', at);
    const size_t stop = sp == std::string_view::npos ? line.size() : sp;
    out.push_back(line.substr(at, stop - at));
    at = stop + 1;
  }
  return out;
}

template <typename T>
T Num(std::string_view s) {
  T v{};
  const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  if (r.ec != std::errc() || r.ptr != s.data() + s.size()) Fail("bad number '" + std::string(s) + "'");
  return v;
}

std::string_view Keyed(Lines& lines, std::string_view key) {
  const std::string_view line = lines.Next();
  if (line.substr(0, key.size()) != key || line.size() <= key.size() || line[key.size()] != ' ')
    Fail("expected " + std::string(key) + ", got '" + std::string(line) + "'");
  return line.substr(key.size() + 1);
}

uint8_t Dir(std::string_view s) {
  if (s.size() == 1)
    for (int d = 0; d < 4; ++d)
      if (s[0] == dragons::kDirChar[d]) return static_cast<uint8_t>(d);
  Fail("bad direction '" + std::string(s) + "'");
}

int32_t EdgeCode(std::string_view s) {
  if (s == ".") return dragons::kEdgeEmpty;
  if (s == "w") return dragons::kEdgeKelp;
  const int32_t id = Num<int32_t>(s);
  if (id < 0) Fail("negative portal id");
  return id;
}

}  // namespace

Spawn ParseSpawn(const std::string& text) {
  Lines lines(text);
  Spawn s;
  s.id = Num<int32_t>(Keyed(lines, "ID"));
  const std::string_view team = Keyed(lines, "TEAM");
  if (team != "A" && team != "B") Fail("bad team");
  s.team = team == "B";
  const auto wh = Split(Keyed(lines, "MAP"));
  if (wh.size() != 2) Fail("bad MAP line");
  s.width = Num<int32_t>(wh[0]);
  s.height = Num<int32_t>(wh[1]);
  s.unit_limit = Num<int32_t>(Keyed(lines, "UNIT_LIMIT"));
  if (s.width < kMinSide || s.width > kMaxSide || s.height < kMinSide || s.height > kMaxSide)
    Fail("board " + std::to_string(s.width) + "x" + std::to_string(s.height) + " is outside 7..64 a side");
  return s;
}

Observation ParseRound(const Spawn& spawn, const std::string& text) {
  Lines lines(text);
  Observation o;
  o.spawn = spawn;
  o.round = Num<int32_t>(Keyed(lines, "ROUND"));
  o.facing = Dir(Keyed(lines, "DIR"));
  o.length = Num<int32_t>(Keyed(lines, "LENGTH"));
  o.unit_count = Num<int32_t>(Keyed(lines, "UNIT_COUNT"));
  const int32_t n = Num<int32_t>(Keyed(lines, "NUM_MSGS"));
  if (n < 0) Fail("negative NUM_MSGS");
  o.inbox.reserve(n);
  for (int32_t i = 0; i < n; ++i) o.inbox.push_back(Num<uint64_t>(lines.Next()));
  if (lines.Peek().substr(0, 7) == "ECHOES ") {
    const auto fields = Split(Keyed(lines, "ECHOES"));
    if (fields.size() != 5) Fail("ECHOES needs five fields");
    o.echo_present = true;
    for (int f = 0; f < 5; ++f) o.echo[f] = std::string(fields[f]);
  }
  for (int k = 0; k < kWindowTiles; ++k) {
    const auto f = Split(lines.Next());
    if (f.size() != 4) Fail("bad tile line");
    o.tile_x[k] = Num<int32_t>(f[0]);
    o.tile_y[k] = Num<int32_t>(f[1]);
    o.pearl[k] = static_cast<uint8_t>(Num<int32_t>(f[2]));
    o.countdown[k] = Num<int32_t>(f[3]);
  }
  const int32_t count = Num<int32_t>(Keyed(lines, "DRAGON_BODIES"));
  for (int32_t i = 0; i < count; ++i) {
    const auto f = Split(lines.Next());
    if (f.size() != 6 || (f[0] != "A" && f[0] != "B")) Fail("bad body line");
    Segment s;
    s.team = f[0] == "B";
    s.id = Num<int32_t>(f[1]);
    s.x = Num<int32_t>(f[2]);
    s.y = Num<int32_t>(f[3]);
    s.facing = Dir(f[4]);
    s.head = Num<int32_t>(f[5]) != 0;
    o.segments.push_back(s);
  }
  for (int r = 0; r <= kWindow; ++r) {
    const auto f = Split(lines.Next());
    if (f.size() != kWindow) Fail("bad horizontal edge row");
    for (int c = 0; c < kWindow; ++c) o.hedge[r * kWindow + c] = EdgeCode(f[c]);
  }
  for (int r = 0; r < kWindow; ++r) {
    const auto f = Split(lines.Next());
    if (f.size() != kWindow + 1) Fail("bad vertical edge row");
    for (int c = 0; c <= kWindow; ++c) o.vedge[r * (kWindow + 1) + c] = EdgeCode(f[c]);
  }
  while (!lines.done())
    if (!lines.Next().empty()) Fail("trailing text in round block");
  return o;
}

Spawn SpawnOf(const dragons::Game& game, int32_t dragon) { return ParseSpawn(game.InitBlock(dragon)); }

Observation Observe(const dragons::Game& game) {
  return ParseRound(SpawnOf(game, game.actor()), game.RoundBlock());
}

}  // namespace rl
