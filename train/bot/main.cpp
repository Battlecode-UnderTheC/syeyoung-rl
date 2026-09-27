// The competition bot: one process per dragon, speaking the unswbc protocol on
// stdin/stdout. Each turn: parse the round block, update this lifetime's facts,
// run the plain-C++ policy (native/policy), print the command.
// Weights: compiled in (weights.cpp, the trainer's actor.bin as little-endian words,
// written by make_bot.sh): the judge's sandbox has no files to read, and a data
// segment costs next to nothing to load where decoding text blew the CPU budget.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "collector.hpp"
#include "policy.hpp"

namespace {

// the next line with its comment removed that has any text ("" at the end of input)
bool data_line(std::string& line) {
  while (std::getline(std::cin, line)) {
    if (const size_t c = line.find('#'); c != std::string::npos) line.erase(c);
    if (line.find_first_not_of(" \t\r") != std::string::npos) return true;
  }
  return false;
}

std::string first_word(const std::string& line) {
  std::istringstream s(line);
  std::string w;
  s >> w;
  return w;
}

int count_after(const std::string& line) {
  std::istringstream s(line);
  std::string w;
  int n = 0;
  s >> w >> n;
  return n;
}

}  // namespace

extern const uint32_t kWeightsWords[];
extern const unsigned long kWeightsBytes;

int main() {
  std::setvbuf(stdout, nullptr, _IOFBF, 1 << 16);
  std::ios::sync_with_stdio(true);
  try {
    // the spawn block: ID, TEAM, MAP, UNIT_LIMIT
    std::string line, init;
    for (int k = 0; k < 4; ++k) {
      if (!data_line(line)) return 0;
      init += line + "\n";
    }
    const rl::Spawn spawn = rl::ParseSpawn(init);
    const rl::Policy policy =
        rl::Policy::FromBytes(reinterpret_cast<const char*>(kWeightsWords), static_cast<size_t>(kWeightsBytes));
    rl::Lifetime life(spawn);
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(spawn.id + 1) ^ spawn.team);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // one decision's buffers
    std::vector<int32_t> win(rl::kWinTiles * rl::kTileFields), wdest(rl::kWinTiles * 4),
        grid(rl::kGridCells * rl::kGridFields), entity(rl::kMaxEntities * rl::kEntityFields),
        segment(rl::kWindowTiles * rl::kSegmentFields), status(rl::kStatusFields), prev_dirs(rl::kPrevDirsFields),
        echo_len(5);
    std::vector<uint8_t> prev_words(32), inbox(rl::kMaxInbox * 8), echo(5 * rl::kMaxEcho);
    std::vector<float> h(rl::kHidden, 0.0f);
    rl::ActorBatch batch{rl::kMaxInbox, rl::kMaxEcho, win.data(), wdest.data(), grid.data(), entity.data(),
                         segment.data(), status.data(), prev_dirs.data(), prev_words.data(), inbox.data(), echo.data(),
                         echo_len.data(), h.data()};

    while (data_line(line)) {
      if (first_word(line) == "ENDGAME") break;
      // the round block, line by line in its fixed structure
      std::string block = line + "\n";
      auto take = [&](int n) {
        for (int i = 0; i < n; ++i) {
          if (!data_line(line)) throw std::runtime_error("round block cut short");
          block += line + "\n";
        }
      };
      take(4);                                      // DIR, LENGTH, UNIT_COUNT, NUM_MSGS
      {
        std::istringstream s(block);
        std::string l, last;
        while (std::getline(s, l)) last = l;
        take(count_after(last));                    // the messages
      }
      if (!data_line(line)) throw std::runtime_error("round block cut short");
      block += line + "\n";
      take(first_word(line) == "ECHOES" ? rl::kWindowTiles : rl::kWindowTiles - 1);
      if (!data_line(line)) throw std::runtime_error("round block cut short");
      block += line + "\n";                         // DRAGON_BODIES n
      take(count_after(line));
      take((rl::kWindow + 1) + rl::kWindow);        // edge rows

      const rl::Observation o = rl::ParseRound(spawn, block);
      life.Observe(o);
      rl::EncodeObservation(life, o, batch, 0);
      float u[rl::kPolicyUniforms];
      for (float& x : u) x = uni(rng);
      const rl::PolicyOutput out = policy.Act(
          {win.data(), wdest.data(), grid.data(), entity.data(), segment.data(), status.data(), prev_dirs.data(),
           prev_words.data(), inbox.data(), rl::kMaxInbox, echo.data(), rl::kMaxEcho, echo_len.data()},
          h.data(), u);
      std::copy(out.h.begin(), out.h.end(), h.begin());

      rl::Command c;
      c.kind = static_cast<uint8_t>(out.kind);
      c.nsteps = static_cast<uint8_t>(out.kind == 0 ? out.nsteps : 0);
      for (int k = 0; k < rl::kMaxMoveTokens; ++k) c.dirs[k] = static_cast<uint8_t>(out.dirs[k]);
      c.split = out.split;
      c.sonar_mask = 0;
      for (int d = 0; d < 4; ++d)
        if (out.sonar[d] > 0) {
          c.sonar_mask |= 1 << d;
          c.words[d] = static_cast<uint64_t>(out.sonar[d] - 1);
        }
      life.Commit(c);

      if (c.kind == 0) {
        std::printf("MOVE ");
        for (int k = 0; k < c.nsteps; ++k) std::printf("%c", dragons::kDirChar[c.dirs[k]]);
        std::printf("\n");
      } else {
        std::printf("SPLIT %d\n", c.split);
      }
      for (int d = 0; d < 4; ++d)
        if (c.sonar_mask & (1 << d))
          std::printf("SONAR %c %llu\n", dragons::kDirChar[d], static_cast<unsigned long long>(c.words[d]));
      std::printf("PROTOCOL 3\nENDTURN\n");
      std::fflush(stdout);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bot error: %s\n", e.what());
    return 1;
  }
  return 0;
}
