// Where the boards come from. The collector asks for one per game.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace train {

class MapSource {
 public:
  virtual ~MapSource() = default;
  // The text of the next game's map; called concurrently from collector threads.
  virtual std::string Next(std::mt19937_64& rng) const = 0;
};

// Every *.map in the given directories.
std::unique_ptr<MapSource> MapFiles(const std::vector<std::string>& dirs, int max_side);

// Random boards (a C++ port of dragons_env.mapgen): size, symmetry, kelp,
// portals, pearl beds, dragons per team and unit limit drawn per board.
struct GeneratorConfig {
  int min_side = 11, max_side = 64;
  // the rest mirrors dragons_env.mapgen.random_map's parameters
};
std::unique_ptr<MapSource> GeneratedMaps(const GeneratorConfig& config);

// A weighted mixture, e.g. 30% contest maps, 70% generated.
std::unique_ptr<MapSource> Mixture(std::vector<std::pair<double, std::unique_ptr<MapSource>>> parts);

}  // namespace train
