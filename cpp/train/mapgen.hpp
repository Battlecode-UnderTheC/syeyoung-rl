// Random boards generated on the fly (spec 11): every match gets a fresh map
// of the thread's board size with random spawns, starting lengths, dragons
// per team, kelp density, portal pairs and pearl timers. A port of
// dragons_env/mapgen.py::random_map with the parameter choices of
// tools/gen_maps.py; the text goes through the engine's own parser.
#pragma once

#include <random>
#include <string>

namespace dragons::train {

std::string RandomMapText(std::mt19937_64& rng, int width, int height, int unit_limit);

}  // namespace dragons::train
