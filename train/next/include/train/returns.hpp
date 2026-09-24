// Advantages and value targets for a Batch: team-outcome GAE along each
// dragon's own turns. Pure: no state, no devices, unit-tested on hand-built
// batches.
#pragma once

#include <vector>

#include "train/batch.hpp"

namespace train {

struct ReturnsConfig {
  double gamma = 0.999;      // per round
  double lambda = 0.95;      // per turn
};

struct Returns {
  std::vector<float> advantage;   // per row, not normalised; 0 for Untrained rows
  std::vector<float> target;      // per row: advantage + value
};

Returns ComputeReturns(const Batch& batch, const ReturnsConfig& config);

// Mean and standard deviation over trained rows, for normalisation at packing.
struct AdvantageScale {
  float mean, stddev;
};
AdvantageScale Scale(const Batch& batch, const Returns& returns);

}  // namespace train
