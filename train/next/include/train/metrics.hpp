// One iteration's numbers, defined once and rendered twice (a log line and a
// JSON object). Non-finite values are written as null.
#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "train/collector.hpp"
#include "train/evaluation.hpp"
#include "train/learner.hpp"

namespace train {

struct IterationMetrics {
  int64_t iteration;
  int64_t turns_total;
  double seconds_total;
  std::vector<CollectionStats> collection;          // per collector
  UpdateReport update;
  std::optional<std::vector<MatchupResult>> evaluation;
};

std::string SummaryLine(const IterationMetrics& m);
void WriteJsonLine(std::ostream& out, const IterationMetrics& m);

}  // namespace train
