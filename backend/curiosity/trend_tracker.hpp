#pragma once

#include "../common/types.hpp"
#include <cstddef>
#include <unordered_map>

namespace brain19 {

// =============================================================================
// METRIC EMA — Exponential Moving Average with trend detection
// =============================================================================

struct MetricEMA {
    double value = 0.0;
    double trend = 0.0;  // positive = improving
    size_t samples = 0;

    void update(double new_val, double alpha = 0.1);
};

// =============================================================================
// TREND TRACKER — Tracks system-wide and per-concept trends over time
// =============================================================================

class TrendTracker {
public:
    MetricEMA avg_chain_quality;
    MetricEMA avg_model_loss;
    MetricEMA graph_density;
    MetricEMA total_pain;

    std::unordered_map<ConceptId, MetricEMA> concept_pain;

    void update_concept(std::unordered_map<ConceptId, MetricEMA>& map,
                        ConceptId cid, double value, double alpha = 0.1);

    // Get concept pain trend (returns 0 if not tracked)
    double get_concept_pain_trend(ConceptId cid) const;
};

} // namespace brain19
