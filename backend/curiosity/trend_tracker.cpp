#include "trend_tracker.hpp"

namespace brain19 {

void MetricEMA::update(double new_val, double alpha) {
    if (samples == 0) {
        value = new_val;
        trend = 0.0;
    } else {
        double old_value = value;
        value = (1.0 - alpha) * value + alpha * new_val;
        trend = value - old_value;  // positive = improving
    }
    ++samples;
}

void TrendTracker::update_concept(std::unordered_map<ConceptId, MetricEMA>& map,
                                  ConceptId cid, double value, double alpha) {
    map[cid].update(value, alpha);
}

double TrendTracker::get_concept_pain_trend(ConceptId cid) const {
    auto it = concept_pain.find(cid);
    if (it == concept_pain.end()) return 0.0;
    return it->second.trend;
}

} // namespace brain19
