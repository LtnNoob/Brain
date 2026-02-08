#include "curiosity_engine.hpp"

namespace brain19 {

CuriosityEngine::CuriosityEngine()
    : shallow_relation_ratio_(0.3)  // Relations < 30% of concepts
    , low_exploration_min_concepts_(5)
{
}

CuriosityEngine::~CuriosityEngine() {
}

std::vector<CuriosityTrigger> CuriosityEngine::observe_and_generate_triggers(
    const std::vector<SystemObservation>& observations
) {
    std::vector<CuriosityTrigger> triggers;
    
    for (const auto& obs : observations) {
        // Pattern detection: Shallow relations
        if (detect_shallow_relations(obs)) {
            CuriosityTrigger trigger(
                TriggerType::SHALLOW_RELATIONS,
                obs.context_id,
                {},  // No specific concepts identified
                "Many concepts activated but few relations"
            );
            triggers.push_back(trigger);
        }
        
        // Pattern detection: Low exploration
        if (detect_low_exploration(obs)) {
            CuriosityTrigger trigger(
                TriggerType::LOW_EXPLORATION,
                obs.context_id,
                {},
                "Stable context with minimal variation"
            );
            triggers.push_back(trigger);
        }
    }
    
    return triggers;
}

void CuriosityEngine::set_shallow_relation_threshold(double ratio) {
    shallow_relation_ratio_ = ratio;
}

void CuriosityEngine::set_low_exploration_threshold(size_t min_concepts) {
    low_exploration_min_concepts_ = min_concepts;
}

bool CuriosityEngine::detect_shallow_relations(const SystemObservation& obs) const {
    if (obs.active_concept_count == 0) {
        return false;
    }
    
    double ratio = static_cast<double>(obs.active_relation_count) 
                 / static_cast<double>(obs.active_concept_count);
    
    return ratio < shallow_relation_ratio_;
}

bool CuriosityEngine::detect_low_exploration(const SystemObservation& obs) const {
    // Low exploration: Few concepts active
    return obs.active_concept_count > 0 
        && obs.active_concept_count < low_exploration_min_concepts_;
}

} // namespace brain19
