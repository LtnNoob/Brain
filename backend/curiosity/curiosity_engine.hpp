#pragma once

#include "curiosity_trigger.hpp"
#include <vector>
#include <cstddef>

namespace brain19 {

// Forward declaration for read-only access
class ShortTermMemory;

// Observation snapshot (read-only)
struct SystemObservation {
    ContextId context_id;
    size_t active_concept_count;
    size_t active_relation_count;
    
    SystemObservation()
        : context_id(0)
        , active_concept_count(0)
        , active_relation_count(0)
    {}
};

// CuriosityEngine: Pure signal generator
// Observes system state, emits triggers
// NO actions, NO learning, NO direct modifications
class CuriosityEngine {
public:
    CuriosityEngine();
    ~CuriosityEngine();
    
    // Observe system state (read-only)
    // Returns curiosity triggers based on observations
    std::vector<CuriosityTrigger> observe_and_generate_triggers(
        const std::vector<SystemObservation>& observations
    );
    
    // Configuration thresholds (mechanical)
    void set_shallow_relation_threshold(double ratio);
    void set_low_exploration_threshold(size_t min_concepts);
    
private:
    // Detection thresholds (not intelligence, just cutoffs)
    double shallow_relation_ratio_;
    size_t low_exploration_min_concepts_;
    
    // Detection helpers (pure pattern matching, no intelligence)
    bool detect_shallow_relations(const SystemObservation& obs) const;
    bool detect_low_exploration(const SystemObservation& obs) const;
};

} // namespace brain19
