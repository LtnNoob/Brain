#pragma once

#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "complexity_analyzer.hpp"
#include <string>
#include <vector>

namespace brain19 {

// Forward declaration — TrustPropagator defined in Phase 3
class TrustPropagator;

struct RetentionStats {
    size_t total_invalidated = 0;
    size_t marked_anti_knowledge = 0;
    size_t gc_candidates = 0;
    size_t actually_removed = 0;
    std::vector<ConceptId> new_anti_knowledge;
};

class RetentionManager {
public:
    RetentionManager(LongTermMemory& ltm,
                     ComplexityAnalyzer& analyzer);

    // Called after invalidation: decide retain or GC
    void on_invalidation(ConceptId invalidated);

    // Periodic GC cycle
    RetentionStats run_gc_cycle(size_t max_removals = 500);

    // Explain why anti-knowledge was retained
    std::string explain_anti_knowledge(ConceptId id) const;

    // Check if a candidate resembles known anti-knowledge
    bool resembles_known_error(ConceptId candidate, float threshold = 0.7f) const;

private:
    LongTermMemory& ltm_;
    ComplexityAnalyzer& analyzer_;
};

} // namespace brain19
