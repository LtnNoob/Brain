#pragma once

#include "colearn_types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../graph_net/graph_reasoner.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brain19 {

// =============================================================================
// KNOWLEDGE EXTRACTOR — Consolidation: Episodes → Graph
// =============================================================================
//
// During consolidation: re-reasons from episode seed → new chain →
// compare quality → differential signal → weight adjustments.
// Uses existing GraphReasoner::extract_signals().
//

class KnowledgeExtractor {
public:
    KnowledgeExtractor(LongTermMemory& ltm, GraphReasoner& reasoner,
                       const CoLearnConfig& config);

    // Consolidate a single episode
    ConsolidationResult consolidate_episode(const Episode& episode);

    // Consolidate a batch of episodes
    ConsolidationResult consolidate_batch(const std::vector<const Episode*>& episodes);

    // Apply accumulated signals from a ChainSignal
    size_t apply_signals(const ChainSignal& signal);

    // Set current cycle for LR decay
    void set_cycle(size_t cycle) { current_cycle_ = cycle; }

    // Cumulative absolute weight change per concept (for retraining threshold)
    const std::unordered_map<ConceptId, double>& cumulative_changes() const { return cumulative_changes_; }
    void clear_retrained(const std::vector<ConceptId>& retrained) {
        for (ConceptId cid : retrained) cumulative_changes_.erase(cid);
    }

private:
    LongTermMemory& ltm_;
    GraphReasoner& reasoner_;
    CoLearnConfig config_;
    size_t current_cycle_ = 0;
    std::unordered_map<RelationId, double> accumulated_deltas_;
    std::unordered_map<ConceptId, double> cumulative_changes_;  // concept → sum of |delta|

    // Find relation between source and target of a given type
    RelationId find_relation(ConceptId source, ConceptId target, RelationType type) const;
};

} // namespace brain19
