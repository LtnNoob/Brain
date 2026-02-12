#pragma once

#include "focus_cursor.hpp"
#include "goal_state.hpp"
#include "../memory/stm.hpp"
#include <vector>
#include <set>

namespace brain19 {

// =============================================================================
// FOCUS CURSOR MANAGER
// =============================================================================
//
// Orchestrates FocusCursor instances for query processing:
//   1. Creates cursors for seed concepts
//   2. Runs each cursor to completion
//   3. Selects best chain
//   4. Persists results to STM
//
class FocusCursorManager {
public:
    FocusCursorManager(
        const LongTermMemory& ltm,
        MicroModelRegistry& registry,
        EmbeddingManager& embeddings,
        ShortTermMemory& stm,
        FocusCursorConfig config = {}
    );

    // Process seeds: create cursors, run to completion, return results
    QueryResult process_seeds(
        const std::vector<ConceptId>& seeds,
        const Vec10& query_context
    );

    // Process seeds with goal
    QueryResult process_seeds(
        const std::vector<ConceptId>& seeds,
        const Vec10& query_context,
        const GoalState& goal
    );

    // Persist traversal results to STM
    void persist_to_stm(ContextId ctx, const TraversalResult& chain);

    // Get config
    const FocusCursorConfig& config() const { return config_; }

private:
    const LongTermMemory& ltm_;
    MicroModelRegistry& registry_;
    EmbeddingManager& embeddings_;
    ShortTermMemory& stm_;
    FocusCursorConfig config_;
};

} // namespace brain19
