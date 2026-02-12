#pragma once

#include "../common/types.hpp"
#include "../micromodel/micro_model.hpp"  // Vec10
#include "../memory/active_relation.hpp"  // RelationType
#include <vector>
#include <string>

namespace brain19 {

// A single step in the traversal chain
struct TraversalStep {
    ConceptId concept_id;           // Which concept we visited
    RelationType relation_from;  // How we got here (from previous step)
    double weight_at_entry;      // MicroModel predict() score when we entered
    Vec10 context_at_entry;      // Context embedding at time of entry
    size_t depth;                // Depth in the chain (0 = seed)
};

// Exploration mode for FocusCursor
enum class ExplorationMode {
    GREEDY,          // Always pick highest-scored neighbor
    EXPLORATORY,     // Some randomness in selection
    GOAL_DIRECTED    // Use GoalState to guide traversal
};

// Result of a single FocusCursor traversal
struct TraversalResult {
    std::vector<TraversalStep> chain;              // Full chain of steps
    std::vector<ConceptId> concept_sequence;       // Just the concept IDs in order
    std::vector<RelationType> relation_sequence;   // Relations between consecutive concepts
    double chain_score = 0.0;                      // Average weight across chain
    size_t total_steps = 0;                        // Length of chain

    // Convenience: is chain non-empty?
    bool empty() const { return chain.empty(); }
};

// View of current cursor state (read-only snapshot)
struct CursorView {
    ConceptId current;             // Current position
    size_t depth;                  // Current depth
    Vec10 context_embedding;       // Current accumulated context
    double accumulated_energy;     // Energy spent so far
    ExplorationMode mode;          // Current exploration mode
    size_t history_size;           // Number of steps taken
};

// Configuration for FocusCursor
struct FocusCursorConfig {
    size_t max_depth = 12;                 // Maximum traversal depth
    double min_weight_threshold = 0.1;     // Stop if best edge < this
    double energy_budget = 10.0;           // Total energy available
    double energy_per_step = 1.0;          // Energy cost per step
    double context_mix_rate = 0.3;         // How much new context blends in
    ExplorationMode default_mode = ExplorationMode::GOAL_DIRECTED;
    size_t max_neighbors_to_evaluate = 30; // Limit neighbor evaluation
};

// Result of FocusCursorManager::process_seeds
struct QueryResult {
    std::vector<TraversalResult> chains;   // All chains (one per seed)
    TraversalResult best_chain;            // Best scoring chain
    std::vector<ConceptId> all_activated;  // Union of all visited concepts
};

} // namespace brain19
