#pragma once

#include "traversal_types.hpp"
#include "goal_state.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include <vector>
#include <optional>
#include <set>

namespace brain19 {

// =============================================================================
// FOCUS CURSOR
// =============================================================================
//
// Sequential graph traversal using MicroModel-scored edges.
// Replaces diffuse spreading activation for targeted queries.
//
// Core loop:
//   1. At current node, evaluate all neighbors via MicroModel::predict(e, c)
//   2. Pick best neighbor (or branch/backtrack)
//   3. Move to neighbor, accumulate context
//   4. Check termination conditions
//   5. Repeat
//
class FocusCursor {
public:
    FocusCursor(
        const LongTermMemory& ltm,
        ConceptModelRegistry& registry,
        EmbeddingManager& embeddings,
        FocusCursorConfig config = {}
    );

    // Seed the cursor at a starting concept
    void seed(ConceptId start);
    void seed(ConceptId start, const Vec10& initial_context);

    // Take one step: evaluate neighbors, pick best, move there
    // Returns the new position, or nullopt if terminated
    std::optional<ConceptId> step();

    // Step to a specific neighbor (forced move)
    bool step_to(ConceptId target);

    // Backtrack one step in history
    bool backtrack();

    // Run to completion: step() until termination
    TraversalResult deepen();

    // Shift focus to follow a specific relation type
    void shift_focus(RelationType preferred_type);

    // Get current state snapshot
    CursorView get_view() const;

    // Current position
    ConceptId position() const { return current_; }

    // Current depth
    size_t depth() const { return depth_; }

    // Has the cursor been seeded?
    bool is_seeded() const { return seeded_; }

    // Has termination been reached?
    bool is_terminated() const { return terminated_; }

    // Branch: create k copies of this cursor at current position
    std::vector<FocusCursor> branch(size_t k) const;

    // Build result from current history
    TraversalResult result() const;

    // Set goal for goal-directed traversal
    void set_goal(const GoalState& goal) { goal_ = goal; }

private:
    const LongTermMemory& ltm_;
    ConceptModelRegistry& registry_;
    EmbeddingManager& embeddings_;
    FocusCursorConfig config_;

    ConceptId current_ = 0;
    size_t depth_ = 0;
    Vec10 context_embedding_{};
    double accumulated_energy_ = 0.0;
    bool seeded_ = false;
    bool terminated_ = false;
    ExplorationMode mode_;

    std::vector<TraversalStep> history_;
    std::set<ConceptId> visited_;

    // Optional goal for goal-directed traversal
    GoalState goal_;

    // Preferred relation type (set by shift_focus)
    std::optional<RelationType> preferred_relation_;

    // Evaluate a single neighbor edge: returns predict() score
    double evaluate_edge(ConceptId from, ConceptId to, RelationType type) const;

    // Accumulate context when moving to a new concept
    void accumulate_context(ConceptId new_concept);

    // Check all termination conditions
    bool check_termination() const;

    // Get sorted candidate neighbors for current position
    struct Candidate {
        ConceptId target;
        RelationType relation;
        double score;
        bool outgoing;  // true = outgoing from current, false = incoming
    };
    std::vector<Candidate> get_candidates() const;
};

} // namespace brain19
