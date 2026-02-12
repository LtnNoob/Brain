#pragma once

#include "goal_state.hpp"
#include "traversal_types.hpp"

namespace brain19 {

// =============================================================================
// STANDALONE TERMINATION CHECK
// =============================================================================
//
// Extracted from FocusCursor::check_termination() so external code
// (e.g., ThinkingPipeline control loop) can evaluate termination
// without owning a FocusCursor.
//
// Checks (in order):
//   1. Max depth exceeded
//   2. Energy budget exhausted
//   3. Goal completed (only in GOAL_DIRECTED mode)
//

inline bool check_termination(
    const GoalState& goal,
    const CursorView& view,
    const FocusCursorConfig& config
) {
    // 1. Max depth
    if (view.depth >= config.max_depth) return true;

    // 2. Energy exhaustion
    if (view.accumulated_energy >= config.energy_budget) return true;

    // 3. Goal completion (only relevant in goal-directed mode)
    if (view.mode == ExplorationMode::GOAL_DIRECTED && goal.is_complete()) return true;

    return false;
}

} // namespace brain19
