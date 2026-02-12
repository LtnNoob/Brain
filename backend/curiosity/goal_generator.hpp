#pragma once

#include "curiosity_trigger.hpp"
#include "../cursor/goal_state.hpp"

#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

namespace brain19 {

// =============================================================================
// GOAL GENERATOR
// =============================================================================
//
// Converts CuriosityTrigger → GoalState using a fixed mapping.
// Pure function: no state, no side effects.
//

class GoalGenerator {
public:
    // Convert a single trigger to a goal
    static GoalState from_trigger(const CuriosityTrigger& trigger);

    // Convert multiple triggers to goals
    static std::vector<GoalState> from_triggers(const std::vector<CuriosityTrigger>& triggers);
};

// =============================================================================
// GOAL QUEUE
// =============================================================================
//
// Priority max-heap for GoalStates with capacity limit and aging.
// Thread-safe.
//

class GoalQueue {
public:
    explicit GoalQueue(size_t max_capacity = 20);

    // Add a goal (drops lowest priority if at capacity)
    void push(GoalState goal);

    // Get and remove highest-priority goal
    std::optional<GoalState> pop();

    // Peek at highest-priority goal without removing
    std::optional<GoalState> peek() const;

    // Remove all goals
    void clear();

    // Number of queued goals
    size_t size() const;

    // Apply aging: reduce priority of all goals by factor
    void age(double factor = 0.95);

    // Remove completed goals (completion_metric >= threshold)
    void prune_completed();

private:
    size_t max_capacity_;
    std::vector<GoalState> heap_;
    mutable std::mutex mtx_;

    void sift_up(size_t i);
    void sift_down(size_t i);
};

} // namespace brain19
