#include "goal_generator.hpp"

#include <algorithm>

namespace brain19 {

// =============================================================================
// GoalGenerator
// =============================================================================

GoalState GoalGenerator::from_trigger(const CuriosityTrigger& trigger) {
    GoalState goal;
    goal.query_text = trigger.description;

    switch (trigger.type) {
        // Legacy types
        case TriggerType::SHALLOW_RELATIONS:
            goal.goal_type = GoalType::EXPLORATION;
            goal.priority_weight = 0.4;
            goal.threshold = 0.6;
            break;

        case TriggerType::MISSING_DEPTH:
            goal.goal_type = GoalType::CAUSAL_CHAIN;
            goal.priority_weight = 0.6;
            goal.threshold = 0.7;
            break;

        case TriggerType::LOW_EXPLORATION:
            goal.goal_type = GoalType::EXPLORATION;
            goal.priority_weight = 0.3;
            goal.threshold = 0.5;
            break;

        case TriggerType::RECURRENT_WITHOUT_FUNCTION:
            goal.goal_type = GoalType::PROPERTY_QUERY;
            goal.priority_weight = 0.5;
            goal.threshold = 0.7;
            break;

        // New 13-signal types — use trigger.priority when available
        case TriggerType::PAIN_CLUSTER:
        case TriggerType::PREDICTION_FAILURE_ZONE:
            goal.goal_type = GoalType::CAUSAL_CHAIN;
            goal.priority_weight = (trigger.priority > 0.0) ? trigger.priority : 0.7;
            goal.threshold = 0.7;
            break;

        case TriggerType::TRUST_DECAY_REGION:
        case TriggerType::CONTRADICTION_REGION:
            goal.goal_type = GoalType::PROPERTY_QUERY;
            goal.priority_weight = (trigger.priority > 0.0) ? trigger.priority : 0.6;
            goal.threshold = 0.7;
            break;

        case TriggerType::MODEL_DIVERGENCE:
        case TriggerType::CROSS_SIGNAL_HOTSPOT:
            goal.goal_type = GoalType::EXPLORATION;
            goal.priority_weight = (trigger.priority > 0.0) ? trigger.priority : 0.8;
            goal.threshold = 0.6;
            break;

        case TriggerType::QUALITY_REGRESSION:
            goal.goal_type = GoalType::CAUSAL_CHAIN;
            goal.priority_weight = (trigger.priority > 0.0) ? trigger.priority : 0.5;
            goal.threshold = 0.6;
            break;

        case TriggerType::EPISODIC_STALENESS:
            goal.goal_type = GoalType::EXPLORATION;
            goal.priority_weight = (trigger.priority > 0.0) ? trigger.priority : 0.4;
            goal.threshold = 0.5;
            break;

        case TriggerType::UNKNOWN:
        default:
            goal.goal_type = GoalType::EXPLORATION;
            goal.priority_weight = 0.2;
            goal.threshold = 0.5;
            break;
    }

    // Use related concepts as targets for non-exploration goals
    if (goal.goal_type != GoalType::EXPLORATION && !trigger.related_concept_ids.empty()) {
        goal.target_concepts = trigger.related_concept_ids;
    }

    return goal;
}

std::vector<GoalState> GoalGenerator::from_triggers(const std::vector<CuriosityTrigger>& triggers) {
    std::vector<GoalState> goals;
    goals.reserve(triggers.size());
    for (const auto& t : triggers) {
        goals.push_back(from_trigger(t));
    }
    return goals;
}

// =============================================================================
// GoalQueue
// =============================================================================

GoalQueue::GoalQueue(size_t max_capacity)
    : max_capacity_(max_capacity)
{}

void GoalQueue::push(GoalState goal) {
    std::lock_guard<std::mutex> lock(mtx_);
    heap_.push_back(std::move(goal));
    sift_up(heap_.size() - 1);

    // If over capacity, remove lowest priority
    while (heap_.size() > max_capacity_) {
        // Find min-priority element (linear scan, small heap)
        size_t min_idx = 0;
        for (size_t i = 1; i < heap_.size(); ++i) {
            if (heap_[i].priority_weight < heap_[min_idx].priority_weight) {
                min_idx = i;
            }
        }
        // Swap with last and pop
        std::swap(heap_[min_idx], heap_.back());
        heap_.pop_back();
        // Re-heapify from scratch (simple, small heap)
        for (size_t i = heap_.size() / 2; i > 0; --i) {
            sift_down(i - 1);
        }
        if (!heap_.empty()) sift_down(0);
    }
}

std::optional<GoalState> GoalQueue::pop() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (heap_.empty()) return std::nullopt;

    GoalState top = std::move(heap_[0]);
    heap_[0] = std::move(heap_.back());
    heap_.pop_back();
    if (!heap_.empty()) sift_down(0);
    return top;
}

std::optional<GoalState> GoalQueue::peek() const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (heap_.empty()) return std::nullopt;
    return heap_[0];
}

void GoalQueue::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    heap_.clear();
}

size_t GoalQueue::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return heap_.size();
}

void GoalQueue::age(double factor) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& g : heap_) {
        g.priority_weight *= factor;
    }
}

void GoalQueue::prune_completed() {
    std::lock_guard<std::mutex> lock(mtx_);
    heap_.erase(
        std::remove_if(heap_.begin(), heap_.end(),
            [](const GoalState& g) { return g.is_complete(); }),
        heap_.end()
    );
    // Rebuild heap
    for (size_t i = heap_.size() / 2; i > 0; --i) {
        sift_down(i - 1);
    }
    if (!heap_.empty()) sift_down(0);
}

void GoalQueue::sift_up(size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (heap_[i].priority_weight > heap_[parent].priority_weight) {
            std::swap(heap_[i], heap_[parent]);
            i = parent;
        } else {
            break;
        }
    }
}

void GoalQueue::sift_down(size_t i) {
    size_t n = heap_.size();
    while (true) {
        size_t largest = i;
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        if (left < n && heap_[left].priority_weight > heap_[largest].priority_weight)
            largest = left;
        if (right < n && heap_[right].priority_weight > heap_[largest].priority_weight)
            largest = right;
        if (largest == i) break;
        std::swap(heap_[i], heap_[largest]);
        i = largest;
    }
}

} // namespace brain19
