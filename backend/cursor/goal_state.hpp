#pragma once

#include "../common/types.hpp"
#include "../micromodel/micro_model.hpp"  // Vec10
#include <vector>
#include <string>
#include <cmath>

namespace brain19 {

// GoalType: What kind of answer are we looking for?
enum class GoalType {
    DEFINITION,        // "Was ist X?" — find definition of a concept
    CAUSAL_CHAIN,      // "Was passiert wenn X?" — follow causal links
    COMPARISON,        // "Was ist der Unterschied zwischen X und Y?"
    PROPERTY_QUERY,    // "Welche Eigenschaft hat X?"
    EXPLORATION,       // Open-ended exploration from seeds
    CUSTOM             // User-defined goal
};

// GoalState: Tracks what the FocusCursor is trying to achieve.
// Guides termination and traversal decisions.
struct GoalState {
    GoalType goal_type = GoalType::EXPLORATION;
    std::vector<ConceptId> target_concepts;  // Concepts we want to reach
    Vec10 query_embedding{};                 // Encoded query vector
    double completion_metric = 0.0;          // [0,1] — how close to done
    double threshold = 0.8;                  // completion_metric >= threshold → done
    double priority_weight = 1.0;            // Higher = pursue more aggressively
    std::string query_text;                  // Original query for template engine

    // Is the goal satisfied?
    bool is_complete() const {
        return completion_metric >= threshold;
    }

    // Update progress based on what the cursor has found so far
    void update_progress(
        const std::vector<ConceptId>& visited_concepts,
        size_t chain_length
    ) {
        if (target_concepts.empty()) {
            // Exploration mode: progress based on chain length
            // Diminishing returns after 6 steps
            completion_metric = 1.0 - std::exp(-0.15 * static_cast<double>(chain_length));
            return;
        }

        // Target mode: fraction of target concepts found
        size_t found = 0;
        for (ConceptId target : target_concepts) {
            for (ConceptId visited : visited_concepts) {
                if (visited == target) {
                    ++found;
                    break;
                }
            }
        }
        completion_metric = static_cast<double>(found) / static_cast<double>(target_concepts.size());
    }

    // Factory: Create goal from a definition query
    static GoalState definition_goal(ConceptId target, const Vec10& query_emb, const std::string& query) {
        GoalState gs;
        gs.goal_type = GoalType::DEFINITION;
        gs.target_concepts = {target};
        gs.query_embedding = query_emb;
        gs.threshold = 0.8;
        gs.query_text = query;
        return gs;
    }

    // Factory: Create goal from a causal query
    static GoalState causal_goal(const std::vector<ConceptId>& seeds, const Vec10& query_emb, const std::string& query) {
        GoalState gs;
        gs.goal_type = GoalType::CAUSAL_CHAIN;
        gs.target_concepts = seeds;  // Seed concepts as starting points
        gs.query_embedding = query_emb;
        gs.threshold = 0.7;
        gs.query_text = query;
        return gs;
    }

    // Factory: Create open exploration goal
    static GoalState exploration_goal(const Vec10& query_emb, const std::string& query) {
        GoalState gs;
        gs.goal_type = GoalType::EXPLORATION;
        gs.target_concepts = {};
        gs.query_embedding = query_emb;
        gs.threshold = 0.6;
        gs.query_text = query;
        return gs;
    }
};

} // namespace brain19
