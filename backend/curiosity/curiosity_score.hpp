#pragma once

#include "../common/types.hpp"
#include "../cursor/goal_state.hpp"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace brain19 {

// =============================================================================
// CURIOSITY DIMENSIONS — The 13 independent signal channels
// =============================================================================

enum class CuriosityDimension {
    PAIN_DRIVEN = 0,
    TRUST_DEFICIT,
    MODEL_UNCERTAINTY,
    NN_KAN_CONFLICT,
    TOPOLOGY_GAP,
    CONTRADICTION_ALERT,
    PREDICTION_ERROR,
    NOVELTY_EXPLORATION,
    EPISODIC_REVISIT,
    ACTIVATION_ANOMALY,
    EDGE_WEIGHT_ANOMALY,
    QUALITY_DEGRADATION,
    CROSS_SIGNAL,
    COUNT  // = 13
};

static constexpr size_t CURIOSITY_DIM_COUNT = static_cast<size_t>(CuriosityDimension::COUNT);

// =============================================================================
// CURIOSITY SCORE — Per-concept multi-dimensional curiosity assessment
// =============================================================================

struct CuriosityScore {
    ConceptId concept_id = 0;
    double total_score = 0.0;
    std::array<double, CURIOSITY_DIM_COUNT> dimension_scores{};
    CuriosityDimension primary_dimension = CuriosityDimension::NOVELTY_EXPLORATION;
    std::string reason;
};

// =============================================================================
// SEED ENTRY — A concept selected as a seed with priority and reason
// =============================================================================

struct SeedEntry {
    ConceptId concept_id = 0;
    double priority = 0.0;
    GoalType suggested_goal = GoalType::EXPLORATION;
    CuriosityDimension primary_reason = CuriosityDimension::NOVELTY_EXPLORATION;
    std::string reason_text;
};

// =============================================================================
// SEED PLAN — The output of the planning phase
// =============================================================================

struct SeedPlan {
    std::vector<SeedEntry> seeds;  // sorted by priority desc
    double system_health = 0.0;
    std::string health_summary;
};

} // namespace brain19
