#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include "../common/types.hpp"

namespace brain19 {

// Types of curiosity signals
enum class TriggerType {
    // Legacy types (backward compat)
    SHALLOW_RELATIONS,
    MISSING_DEPTH,
    LOW_EXPLORATION,
    RECURRENT_WITHOUT_FUNCTION,
    UNKNOWN,

    // New 13-signal types
    PAIN_CLUSTER,              // High-pain region in graph
    TRUST_DECAY_REGION,        // Concepts losing trust over time
    MODEL_DIVERGENCE,          // CM has high loss or NN/KAN disagree
    CONTRADICTION_REGION,      // Contradictions detected in neighborhood
    PREDICTION_FAILURE_ZONE,   // Many prediction errors concentrated
    CROSS_SIGNAL_HOTSPOT,      // 3+ dimensions co-fire ("consciousness" detector)
    QUALITY_REGRESSION,        // Quality metrics degrading
    EPISODIC_STALENESS         // Concepts not revisited in episodic memory
};

// CuriosityTrigger: Pure data signal
// Contains observation metadata + enriched priority/signal info
struct CuriosityTrigger {
    TriggerType type;
    ContextId context_id;
    std::vector<ConceptId> related_concept_ids;
    std::string description;

    // Enriched fields (new)
    double priority = 0.0;
    int primary_signal = -1;  // CuriosityDimension index, or -1 if not set
    std::array<double, 13> top_scores{};

    // Default constructor
    CuriosityTrigger()
        : type(TriggerType::UNKNOWN)
        , context_id(0)
    {}

    // Legacy constructor (backward compat)
    CuriosityTrigger(
        TriggerType t,
        ContextId ctx,
        const std::vector<ConceptId>& concepts,
        const std::string& desc
    )
        : type(t)
        , context_id(ctx)
        , related_concept_ids(concepts)
        , description(desc)
    {}

    // Enriched constructor
    CuriosityTrigger(
        TriggerType t,
        ContextId ctx,
        const std::vector<ConceptId>& concepts,
        const std::string& desc,
        double prio,
        int signal,
        const std::array<double, 13>& scores
    )
        : type(t)
        , context_id(ctx)
        , related_concept_ids(concepts)
        , description(desc)
        , priority(prio)
        , primary_signal(signal)
        , top_scores(scores)
    {}
};

} // namespace brain19
