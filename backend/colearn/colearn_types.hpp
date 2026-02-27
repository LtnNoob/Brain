#pragma once

#include "error_collector.hpp"
#include "../common/types.hpp"
#include "../memory/active_relation.hpp"
#include "../micromodel/flex_embedding.hpp"
#include "../graph_net/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace brain19 {

// =============================================================================
// CO-LEARNING TYPES — Episodes, Signals, Configuration
// =============================================================================
//
// Episodic Memory (hippocampus): temporal sequences of experiences.
// Episodes are variable-length — a single moment (duration=0) or a long sequence.
// Each has a timestamp + duration.
//
// Co-Learning Loop: wake (reason) → sleep (replay/consolidate) → train (retrain CMs).
//

// =============================================================================
// EpisodeStep — One moment in a temporal sequence
// =============================================================================

struct EpisodeStep {
    ConceptId concept_id = 0;
    RelationType relation = RelationType::CUSTOM;  // How we got here
    ConceptId from_concept = 0;                     // Previous concept (0 = seed)
    CoreVec activation{};                           // Full core activation vector
    double step_trust = 0.0;
    double nn_quality = 0.0;
    double kan_quality = 0.0;
    double kan_gate = 1.0;
};

// =============================================================================
// Episode — Variable-length temporal sequence
// =============================================================================

struct Episode {
    uint64_t id = 0;
    uint64_t timestamp_ms = 0;
    uint64_t duration_ms = 0;           // Can be 0 for instantaneous
    ConceptId seed = 0;

    std::vector<EpisodeStep> steps;
    double quality = 0.0;               // Chain quality metric
    TerminationReason termination = TerminationReason::STILL_RUNNING;

    uint32_t replay_count = 0;
    double consolidation_strength = 0.0; // [0,1] — how well consolidated

    size_t length() const { return steps.empty() ? 0 : steps.size() - 1; }
    bool empty() const { return steps.empty(); }
};

// =============================================================================
// ConsolidationResult — Feedback from consolidation
// =============================================================================

struct ConsolidationResult {
    size_t edges_strengthened = 0;
    size_t edges_weakened = 0;
    size_t edges_pruned = 0;
    size_t episodes_replayed = 0;
    size_t episodes_consolidated = 0;

    ConsolidationResult& operator+=(const ConsolidationResult& other) {
        edges_strengthened += other.edges_strengthened;
        edges_weakened += other.edges_weakened;
        edges_pruned += other.edges_pruned;
        episodes_replayed += other.episodes_replayed;
        episodes_consolidated += other.episodes_consolidated;
        return *this;
    }
};

// =============================================================================
// CoLearnConfig — All hyperparameters for the Co-Learning loop
// =============================================================================

struct CoLearnConfig {
    // Wake phase
    size_t wake_chains_per_cycle = 50;

    // Sleep phase (replay)
    size_t sleep_replay_count = 50;
    double replay_weight_quality = 0.4;
    double replay_weight_recency = 0.3;
    double replay_weight_novelty = 0.3;

    // Consolidation thresholds
    double strengthen_threshold = 0.6;   // Quality above this → strengthen
    double weaken_threshold = 0.3;       // Quality below this → weaken
    double prune_threshold = 0.05;       // Weight below this → prune
    double weight_delta = 0.05;          // Base weight change per consolidation

    // Training phase
    size_t retrain_epochs = 50;            // Base model epochs (needs convergence, not noise)
    size_t retrain_refined_epochs = 5;     // Refined/KAN epochs per retrained model
    double retrain_threshold = 1.0;        // Cumulative |delta| before retraining fires
    double retrain_learning_rate = 0.001;  // Fine-tuning LR (10x smaller than initial training)
    double retrain_kan_lr = 0.0005;        // Fine-tuning KAN LR
    double pain_learning_rate_boost = 1.5;
    double lr_decay_rate = 0.05;          // Cycle-based LR decay: 1/(1 + rate*cycle)

    // Episodic memory
    size_t max_episodes = 10000;

    // Error-driven learning (prediction error → corrective training)
    ErrorCorrectionConfig error_correction;
};

} // namespace brain19
