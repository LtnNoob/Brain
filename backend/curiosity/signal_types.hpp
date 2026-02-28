#pragma once

#include "../common/types.hpp"
#include "../epistemic/epistemic_metadata.hpp"
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// CONCEPT SIGNALS — Per-concept observation from all 13 signal channels
// =============================================================================

struct ConceptSignals {
    ConceptId concept_id = 0;

    // [1] Pain/Reward
    double avg_edge_pain = 0.0;
    double seed_pain_ema = 0.0;

    // [2] Prediction Error
    size_t correction_count = 0;

    // [3] CM Confidence
    bool has_model = false;
    bool model_converged = false;
    double model_loss = 1.0;

    // [4] NN vs KAN Divergence
    double nn_kan_divergence = 0.0;

    // [5] Graph Topology
    size_t relation_count = 0;

    // [6] Episodic Memory
    size_t episode_count = 0;
    double avg_episode_quality = 0.0;

    // [7] Trust
    double trust = 0.5;
    EpistemicType epistemic_type = EpistemicType::HYPOTHESIS;

    // [8] Novelty (low episodes = high novelty)
    double novelty_score = 0.0;

    // [9] Contradictions
    bool has_contradictions = false;
    double contradiction_ratio = 0.0;

    // [10-12] Edge weights, chain stats
    double avg_edge_weight = 0.0;
    double edge_weight_variance = 0.0;
};

// =============================================================================
// SYSTEM SIGNALS — Global system-level metrics
// =============================================================================

struct SystemSignals {
    size_t total_concepts = 0;
    size_t total_relations = 0;
    double graph_density = 0.0;
    double avg_degree = 0.0;
    size_t converged_models = 0;
    double avg_model_loss = 0.0;
    size_t total_episodes = 0;
    size_t total_corrections = 0;
};

// =============================================================================
// SYSTEM SNAPSHOT — Complete observation of the system at one point in time
// =============================================================================

struct SystemSnapshot {
    SystemSignals system;
    std::vector<ConceptSignals> concepts;
    std::unordered_map<ConceptId, size_t> concept_index;  // cid -> index into concepts
    uint64_t timestamp_ms = 0;

    const ConceptSignals* get_concept(ConceptId cid) const {
        auto it = concept_index.find(cid);
        if (it == concept_index.end()) return nullptr;
        return &concepts[it->second];
    }
};

} // namespace brain19
