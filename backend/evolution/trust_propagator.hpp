#pragma once

#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "epistemic_promotion.hpp"
#include "graph_densifier.hpp"
#include <unordered_map>
#include <vector>

namespace brain19 {

struct PropagationResult {
    ConceptId source = 0;                                     // The invalidated concept
    std::vector<std::pair<ConceptId, double>> affected;       // (concept, new_trust)
    std::vector<ConceptId> force_invalidated;                 // Trust → 0 by cumulation
    size_t concepts_checked = 0;
    size_t concepts_adjusted = 0;
};

struct PropagationConfig {
    float similarity_threshold = 0.5f;         // Min similarity for propagation
    float max_trust_reduction = 0.3f;          // Max trust reduction per event
    float cumulative_invalidation_threshold = 0.1f;  // Trust below this → force invalidate
    size_t max_hop_distance = 3;               // Max graph distance for candidates
    bool propagate_to_linguistic = false;      // Include linguistic layer?
    size_t max_recursion_depth = 3;            // Prevent infinite cascades

    PropagationConfig() {}
};

class TrustPropagator {
public:
    TrustPropagator(LongTermMemory& ltm,
                    EpistemicPromotion& promotion,
                    GraphDensifier& densifier,
                    PropagationConfig config = {});

    // Main method: propagate trust reduction from an invalidated concept
    PropagationResult propagate(ConceptId invalidated);

    // Combined similarity score between two concepts
    // Weights: structural 0.4, co-activation 0.35, shared_source 0.25
    float combined_similarity(ConceptId a, ConceptId b) const;

    // Compute trust reduction (without applying)
    float compute_reduction(ConceptId target, float similarity_to_invalidated) const;

    // History: which propagations have affected a concept?
    std::vector<ConceptId> get_propagation_sources(ConceptId target) const;

private:
    LongTermMemory& ltm_;
    EpistemicPromotion& promotion_;
    GraphDensifier& densifier_;
    PropagationConfig config_;

    // Tracking: ConceptId → list of (source_invalidation, reduction_applied)
    std::unordered_map<ConceptId,
        std::vector<std::pair<ConceptId, float>>> propagation_history_;

    // Current recursion depth (to prevent cascading loops)
    size_t current_depth_ = 0;

    // Find candidate concepts within max_hop_distance (BFS)
    std::vector<ConceptId> find_candidates(ConceptId source) const;

    // Co-activation score using shared neighbors (from GraphDensifier logic)
    float co_activation_score(ConceptId a, ConceptId b) const;

    // Shared-source score: fraction of shared incoming relations
    float shared_source_score(ConceptId a, ConceptId b) const;

    // Structural similarity: same relation type pattern
    float structural_similarity(ConceptId a, ConceptId b) const;
};

} // namespace brain19
