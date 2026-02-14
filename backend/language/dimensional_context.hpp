#pragma once

#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../memory/relation_type_registry.hpp"
#include <cmath>
#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// DIMENSIONAL CONTEXT — Variable-dimensionality concept profiles
// =============================================================================
//
// Each concept discovers its own relevant dimensions from its graph structure.
// A concept with only taxonomic (IS_A) relations has 1 dimension.
// A concept like Photosynthesis with causal, functional, compositional, and
// hierarchical relations has 4+ dimensions. Dimensionality is EMERGENT.
//
// Internal representation: sparse map<category, weight> per concept.
// The number of observed categories across the entire graph determines the
// decoder vector size (runtime, not compile-time). Concepts with fewer active
// dimensions simply have zeros in those slots.
//
// Decoder output: [weights for each observed category | richness | entropy]
// where richness = concept_dims / max_dims and entropy = normalized Shannon H.
//

class DimensionalContext {
public:
    // Per-concept dimensional profile: sparse, variable-length
    struct DimProfile {
        // Only categories this concept actually participates in
        std::unordered_map<RelationCategory, double> weights;

        size_t dimensionality() const { return weights.size(); }

        double get(RelationCategory cat) const {
            auto it = weights.find(cat);
            return (it != weights.end()) ? it->second : 0.0;
        }
    };

    // Build profiles for all concepts. O(|C|+|R|), ~10ms.
    // Discovers dimensions from graph — number of observed categories is data-driven.
    void build(const LongTermMemory& ltm);

    // Get sparse profile for a concept (returns empty profile if not found)
    const DimProfile& get_profile(ConceptId cid) const;

    // Number of distinct relation categories observed across entire graph.
    // This is NOT a constant — it depends on what's in the knowledge graph.
    size_t observed_dimensions() const { return dim_order_.size(); }

    // Maximum dimensionality of any single concept in the graph
    size_t max_dimensionality() const { return max_dimensionality_; }

    // Size of decoder vector: observed_dims + 2 (richness + entropy)
    size_t decoder_dim() const { return dim_order_.size() + 2; }

    // Project a concept's sparse profile to a fixed-size decoder vector.
    // Size = decoder_dim(). Inactive dimensions are 0.0 — the decoder
    // learns that zeros mean "this dimension is irrelevant for this concept."
    std::vector<double> to_decoder_vec(ConceptId cid) const;

    // Average decoder vector across multiple concepts (for multi-seed queries).
    // Element-wise average; inactive dims contribute 0.0.
    std::vector<double> average_decoder_vec(const std::vector<ConceptId>& cids) const;

    bool is_built() const { return built_; }

private:
    std::unordered_map<ConceptId, DimProfile> profiles_;

    // Ordered list of categories observed in the graph (determines decoder vec layout)
    std::vector<RelationCategory> dim_order_;

    // Reverse index: category → position in dim_order_
    std::unordered_map<RelationCategory, size_t> dim_index_;

    size_t max_dimensionality_ = 0;
    bool built_ = false;

    static const DimProfile empty_profile_;
};

} // namespace brain19
