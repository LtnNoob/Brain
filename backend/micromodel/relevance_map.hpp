#pragma once

#include "micro_model.hpp"
#include "micro_model_registry.hpp"
#include "embedding_manager.hpp"
#include "../ltm/long_term_memory.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brain19 {

// =============================================================================
// RELEVANCE MAP
// =============================================================================
//
// Evaluates a concept's micro-model over all KG nodes to produce a scored map
// of relevance from that concept's "perspective".
//
// Overlay operations support Phase 3 Creativity by combining multiple
// concept perspectives into novel relevance landscapes.
//

enum class OverlayMode {
    ADDITION,          // sum scores
    MAX,               // take maximum
    WEIGHTED_AVERAGE   // weighted average
};

class RelevanceMap {
public:
    RelevanceMap() : source_cid_(0) {}
    explicit RelevanceMap(ConceptId source) : source_cid_(source) {}

    // Compute relevance map for a source concept over all KG nodes
    static RelevanceMap compute(ConceptId source,
                                MicroModelRegistry& registry,
                                EmbeddingManager& embeddings,
                                const LongTermMemory& ltm,
                                RelationType rel_type,
                                const std::string& context);

    // Query results
    double score(ConceptId cid) const;
    std::vector<std::pair<ConceptId, double>> top_k(size_t k) const;
    std::vector<std::pair<ConceptId, double>> above_threshold(double threshold) const;

    // Overlay operations (for Phase 3 Creativity)
    void overlay(const RelevanceMap& other, OverlayMode mode, double weight = 0.5);

    // Combine multiple maps
    static RelevanceMap combine(const std::vector<RelevanceMap>& maps,
                                OverlayMode mode,
                                const std::vector<double>& weights = {});

    // Normalize scores to [0, 1]
    void normalize();

    // Accessors
    ConceptId source() const { return source_cid_; }
    size_t size() const { return scores_.size(); }
    bool empty() const { return scores_.empty(); }

    const std::unordered_map<ConceptId, double>& scores() const { return scores_; }

private:
    ConceptId source_cid_;
    std::unordered_map<ConceptId, double> scores_;
};

} // namespace brain19
