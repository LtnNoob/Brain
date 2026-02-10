#include "embedding_manager.hpp"

#include <cmath>
#include <numeric>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

EmbeddingManager::EmbeddingManager() {
    init_relation_embeddings();
}

// =============================================================================
// Relation embedding initialization
// =============================================================================
//
// Each RelationType gets a heuristic 10D embedding that encodes its semantic
// character. Dimensions loosely represent:
//   0: hierarchical  1: causal  2: compositional  3: similarity
//   4: temporal       5: support/opposition  6: specificity
//   7: directionality 8: abstractness  9: strength
//

void EmbeddingManager::init_relation_embeddings() {
    // IS_A: strong hierarchical, moderate specificity
    relation_embeddings_[static_cast<size_t>(RelationType::IS_A)] =
        {0.9, 0.0, 0.1, 0.3, 0.0, 0.1, 0.7, 0.8, 0.5, 0.7};

    // HAS_PROPERTY: compositional, moderate
    relation_embeddings_[static_cast<size_t>(RelationType::HAS_PROPERTY)] =
        {0.2, 0.0, 0.8, 0.2, 0.0, 0.1, 0.5, 0.6, 0.3, 0.5};

    // CAUSES: strong causal, directional, temporal
    relation_embeddings_[static_cast<size_t>(RelationType::CAUSES)] =
        {0.0, 0.9, 0.0, 0.1, 0.7, 0.1, 0.6, 0.9, 0.4, 0.8};

    // ENABLES: moderate causal, enabling
    relation_embeddings_[static_cast<size_t>(RelationType::ENABLES)] =
        {0.0, 0.6, 0.1, 0.2, 0.4, 0.3, 0.4, 0.7, 0.3, 0.5};

    // PART_OF: strong compositional, hierarchical
    relation_embeddings_[static_cast<size_t>(RelationType::PART_OF)] =
        {0.6, 0.0, 0.9, 0.2, 0.0, 0.1, 0.6, 0.7, 0.2, 0.6};

    // SIMILAR_TO: strong similarity, symmetric
    relation_embeddings_[static_cast<size_t>(RelationType::SIMILAR_TO)] =
        {0.1, 0.0, 0.1, 0.9, 0.0, 0.2, 0.3, 0.1, 0.5, 0.4};

    // CONTRADICTS: opposition, strong directionality
    relation_embeddings_[static_cast<size_t>(RelationType::CONTRADICTS)] =
        {0.0, 0.1, 0.0, -0.5, 0.0, -0.9, 0.7, 0.5, 0.6, 0.8};

    // SUPPORTS: positive support
    relation_embeddings_[static_cast<size_t>(RelationType::SUPPORTS)] =
        {0.1, 0.2, 0.1, 0.4, 0.0, 0.9, 0.4, 0.5, 0.5, 0.6};

    // TEMPORAL_BEFORE: strong temporal, directional
    relation_embeddings_[static_cast<size_t>(RelationType::TEMPORAL_BEFORE)] =
        {0.0, 0.3, 0.0, 0.1, 0.9, 0.0, 0.3, 0.8, 0.2, 0.5};

    // CUSTOM: neutral baseline
    relation_embeddings_[static_cast<size_t>(RelationType::CUSTOM)] =
        {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
}

// =============================================================================
// Relation embeddings
// =============================================================================

const Vec10& EmbeddingManager::get_relation_embedding(RelationType type) const {
    size_t idx = static_cast<size_t>(type);
    // Safety: clamp to CUSTOM if out of range
    if (idx >= NUM_RELATION_TYPES) {
        idx = static_cast<size_t>(RelationType::CUSTOM);
    }
    return relation_embeddings_[idx];
}

// =============================================================================
// Context embeddings
// =============================================================================

Vec10 EmbeddingManager::make_context_embedding(const std::string& name) const {
    // Deterministic pseudo-random based on name hash
    Vec10 emb;
    size_t hash = std::hash<std::string>{}(name);
    for (size_t i = 0; i < EMBED_DIM; ++i) {
        // Mix hash with index for variety
        size_t mixed = hash ^ (i * 2654435761u);
        // NOTE: Range is actually [-0.1, +0.0999] (asymmetric). Not fixed to preserve
        // deterministic embedding compatibility with persisted models. See FIX_PLAN_ROUND3 Bug #4.
        double val = static_cast<double>(mixed % 10000) / 50000.0 - 0.1;
        emb[i] = val;
    }
    return emb;
}

const Vec10& EmbeddingManager::get_context_embedding(const std::string& name) {
    auto it = context_embeddings_.find(name);
    if (it == context_embeddings_.end()) {
        auto [inserted, ok] = context_embeddings_.emplace(name, make_context_embedding(name));
        return inserted->second;
    }
    return it->second;
}

bool EmbeddingManager::has_context(const std::string& name) const {
    return context_embeddings_.find(name) != context_embeddings_.end();
}

std::vector<std::string> EmbeddingManager::get_context_names() const {
    std::vector<std::string> names;
    names.reserve(context_embeddings_.size());
    for (const auto& [name, emb] : context_embeddings_) {
        names.push_back(name);
    }
    return names;
}

} // namespace brain19
