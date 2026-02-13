#include "embedding_manager.hpp"
#include "../memory/relation_type_registry.hpp"

#include <cmath>
#include <numeric>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

EmbeddingManager::EmbeddingManager() {
    // Relation embeddings are now managed by RelationTypeRegistry::instance()
    // No local initialization needed.
}

// =============================================================================
// Relation embeddings — delegate to registry
// =============================================================================

const FlexEmbedding& EmbeddingManager::get_relation_embedding(RelationType type) const {
    return RelationTypeRegistry::instance().get_embedding(type);
}

// =============================================================================
// Context embeddings
// =============================================================================

FlexEmbedding EmbeddingManager::make_context_embedding(const std::string& name) const {
    // Deterministic pseudo-random based on name hash
    FlexEmbedding emb;
    size_t hash = std::hash<std::string>{}(name);
    for (size_t i = 0; i < CORE_DIM; ++i) {
        // Mix hash with index for variety
        size_t mixed = hash ^ (i * 2654435761u);
        // NOTE: Range is actually [-0.1, +0.0999] (asymmetric). Not fixed to preserve
        // deterministic embedding compatibility with persisted models. See FIX_PLAN_ROUND3 Bug #4.
        double val = static_cast<double>(mixed % 10000) / 50000.0 - 0.1;
        emb.core[i] = val;
    }
    // detail stays empty
    return emb;
}

const FlexEmbedding& EmbeddingManager::get_context_embedding(const std::string& name) {
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


FlexEmbedding EmbeddingManager::make_target_embedding(size_t context_hash, uint64_t source_id, uint64_t target_id) const {
    // Deterministic pseudo-random from numeric values only (zero string allocation)
    FlexEmbedding emb;
    size_t seed = context_hash ^ (source_id * 2654435761ULL) ^ (target_id * 2246822519ULL);
    for (size_t i = 0; i < CORE_DIM; ++i) {
        size_t mixed = seed ^ (i * 2654435761ULL);
        double val = static_cast<double>(mixed % 10000) / 50000.0 - 0.1;
        emb.core[i] = val;
    }
    return emb;
}

} // namespace brain19
