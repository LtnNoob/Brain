#pragma once

#include "micro_model.hpp"
#include "../memory/active_relation.hpp"

#include <array>
#include <string>
#include <unordered_map>

namespace brain19 {

// =============================================================================
// EMBEDDING MANAGER
// =============================================================================
//
// Provides 10D embeddings for:
//   - Relation types (one per RelationType, 10 total)
//   - Named contexts (auto-created on first access)
//
// Relation embeddings are initialized with heuristic patterns that encode
// semantic character (hierarchical, causal, etc.). Context embeddings are
// initialized with small random values for symmetry breaking.
//

static constexpr size_t NUM_RELATION_TYPES = 10;

class EmbeddingManager {
public:
    EmbeddingManager();

    // Relation type embeddings (fixed set of 10)
    const Vec10& get_relation_embedding(RelationType type) const;

    // Context embeddings (auto-created on first access)
    const Vec10& get_context_embedding(const std::string& name);

    // Create a context embedding without caching (for temporary use)
    Vec10 make_context_embedding(const std::string& name) const;
    Vec10 make_target_embedding(size_t context_hash, uint64_t source_id, uint64_t target_id) const;
    
    // Compute target-specific embedding without string allocation

    // Convenience context accessors
    const Vec10& query_context() { return get_context_embedding("query"); }
    const Vec10& recall_context() { return get_context_embedding("recall"); }
    const Vec10& creative_context() { return get_context_embedding("creative"); }
    const Vec10& analytical_context() { return get_context_embedding("analytical"); }

    // Check if context exists
    bool has_context(const std::string& name) const;

    // Get all context names
    std::vector<std::string> get_context_names() const;

    // Direct access for persistence
    const std::array<Vec10, NUM_RELATION_TYPES>& relation_embeddings() const { return relation_embeddings_; }
    std::array<Vec10, NUM_RELATION_TYPES>& relation_embeddings_mut() { return relation_embeddings_; }

    const std::unordered_map<std::string, Vec10>& context_embeddings() const { return context_embeddings_; }
    std::unordered_map<std::string, Vec10>& context_embeddings_mut() { return context_embeddings_; }

private:
    void init_relation_embeddings();

    std::array<Vec10, NUM_RELATION_TYPES> relation_embeddings_;
    std::unordered_map<std::string, Vec10> context_embeddings_;
};

} // namespace brain19
