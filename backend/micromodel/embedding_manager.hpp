#pragma once

#include "micro_model.hpp"
#include "concept_embedding_store.hpp"
#include "../memory/active_relation.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// EMBEDDING MANAGER
// =============================================================================
//
// Provides 10D embeddings for:
//   - Relation types (delegated to RelationTypeRegistry — supports dynamic types)
//   - Named contexts (auto-created on first access)
//
// Relation embeddings are managed by the RelationTypeRegistry singleton.
// Context embeddings are initialized with small random values for symmetry breaking.
//

class EmbeddingManager {
public:
    EmbeddingManager();

    // Relation type embeddings (delegates to RelationTypeRegistry)
    const Vec10& get_relation_embedding(RelationType type) const;

    // Context embeddings (auto-created on first access)
    const Vec10& get_context_embedding(const std::string& name);

    // Create a context embedding without caching (for temporary use)
    Vec10 make_context_embedding(const std::string& name) const;
    /// Compute target-specific embedding without string allocation
    Vec10 make_target_embedding(size_t context_hash, uint64_t source_id, uint64_t target_id) const;

    // Convenience context accessors
    const Vec10& query_context() { return get_context_embedding("query"); }
    const Vec10& recall_context() { return get_context_embedding("recall"); }
    const Vec10& creative_context() { return get_context_embedding("creative"); }
    const Vec10& analytical_context() { return get_context_embedding("analytical"); }

    // Check if context exists
    bool has_context(const std::string& name) const;

    // Get all context names
    std::vector<std::string> get_context_names() const;

    // Context embedding direct access for persistence
    const std::unordered_map<std::string, Vec10>& context_embeddings() const { return context_embeddings_; }
    std::unordered_map<std::string, Vec10>& context_embeddings_mut() { return context_embeddings_; }

    // Concept embeddings
    ConceptEmbeddingStore& concept_embeddings() { return concept_embeddings_; }
    const ConceptEmbeddingStore& concept_embeddings() const { return concept_embeddings_; }

private:
    std::unordered_map<std::string, Vec10> context_embeddings_;
    ConceptEmbeddingStore concept_embeddings_;
};

} // namespace brain19
