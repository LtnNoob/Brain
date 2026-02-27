#pragma once

#include "micro_model.hpp"
#include "concept_embedding_store.hpp"
#include "../memory/active_relation.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace brain19 {

class LongTermMemory;  // forward declaration

// =============================================================================
// EMBEDDING MANAGER
// =============================================================================
//
// Provides FlexEmbeddings (16D core + variable detail) for:
//   - Relation types (delegated to RelationTypeRegistry)
//   - Named contexts (auto-created on first access)
//

class EmbeddingManager {
public:
    EmbeddingManager();

    // Relation type embeddings (delegates to RelationTypeRegistry)
    const FlexEmbedding& get_relation_embedding(RelationType type) const;

    // Context embeddings (auto-created on first access)
    const FlexEmbedding& get_context_embedding(const std::string& name);

    // Create a context embedding without caching (for temporary use)
    FlexEmbedding make_context_embedding(const std::string& name) const;
    /// Compute target-specific embedding without string allocation
    FlexEmbedding make_target_embedding(size_t context_hash, uint64_t source_id, uint64_t target_id) const;

    // Convenience context accessors
    const FlexEmbedding& query_context() { return get_context_embedding("query"); }
    const FlexEmbedding& recall_context() { return get_context_embedding("recall"); }
    const FlexEmbedding& creative_context() { return get_context_embedding("creative"); }
    const FlexEmbedding& analytical_context() { return get_context_embedding("analytical"); }

    // Check if context exists
    bool has_context(const std::string& name) const;

    // Get all context names
    std::vector<std::string> get_context_names() const;

    // Context embedding direct access for persistence
    const std::unordered_map<std::string, FlexEmbedding>& context_embeddings() const { return context_embeddings_; }
    std::unordered_map<std::string, FlexEmbedding>& context_embeddings_mut() { return context_embeddings_; }

    // Concept embeddings
    ConceptEmbeddingStore& concept_embeddings() { return concept_embeddings_; }
    const ConceptEmbeddingStore& concept_embeddings() const { return concept_embeddings_; }

    // Train concept embeddings from KG structure (delegates to ConceptEmbeddingStore)
    ConceptEmbeddingStore::LearnResult train_embeddings(const LongTermMemory& ltm,
                                                         double alpha = 0.05,
                                                         size_t iterations = 3);
    ConceptEmbeddingStore::LearnResult train_embeddings(
        const LongTermMemory& ltm,
        const ConceptEmbeddingStore::LearnConfig& config);

private:
    std::unordered_map<std::string, FlexEmbedding> context_embeddings_;
    ConceptEmbeddingStore concept_embeddings_;
};

} // namespace brain19
