#pragma once

#include "micro_model.hpp"  // Vec10, EMBED_DIM
#include "../common/types.hpp"

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brain19 {

// =============================================================================
// ConceptEmbeddingStore
// =============================================================================
//
// Per-concept 10D embeddings, auto-created from hash on first access.
// Supports nudge (gradient-free update) and cosine similarity search.
//

class ConceptEmbeddingStore {
public:
    // Get embedding for concept (auto-created from hash if missing)
    const Vec10& get(ConceptId cid);

    // Explicitly set an embedding
    void set(ConceptId cid, const Vec10& emb);

    // Nudge embedding toward a target: emb = (1-alpha)*emb + alpha*target
    void nudge(ConceptId cid, const Vec10& target, double alpha = 0.1);

    // Cosine similarity between two concept embeddings
    double similarity(ConceptId a, ConceptId b);

    // Find k most similar concepts to the given one
    std::vector<std::pair<ConceptId, double>> most_similar(ConceptId cid, size_t k);

    // Check if concept has an embedding
    bool has(ConceptId cid) const;

    // Number of stored embeddings
    size_t size() const { return store_.size(); }

    // Clear all embeddings
    void clear() { store_.clear(); }

    // Direct access for persistence
    const std::unordered_map<ConceptId, Vec10>& data() const { return store_; }
    std::unordered_map<ConceptId, Vec10>& data_mut() { return store_; }

private:
    std::unordered_map<ConceptId, Vec10> store_;

    // Create a deterministic embedding from concept ID hash
    static Vec10 hash_init(ConceptId cid);
};

} // namespace brain19
