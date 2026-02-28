#pragma once

#include "micro_model.hpp"  // FlexEmbedding, CORE_DIM
#include "flex_embedding.hpp"  // EmbeddingMeta, DimensionManager
#include "../common/types.hpp"

#include <cstddef>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brain19 {

class LongTermMemory;  // forward declaration

// =============================================================================
// ConceptEmbeddingStore
// =============================================================================
//
// Per-concept FlexEmbeddings (16D core + variable detail), auto-created from
// hash on first access. Supports nudge (gradient-free update) and two-phase
// cosine similarity search (core-filter then full-rerank).
//

class ConceptEmbeddingStore {
public:
    // Get embedding for concept (auto-created from hash if missing)
    const FlexEmbedding& get(ConceptId cid);

    // Explicitly set an embedding
    void set(ConceptId cid, const FlexEmbedding& emb);

    // Nudge embedding toward a target: emb = (1-alpha)*emb + alpha*target
    void nudge(ConceptId cid, const FlexEmbedding& target, double alpha = 0.1);

    // Full cosine similarity between two concept embeddings
    double similarity(ConceptId a, ConceptId b);

    // Find k most similar concepts using two-phase search:
    //   Phase 1: core_similarity for all -> top 50 candidates
    //   Phase 2: full_similarity for top 50 -> top K
    std::vector<std::pair<ConceptId, double>> most_similar(ConceptId cid, size_t k);

    // Get embedding for concept (returns hash_init if missing, does NOT modify store)
    FlexEmbedding get_or_default(ConceptId cid) const;

    // Check if concept has an embedding
    bool has(ConceptId cid) const;

    // Number of stored embeddings
    size_t size() const { return store_.size(); }

    // Clear all embeddings
    void clear() { store_.clear(); }

    // Configuration for learn_from_graph
    struct LearnConfig {
        double alpha = 0.1;                // Positive nudge strength
        size_t iterations = 15;            // Training iterations
        size_t negative_samples = 5;       // Negatives per concept per iteration
        double negative_alpha_ratio = 0.3; // neg_alpha = alpha * this
        uint64_t rng_seed = 42;            // For reproducibility
    };

    // Learn embeddings from KG structure: nudge each concept toward
    // the weighted average of its neighbors' embeddings.
    // Returns number of concepts updated.
    struct LearnResult {
        size_t concepts_updated = 0;
        size_t total_neighbors = 0;
        size_t iterations = 0;
    };
    LearnResult learn_from_graph(const LongTermMemory& ltm,
                                 double alpha = 0.05, size_t iterations = 3);
    LearnResult learn_from_graph(const LongTermMemory& ltm, const LearnConfig& config);

    // === Flex Growth/Shrink ===

    // Record that a concept was activated (for growth tracking)
    void record_activation(ConceptId cid, uint32_t tick);

    // Record gradient magnitude for a concept (EMA update)
    void record_gradient(ConceptId cid, float magnitude);

    // Update relation count for a concept
    void update_relation_count(ConceptId cid, uint32_t count);

    // Evaluate all concepts for growth/shrink and apply changes.
    // Returns number of concepts resized.
    struct ResizeResult {
        size_t grown = 0;
        size_t shrunk = 0;
    };
    ResizeResult evaluate_and_resize(uint32_t current_tick);

    // Access metadata
    const EmbeddingMeta& meta(ConceptId cid) const;

    // Direct access for persistence
    const std::unordered_map<ConceptId, FlexEmbedding>& data() const { return store_; }
    std::unordered_map<ConceptId, FlexEmbedding>& data_mut() { return store_; }

private:
    std::unordered_map<ConceptId, FlexEmbedding> store_;
    std::unordered_map<ConceptId, EmbeddingMeta> meta_;
    std::mt19937 resize_rng_{42};

    // Create a deterministic embedding from concept ID hash
    static FlexEmbedding hash_init(ConceptId cid);
};

} // namespace brain19
