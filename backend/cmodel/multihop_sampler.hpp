#pragma once

#include "../common/types.hpp"
#include "../ltm/long_term_memory.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../micromodel/micro_model.hpp"
#include "../core/relation_config.hpp"

#include <vector>
#include <cstddef>

namespace brain19 {

// =============================================================================
// MULTI-HOP SAMPLER
// =============================================================================
//
// BFS path extraction through the KG for multi-hop training samples.
// Paths decay naturally via weight * epistemic_trust * trust_decay_per_hop.
// No hardcoded depth limit — termination is structural:
//   - Weight floor: path_weight < 0.01 → stop
//   - OPPOSITION decay = 0.0 → kills path immediately
//   - Per-concept cap: 50 highest-weight paths
//   - Dedup by terminus: best path per source→terminus pair
//

struct HopEdge {
    ConceptId from;
    ConceptId to;
    RelationType type;
    double weight;            // LTM edge weight
    float epistemic_factor;   // target concept's epistemic trust
};

struct MultiHopPath {
    ConceptId source;
    ConceptId terminus;
    std::vector<HopEdge> edges;
    double path_weight;       // product of (weight * epistemic * decay) per hop
};

struct MultiHopConfig {
    double weight_floor = 0.01;       // Stop extending when path_weight drops below
    size_t max_paths_per_concept = 50; // Keep top-N paths per source concept
    size_t max_bfs_queue = 5000;       // BFS queue cap to prevent runaway on dense graphs
    size_t max_hops = 3;              // Hard depth limit (prevents exponential fan-out)
};

class MultiHopSampler {
public:
    explicit MultiHopSampler(const MultiHopConfig& config = MultiHopConfig{});

    // Extract multi-hop paths from a source concept via BFS
    std::vector<MultiHopPath> extract_paths(ConceptId source,
                                             const LongTermMemory& ltm) const;

    // Create a composite relation embedding from a multi-hop path
    FlexEmbedding compose_path_embedding(const std::vector<HopEdge>& edges,
                                          const EmbeddingManager& embeddings) const;

    // Generate TrainingSamples from multi-hop paths for a concept
    std::vector<TrainingSample> generate_samples(ConceptId source,
                                                  EmbeddingManager& embeddings,
                                                  const LongTermMemory& ltm) const;

private:
    MultiHopConfig config_;

    // Epistemic trust factor (same scale as ConceptTrainer)
    static float epistemic_trust(EpistemicType type);
};

} // namespace brain19
