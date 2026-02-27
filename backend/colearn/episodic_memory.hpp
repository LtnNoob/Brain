#pragma once

#include "colearn_types.hpp"
#include "../graph_net/epistemic_trace.hpp"

#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// EPISODIC MEMORY — Hippocampus Layer
// =============================================================================
//
// Stores temporal sequences of reasoning experiences (Episodes).
// Supports weighted replay selection (quality, recency, novelty).
// Concept index enables fast lookup by concept.
// Eviction prioritizes fully-consolidated episodes.
//

class EpisodicMemory {
public:
    explicit EpisodicMemory(size_t max_episodes = 10000);

    // Store an episode, returns its assigned ID
    uint64_t store(const Episode& episode);

    // Convert a GraphChain to an Episode
    Episode from_chain(const GraphChain& chain, ConceptId seed) const;

    // Retrieve episode by ID
    const Episode* get(uint64_t id) const;

    // Select episodes for replay (weighted by quality, recency, novelty)
    std::vector<const Episode*> select_for_replay(
        size_t count,
        double w_quality, double w_recency, double w_novelty) const;

    // Get all episodes involving a concept
    std::vector<const Episode*> episodes_for_concept(ConceptId cid) const;

    // Mark episode as replayed (increments replay_count)
    void mark_replayed(uint64_t id);

    // Mark episode as consolidated with given strength
    void mark_consolidated(uint64_t id, double strength);

    // Evict fully-consolidated episodes to reach target_count
    size_t evict_consolidated(size_t target_count);

    // Current episode count
    size_t episode_count() const { return episodes_.size(); }

private:
    std::unordered_map<uint64_t, Episode> episodes_;
    std::unordered_map<ConceptId, std::vector<uint64_t>> concept_index_;
    uint64_t next_id_ = 1;
    size_t max_episodes_;
};

} // namespace brain19
