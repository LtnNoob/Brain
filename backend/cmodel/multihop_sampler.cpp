#include "multihop_sampler.hpp"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace brain19 {

MultiHopSampler::MultiHopSampler(const MultiHopConfig& config)
    : config_(config)
{}

float MultiHopSampler::epistemic_trust(EpistemicType type) {
    switch (type) {
        case EpistemicType::FACT:        return 1.0f;
        case EpistemicType::DEFINITION:  return 1.0f;
        case EpistemicType::THEORY:      return 0.7f;
        case EpistemicType::INFERENCE:   return 0.6f;
        case EpistemicType::HYPOTHESIS:  return 0.4f;
        case EpistemicType::SPECULATION: return 0.2f;
    }
    return 0.5f;
}

// =============================================================================
// BFS path extraction
// =============================================================================

std::vector<MultiHopPath> MultiHopSampler::extract_paths(
        ConceptId source,
        const LongTermMemory& ltm) const {

    // BFS state: partial paths being extended
    struct BFSEntry {
        MultiHopPath path;
    };

    // Best path per terminus (dedup: only keep highest path_weight)
    std::unordered_map<ConceptId, MultiHopPath> best_by_terminus;

    // Visited set to prevent cycles within a single path is handled by
    // checking edges — but we need to avoid revisiting nodes on the same path.
    // We'll track visited per-path via the edge list.

    std::queue<BFSEntry> queue;

    // Seed: extend from source's outgoing relations
    auto outgoing = ltm.get_outgoing_relations(source);
    for (const auto& rel : outgoing) {
        auto target_info = ltm.retrieve_concept(rel.target);
        if (!target_info) continue;

        // Anti-Knowledge handling: simple AK = skip, complex AK = negative path
        if (target_info->is_anti_knowledge && target_info->complexity_score < 0.3f) continue;
        bool is_ak = target_info->is_anti_knowledge;

        float target_trust = epistemic_trust(target_info->epistemic.type);

        // Get trust decay for this relation's category
        const auto& behavior = get_behavior(rel.type);
        double decay = static_cast<double>(behavior.trust_decay_per_hop);

        // OPPOSITION kills path immediately (decay = 0.0)
        double hop_weight = rel.weight * target_trust * decay;
        // Complex AK: negate the path weight (negative training signal)
        if (is_ak) hop_weight = -hop_weight;
        if (std::abs(hop_weight) < config_.weight_floor) continue;

        HopEdge edge;
        edge.from = source;
        edge.to = rel.target;
        edge.type = rel.type;
        edge.weight = rel.weight;
        edge.epistemic_factor = target_trust;

        MultiHopPath path;
        path.source = source;
        path.terminus = rel.target;
        path.edges.push_back(edge);
        path.path_weight = hop_weight;

        // 1-hop paths are already covered by direct training —
        // only record paths with >= 2 hops as multi-hop
        // But still enqueue for extension
        BFSEntry entry;
        entry.path = std::move(path);
        queue.push(std::move(entry));
    }

    size_t queue_ops = 0;

    while (!queue.empty() && queue_ops < config_.max_bfs_queue) {
        auto entry = std::move(queue.front());
        queue.pop();
        ++queue_ops;

        ConceptId current = entry.path.terminus;

        // Record this path if it's multi-hop (>= 2 edges)
        if (entry.path.edges.size() >= 2) {
            auto it = best_by_terminus.find(entry.path.terminus);
            if (it == best_by_terminus.end() ||
                entry.path.path_weight > it->second.path_weight) {
                best_by_terminus[entry.path.terminus] = entry.path;
            }
        }

        // Extend: try outgoing relations from current terminus
        if (entry.path.edges.size() >= config_.max_hops) continue;  // depth limit
        auto next_rels = ltm.get_outgoing_relations(current);
        for (const auto& rel : next_rels) {
            // Prevent cycles: don't revisit source or any node in current path
            if (rel.target == source) continue;
            bool in_path = false;
            for (const auto& e : entry.path.edges) {
                if (e.from == rel.target || e.to == rel.target) {
                    in_path = true;
                    break;
                }
            }
            if (in_path) continue;

            auto target_info = ltm.retrieve_concept(rel.target);
            if (!target_info) continue;

            // Anti-Knowledge handling in path extension
            if (target_info->is_anti_knowledge && target_info->complexity_score < 0.3f) continue;
            bool is_ak = target_info->is_anti_knowledge;

            float target_trust = epistemic_trust(target_info->epistemic.type);

            const auto& behavior = get_behavior(rel.type);
            double decay = static_cast<double>(behavior.trust_decay_per_hop);
            double hop_weight = rel.weight * target_trust * decay;
            if (is_ak) hop_weight = -hop_weight;

            double new_path_weight = entry.path.path_weight * hop_weight;
            if (std::abs(new_path_weight) < config_.weight_floor) continue;

            HopEdge edge;
            edge.from = current;
            edge.to = rel.target;
            edge.type = rel.type;
            edge.weight = rel.weight;
            edge.epistemic_factor = target_trust;

            MultiHopPath new_path;
            new_path.source = source;
            new_path.terminus = rel.target;
            new_path.edges = entry.path.edges;
            new_path.edges.push_back(edge);
            new_path.path_weight = new_path_weight;

            BFSEntry new_entry;
            new_entry.path = std::move(new_path);
            queue.push(std::move(new_entry));
        }
    }

    // Collect all paths, sort by weight descending, cap at max_paths_per_concept
    std::vector<MultiHopPath> result;
    result.reserve(best_by_terminus.size());
    for (auto& [tid, path] : best_by_terminus) {
        result.push_back(std::move(path));
    }

    if (result.size() > config_.max_paths_per_concept) {
        std::partial_sort(result.begin(),
                          result.begin() + config_.max_paths_per_concept,
                          result.end(),
                          [](const MultiHopPath& a, const MultiHopPath& b) {
                              return a.path_weight > b.path_weight;
                          });
        result.resize(config_.max_paths_per_concept);
    }

    return result;
}

// =============================================================================
// Composite embedding: weighted mean of relation embeddings along path
// =============================================================================

FlexEmbedding MultiHopSampler::compose_path_embedding(
        const std::vector<HopEdge>& edges,
        const EmbeddingManager& embeddings) const {

    FlexEmbedding composite;
    composite.core.fill(0.0);

    double weight_sum = 0.0;
    for (const auto& edge : edges) {
        const auto& rel_emb = embeddings.get_relation_embedding(edge.type);
        double w = edge.weight;
        for (size_t i = 0; i < CORE_DIM; ++i) {
            composite.core[i] += rel_emb.core[i] * w;
        }
        weight_sum += w;
    }

    if (weight_sum > 0.0) {
        for (size_t i = 0; i < CORE_DIM; ++i) {
            composite.core[i] /= weight_sum;
        }
    }

    return composite;
}

// =============================================================================
// Generate TrainingSamples from multi-hop paths
// =============================================================================

std::vector<TrainingSample> MultiHopSampler::generate_samples(
        ConceptId source,
        EmbeddingManager& embeddings,
        const LongTermMemory& ltm) const {

    static const size_t RECALL_HASH = std::hash<std::string>{}("recall");

    auto paths = extract_paths(source, ltm);

    std::vector<TrainingSample> samples;
    samples.reserve(paths.size());

    for (const auto& path : paths) {
        TrainingSample sample;
        sample.relation_embedding = compose_path_embedding(path.edges, embeddings);
        sample.context_embedding = embeddings.make_target_embedding(
            RECALL_HASH, source, path.terminus);
        sample.target = path.path_weight;  // decayed weight as training target
        samples.push_back(std::move(sample));
    }

    return samples;
}

} // namespace brain19
