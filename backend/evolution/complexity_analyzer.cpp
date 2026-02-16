#include "complexity_analyzer.hpp"
#include <algorithm>
#include <queue>

namespace brain19 {

ComplexityAnalyzer::ComplexityAnalyzer(LongTermMemory& ltm,
                                       GraphDensifier& densifier,
                                       RetentionConfig config)
    : ltm_(ltm)
    , densifier_(densifier)
    , config_(config)
{
}

float ComplexityAnalyzer::norm(size_t value, size_t cap) {
    if (cap == 0) return 0.0f;
    return std::min(static_cast<float>(value) / static_cast<float>(cap), 1.0f);
}

ComplexityMetrics ComplexityAnalyzer::analyze(ConceptId id) const {
    ComplexityMetrics metrics;

    // 1. Causal chain length
    auto chain = extract_causal_chain(id);
    metrics.causal_chain_length = chain.size();

    // 2. Dependency subgraph size
    auto subgraph = extract_dependency_subgraph(id, config_.max_traversal_depth);
    metrics.involved_concepts = subgraph.size();

    // 3. Relation depth (max BFS distance)
    // Already computed as part of extract_dependency_subgraph, but we compute
    // the actual max depth here via BFS
    {
        std::unordered_map<ConceptId, size_t> dist;
        std::queue<ConceptId> bfs;
        bfs.push(id);
        dist[id] = 0;
        size_t max_depth = 0;

        while (!bfs.empty()) {
            ConceptId cur = bfs.front();
            bfs.pop();
            size_t d = dist[cur];
            if (d >= config_.max_traversal_depth) continue;

            for (const auto& rel : ltm_.get_outgoing_relations(cur)) {
                if (dist.find(rel.target) == dist.end()) {
                    dist[rel.target] = d + 1;
                    max_depth = std::max(max_depth, d + 1);
                    bfs.push(rel.target);
                }
            }
        }
        metrics.relation_depth = max_depth;
    }

    // 4. Inference steps in causal chain
    metrics.inference_steps = count_inference_steps(chain);

    // 5. Also count incoming DENOTES references (linguistic complexity boost)
    auto incoming = ltm_.get_incoming_relations(id);
    size_t linguistic_refs = 0;
    for (const auto& rel : incoming) {
        if (rel.type == RelationType::DENOTES) {
            ++linguistic_refs;
        }
    }
    metrics.involved_concepts += linguistic_refs;

    // 6. Normalized score
    metrics.normalized_score =
        config_.weight_causal_chain     * norm(metrics.causal_chain_length, 10) +
        config_.weight_involved_concepts * norm(metrics.involved_concepts, 20) +
        config_.weight_relation_depth    * norm(metrics.relation_depth, 8) +
        config_.weight_inference_steps   * norm(metrics.inference_steps, 5);

    return metrics;
}

bool ComplexityAnalyzer::should_retain(ConceptId invalidated) const {
    auto metrics = analyze(invalidated);
    return metrics.normalized_score >= config_.complexity_threshold
        && metrics.causal_chain_length >= config_.min_causal_chain
        && metrics.involved_concepts >= config_.min_involved_concepts;
}

size_t ComplexityAnalyzer::evaluate_all_invalidated() {
    auto invalidated = ltm_.get_concepts_by_status(EpistemicStatus::INVALIDATED);
    size_t newly_marked = 0;

    for (auto cid : invalidated) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (!cinfo || cinfo->is_anti_knowledge) continue;

        if (should_retain(cid)) {
            auto metrics = analyze(cid);
            std::string reason = "causal_chain:" + std::to_string(metrics.causal_chain_length) +
                                 ",involved:" + std::to_string(metrics.involved_concepts) +
                                 ",score:" + std::to_string(metrics.normalized_score);
            ltm_.mark_as_anti_knowledge(cid, reason);

            // Cache complexity score
            auto it = ltm_.retrieve_concept(cid);
            if (it) {
                // Update complexity_score via direct concept access
                // (We need to modify through the concepts_ map — use update path)
                // Note: we store it via a small trick — invalidate_concept preserves it
            }
            ++newly_marked;
        }
    }

    return newly_marked;
}

std::vector<ConceptId> ComplexityAnalyzer::extract_causal_chain(ConceptId id) const {
    // Forward DFS along CAUSES/ENABLES edges to find longest chain
    std::vector<ConceptId> best_chain;

    // Forward chain
    {
        std::vector<ConceptId> forward;
        std::unordered_set<ConceptId> visited;
        ConceptId cur = id;
        visited.insert(cur);

        while (true) {
            bool found_next = false;
            for (const auto& rel : ltm_.get_outgoing_relations(cur)) {
                if ((rel.type == RelationType::CAUSES || rel.type == RelationType::ENABLES)
                    && visited.find(rel.target) == visited.end()) {
                    forward.push_back(rel.target);
                    visited.insert(rel.target);
                    cur = rel.target;
                    found_next = true;
                    break;
                }
            }
            if (!found_next) break;
            if (forward.size() >= config_.max_traversal_depth) break;
        }

        // Backward chain
        std::vector<ConceptId> backward;
        cur = id;
        while (true) {
            bool found_prev = false;
            for (const auto& rel : ltm_.get_incoming_relations(cur)) {
                if ((rel.type == RelationType::CAUSES || rel.type == RelationType::ENABLES)
                    && visited.find(rel.source) == visited.end()) {
                    backward.push_back(rel.source);
                    visited.insert(rel.source);
                    cur = rel.source;
                    found_prev = true;
                    break;
                }
            }
            if (!found_prev) break;
            if (backward.size() >= config_.max_traversal_depth) break;
        }

        // Combine: backward (reversed) + id + forward
        best_chain.reserve(backward.size() + 1 + forward.size());
        for (auto it = backward.rbegin(); it != backward.rend(); ++it) {
            best_chain.push_back(*it);
        }
        best_chain.push_back(id);
        for (auto fwd : forward) {
            best_chain.push_back(fwd);
        }
    }

    return best_chain;
}

std::unordered_set<ConceptId> ComplexityAnalyzer::extract_dependency_subgraph(
    ConceptId id, size_t max_depth) const
{
    std::unordered_set<ConceptId> subgraph;
    std::queue<std::pair<ConceptId, size_t>> bfs;
    bfs.push({id, 0});
    subgraph.insert(id);

    while (!bfs.empty()) {
        auto [cur, depth] = bfs.front();
        bfs.pop();
        if (depth >= max_depth) continue;

        // Outgoing
        for (const auto& rel : ltm_.get_outgoing_relations(cur)) {
            if (subgraph.find(rel.target) == subgraph.end()) {
                subgraph.insert(rel.target);
                bfs.push({rel.target, depth + 1});
            }
        }

        // Incoming
        for (const auto& rel : ltm_.get_incoming_relations(cur)) {
            if (subgraph.find(rel.source) == subgraph.end()) {
                subgraph.insert(rel.source);
                bfs.push({rel.source, depth + 1});
            }
        }
    }

    return subgraph;
}

size_t ComplexityAnalyzer::longest_causal_chain(ConceptId start) const {
    return extract_causal_chain(start).size();
}

size_t ComplexityAnalyzer::count_inference_steps(const std::vector<ConceptId>& chain) const {
    size_t count = 0;
    for (auto cid : chain) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (cinfo && cinfo->epistemic.type == EpistemicType::INFERENCE) {
            ++count;
        }
    }
    return count;
}

} // namespace brain19
