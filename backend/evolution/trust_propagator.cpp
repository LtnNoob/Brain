#include "trust_propagator.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

namespace brain19 {

TrustPropagator::TrustPropagator(LongTermMemory& ltm,
                                  EpistemicPromotion& promotion,
                                  GraphDensifier& densifier,
                                  PropagationConfig config)
    : ltm_(ltm)
    , promotion_(promotion)
    , densifier_(densifier)
    , config_(config)
{
}

PropagationResult TrustPropagator::propagate(ConceptId invalidated) {
    PropagationResult result;
    result.source = invalidated;

    // Depth guard: prevent infinite cascades
    if (current_depth_ >= config_.max_recursion_depth) {
        return result;
    }
    ++current_depth_;

    // Find candidate concepts within hop distance
    auto candidates = find_candidates(invalidated);
    result.concepts_checked = candidates.size();

    for (auto cid : candidates) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (!cinfo) continue;

        // Skip already-invalidated concepts
        if (cinfo->epistemic.is_invalidated()) continue;

        // Skip linguistic layer concepts unless configured
        // (Check if concept has linguistic properties via outgoing DENOTES)
        if (!config_.propagate_to_linguistic) {
            bool is_linguistic = false;
            for (const auto& rel : ltm_.get_outgoing_relations(cid)) {
                if (rel.type == RelationType::DENOTES ||
                    rel.type == RelationType::SUBJECT_OF ||
                    rel.type == RelationType::VERB_OF ||
                    rel.type == RelationType::OBJECT_OF ||
                    rel.type == RelationType::PART_OF_SENTENCE) {
                    is_linguistic = true;
                    break;
                }
            }
            if (is_linguistic) continue;
        }

        float similarity = combined_similarity(invalidated, cid);
        if (similarity < config_.similarity_threshold) continue;

        float reduction = compute_reduction(cid, similarity);
        if (reduction <= 0.0f) continue;

        double new_trust = std::max(0.0, cinfo->epistemic.trust - reduction);

        // Apply trust reduction
        EpistemicMetadata adjusted(
            cinfo->epistemic.type,
            cinfo->epistemic.status,
            new_trust
        );
        ltm_.update_epistemic_metadata(cid, adjusted);

        result.affected.emplace_back(cid, new_trust);
        ++result.concepts_adjusted;

        // Track history
        propagation_history_[cid].emplace_back(invalidated, reduction);

        // Force-invalidate if trust drops below threshold
        if (new_trust < config_.cumulative_invalidation_threshold) {
            ltm_.invalidate_concept(cid, new_trust);
            result.force_invalidated.push_back(cid);

            // Recursive propagation (depth-limited)
            auto sub_result = propagate(cid);
            for (auto& aff : sub_result.affected) {
                result.affected.push_back(aff);
            }
            for (auto fi : sub_result.force_invalidated) {
                result.force_invalidated.push_back(fi);
            }
            result.concepts_checked += sub_result.concepts_checked;
            result.concepts_adjusted += sub_result.concepts_adjusted;
        }
    }

    --current_depth_;
    return result;
}

float TrustPropagator::combined_similarity(ConceptId a, ConceptId b) const {
    float structural = structural_similarity(a, b);
    float coact = co_activation_score(a, b);
    float shared = shared_source_score(a, b);

    return 0.40f * structural + 0.35f * coact + 0.25f * shared;
}

float TrustPropagator::compute_reduction(ConceptId target, float similarity_to_invalidated) const {
    auto cinfo = ltm_.retrieve_concept(target);
    if (!cinfo) return 0.0f;

    // support ratio: fraction of supporting relations
    auto incoming = ltm_.get_incoming_relations(target);
    size_t support_count = 0;
    size_t total_epistemic = 0;
    size_t contradict_count = 0;

    for (const auto& rel : incoming) {
        if (rel.type == RelationType::SUPPORTS) {
            auto src = ltm_.retrieve_concept(rel.source);
            if (src && src->epistemic.is_active()) {
                ++support_count;
                ++total_epistemic;
            }
        }
        if (rel.type == RelationType::CONTRADICTS) {
            auto src = ltm_.retrieve_concept(rel.source);
            if (src && src->epistemic.is_active()) {
                ++contradict_count;
                ++total_epistemic;
            }
        }
    }

    float support_ratio = total_epistemic > 0
        ? static_cast<float>(support_count) / static_cast<float>(total_epistemic)
        : 0.0f;
    float contradiction_ratio = total_epistemic > 0
        ? static_cast<float>(contradict_count) / static_cast<float>(total_epistemic)
        : 0.0f;

    // Formula from design doc:
    // reduction = trust * similarity * (1 - support) * (1 + contradiction)
    float reduction = static_cast<float>(cinfo->epistemic.trust)
                    * similarity_to_invalidated
                    * (1.0f - support_ratio)
                    * (1.0f + contradiction_ratio);

    return std::min(reduction, config_.max_trust_reduction);
}

std::vector<ConceptId> TrustPropagator::get_propagation_sources(ConceptId target) const {
    std::vector<ConceptId> sources;
    auto it = propagation_history_.find(target);
    if (it != propagation_history_.end()) {
        for (const auto& [src, _reduction] : it->second) {
            sources.push_back(src);
        }
    }
    return sources;
}

std::vector<ConceptId> TrustPropagator::find_candidates(ConceptId source) const {
    std::vector<ConceptId> candidates;
    std::unordered_set<ConceptId> visited;
    std::queue<std::pair<ConceptId, size_t>> bfs;

    bfs.push({source, 0});
    visited.insert(source);

    while (!bfs.empty()) {
        auto [cur, depth] = bfs.front();
        bfs.pop();

        if (depth > 0) {
            candidates.push_back(cur);
        }

        if (depth >= config_.max_hop_distance) continue;

        // Traverse outgoing relations
        for (const auto& rel : ltm_.get_outgoing_relations(cur)) {
            if (visited.find(rel.target) == visited.end()) {
                visited.insert(rel.target);
                bfs.push({rel.target, depth + 1});
            }
        }

        // Traverse incoming relations
        for (const auto& rel : ltm_.get_incoming_relations(cur)) {
            if (visited.find(rel.source) == visited.end()) {
                visited.insert(rel.source);
                bfs.push({rel.source, depth + 1});
            }
        }
    }

    return candidates;
}

float TrustPropagator::co_activation_score(ConceptId a, ConceptId b) const {
    // Shared neighbor ratio (Jaccard-like)
    std::unordered_set<ConceptId> neighbors_a;
    for (const auto& rel : ltm_.get_outgoing_relations(a)) {
        neighbors_a.insert(rel.target);
    }
    for (const auto& rel : ltm_.get_incoming_relations(a)) {
        neighbors_a.insert(rel.source);
    }

    std::unordered_set<ConceptId> neighbors_b;
    for (const auto& rel : ltm_.get_outgoing_relations(b)) {
        neighbors_b.insert(rel.target);
    }
    for (const auto& rel : ltm_.get_incoming_relations(b)) {
        neighbors_b.insert(rel.source);
    }

    if (neighbors_a.empty() && neighbors_b.empty()) return 0.0f;

    size_t intersection = 0;
    for (auto n : neighbors_a) {
        if (neighbors_b.count(n)) ++intersection;
    }

    size_t union_size = neighbors_a.size() + neighbors_b.size() - intersection;
    if (union_size == 0) return 0.0f;

    return static_cast<float>(intersection) / static_cast<float>(union_size);
}

float TrustPropagator::shared_source_score(ConceptId a, ConceptId b) const {
    // Fraction of shared incoming relation sources
    std::unordered_set<ConceptId> sources_a;
    for (const auto& rel : ltm_.get_incoming_relations(a)) {
        sources_a.insert(rel.source);
    }

    std::unordered_set<ConceptId> sources_b;
    for (const auto& rel : ltm_.get_incoming_relations(b)) {
        sources_b.insert(rel.source);
    }

    if (sources_a.empty() && sources_b.empty()) return 0.0f;

    size_t intersection = 0;
    for (auto s : sources_a) {
        if (sources_b.count(s)) ++intersection;
    }

    size_t union_size = sources_a.size() + sources_b.size() - intersection;
    if (union_size == 0) return 0.0f;

    return static_cast<float>(intersection) / static_cast<float>(union_size);
}

float TrustPropagator::structural_similarity(ConceptId a, ConceptId b) const {
    // Compare relation type patterns (outgoing)
    std::unordered_map<uint16_t, size_t> types_a, types_b;

    for (const auto& rel : ltm_.get_outgoing_relations(a)) {
        types_a[static_cast<uint16_t>(rel.type)]++;
    }
    for (const auto& rel : ltm_.get_outgoing_relations(b)) {
        types_b[static_cast<uint16_t>(rel.type)]++;
    }

    if (types_a.empty() && types_b.empty()) return 0.0f;

    // Cosine similarity of type frequency vectors
    double dot = 0.0, mag_a = 0.0, mag_b = 0.0;

    std::unordered_set<uint16_t> all_types;
    for (const auto& [t, _c] : types_a) all_types.insert(t);
    for (const auto& [t, _c] : types_b) all_types.insert(t);

    for (auto t : all_types) {
        double va = types_a.count(t) ? static_cast<double>(types_a[t]) : 0.0;
        double vb = types_b.count(t) ? static_cast<double>(types_b[t]) : 0.0;
        dot += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
    }

    if (mag_a < 1e-9 || mag_b < 1e-9) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(mag_a) * std::sqrt(mag_b)));
}

} // namespace brain19
