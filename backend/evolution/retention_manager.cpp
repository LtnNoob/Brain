#include "retention_manager.hpp"
#include <algorithm>
#include <sstream>

namespace brain19 {

RetentionManager::RetentionManager(LongTermMemory& ltm,
                                   ComplexityAnalyzer& analyzer)
    : ltm_(ltm)
    , analyzer_(analyzer)
{
}

void RetentionManager::on_invalidation(ConceptId invalidated) {
    auto cinfo = ltm_.retrieve_concept(invalidated);
    if (!cinfo) return;
    if (!cinfo->epistemic.is_invalidated()) return;

    if (analyzer_.should_retain(invalidated)) {
        auto metrics = analyzer_.analyze(invalidated);
        std::string reason = "causal_chain:" + std::to_string(metrics.causal_chain_length) +
                             ",involved:" + std::to_string(metrics.involved_concepts) +
                             ",depth:" + std::to_string(metrics.relation_depth) +
                             ",inferences:" + std::to_string(metrics.inference_steps) +
                             ",score:" + std::to_string(metrics.normalized_score);
        ltm_.mark_as_anti_knowledge(invalidated, reason);
    }
}

RetentionStats RetentionManager::run_gc_cycle(size_t max_removals) {
    RetentionStats stats;

    auto invalidated = ltm_.get_concepts_by_status(EpistemicStatus::INVALIDATED);
    stats.total_invalidated = invalidated.size();

    // Re-evaluate all invalidated concepts (complexity may have changed)
    for (auto cid : invalidated) {
        auto cinfo = ltm_.retrieve_concept(cid);
        if (!cinfo) continue;

        if (!cinfo->is_anti_knowledge && analyzer_.should_retain(cid)) {
            auto metrics = analyzer_.analyze(cid);
            std::string reason = "gc_cycle:causal_chain:" +
                                 std::to_string(metrics.causal_chain_length) +
                                 ",score:" + std::to_string(metrics.normalized_score);
            ltm_.mark_as_anti_knowledge(cid, reason);
            stats.new_anti_knowledge.push_back(cid);
            ++stats.marked_anti_knowledge;
        }
    }

    // Count GC candidates
    auto gc_candidates = ltm_.get_gc_candidates();
    stats.gc_candidates = gc_candidates.size();

    // Perform GC
    stats.actually_removed = ltm_.garbage_collect(max_removals);

    return stats;
}

std::string RetentionManager::explain_anti_knowledge(ConceptId id) const {
    auto cinfo = ltm_.retrieve_concept(id);
    if (!cinfo || !cinfo->is_anti_knowledge) {
        return "Not anti-knowledge";
    }

    auto chain = analyzer_.extract_causal_chain(id);
    auto metrics = analyzer_.analyze(id);

    std::ostringstream oss;
    oss << "Anti-Knowledge: " << cinfo->label << "\n"
        << "  Complexity Score: " << metrics.normalized_score << "\n"
        << "  Causal Chain Length: " << metrics.causal_chain_length << "\n"
        << "  Involved Concepts: " << metrics.involved_concepts << "\n"
        << "  Relation Depth: " << metrics.relation_depth << "\n"
        << "  Inference Steps: " << metrics.inference_steps << "\n"
        << "  Causal Chain: ";

    for (size_t i = 0; i < chain.size(); ++i) {
        auto c = ltm_.retrieve_concept(chain[i]);
        if (c) oss << c->label;
        else oss << "#" << chain[i];
        if (i + 1 < chain.size()) oss << " -> ";
    }

    return oss.str();
}

bool RetentionManager::resembles_known_error(ConceptId candidate, float threshold) const {
    auto anti_knowledge = ltm_.get_anti_knowledge();
    if (anti_knowledge.empty()) return false;

    // Compare structural similarity: same relation types, overlapping targets
    auto candidate_out = ltm_.get_outgoing_relations(candidate);
    if (candidate_out.empty()) return false;

    // Build candidate's relation signature
    std::unordered_set<uint64_t> candidate_sig;
    for (const auto& rel : candidate_out) {
        // Encode (type, target) as signature
        uint64_t sig = (static_cast<uint64_t>(static_cast<uint16_t>(rel.type)) << 48)
                     | rel.target;
        candidate_sig.insert(sig);
    }

    for (auto ak_id : anti_knowledge) {
        auto ak_out = ltm_.get_outgoing_relations(ak_id);
        if (ak_out.empty()) continue;

        // Build anti-knowledge signature
        std::unordered_set<uint64_t> ak_sig;
        for (const auto& rel : ak_out) {
            uint64_t sig = (static_cast<uint64_t>(static_cast<uint16_t>(rel.type)) << 48)
                         | rel.target;
            ak_sig.insert(sig);
        }

        // Jaccard similarity of relation signatures
        size_t intersection = 0;
        for (auto s : candidate_sig) {
            if (ak_sig.count(s)) ++intersection;
        }
        size_t union_size = candidate_sig.size() + ak_sig.size() - intersection;
        if (union_size == 0) continue;

        float similarity = static_cast<float>(intersection) / static_cast<float>(union_size);
        if (similarity >= threshold) {
            return true;
        }
    }

    return false;
}

} // namespace brain19
