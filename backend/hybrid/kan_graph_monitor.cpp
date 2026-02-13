#include "kan_graph_monitor.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>

namespace brain19 {

KanGraphMonitor::KanGraphMonitor(
    const MicroModelRegistry& registry,
    const EmbeddingManager& embeddings,
    Config config)
    : registry_(registry)
    , embeddings_(embeddings)
    , config_(config)
{}

// =============================================================================
// PREDICT EDGE VIA MICROMODEL
// =============================================================================

double KanGraphMonitor::predict_edge(ConceptId from, RelationType type) const {
    const MicroModel* model = registry_.get_model(from);
    if (!model) return 0.0;

    const auto& rel_emb = embeddings_.get_relation_embedding(type);
    auto ctx_emb = embeddings_.make_context_embedding("query");

    return model->predict(rel_emb, ctx_emb);
}

// =============================================================================
// SCAN
// =============================================================================

std::vector<InvestigationRequest> KanGraphMonitor::scan(
    const std::vector<ConceptId>& focus_concepts,
    const LongTermMemory& ltm) const
{
    std::vector<InvestigationRequest> results;

    for (auto cid : focus_concepts) {
        detect_weak_edges(cid, ltm, results);
        detect_contradictions(cid, ltm, results);
        detect_stale_relations(cid, ltm, results);
    }

    // Missing links: check between focus concept pairs
    for (auto cid : focus_concepts) {
        detect_missing_links(cid, focus_concepts, ltm, results);
    }

    // Sort by anomaly_strength descending
    std::sort(results.begin(), results.end(),
        [](const InvestigationRequest& a, const InvestigationRequest& b) {
            return a.anomaly_strength > b.anomaly_strength;
        });

    // Cap at max_results
    if (results.size() > config_.max_results) {
        results.erase(results.begin() + config_.max_results, results.end());
    }

    return results;
}

// =============================================================================
// DETECT WEAK EDGES
// =============================================================================
// LTM weight <= threshold but KAN predicts >= threshold

void KanGraphMonitor::detect_weak_edges(
    ConceptId cid, const LongTermMemory& ltm,
    std::vector<InvestigationRequest>& results) const
{
    auto rels = ltm.get_outgoing_relations(cid);
    for (const auto& r : rels) {
        if (r.weight > config_.weak_edge_ltm_max) continue;

        double kan = predict_edge(r.source, r.type);
        if (kan < config_.weak_edge_kan_min) continue;

        double strength = kan - r.weight;

        auto src_info = ltm.retrieve_concept(r.source);
        auto tgt_info = ltm.retrieve_concept(r.target);

        std::ostringstream desc;
        desc << "Weak edge: "
             << (src_info ? src_info->label : "?") << " --"
             << relation_type_to_string(r.type) << "--> "
             << (tgt_info ? tgt_info->label : "?")
             << " LTM=" << std::fixed;
        desc.precision(2);
        desc << r.weight << " KAN=" << kan;

        results.emplace_back(
            ++request_counter_,
            AnomalyType::WEAK_EDGE,
            r.source, r.target, r.type,
            r.weight, kan, strength,
            desc.str()
        );
    }
}

// =============================================================================
// DETECT CONTRADICTIONS
// =============================================================================
// |LTM - KAN| >= threshold (but not already caught by weak/stale)

void KanGraphMonitor::detect_contradictions(
    ConceptId cid, const LongTermMemory& ltm,
    std::vector<InvestigationRequest>& results) const
{
    auto rels = ltm.get_outgoing_relations(cid);
    for (const auto& r : rels) {
        double kan = predict_edge(r.source, r.type);
        double mismatch = std::abs(r.weight - kan);

        if (mismatch < config_.contradiction_mismatch_min) continue;

        // Skip if already categorized as weak or stale
        bool is_weak = (r.weight <= config_.weak_edge_ltm_max && kan >= config_.weak_edge_kan_min);
        bool is_stale = (r.weight >= config_.stale_ltm_min && kan <= config_.stale_kan_max);
        if (is_weak || is_stale) continue;

        auto src_info = ltm.retrieve_concept(r.source);
        auto tgt_info = ltm.retrieve_concept(r.target);

        std::ostringstream desc;
        desc << "Contradiction: "
             << (src_info ? src_info->label : "?") << " --"
             << relation_type_to_string(r.type) << "--> "
             << (tgt_info ? tgt_info->label : "?")
             << " LTM=" << std::fixed;
        desc.precision(2);
        desc << r.weight << " KAN=" << kan
             << " (mismatch=" << mismatch << ")";

        results.emplace_back(
            ++request_counter_,
            AnomalyType::CONTRADICTION,
            r.source, r.target, r.type,
            r.weight, kan, mismatch,
            desc.str()
        );
    }
}

// =============================================================================
// DETECT STALE RELATIONS
// =============================================================================
// LTM weight >= threshold but KAN predicts <= threshold

void KanGraphMonitor::detect_stale_relations(
    ConceptId cid, const LongTermMemory& ltm,
    std::vector<InvestigationRequest>& results) const
{
    auto rels = ltm.get_outgoing_relations(cid);
    for (const auto& r : rels) {
        if (r.weight < config_.stale_ltm_min) continue;

        double kan = predict_edge(r.source, r.type);
        if (kan > config_.stale_kan_max) continue;

        double strength = r.weight - kan;

        auto src_info = ltm.retrieve_concept(r.source);
        auto tgt_info = ltm.retrieve_concept(r.target);

        std::ostringstream desc;
        desc << "Stale: "
             << (src_info ? src_info->label : "?") << " --"
             << relation_type_to_string(r.type) << "--> "
             << (tgt_info ? tgt_info->label : "?")
             << " LTM=" << std::fixed;
        desc.precision(2);
        desc << r.weight << " KAN=" << kan;

        results.emplace_back(
            ++request_counter_,
            AnomalyType::STALE_RELATION,
            r.source, r.target, r.type,
            r.weight, kan, strength,
            desc.str()
        );
    }
}

// =============================================================================
// DETECT MISSING LINKS
// =============================================================================
// No LTM relation, but KAN predicts >= threshold

void KanGraphMonitor::detect_missing_links(
    ConceptId cid, const std::vector<ConceptId>& focus_concepts,
    const LongTermMemory& ltm,
    std::vector<InvestigationRequest>& results) const
{
    // Build set of existing targets for this concept
    auto existing_rels = ltm.get_outgoing_relations(cid);
    std::unordered_set<ConceptId> existing_targets;
    for (const auto& r : existing_rels) {
        existing_targets.insert(r.target);
    }

    size_t checks = 0;
    RelationType try_types[] = {
        RelationType::CAUSES, RelationType::ENABLES,
        RelationType::SUPPORTS, RelationType::IS_A
    };

    for (auto other_cid : focus_concepts) {
        if (other_cid == cid) continue;
        if (existing_targets.count(other_cid)) continue;
        if (checks >= config_.max_missing_link_checks) break;

        for (auto rtype : try_types) {
            ++checks;
            if (checks > config_.max_missing_link_checks) break;

            double kan = predict_edge(cid, rtype);
            if (kan < config_.missing_link_kan_min) continue;

            auto src_info = ltm.retrieve_concept(cid);
            auto tgt_info = ltm.retrieve_concept(other_cid);

            std::ostringstream desc;
            desc << "Missing link: "
                 << (src_info ? src_info->label : "?") << " --"
                 << relation_type_to_string(rtype) << "--> "
                 << (tgt_info ? tgt_info->label : "?")
                 << " KAN=" << std::fixed;
            desc.precision(2);
            desc << kan << " (no LTM relation)";

            results.emplace_back(
                ++request_counter_,
                AnomalyType::MISSING_LINK,
                cid, other_cid, rtype,
                0.0, kan, kan,
                desc.str()
            );
            break;  // One per concept pair
        }
    }
}

} // namespace brain19
