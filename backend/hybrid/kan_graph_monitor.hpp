#pragma once

#include "investigation_request.hpp"
#include "../cmodel/concept_model_registry.hpp"
#include "../micromodel/embedding_manager.hpp"
#include "../ltm/long_term_memory.hpp"
#include <vector>
#include <cstdint>

namespace brain19 {

// =============================================================================
// KAN GRAPH MONITOR (Topology A — Source)
// =============================================================================
//
// Scans the knowledge graph using MicroModel predictions to detect
// anomalies: discrepancies between LTM weights and KAN predictions.
//
// ARCHITECTURE CONTRACT:
// - READ-ONLY access to Registry, Embeddings, LTM
// - Output = InvestigationRequest signals (not knowledge)
// - Stateless scan — no learning, no persistence
// - "Tool" in Brain19 philosophy
//
class KanGraphMonitor {
public:
    struct Config {
        double weak_edge_kan_min = 0.6;        // KAN must predict >= this
        double weak_edge_ltm_max = 0.3;        // LTM weight must be <= this
        double contradiction_mismatch_min = 0.4; // |LTM - KAN| >= this
        double missing_link_kan_min = 0.5;      // KAN prediction for missing links
        double stale_ltm_min = 0.7;            // LTM weight >= this for stale
        double stale_kan_max = 0.3;            // KAN must predict <= this for stale
        size_t max_results = 10;               // Cap output size
        size_t max_missing_link_checks = 50;   // Bound missing link search cost

        Config() = default;
    };

    KanGraphMonitor(const ConceptModelRegistry& registry,
                    const EmbeddingManager& embeddings)
        : KanGraphMonitor(registry, embeddings, Config{}) {}

    KanGraphMonitor(const ConceptModelRegistry& registry,
                    const EmbeddingManager& embeddings,
                    Config config);

    // Scan focus concepts for anomalies. Returns sorted by strength desc.
    std::vector<InvestigationRequest> scan(
        const std::vector<ConceptId>& focus_concepts,
        const LongTermMemory& ltm) const;

    const Config& get_config() const { return config_; }

private:
    const ConceptModelRegistry& registry_;
    const EmbeddingManager& embeddings_;
    Config config_;
    mutable uint64_t request_counter_ = 0;

    // Core helper: predict edge strength via MicroModel (target-aware)
    double predict_edge(ConceptId from, ConceptId to, RelationType type) const;

    // Detection routines (append to results vector)
    void detect_weak_edges(
        ConceptId cid, const LongTermMemory& ltm,
        std::vector<InvestigationRequest>& results) const;

    void detect_contradictions(
        ConceptId cid, const LongTermMemory& ltm,
        std::vector<InvestigationRequest>& results) const;

    void detect_stale_relations(
        ConceptId cid, const LongTermMemory& ltm,
        std::vector<InvestigationRequest>& results) const;

    void detect_missing_links(
        ConceptId cid, const std::vector<ConceptId>& focus_concepts,
        const LongTermMemory& ltm,
        std::vector<InvestigationRequest>& results) const;
};

} // namespace brain19
