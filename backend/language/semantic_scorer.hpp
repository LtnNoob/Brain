#pragma once

#include "language_config.hpp"
#include "../kan/kan_module.hpp"
#include "../common/types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// SEMANTIC SCORES — output of SemanticScorer
// =============================================================================

struct SemanticScores {
    // Per-concept relevance to query [0,1]
    std::unordered_map<ConceptId, double> relevance;

    // Per-concept-pair causal strength [0,1]
    // Key: "source_id:target_id"
    std::unordered_map<std::string, double> causality;

    // Template type probabilities (softmax over 4 types)
    // [KAUSAL_ERKLAEREND, DEFINITIONAL, AUFZAEHLEND, VERGLEICHEND]
    std::vector<double> template_probs;

    // Best template type index
    size_t best_template() const;
};

// =============================================================================
// SEMANTIC SCORER — 3 KAN-based scoring modules
// =============================================================================
//
// Scorer 1 (RelevanceScorer): concat(a_i, query) ∈ R^32 → KAN(32→16→1)
// Scorer 2 (CausalityScorer): concat(a_i, a_j, e_rel) ∈ R^48 → KAN(48→16→1)
// Scorer 3 (TemplateClassifier): bag(a_active) ∈ R^16 → KAN(16→8→4)
//
// Total: 14,720 parameters
//

class SemanticScorer {
public:
    explicit SemanticScorer(const LanguageConfig& config = LanguageConfig{});

    // Score all active concepts for relevance to query
    SemanticScores score(
        const std::unordered_map<ConceptId, std::vector<double>>& activations,
        const std::vector<double>& query,
        const std::vector<std::pair<ConceptId, ConceptId>>& causal_pairs,
        const std::unordered_map<std::string, std::vector<double>>& relation_embeddings
    ) const;

    // Individual scorers (exposed for training)
    double score_relevance(const std::vector<double>& activation,
                           const std::vector<double>& query) const;

    double score_causality(const std::vector<double>& act_source,
                           const std::vector<double>& act_target,
                           const std::vector<double>& rel_embedding) const;

    std::vector<double> classify_template(
        const std::vector<double>& aggregated_activation) const;

    // Access KAN modules for training
    KANModule& relevance_kan() { return relevance_kan_; }
    KANModule& causality_kan() { return causality_kan_; }
    KANModule& template_kan() { return template_kan_; }

private:
    LanguageConfig config_;

    // Scorer 1: Relevance — (16+16)=32 → 16 → 1
    KANModule relevance_kan_;

    // Scorer 2: Causality — (16+16+16)=48 → 16 → 1
    KANModule causality_kan_;

    // Scorer 3: Template classifier — 16 → 8 → 4
    KANModule template_kan_;

    // Sigmoid activation
    static double sigmoid(double x);

    // Softmax
    static std::vector<double> softmax(const std::vector<double>& logits);
};

} // namespace brain19
