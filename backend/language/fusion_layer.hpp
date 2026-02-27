#pragma once

#include "language_config.hpp"
#include "semantic_scorer.hpp"
#include "../common/types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace brain19 {

// =============================================================================
// FUSED REPRESENTATION — output of FusionLayer
// =============================================================================

struct FusedRepresentation {
    // Ordered concept chain (causal order, then by score)
    std::vector<ConceptId> ordered_concepts;

    // Per-concept gate scores
    std::unordered_map<ConceptId, double> gate_scores;

    // Fused vector ∈ R^64 (input to decoder)
    std::vector<double> fused_vector;

    // 16D FlexDetail vector (injected between fused and dim_context for v11)
    std::vector<double> flex_detail;

    // Dimensional context: variable-length, set by DimensionalContext.
    // Empty if no dimensional context is available (backward compatible).
    std::vector<double> dimensional_context;

    // 32D convergence state from reasoning chain (ConvergencePort composition).
    // Filled by ConceptReasoner's chain_state when available.
    std::vector<double> convergence_state;

    // Extended fused: [fused_vector | flex_detail | dimensional_context | convergence_state]
    // Size = FUSED_DIM + flex_detail.size() + dim_context_size + CONVERGENCE_DIM
    std::vector<double> extended_fused_vector() const;

    // Best template type index
    size_t template_type = 1;  // default: DEFINITIONAL
};

// =============================================================================
// FUSION LAYER — Gated merge of KAN activations + semantic scores
// =============================================================================
//
// For each concept i:
//   gate_i = σ(W_gate · [‖a*_i‖; rel_i; causal_i] + b_gate)
//   score_i = gate_i × ‖a*_i‖
//
// Fused vector: concat(top-3 a*_i weighted, gate_scores, template_one_hot)
// Projected to R^64
//
// Total: 3,652 parameters
//

class FusionLayer {
public:
    explicit FusionLayer(const LanguageConfig& config = LanguageConfig{});

    // Fuse activations with semantic scores
    FusedRepresentation fuse(
        const std::unordered_map<ConceptId, std::vector<double>>& activations,
        const SemanticScores& scores,
        const std::vector<ConceptId>& causal_chain
    ) const;

    // Access gate weights for training
    std::vector<double>& gate_weights() { return gate_w_; }
    double& gate_bias() { return gate_b_; }

    // Access projection matrix for training
    std::vector<std::vector<double>>& projection() { return projection_; }

private:
    LanguageConfig config_;

    // Gate: w ∈ R^3 + b ∈ R (4 params)
    std::vector<double> gate_w_;
    double gate_b_ = 0.0;

    // Projection: R^57 → R^64 (3×16 activations + 5 gates + 4 template = 57 → 64)
    // 57 × 64 = 3,648 params
    std::vector<std::vector<double>> projection_;

    // Compute gate value for one concept
    double compute_gate(double activation_norm, double relevance, double causality) const;

    // Build raw fused vector before projection (R^57)
    std::vector<double> build_raw_fused(
        const std::vector<std::pair<ConceptId, double>>& top_concepts,
        const std::unordered_map<ConceptId, std::vector<double>>& activations,
        const std::unordered_map<ConceptId, double>& gates,
        const std::vector<double>& template_probs
    ) const;

    static double sigmoid(double x);
    static double vec_norm(const std::vector<double>& v);
};

} // namespace brain19
