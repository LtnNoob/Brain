#pragma once

#include "convergence_config.hpp"
#include "../common/types.hpp"
#include <vector>
#include <unordered_map>
#include <memory>

namespace brain19 {
namespace convergence {

// =============================================================================
// CONCEPT BANK — Per-concept local experts for convergence pipeline
// =============================================================================
//
// Wraps per-concept models that produce CM_OUTPUT_DIM outputs from
// CM_INPUT_DIM inputs (h ⊕ k1_proj = 90 + 32 = 122 dims).
//
// This is the convergence pipeline's interface to concept-level processing.
// Each concept has a small bilinear model that maps 122→32.
//
// In Phase 2, this will be replaced by the full ConceptModel redesign.
// For now, each concept has a lightweight linear+bias: W[32×122] + b[32].
//

class ConceptBank {
public:
    ConceptBank() = default;

    // Forward: compute weighted sum of activated concept outputs
    // cm_input: h ⊕ k1_proj [CM_INPUT_DIM = 122]
    // concept_ids: top-K concept IDs from router
    // weights: routing weights (softmax-normalized)
    // Returns: aggregated L(h) ∈ ℝ^CM_OUTPUT_DIM [32]
    std::vector<double> forward(const std::vector<double>& cm_input,
                                 const std::vector<ConceptId>& concept_ids,
                                 const std::vector<double>& weights);

    // Forward single concept (for backward pass)
    std::vector<double> forward_single(const std::vector<double>& cm_input,
                                        ConceptId concept_id);

    // Backward: given d_L_out, compute gradients and update per-concept weights
    // Returns d_cm_input for further backprop
    std::vector<double> backward(const std::vector<double>& cm_input,
                                  const std::vector<ConceptId>& concept_ids,
                                  const std::vector<double>& weights,
                                  const std::vector<double>& d_L_out,
                                  double lr);

    // Ensure concept has a model (creates if missing)
    void ensure_concept(ConceptId id);

    // Bulk creation
    void ensure_concepts(const std::vector<ConceptId>& ids);

    bool has_concept(ConceptId id) const;
    size_t num_concepts() const { return models_.size(); }
    size_t params_per_concept() const { return CM_INPUT_DIM * CM_OUTPUT_DIM + CM_OUTPUT_DIM; }

private:
    struct ConceptExpert {
        std::vector<double> W;  // [CM_OUTPUT_DIM × CM_INPUT_DIM]
        std::vector<double> b;  // [CM_OUTPUT_DIM]

        ConceptExpert();

        // Forward: out = tanh(W · input + b)
        std::vector<double> forward(const std::vector<double>& input) const;
    };

    std::unordered_map<ConceptId, ConceptExpert> models_;
};

} // namespace convergence
} // namespace brain19
