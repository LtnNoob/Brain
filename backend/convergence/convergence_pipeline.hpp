#pragma once

#include "convergence_config.hpp"
#include "convergence_kan.hpp"
#include "concept_router.hpp"
#include "concept_bank.hpp"
#include "gated_residual.hpp"
#include <vector>

namespace brain19 {
namespace convergence {

// =============================================================================
// CONVERGENCE PIPELINE — Deep KAN ↔ ConceptModel Integration
// =============================================================================
//
// End-to-end pipeline implementing v2 architecture:
//
//   h ∈ ℝ⁹⁰ (input)
//   ├── KAN Layer 1 → k1 ∈ ℝ²⁵⁶
//   ├── Router(h) → Top-K concept IDs + weights       (parallel with L1)
//   ├── KAN Projection(k1) → k1_proj ∈ ℝ³²
//   ├── ConceptBank(h ⊕ k1_proj, Top-K) → L(h) ∈ ℝ³²
//   ├── KAN Layer 2(k1 ⊕ L(h)) → k2 ∈ ℝ¹²⁸          (CM-Feedback-Port)
//   ├── KAN Layer 3(k2) → G(h) ∈ ℝ³²
//   └── Gated Residual PoE(h, G(h), L(h)) → fused ∈ ℝ³²
//

struct PipelineOutput {
    std::vector<double> fused;           // Final fused output [OUTPUT_DIM]
    std::vector<double> G_out;           // Global KAN output [OUTPUT_DIM]
    std::vector<double> L_out;           // Local CM output [OUTPUT_DIM]
    float agreement;                     // Agreement between G and L
    IgnitionMode ignition;               // FAST / DELIBERATE / CONFLICT
    std::vector<RouteResult> routes;     // Routing decisions
};

class ConvergencePipeline {
public:
    ConvergencePipeline();

    // ─── Inference ───────────────────────────────────────────────────────

    // Full forward pass
    PipelineOutput forward(const std::vector<double>& h);

    // ─── Training ────────────────────────────────────────────────────────

    // Training step with target output
    // Returns loss (MSE)
    double train_step(const std::vector<double>& h,
                      const std::vector<double>& target);

    // Training with per-component learning rates
    struct TrainingConfig {
        double lr_kan_l1   = DEFAULT_LR_KAN_L1;
        double lr_kan_proj = DEFAULT_LR_KAN_PROJ;
        double lr_kan_l2l3 = DEFAULT_LR_KAN_L2L3;
        double lr_cm       = DEFAULT_LR_CM;
        double lr_router   = DEFAULT_LR_ROUTER;
        double lr_gate     = DEFAULT_LR_GATE;
    };

    double train_step(const std::vector<double>& h,
                      const std::vector<double>& target,
                      const TrainingConfig& config);

    // ─── Component Access ────────────────────────────────────────────────

    ConvergenceKAN& kan() { return kan_; }
    CentroidRouter& router() { return router_; }
    ConceptBank& concept_bank() { return concept_bank_; }
    GatedResidualPoE& gate() { return gate_; }

    const ConvergenceKAN& kan() const { return kan_; }
    const CentroidRouter& router() const { return router_; }
    const ConceptBank& concept_bank() const { return concept_bank_; }
    const GatedResidualPoE& gate() const { return gate_; }

    size_t total_params() const;

private:
    ConvergenceKAN kan_;
    CentroidRouter router_;
    ConceptBank concept_bank_;
    GatedResidualPoE gate_;

    // Cached forward state for backward
    struct ForwardCache {
        std::vector<double> h;
        std::vector<double> k1;
        std::vector<double> k1_proj;
        std::vector<double> cm_input;
        std::vector<double> L_out;
        std::vector<double> G_out;
        std::vector<double> gate_values;
        std::vector<ConceptId> concept_ids;
        std::vector<double> concept_weights;
    };
    ForwardCache cache_;
};

} // namespace convergence
} // namespace brain19
