#pragma once

#include "convergence_config.hpp"
#include "../language/deep_kan.hpp"
#include <vector>

namespace brain19 {
namespace convergence {

// =============================================================================
// CONVERGENCE KAN — 3-Layer KAN with CM Feedback Port
// =============================================================================
//
// Deep KAN ↔ ConceptModel Integration (v2 Architecture)
//
// Layer 1:  h ∈ ℝ⁹⁰  →  k1 ∈ ℝ²⁵⁶     (EfficientKANLayer)
// Projection: k1 → k1_proj ∈ ℝ³²         (Linear, shared for CM input)
// Layer 2:  (k1 ⊕ cm) ∈ ℝ²⁸⁸  →  k2 ∈ ℝ¹²⁸  (EfficientKANLayer, CM-Feedback-Port)
// Layer 3:  k2 ∈ ℝ¹²⁸ → G(h) ∈ ℝ³²      (EfficientKANLayer)
//

class ConvergenceKAN {
public:
    ConvergenceKAN();

    // ─── Forward pass (split into stages for deep integration) ───────────

    // Stage 1: h → k1 (can run in parallel with Router)
    std::vector<double> forward_layer1(const std::vector<double>& h);

    // Stage 1.5: k1 → k1_proj (shared linear projection for CM input)
    std::vector<double> project_for_cm(const std::vector<double>& k1) const;

    // Stage 2+3: (k1, cm_output) → G(h)
    // k1 from Layer 1, cm_output from ConceptModels
    std::vector<double> forward_layer2_3(const std::vector<double>& k1,
                                          const std::vector<double>& cm_output);

    // ─── Backward pass ───────────────────────────────────────────────────

    // Backward through Layer 3 + Layer 2, returns d_k1 and d_cm
    struct BackwardResult {
        std::vector<double> d_k1;
        std::vector<double> d_cm;
    };
    BackwardResult backward_layer2_3(const std::vector<double>& d_output,
                                      double lr_l2, double lr_l3);

    // Backward through Layer 1, returns d_h (usually discarded)
    std::vector<double> backward_layer1(const std::vector<double>& d_k1, double lr_l1);

    // Backward through projection, returns d_k1_proj_contrib
    void backward_projection(const std::vector<double>& d_cm_input_proj, double lr_proj);

    // ─── Accessors ───────────────────────────────────────────────────────

    size_t num_params() const;
    EfficientKANLayer& layer1() { return layer1_; }
    EfficientKANLayer& layer2() { return layer2_; }
    EfficientKANLayer& layer3() { return layer3_; }

    // Projection weights (W_proj: [KAN_PROJ_OUT × KAN_L1_OUT] + b_proj: [KAN_PROJ_OUT])
    std::vector<double>& proj_W() { return proj_W_; }
    std::vector<double>& proj_b() { return proj_b_; }
    const std::vector<double>& proj_W() const { return proj_W_; }
    const std::vector<double>& proj_b() const { return proj_b_; }

private:
    EfficientKANLayer layer1_;   // 90→256
    EfficientKANLayer layer2_;   // 288→128 (CM-Feedback-Port)
    EfficientKANLayer layer3_;   // 128→32

    // Shared linear projection: 256→32
    std::vector<double> proj_W_;  // [KAN_PROJ_OUT × KAN_L1_OUT]
    std::vector<double> proj_b_;  // [KAN_PROJ_OUT]

    // Forward caches
    EfficientKANLayerCache cache1_;
    EfficientKANLayerCache cache2_;
    EfficientKANLayerCache cache3_;

    // Cached intermediates for backward
    std::vector<double> cached_k1_;
    std::vector<double> cached_l2_input_;
};

} // namespace convergence
} // namespace brain19
