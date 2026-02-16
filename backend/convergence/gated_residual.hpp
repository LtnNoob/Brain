#pragma once

#include "convergence_config.hpp"
#include <vector>

namespace brain19 {
namespace convergence {

// =============================================================================
// GATED RESIDUAL PRODUCT OF EXPERTS + IGNITION
// =============================================================================
//
// Convergence mechanism:
//   ε = L(h) - G(h)                    prediction error
//   γ = σ(W_gate · h + b_gate)         per-dimension gate ∈ (0,1)
//   fused = G(h) + γ ⊙ ε              gated correction
//
// Ignition (adaptive iterations):
//   agreement = 1 - ||ε|| / (||G(h)|| + ||L(h)||)
//   > 0.85 → FAST (use G(h) directly, skip gate)
//   > 0.40 → DELIBERATE (1 iteration with gate)
//   else   → CONFLICT (expand neighborhood, iterate)
//

enum class IgnitionMode {
    FAST,        // Systems agree — use global prediction directly
    DELIBERATE,  // Moderate disagreement — apply gated correction once
    CONFLICT     // Strong disagreement — iterate with expanded neighborhood
};

struct ConvergenceResult {
    std::vector<double> fused;       // Final output [OUTPUT_DIM]
    std::vector<double> gate_values; // γ values [OUTPUT_DIM]
    float agreement;                 // Agreement score [0..1]
    IgnitionMode mode;               // Ignition decision
};

class GatedResidualPoE {
public:
    GatedResidualPoE();

    // Main convergence: combine global KAN output with local CM output
    ConvergenceResult converge(const std::vector<double>& h,
                               const std::vector<double>& G_out,
                               const std::vector<double>& L_out) const;

    // Compute agreement score between G and L
    static float compute_agreement(const std::vector<double>& G_out,
                                   const std::vector<double>& L_out);

    // Determine ignition mode from agreement
    static IgnitionMode check_ignition(float agreement);

    // ─── Backward ────────────────────────────────────────────────────────

    struct BackwardResult {
        std::vector<double> d_h;      // Gradient w.r.t. input h
        std::vector<double> d_G;      // Gradient w.r.t. G(h)
        std::vector<double> d_L;      // Gradient w.r.t. L(h)
    };

    // Backward through gate: given d_fused, compute gradients
    BackwardResult backward(const std::vector<double>& h,
                            const std::vector<double>& G_out,
                            const std::vector<double>& L_out,
                            const std::vector<double>& gate_values,
                            const std::vector<double>& d_fused,
                            double lr);

    // ─── Accessors ───────────────────────────────────────────────────────

    std::vector<double>& gate_W() { return W_gate_; }
    std::vector<double>& gate_b() { return b_gate_; }
    const std::vector<double>& gate_W() const { return W_gate_; }
    const std::vector<double>& gate_b() const { return b_gate_; }

    size_t num_params() const { return W_gate_.size() + b_gate_.size(); }

    // Initialize gate bias from known precision ratio
    void init_bias_from_precision(double local_precision, double global_precision);

private:
    std::vector<double> W_gate_;  // [OUTPUT_DIM × QUERY_DIM]
    std::vector<double> b_gate_;  // [OUTPUT_DIM]

    static double sigmoid(double x);
    static double vector_norm(const std::vector<double>& v);
};

} // namespace convergence
} // namespace brain19
