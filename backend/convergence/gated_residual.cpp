#include "gated_residual.hpp"
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>

namespace brain19 {
namespace convergence {

GatedResidualPoE::GatedResidualPoE()
    : W_gate_(OUTPUT_DIM * QUERY_DIM, 0.0)
    , b_gate_(OUTPUT_DIM, 0.0)
{
    // Xavier init for gate weights
    std::mt19937 rng(123);
    double limit = std::sqrt(6.0 / (QUERY_DIM + OUTPUT_DIM));
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (auto& w : W_gate_) {
        w = dist(rng);
    }
    // Bias = 0 → gate starts at 0.5 (equal trust in both systems)
}

ConvergenceResult GatedResidualPoE::converge(
    const std::vector<double>& h,
    const std::vector<double>& G_out,
    const std::vector<double>& L_out) const
{
    assert(h.size() == QUERY_DIM);
    assert(G_out.size() == OUTPUT_DIM);
    assert(L_out.size() == OUTPUT_DIM);

    ConvergenceResult result;
    result.agreement = compute_agreement(G_out, L_out);
    result.mode = check_ignition(result.agreement);

    if (result.mode == IgnitionMode::FAST) {
        // Systems agree — use global prediction, skip gate
        result.fused = G_out;
        result.gate_values.assign(OUTPUT_DIM, 0.0);
        return result;
    }

    // Compute gate: γ = σ(W_gate · h + b_gate)
    result.gate_values.resize(OUTPUT_DIM);
    for (size_t i = 0; i < OUTPUT_DIM; ++i) {
        double z = b_gate_[i];
        for (size_t j = 0; j < QUERY_DIM; ++j) {
            z += W_gate_[i * QUERY_DIM + j] * h[j];
        }
        result.gate_values[i] = sigmoid(z);
    }

    // Fused = G(h) + γ ⊙ (L(h) - G(h))
    result.fused.resize(OUTPUT_DIM);
    for (size_t i = 0; i < OUTPUT_DIM; ++i) {
        double epsilon = L_out[i] - G_out[i];
        result.fused[i] = G_out[i] + result.gate_values[i] * epsilon;
    }

    return result;
}

float GatedResidualPoE::compute_agreement(
    const std::vector<double>& G_out,
    const std::vector<double>& L_out)
{
    assert(G_out.size() == L_out.size());

    // agreement = 1 - ||ε|| / (||G|| + ||L||)
    double norm_G = vector_norm(G_out);
    double norm_L = vector_norm(L_out);

    if (norm_G + norm_L < 1e-10) return 1.0f;  // Both zero → perfect agreement

    std::vector<double> epsilon(G_out.size());
    for (size_t i = 0; i < G_out.size(); ++i) {
        epsilon[i] = L_out[i] - G_out[i];
    }
    double norm_eps = vector_norm(epsilon);

    float agreement = static_cast<float>(1.0 - norm_eps / (norm_G + norm_L));
    return std::max(0.0f, std::min(1.0f, agreement));
}

IgnitionMode GatedResidualPoE::check_ignition(float agreement) {
    if (agreement > IGNITION_FAST) return IgnitionMode::FAST;
    if (agreement > IGNITION_DELIBERATE) return IgnitionMode::DELIBERATE;
    return IgnitionMode::CONFLICT;
}

GatedResidualPoE::BackwardResult GatedResidualPoE::backward(
    const std::vector<double>& h,
    const std::vector<double>& G_out,
    const std::vector<double>& L_out,
    const std::vector<double>& gate_values,
    const std::vector<double>& d_fused,
    double lr)
{
    BackwardResult result;
    result.d_h.resize(QUERY_DIM, 0.0);
    result.d_G.resize(OUTPUT_DIM);
    result.d_L.resize(OUTPUT_DIM);

    // fused_i = G_i + γ_i * (L_i - G_i)
    // d_fused/d_G_i = 1 - γ_i
    // d_fused/d_L_i = γ_i
    // d_fused/d_γ_i = L_i - G_i = ε_i

    for (size_t i = 0; i < OUTPUT_DIM; ++i) {
        double gamma = gate_values[i];
        double epsilon = L_out[i] - G_out[i];

        result.d_G[i] = d_fused[i] * (1.0 - gamma);
        result.d_L[i] = d_fused[i] * gamma;

        // d_fused/d_γ_i * d_γ/d_z_i where z = W·h+b, dγ/dz = γ(1-γ)
        double d_gamma = d_fused[i] * epsilon;
        double d_z = d_gamma * gamma * (1.0 - gamma);

        // Update gate weights
        for (size_t j = 0; j < QUERY_DIM; ++j) {
            W_gate_[i * QUERY_DIM + j] -= lr * d_z * h[j];
            result.d_h[j] += d_z * W_gate_[i * QUERY_DIM + j];
        }
        b_gate_[i] -= lr * d_z;
    }

    return result;
}

void GatedResidualPoE::init_bias_from_precision(
    double local_precision, double global_precision)
{
    if (global_precision <= 0 || local_precision <= 0) return;
    double bias = std::log(local_precision / global_precision);
    std::fill(b_gate_.begin(), b_gate_.end(), bias);
}

double GatedResidualPoE::sigmoid(double x) {
    if (x >= 0) {
        double ez = std::exp(-x);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
}

double GatedResidualPoE::vector_norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

} // namespace convergence
} // namespace brain19
