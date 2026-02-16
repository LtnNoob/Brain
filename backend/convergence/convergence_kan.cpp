#include "convergence_kan.hpp"
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>

namespace brain19 {
namespace convergence {

ConvergenceKAN::ConvergenceKAN()
    : layer1_(QUERY_DIM, KAN_L1_OUT, KAN_L1_GRID, SPLINE_ORDER)
    , layer2_(KAN_L2_IN, KAN_L2_OUT, KAN_L2_GRID, SPLINE_ORDER)
    , layer3_(KAN_L2_OUT, KAN_L3_OUT, KAN_L3_GRID, SPLINE_ORDER)
    , proj_W_(KAN_PROJ_OUT * KAN_L1_OUT)
    , proj_b_(KAN_PROJ_OUT, 0.0)
{
    // Initialize projection with Xavier uniform
    std::mt19937 rng(42);
    double limit = std::sqrt(6.0 / (KAN_L1_OUT + KAN_PROJ_OUT));
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (auto& w : proj_W_) {
        w = dist(rng);
    }
}

// ─── Forward ─────────────────────────────────────────────────────────────────

std::vector<double> ConvergenceKAN::forward_layer1(const std::vector<double>& h) {
    assert(h.size() == QUERY_DIM);
    auto k1 = layer1_.forward(h, cache1_);
    cached_k1_ = k1;
    return k1;
}

std::vector<double> ConvergenceKAN::project_for_cm(const std::vector<double>& k1) const {
    assert(k1.size() == KAN_L1_OUT);

    // Linear: out = W @ k1 + b
    std::vector<double> proj(KAN_PROJ_OUT);
    for (size_t i = 0; i < KAN_PROJ_OUT; ++i) {
        double sum = proj_b_[i];
        for (size_t j = 0; j < KAN_L1_OUT; ++j) {
            sum += proj_W_[i * KAN_L1_OUT + j] * k1[j];
        }
        proj[i] = sum;
    }
    return proj;
}

std::vector<double> ConvergenceKAN::forward_layer2_3(
    const std::vector<double>& k1,
    const std::vector<double>& cm_output)
{
    assert(k1.size() == KAN_L1_OUT);
    assert(cm_output.size() == CM_OUTPUT_DIM);

    // Concatenate k1 ⊕ cm_output → [288]
    std::vector<double> l2_input;
    l2_input.reserve(KAN_L2_IN);
    l2_input.insert(l2_input.end(), k1.begin(), k1.end());
    l2_input.insert(l2_input.end(), cm_output.begin(), cm_output.end());
    cached_l2_input_ = l2_input;

    auto k2 = layer2_.forward(l2_input, cache2_);
    auto g_out = layer3_.forward(k2, cache3_);
    return g_out;
}

// ─── Backward ────────────────────────────────────────────────────────────────

ConvergenceKAN::BackwardResult ConvergenceKAN::backward_layer2_3(
    const std::vector<double>& d_output,
    double lr_l2, double lr_l3)
{
    // Backward through Layer 3
    auto d_k2 = layer3_.backward(cache3_, d_output, lr_l3);

    // Backward through Layer 2
    auto d_l2_input = layer2_.backward(cache2_, d_k2, lr_l2);

    // Split gradient: d_l2_input = [d_k1 (256) | d_cm (32)]
    BackwardResult result;
    result.d_k1.assign(d_l2_input.begin(), d_l2_input.begin() + KAN_L1_OUT);
    result.d_cm.assign(d_l2_input.begin() + KAN_L1_OUT, d_l2_input.end());
    return result;
}

std::vector<double> ConvergenceKAN::backward_layer1(
    const std::vector<double>& d_k1,
    double lr_l1)
{
    return layer1_.backward(cache1_, d_k1, lr_l1);
}

void ConvergenceKAN::backward_projection(
    const std::vector<double>& d_proj_out,
    double lr_proj)
{
    assert(d_proj_out.size() == KAN_PROJ_OUT);

    // d_W[i][j] = d_proj_out[i] * cached_k1_[j]
    // d_b[i] = d_proj_out[i]
    for (size_t i = 0; i < KAN_PROJ_OUT; ++i) {
        for (size_t j = 0; j < KAN_L1_OUT; ++j) {
            proj_W_[i * KAN_L1_OUT + j] -= lr_proj * d_proj_out[i] * cached_k1_[j];
        }
        proj_b_[i] -= lr_proj * d_proj_out[i];
    }
}

// ─── Accessors ───────────────────────────────────────────────────────────────

size_t ConvergenceKAN::num_params() const {
    return layer1_.num_params()
         + layer2_.num_params()
         + layer3_.num_params()
         + proj_W_.size() + proj_b_.size();
}

} // namespace convergence
} // namespace brain19
