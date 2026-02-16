// libtorch/torch_kan.hpp — EfficientKAN + DeepKANv2 Decoder as torch::nn::Module
// LibTorch autograd-based KAN with CM-Feedback-Port (Convergence v2)
#pragma once

#include <torch/torch.h>
#include <cstddef>

namespace brain19 {
namespace libtorch {

// =============================================================================
// EfficientKANLayer — B-spline + residual + LayerNorm
// =============================================================================

struct EfficientKANLayerImpl : torch::nn::Module {
    EfficientKANLayerImpl(size_t in_dim, size_t out_dim,
                          size_t grid_size = 5, size_t spline_order = 3);

    torch::Tensor compute_basis(torch::Tensor x);
    torch::Tensor forward(torch::Tensor x);

    size_t in_dim_, out_dim_, grid_size_, spline_order_, basis_size_;

    torch::Tensor spline_weights;   // [out_dim, in_dim * basis_size], parameter
    torch::Tensor residual_W;       // [in_dim, out_dim], parameter
    torch::Tensor ln_gamma;         // [out_dim], parameter
    torch::Tensor ln_beta;          // [out_dim], parameter
    torch::Tensor knots;            // [n_knots], buffer (not trained)
};
TORCH_MODULE(EfficientKANLayer);

// =============================================================================
// DeepKANv2Decoder — 3 KAN layers + Shared ConvergencePort + Output
// =============================================================================
//
// Pipeline:
//   h ∈ ℝ^90 (precomputed, Blocks 1-3 only)
//   → KAN L1 (90→256, G=8) → k1
//   → Projection (256→32) → k1_proj
//   → ConvergencePort: cm = tanh(conv_linear(cat(h, k1_proj, conv_emb(tok))))
//       conv_emb: Embedding(V, 16)       — token identity
//       conv_linear: Linear(122+16, 32)  — SHARED across all tokens
//   → KAN L2 (cat(k1, cm)=288→128, G=5)
//   → KAN L3 (128→128, G=5)
//   → Linear (128→VA) → logits
//

static constexpr size_t CONV_EMB_DIM = 16;
static constexpr size_t CONV_INPUT_DIM = 122;  // 90 + 32
static constexpr size_t CONV_OUTPUT_DIM = 32;

struct DeepKANv2DecoderImpl : torch::nn::Module {
    DeepKANv2DecoderImpl(size_t vocab_active, size_t vocab_total,
                          double dropout_p = 0.15);

    // Token forward: h=[B,90], tok_ids=[B] -> logits=[B,VA]
    torch::Tensor forward(torch::Tensor h, torch::Tensor tok_ids);

    // Concept forward: h=[B,90], concept_emb=[B,16] -> logits=[B,N_concepts]
    // concept_emb is the current concept's 16D core embedding (replaces conv_emb lookup)
    // Gradient flows through: concept_proj → KAN L3 → KAN L2 → CM → KAN L1
    torch::Tensor forward_concepts(torch::Tensor h, torch::Tensor concept_emb_16d);

    // Set concept embedding matrix for cosine similarity output
    void set_concept_matrix(torch::Tensor matrix, float temperature);

    EfficientKANLayer kan_l1{nullptr};          // 90 → 256, G=8
    torch::nn::Dropout drop1{nullptr};          // after L1
    torch::nn::Linear k1_proj{nullptr};         // 256 → 32
    // Shared ConvergencePort:
    torch::nn::Embedding conv_emb{nullptr};     // V → 16
    torch::nn::Linear conv_linear{nullptr};     // 138 (122+16) → 32
    torch::nn::Dropout drop_cm{nullptr};        // after ConvergencePort
    EfficientKANLayer kan_l2{nullptr};          // 288 → 128, G=5
    torch::nn::Dropout drop2{nullptr};          // after L2
    EfficientKANLayer kan_l3{nullptr};          // 128 → 128, G=5
    torch::nn::Dropout drop3{nullptr};          // after L3
    torch::nn::Linear output{nullptr};          // 128 → VA (token prediction)

    // Concept prediction head: 128 → 16 (concept embedding space)
    torch::nn::Linear concept_proj{nullptr};    // 128 → 16
    torch::Tensor concept_matrix_;              // [N_concepts, 16] L2-normed, buffer
    float concept_temperature_ = 0.1f;

    size_t VA_;
    size_t V_total_;
    double dropout_p_;
};
TORCH_MODULE(DeepKANv2Decoder);

} // namespace libtorch
} // namespace brain19
