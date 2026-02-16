// cuda/cuda_training.h — CUDA-accelerated SGD training loop
// CPU-safe header: works with or without USE_CUDA
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace brain19 {
namespace cuda {

struct TrainingData {
    // Flattened sample tokens: all tokens concatenated
    std::vector<uint16_t> all_tokens;
    // Per-sample: start offset into all_tokens
    std::vector<size_t> sample_offsets;
    // Per-sample: number of tokens
    std::vector<size_t> sample_lengths;
    // Per-sample: embedding (flattened, H per sample)
    std::vector<double> embeddings;  // [num_samples * H]
    // Token compression map: original_token -> compressed index (VA)
    std::vector<size_t> compress;    // [V]
    // Active token list
    std::vector<uint16_t> active_tokens;
    // Embedding table (flattened): [V * FUSED_BASE]
    std::vector<double> emb_table;
    // FlexDetail table (flattened): [V * flex_dim], precomputed per token
    std::vector<double> flex_table;
    // ConvergencePort table (flattened): [V * conv_dim], precomputed per token
    std::vector<double> conv_table;

    size_t num_samples;
    size_t V;          // vocab size
    size_t VA;         // active vocab size
    size_t H;          // extended fused dim (before quadratic)
    size_t H_EXT;      // 2*H (with quadratic features)
    size_t K;          // transform bottleneck dim
    size_t FUSED_BASE; // base fused dim (for hidden state evolution)
    size_t flex_dim;   // v11: 16 (FlexDetail dims), v10: 0
    size_t conv_dim;   // v12: 32 (ConvergencePort dims), 0 = disabled
};

struct TrainingWeights {
    // All flattened row-major
    std::vector<double> W_a;    // [H_EXT * VA]
    std::vector<double> W1;     // [H * K]
    std::vector<double> b1;     // [K]
    std::vector<double> W2;     // [K * H]
    std::vector<double> b2;     // [H]
};

struct TrainingConfig {
    size_t num_epochs;
    double base_lr;
    double lr_transform_base;
    size_t transform_warmup;  // freeze transform for first N epochs
};

struct TrainingResult {
    double best_loss;
    // Weights are updated in-place in TrainingWeights
};

// Run the full SGD training loop on GPU.
// Returns true if GPU was used, false if not available.
bool train_sgd_gpu(const TrainingData& data,
                   TrainingWeights& weights,
                   const TrainingConfig& config,
                   TrainingResult& result);

// V11 3-block SM-parallel training: each dimension block on its own SM.
// Block A (dims 0..63, LR×1.0), Block B (dims 64..79, LR×0.3), Block C (dims 80..H-1, LR×0.1)
// Requires flex_table and flex_dim to be set in TrainingData.
bool train_sgd_v11_gpu(const TrainingData& data,
                       TrainingWeights& weights,
                       const TrainingConfig& config,
                       TrainingResult& result);

// =============================================================================
// V12 Deep KAN Training — 3-layer EfficientKAN + Linear→VA
// V12:    90→256→128→128 (L2 input=256)
// V12v2:  90→256→(CM-Feedback-Port)→128→128 (L2 input=288, via LibTorch)
// =============================================================================

struct DeepKANWeights {
    // Layer 1: 90→256, G=8, k=3, basis_size=11
    std::vector<double> k1_weights;    // [256 * (90*11)]
    std::vector<double> k1_residual;   // [90 * 256]
    std::vector<double> k1_gamma;      // [256]
    std::vector<double> k1_beta;       // [256]
    std::vector<double> k1_knots;      // [15]

    // Layer 2: v12=256→128, v2=288→128, G=5, k=3, basis_size=8
    std::vector<double> k2_weights;    // [128 * (in*8)]
    std::vector<double> k2_residual;   // [in * 128]
    std::vector<double> k2_gamma;      // [128]
    std::vector<double> k2_beta;       // [128]
    std::vector<double> k2_knots;      // [12]

    // Layer 3: 128→128, G=5, k=3, basis_size=8
    std::vector<double> k3_weights;    // [128 * (128*8)]
    std::vector<double> k3_residual;   // [128 * 128]
    std::vector<double> k3_gamma;      // [128]
    std::vector<double> k3_beta;       // [128]
    std::vector<double> k3_knots;      // [12]

    // Output projection: [128 * VA]
    std::vector<double> W_a;
};

struct DeepKANConfig {
    size_t num_epochs;
    double lr_output;              // W_a LR (2.0)
    double lr_kan;                 // KAN weight LR (0.01)
    size_t warmup_epochs;          // freeze KAN backward for first N epochs
    std::vector<double> lr_scale;  // [H] per-input-dim LR scale for Layer 1
};

// V12 Deep KAN GPU training.
// Returns true if GPU was used, false if not available.
bool train_deep_kan_gpu(const TrainingData& data,
                        DeepKANWeights& weights,
                        const DeepKANConfig& config,
                        TrainingResult& result);

} // namespace cuda
} // namespace brain19
