// libtorch/torch_training.hpp — LibTorch Deep KAN v2 training interface
#pragma once

#include "../cuda/cuda_training.h"
#include <vector>
#include <cstddef>

namespace brain19 {
namespace libtorch {

// ConvergencePort data bridge (v2: shared W + token embedding)
// conv_emb_weights: [V * 16] token embeddings for ConvergencePort conditioning
// conv_linear_W: [32 * 138] shared linear weights
// conv_linear_b: [32] shared linear bias
struct ConvergencePortData {
    std::vector<double> conv_emb_weights;  // [V * 16]
    std::vector<double> conv_linear_W;     // [32 * 138]
    std::vector<double> conv_linear_b;     // [32]
};

struct DeepKANv2Config {
    size_t num_epochs = 300;
    double lr_output = 0.002;     // W_a (output projection)
    double lr_kan = 0.001;        // KAN spline + residual + LN + k1_proj
    double lr_conv = 0.0005;      // ConvergencePort (shared linear + embedding)
    size_t warmup_epochs = 10;    // freeze KAN+CM, train W_a only
    size_t batch_size = 2048;
    std::vector<double> lr_scale; // [90] per-input-dim LR scale for KAN L1
    double dropout_p = 0.08;      // dropout probability (0 = no dropout)
    double weight_decay = 0.01;   // L2 regularization (Adam weight decay)
    size_t patience = 25;         // early stopping: stop if val loss doesn't improve for N epochs
    double max_val_gap = 0.12;    // early stopping: stop if (val - train) > this after warmup
};

// Run Deep KAN v2 training with LibTorch (autograd + Adam).
// Returns true if training succeeded (GPU or CPU).
// Weights are updated in-place: dkw (KAN layers), cpd (CM convergence weights).
bool train_deep_kan_v2(const cuda::TrainingData& data,
                       cuda::DeepKANWeights& dkw,
                       ConvergencePortData& cpd,
                       const DeepKANv2Config& config,
                       cuda::TrainingResult& result);

// Autoregressive generation using trained DeepKAN v2 model.
// Returns generated token IDs (in original vocabulary space, not active-token space).
struct GenerateResult {
    std::vector<uint16_t> tokens;
    std::vector<double> probs;      // softmax probability per token
};

GenerateResult generate_deep_kan_v2(
    const cuda::DeepKANWeights& dkw,
    const ConvergencePortData& cpd,
    size_t VA, size_t V,
    const std::vector<double>& emb_table,   // [V * FUSED_BASE]
    const std::vector<double>& flex_table,  // [V * flex_dim]
    size_t FUSED_BASE, size_t flex_dim,
    const std::vector<uint16_t>& active_tokens,  // [VA] active_idx → token_id
    const std::vector<float>& initial_h,    // [90] initial hidden state
    uint16_t start_token,                   // initial token for conv_emb
    size_t max_tokens);

// =============================================================================
// Concept Prediction — DeepKAN v2 with concept output head
// =============================================================================
// Concept loss flows through: concept_proj → KAN L3 → KAN L2 → conv_linear → KAN L1
// Same gradient path as token prediction, but target is concept-ID (cosine similarity)
// instead of token-ID (softmax over VA).

struct ConceptTrainingData {
    // Concept embedding matrix: [num_concepts × 32], L2-normalized rows
    // First 16D = core embedding, next 16D = detail embedding
    std::vector<double> concept_matrix;     // [num_concepts * 32]

    // For hidden state evolution (concept→h feedback):
    std::vector<double> concept_emb_64d;    // [num_concepts * 64] Block 1
    std::vector<double> concept_flex_16d;   // [num_concepts * 16] Block 2

    // Per-sample data (one sample = one source concept with target sequence)
    std::vector<double> initial_h;          // [num_samples * 90]
    std::vector<int64_t> concept_seqs;      // flat concept index sequences
    std::vector<size_t> seq_offsets;         // [num_samples] start offsets
    std::vector<size_t> seq_lengths;         // [num_samples] lengths
    std::vector<double> trust_weights;      // [num_samples] epistemic trust

    size_t num_samples;
    size_t num_concepts;
};

// Concept-specific weights (concept_proj + k1_proj, not in DeepKANWeights)
struct ConceptWeights {
    std::vector<double> concept_proj_W;     // [32 * 128]
    std::vector<double> k1_proj_W;          // [32 * 256]
    std::vector<double> k1_proj_b;          // [32]
};

// Train concept prediction through DeepKAN v2 backbone.
// Trust-weighted cross-entropy on concept prediction targets.
bool train_concept_deep_kan_v2(const ConceptTrainingData& data,
                                cuda::DeepKANWeights& dkw,
                                ConvergencePortData& cpd,
                                ConceptWeights& cw,
                                const DeepKANv2Config& config,
                                float concept_temperature,
                                cuda::TrainingResult& result);

// Concept generation result
struct ConceptGenerateResult {
    std::vector<int64_t> concept_indices;
    std::vector<double> confidences;
};

// Autoregressive concept generation using trained model.
ConceptGenerateResult generate_concept_deep_kan_v2(
    const cuda::DeepKANWeights& dkw,
    const ConvergencePortData& cpd,
    const ConceptWeights& cw,
    const std::vector<double>& concept_matrix,   // [N * 32]
    const std::vector<double>& concept_emb_64d,  // [N * 64]
    const std::vector<double>& concept_flex_16d,  // [N * 16]
    size_t num_concepts,
    const std::vector<float>& initial_h,          // [90]
    int64_t start_concept_idx,
    size_t max_concepts,
    float temperature = 0.1f);

} // namespace libtorch
} // namespace brain19
