#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace brain19 {

// =============================================================================
// BILINEAR MICRO-MODEL
// =============================================================================
//
// Each concept in the KG gets its own MicroModel that computes a personalized
// relevance score given a relation embedding (e) and a context embedding (c).
//
// Forward pass:  v = W·c + b  (10D),  z = eᵀ·v  (scalar),  w = σ(z) ∈ (0,1)
//
// CONSTRAINTS:
// - No shared weights between models (each is fully independent)
// - Models point AT the KG (via ConceptIds), never store knowledge
// - No external dependencies: hand-written matmul, sigmoid, Adam
//

static constexpr size_t EMBED_DIM = 10;
static constexpr size_t FLAT_SIZE = 430;  // 100 W + 10 b + 10 e_init + 10 c_init + 300 state

using Vec10 = std::array<double, EMBED_DIM>;
using Mat10x10 = std::array<double, EMBED_DIM * EMBED_DIM>;  // row-major

// Training configuration
struct TrainingConfig {
    double learning_rate = 0.01;
    size_t max_epochs = 100;
    double convergence_threshold = 1e-6;
    double adam_beta1 = 0.9;
    double adam_beta2 = 0.999;
    double adam_epsilon = 1e-8;

    TrainingConfig() = default;
};

// Training sample: (relation_embedding, context_embedding, target_score)
struct TrainingSample {
    Vec10 relation_embedding;
    Vec10 context_embedding;
    double target;  // desired output in [0, 1]
};

// Result of a training run
struct TrainingResult {
    size_t epochs_run = 0;
    double final_loss = 0.0;
    bool converged = false;
};

// Adam optimizer state for a single MicroModel
// Total: 300 doubles
struct TrainingState {
    Mat10x10 dW_momentum{};       // 100 - first moment of W gradients
    Vec10    db_momentum{};       //  10 - first moment of b gradients
    Mat10x10 dW_variance{};       // 100 - second moment of W gradients
    Vec10    db_variance{};       //  10 - second moment of b gradients
    Vec10    e_grad_accum{};      //  10 - accumulated e gradients (unused for now)
    Vec10    c_grad_accum{};      //  10 - accumulated c gradients (unused for now)
    // Scalars (5)
    double   timestep = 0.0;
    double   last_loss = 0.0;
    double   best_loss = 1e9;
    double   total_samples = 0.0;
    double   reserved_scalar = 0.0;
    // Reserved (55)
    std::array<double, 55> reserved{};

    TrainingState() {
        dW_momentum.fill(0.0);
        db_momentum.fill(0.0);
        dW_variance.fill(0.0);
        db_variance.fill(0.0);
        e_grad_accum.fill(0.0);
        c_grad_accum.fill(0.0);
        reserved.fill(0.0);
    }
};

// Sigmoid activation
inline double sigmoid(double x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    // Numerically stable for negative x
    double ex = std::exp(x);
    return ex / (1.0 + ex);
}

class MicroModel {
public:
    MicroModel();

    // Forward pass: predict relevance given relation and context embeddings
    // Returns value in (0, 1)
    double predict(const Vec10& e, const Vec10& c) const;

    // Single training step with Adam optimizer
    // Returns MSE loss for this sample
    double train_step(const Vec10& e, const Vec10& c, double target, const TrainingConfig& config);

    // Train on a batch of samples for multiple epochs
    TrainingResult train(const std::vector<TrainingSample>& samples, const TrainingConfig& config);

    // Serialization to/from flat array (430 doubles)
    void to_flat(std::array<double, FLAT_SIZE>& out) const;
    void from_flat(const std::array<double, FLAT_SIZE>& in);

    // Direct access for testing
    const Mat10x10& weights() const { return W_; }
    const Vec10& bias() const { return b_; }
    const Vec10& e_init() const { return e_init_; }
    const Vec10& c_init() const { return c_init_; }

private:
    Mat10x10 W_;       // 100 params - weight matrix
    Vec10 b_;          // 10 params  - bias vector
    Vec10 e_init_;     // 10 params  - default relation embedding
    Vec10 c_init_;     // 10 params  - default context embedding
    TrainingState state_;  // 300 params - optimizer state
};

} // namespace brain19
