#pragma once

#include "flex_embedding.hpp"

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
// Forward pass:  v = W·c_core + b  (16D),  z = e_core^T·v  (scalar),  w = sigma(z) in (0,1)
//
// CONSTRAINTS:
// - No shared weights between models (each is fully independent)
// - Models point AT the KG (via ConceptIds), never store knowledge
// - No external dependencies: hand-written matmul, sigmoid, Adam
//

// Backward compatibility aliases
static constexpr size_t EMBED_DIM = CORE_DIM;  // 16 (was 10)

// MicroModel operates on CoreVec (16D) for its weight matrix
using CoreMat = std::array<double, CORE_DIM * CORE_DIM>;  // 256 elements, row-major

// Type alias: Vec10 -> FlexEmbedding for transition compatibility
using Vec10 = FlexEmbedding;

// Old alias (deprecated)
using Mat10x10 = CoreMat;

// New FLAT_SIZE for 16D:
// W(256) + b(16) + e_init(16) + c_init(16) +
// dW_momentum(256) + db_momentum(16) + dW_variance(256) + db_variance(16) +
// e_grad_accum(16) + c_grad_accum(16) + scalars(5) + reserved(55) = 940
static constexpr size_t FLAT_SIZE = 940;

// Training configuration
struct MicroTrainingConfig {
    double learning_rate = 0.01;
    size_t max_epochs = 500;
    double convergence_threshold = 1e-4;
    double adam_beta1 = 0.9;
    double adam_beta2 = 0.999;
    double adam_epsilon = 1e-8;

    MicroTrainingConfig() = default;
};

// Training sample: (relation_embedding, context_embedding, target_score)
struct TrainingSample {
    FlexEmbedding relation_embedding;
    FlexEmbedding context_embedding;
    double target;  // desired output in [0, 1]
};

// Result of a training run
struct MicroTrainingResult {
    size_t epochs_run = 0;
    double final_loss = 0.0;
    bool converged = false;
};

// Adam optimizer state for a single MicroModel
// Total: 636 doubles (256+16+256+16+16+16+5+55)
struct TrainingState {
    CoreMat  dW_momentum{};       // 256 - first moment of W gradients
    CoreVec  db_momentum{};       //  16 - first moment of b gradients
    CoreMat  dW_variance{};       // 256 - second moment of W gradients
    CoreVec  db_variance{};       //  16 - second moment of b gradients
    CoreVec  e_grad_accum{};      //  16 - accumulated e gradients (unused for now)
    CoreVec  c_grad_accum{};      //  16 - accumulated c gradients (unused for now)
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
    // Returns value in (0, 1). Operates on core dimensions only.
    double predict(const FlexEmbedding& e, const FlexEmbedding& c) const;

    // Single training step with Adam optimizer
    // Returns MSE loss for this sample
    double train_step(const FlexEmbedding& e, const FlexEmbedding& c, double target, const MicroTrainingConfig& config);

    // Train on a batch of samples for multiple epochs
    MicroTrainingResult train(const std::vector<TrainingSample>& samples, const MicroTrainingConfig& config);

    // Serialization to/from flat array (940 doubles)
    void to_flat(std::array<double, FLAT_SIZE>& out) const;
    void from_flat(const std::array<double, FLAT_SIZE>& in);

    // Direct access for testing
    const CoreMat& weights() const { return W_; }
    const CoreVec& bias() const { return b_; }
    const CoreVec& e_init() const { return e_init_; }
    const CoreVec& c_init() const { return c_init_; }

private:
    CoreMat W_;        // 256 params - weight matrix (16x16)
    CoreVec b_;        // 16 params  - bias vector
    CoreVec e_init_;   // 16 params  - default relation embedding
    CoreVec c_init_;   // 16 params  - default context embedding
    TrainingState state_;  // 636 params - optimizer state
};

} // namespace brain19
