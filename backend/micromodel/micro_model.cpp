#include "micro_model.hpp"

#include <algorithm>
#include <numeric>

namespace brain19 {

// =============================================================================
// Construction
// =============================================================================

MicroModel::MicroModel() {
    // Initialize W with small values (Xavier-like for 16x16)
    // Use deterministic pattern: W[i][j] = 0.1 * (i == j) + 0.01 * sin(i*10+j)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        for (size_t j = 0; j < CORE_DIM; ++j) {
            double diag = (i == j) ? 0.1 : 0.0;
            double off = 0.01 * std::sin(static_cast<double>(i * 10 + j));
            W_[i * CORE_DIM + j] = diag + off;
        }
    }

    // Bias initialized to small values
    for (size_t i = 0; i < CORE_DIM; ++i) {
        b_[i] = 0.01 * std::cos(static_cast<double>(i));
    }

    // Default embeddings (small nonzero values for symmetry breaking)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        e_init_[i] = 0.1 * std::sin(static_cast<double>(i * 3 + 1));
        c_init_[i] = 0.1 * std::cos(static_cast<double>(i * 7 + 2));
    }
}

// =============================================================================
// Forward pass — operates on Core dimensions only
// =============================================================================

double MicroModel::predict(const FlexEmbedding& e, const FlexEmbedding& c) const {
    // v = W·c_core + b
    CoreVec v;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = b_[i];
        for (size_t j = 0; j < CORE_DIM; ++j) {
            sum += W_[i * CORE_DIM + j] * c.core[j];
        }
        v[i] = sum;
    }

    // z = e_core^T · v (dot product)
    double z = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        z += e.core[i] * v[i];
    }

    // w = sigma(z)
    return sigmoid(z);
}

// =============================================================================
// Training step (Adam optimizer)
// =============================================================================

double MicroModel::train_step(const FlexEmbedding& e, const FlexEmbedding& c, double target,
                               const MicroTrainingConfig& config) {
    // --- Forward pass ---
    CoreVec v;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double sum = b_[i];
        for (size_t j = 0; j < CORE_DIM; ++j) {
            sum += W_[i * CORE_DIM + j] * c.core[j];
        }
        v[i] = sum;
    }

    double z = 0.0;
    for (size_t i = 0; i < CORE_DIM; ++i) {
        z += e.core[i] * v[i];
    }

    double w = sigmoid(z);

    // --- Loss (MSE) ---
    double error = w - target;
    double loss = 0.5 * error * error;

    // --- Backward pass ---
    // delta = (w - target) * w * (1 - w)   [sigmoid derivative chain rule]
    double delta = error * w * (1.0 - w);

    // Gradient for W: dL/dW[i,j] = delta * e[i] * c[j]
    // Gradient for b: dL/db[i] = delta * e[i]
    state_.timestep += 1.0;
    double t = state_.timestep;

    double beta1 = config.adam_beta1;
    double beta2 = config.adam_beta2;
    double eps = config.adam_epsilon;
    double lr = config.learning_rate;

    // Bias correction factors
    double bc1 = 1.0 - std::pow(beta1, t);
    double bc2 = 1.0 - std::pow(beta2, t);

    // Update W with Adam
    for (size_t i = 0; i < CORE_DIM; ++i) {
        double grad_b = delta * e.core[i];

        // Update bias momentum and variance
        state_.db_momentum[i] = beta1 * state_.db_momentum[i] + (1.0 - beta1) * grad_b;
        state_.db_variance[i] = beta2 * state_.db_variance[i] + (1.0 - beta2) * grad_b * grad_b;

        // Bias-corrected estimates
        double m_hat = state_.db_momentum[i] / bc1;
        double v_hat = state_.db_variance[i] / bc2;

        b_[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);

        for (size_t j = 0; j < CORE_DIM; ++j) {
            double grad_w = delta * e.core[i] * c.core[j];
            size_t idx = i * CORE_DIM + j;

            // Update momentum and variance
            state_.dW_momentum[idx] = beta1 * state_.dW_momentum[idx] + (1.0 - beta1) * grad_w;
            state_.dW_variance[idx] = beta2 * state_.dW_variance[idx] + (1.0 - beta2) * grad_w * grad_w;

            // Bias-corrected estimates
            double mw_hat = state_.dW_momentum[idx] / bc1;
            double vw_hat = state_.dW_variance[idx] / bc2;

            W_[idx] -= lr * mw_hat / (std::sqrt(vw_hat) + eps);
        }
    }

    state_.last_loss = loss;
    state_.total_samples += 1.0;
    if (loss < state_.best_loss) {
        state_.best_loss = loss;
    }

    return loss;
}

// =============================================================================
// Batch training
// =============================================================================

MicroTrainingResult MicroModel::train(const std::vector<TrainingSample>& samples,
                                  const MicroTrainingConfig& config) {
    MicroTrainingResult result;

    if (samples.empty()) {
        result.converged = true;
        return result;
    }

    double prev_avg_loss = 1e9;

    for (size_t epoch = 0; epoch < config.max_epochs; ++epoch) {
        double total_loss = 0.0;

        for (const auto& sample : samples) {
            total_loss += train_step(sample.relation_embedding,
                                     sample.context_embedding,
                                     sample.target, config);
        }

        double avg_loss = total_loss / static_cast<double>(samples.size());
        result.epochs_run = epoch + 1;
        result.final_loss = avg_loss;

        // Check convergence
        double improvement = std::abs(prev_avg_loss - avg_loss);
        if (improvement < config.convergence_threshold && epoch > 0) {
            result.converged = true;
            break;
        }

        prev_avg_loss = avg_loss;
    }

    return result;
}

// =============================================================================
// Serialization
// =============================================================================

void MicroModel::to_flat(std::array<double, FLAT_SIZE>& out) const {
    size_t idx = 0;

    // W (256)
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) {
        out[idx++] = W_[i];
    }
    // b (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        out[idx++] = b_[i];
    }
    // e_init (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        out[idx++] = e_init_[i];
    }
    // c_init (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        out[idx++] = c_init_[i];
    }

    // TrainingState
    // dW_momentum (256)
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) {
        out[idx++] = state_.dW_momentum[i];
    }
    // db_momentum (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        out[idx++] = state_.db_momentum[i];
    }
    // dW_variance (256)
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) {
        out[idx++] = state_.dW_variance[i];
    }
    // db_variance (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        out[idx++] = state_.db_variance[i];
    }
    // e_grad_accum (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        out[idx++] = state_.e_grad_accum[i];
    }
    // c_grad_accum (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        out[idx++] = state_.c_grad_accum[i];
    }
    // scalars (5)
    out[idx++] = state_.timestep;
    out[idx++] = state_.last_loss;
    out[idx++] = state_.best_loss;
    out[idx++] = state_.total_samples;
    out[idx++] = state_.reserved_scalar;
    // reserved (55)
    for (size_t i = 0; i < 55; ++i) {
        out[idx++] = state_.reserved[i];
    }
}

void MicroModel::from_flat(const std::array<double, FLAT_SIZE>& in) {
    size_t idx = 0;

    // W (256)
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) {
        W_[i] = in[idx++];
    }
    // b (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        b_[i] = in[idx++];
    }
    // e_init (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        e_init_[i] = in[idx++];
    }
    // c_init (16)
    for (size_t i = 0; i < CORE_DIM; ++i) {
        c_init_[i] = in[idx++];
    }

    // TrainingState
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) {
        state_.dW_momentum[i] = in[idx++];
    }
    for (size_t i = 0; i < CORE_DIM; ++i) {
        state_.db_momentum[i] = in[idx++];
    }
    for (size_t i = 0; i < CORE_DIM * CORE_DIM; ++i) {
        state_.dW_variance[i] = in[idx++];
    }
    for (size_t i = 0; i < CORE_DIM; ++i) {
        state_.db_variance[i] = in[idx++];
    }
    for (size_t i = 0; i < CORE_DIM; ++i) {
        state_.e_grad_accum[i] = in[idx++];
    }
    for (size_t i = 0; i < CORE_DIM; ++i) {
        state_.c_grad_accum[i] = in[idx++];
    }
    state_.timestep = in[idx++];
    state_.last_loss = in[idx++];
    state_.best_loss = in[idx++];
    state_.total_samples = in[idx++];
    state_.reserved_scalar = in[idx++];
    for (size_t i = 0; i < 55; ++i) {
        state_.reserved[i] = in[idx++];
    }
}

} // namespace brain19
